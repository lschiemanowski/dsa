"""One small native MLflow evaluation path for canonical DSA runs."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import deepcopy
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, cast

from pydantic import Field, JsonValue, TypeAdapter, model_validator
from pydantic_ai.models import Model

from dsa.contract import (
    ContractModel,
    DerivationRequest,
    ModelConfiguration,
    RunPolicy,
    RunRequest,
)
from dsa.environment import PythonExecutor
from dsa.mlflow_config import mlflow_configuration_failure
from dsa.pack import (
    EvaluationPackCase,
    ExactJsonScorer,
    JsonNumericToleranceScorer,
    LoadedEvaluationPack,
    PackScorer,
)
from dsa.record import FailureStage, RunSuccess
from dsa.reporting import MlflowReporting
from dsa.runner import RunCompletion, run_analysis


class MlflowEvaluationPrediction(ContractModel):
    """Bounded scorer input projected from one completed DSA run."""

    case_id: str = Field(pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    run_id: str = Field(pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    terminal_sha256: str = Field(pattern=r"[0-9a-f]{64}")
    accepted: bool
    answer: JsonValue = None
    derivation_sha256: str | None = Field(
        default=None,
        pattern=r"[0-9a-f]{64}",
        exclude_if=lambda value: value is None,
    )
    failure_stage: FailureStage | None = None
    failure_code: str | None = None
    reporting: MlflowReporting

    @model_validator(mode="after")
    def validate_outcome_projection(self) -> MlflowEvaluationPrediction:
        if self.accepted:
            if self.failure_stage is not None or self.failure_code is not None:
                raise ValueError("accepted predictions must not contain failure state")
        elif (
            self.failure_stage is None
            or not self.failure_code
            or self.answer is not None
            or self.derivation_sha256 is not None
        ):
            raise ValueError("failed predictions require only classified failure state")
        return self


class MlflowEvaluationResult(ContractModel):
    """Stable identities returned by one native MLflow evaluation."""

    dataset_id: str
    dataset_digest: str
    evaluation_run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    predictions: tuple[MlflowEvaluationPrediction, ...] = ()


class MlflowEvaluationError(RuntimeError):
    """Stable infrastructure failure at the native evaluation boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(f"MLflow evaluation failed ({code})")
        self.code = code


class _NativeDataset(Protocol):
    dataset_id: str
    digest: str

    def merge_records(self, records: list[dict[str, JsonValue]]) -> _NativeDataset: ...


class _EvaluationApi(Protocol):
    def create_dataset(self, *, name: str, experiment_id: str) -> _NativeDataset: ...

    def get_dataset(self, *, name: str) -> _NativeDataset: ...

    def scorer(self, function: Callable[..., object], *, name: str) -> object: ...

    def invalid_feedback(self, name: str, rationale: str) -> object: ...

    def evaluation_context(
        self,
        tags: dict[str, str],
    ) -> AbstractContextManager[object]: ...

    def evaluate(
        self,
        *,
        data: _NativeDataset,
        predict_fn: Callable[..., object],
        scorers: list[object],
    ) -> object: ...


ModelFactory = Callable[[EvaluationPackCase], Model | str | None]
_PACK_SCORER_ADAPTER: TypeAdapter[PackScorer] = TypeAdapter(
    ExactJsonScorer | JsonNumericToleranceScorer
)


def run_mlflow_evaluation(
    pack: LoadedEvaluationPack | object,
    *,
    dataset_name: str,
    runs_directory: Path,
    model_configuration: ModelConfiguration | object,
    policy: RunPolicy | object,
    model_factory: ModelFactory | None = None,
    python_executor: PythonExecutor | None = None,
    api: _EvaluationApi | None = None,
    run_tags: dict[str, str] | None = None,
) -> MlflowEvaluationResult:
    """Create a native dataset from a verified pack and run ordinary DSA analyses."""
    canonical_pack = _canonical_pack(pack)
    canonical_cases = canonical_pack.cases
    if len({case.case_id for case in canonical_cases}) != len(canonical_cases):
        raise ValueError("evaluation case IDs must be unique")
    if len(canonical_cases) != canonical_pack.manifest.cases.case_count:
        raise ValueError("evaluation pack case count does not match its manifest")
    if not dataset_name.strip():
        raise ValueError("dataset_name must not be blank")
    canonical_model = _canonical_model_configuration(model_configuration)
    canonical_policy = _canonical_policy(policy)
    canonical_tags = _canonical_run_tags(run_tags)
    _verify_pack_database(canonical_pack)
    configuration_failure = _evaluation_configuration_failure()
    if configuration_failure is not None:
        raise MlflowEvaluationError(configuration_failure)
    experiment_id = os.environ["MLFLOW_EXPERIMENT_ID"]

    try:
        selected_api = api or _MlflowEvaluationApi()
        dataset = selected_api.create_dataset(
            name=dataset_name,
            experiment_id=experiment_id,
        )
        dataset = dataset.merge_records(
            [case.dataset_record(canonical_pack.manifest) for case in canonical_cases]
        )
        dataset = selected_api.get_dataset(name=dataset_name)
    except ModuleNotFoundError:
        raise MlflowEvaluationError("mlflow_dependency_missing") from None
    except Exception:
        raise MlflowEvaluationError("mlflow_dataset_failed") from None
    indexed = {case.case_id: case for case in canonical_cases}
    reserved: set[str] = set()
    captured: dict[str, MlflowEvaluationPrediction] = {}
    capture_lock = Lock()

    async def predict_fn(
        case_id: str,
        case_version: str,
        database_id: str,
        database_sha256: str,
        question: str,
        answer_schema: dict[str, JsonValue],
        derivation: dict[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        case = indexed.get(case_id)
        if case is None:
            raise ValueError("evaluation input contains an unknown case ID")
        if (
            case_version != case.case_version
            or database_id != canonical_pack.manifest.database.id
            or database_sha256 != canonical_pack.manifest.database.sha256
            or question != case.question
            or answer_schema != case.answer_schema
            or derivation != _derivation_input(case.derivation)
        ):
            raise ValueError("evaluation input does not match its verified pack case")
        with capture_lock:
            if case_id in reserved:
                raise ValueError("evaluation invoked a case more than once")
            reserved.add(case_id)
        canonical_request = RunRequest(
            database_path=canonical_pack.database_path,
            question=case.question,
            answer_schema=case.answer_schema,
            derivation=case.derivation,
            model=canonical_model,
            policy=canonical_policy,
        )
        selected_model = model_factory(case) if model_factory is not None else None
        completion = await run_analysis(
            canonical_request,
            runs_directory=runs_directory,
            model=selected_model,
            python_executor=python_executor,
            report_to_mlflow=True,
        )
        prediction = evaluation_prediction(case_id, completion)
        with capture_lock:
            captured[case_id] = prediction
        return prediction.model_dump(mode="json")

    scorers = _native_scorers(selected_api)
    try:
        evaluation_context = (
            selected_api.evaluation_context(canonical_tags)
            if canonical_tags
            else nullcontext()
        )
        with evaluation_context, _skip_mlflow_prediction_preflight():
            raw_result = selected_api.evaluate(
                data=dataset,
                predict_fn=predict_fn,
                scorers=scorers,
            )
    except Exception:
        raise MlflowEvaluationError("mlflow_evaluation_failed") from None
    metrics = {
        str(key): float(cast(int | float, value))
        for key, value in cast(dict[str, object], getattr(raw_result, "metrics", {})).items()
        if type(value) in (int, float) and math.isfinite(cast(float, value))
    }
    evaluation_run_id = getattr(raw_result, "run_id", None)
    with capture_lock:
        if set(captured) != set(indexed):
            raise MlflowEvaluationError("mlflow_prediction_set_mismatch")
        predictions = tuple(captured[case.case_id] for case in canonical_cases)
    return MlflowEvaluationResult(
        dataset_id=dataset.dataset_id,
        dataset_digest=dataset.digest,
        evaluation_run_id=(
            evaluation_run_id if isinstance(evaluation_run_id, str) else None
        ),
        metrics=metrics,
        predictions=predictions,
    )


def evaluation_prediction(
    case_id: str,
    completion: RunCompletion,
) -> MlflowEvaluationPrediction:
    """Project canonical completion state into bounded evaluation output."""
    if isinstance(completion.outcome, RunSuccess):
        verification = completion.outcome.derivation_verification
        return MlflowEvaluationPrediction(
            case_id=case_id,
            run_id=completion.record.run_id,
            terminal_sha256=completion.retained_record.sha256,
            accepted=True,
            answer=deepcopy(completion.outcome.answer),
            derivation_sha256=(
                verification.derivation_sha256 if verification is not None else None
            ),
            reporting=completion.reporting,
        )
    return MlflowEvaluationPrediction(
        case_id=case_id,
        run_id=completion.record.run_id,
        terminal_sha256=completion.retained_record.sha256,
        accepted=False,
        failure_stage=completion.outcome.failure.stage,
        failure_code=completion.outcome.failure.code,
        reporting=completion.reporting,
    )


def _derivation_input(
    value: DerivationRequest | None,
) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    return cast(dict[str, JsonValue], value.model_dump(mode="json"))


def end_to_end_exact_success(
    outputs: MlflowEvaluationPrediction | dict[str, JsonValue],
    expectations: dict[str, JsonValue],
) -> bool:
    prediction = _prediction(outputs)
    expected, _scorer = _decode_scoring_expectations(expectations)
    return (
        prediction.accepted
        and exact_json_equal(prediction.answer, expected)
        and prediction.reporting.status == "reported"
    )


def conditional_exact_json(
    outputs: MlflowEvaluationPrediction | dict[str, JsonValue],
    expectations: dict[str, JsonValue],
) -> bool | None:
    prediction = _prediction(outputs)
    if not prediction.accepted:
        return None
    expected, _scorer = _decode_scoring_expectations(expectations)
    return exact_json_equal(prediction.answer, expected)


def end_to_end_policy_success(
    outputs: MlflowEvaluationPrediction | dict[str, JsonValue],
    expectations: dict[str, JsonValue],
) -> bool:
    """Score accepted, reported answers under the pack comparison policy."""
    prediction = _prediction(outputs)
    return (
        prediction.accepted
        and json_answers_equal(prediction.answer, expectations)
        and prediction.reporting.status == "reported"
    )


def conditional_policy_match(
    outputs: MlflowEvaluationPrediction | dict[str, JsonValue],
    expectations: dict[str, JsonValue],
) -> bool | None:
    """Score accepted answers under the pack policy without failures in the denominator."""
    prediction = _prediction(outputs)
    if not prediction.accepted:
        return None
    return json_answers_equal(prediction.answer, expectations)


def agent_failure(
    outputs: MlflowEvaluationPrediction | dict[str, JsonValue],
    expectations: dict[str, JsonValue],
) -> bool:
    prediction = _prediction(outputs)
    if infrastructure_failure(prediction):
        return False
    if not prediction.accepted:
        return True
    return not json_answers_equal(prediction.answer, expectations)


def infrastructure_failure(
    outputs: MlflowEvaluationPrediction | dict[str, JsonValue],
) -> bool:
    prediction = _prediction(outputs)
    if prediction.reporting.status != "reported":
        return True
    if prediction.accepted:
        return False
    if prediction.failure_stage in {"analysis_environment", "cancelled"}:
        return True
    if prediction.failure_stage == "model" and prediction.failure_code in {
        "model_api_error",
        "model_http_error",
    }:
        return True
    return prediction.failure_stage == "orchestration" and prediction.failure_code in {
        "internal_error",
        "run_timeout",
        "tool_result_limit_exceeded",
        "usage_limit_exceeded",
    }


def exact_json_equal(left: JsonValue, right: JsonValue) -> bool:
    """Compare finite JSON structurally while preserving numeric JSON values."""
    return _canonical_json(left) == _canonical_json(right)


def json_answers_equal(
    actual: JsonValue,
    expectations: dict[str, JsonValue],
) -> bool:
    """Compare one answer under the canonical policy retained in the dataset."""
    expected, scorer = _decode_scoring_expectations(expectations)
    if isinstance(scorer, JsonNumericToleranceScorer):
        return _tolerant_json_equal(actual, expected, scorer)
    return exact_json_equal(actual, expected)


def _decode_scoring_expectations(
    value: dict[str, JsonValue],
) -> tuple[JsonValue, PackScorer]:
    if set(value) != {"answer", "scorer"}:
        raise ValueError("evaluation expectations have an invalid shape")
    answer_text = value["answer"]
    scorer_text = value["scorer"]
    if not isinstance(answer_text, str) or not isinstance(scorer_text, str):
        raise ValueError("evaluation expectations must be canonical JSON strings")
    try:
        answer = cast(JsonValue, json.loads(answer_text))
        scorer_value = json.loads(scorer_text)
        scorer = _PACK_SCORER_ADAPTER.validate_python(scorer_value)
    except Exception:
        raise ValueError("evaluation expectations are invalid") from None
    if answer_text != _canonical_json(answer) or scorer_text != _canonical_json(
        cast(JsonValue, scorer.model_dump(mode="json"))
    ):
        raise ValueError("evaluation expectations must be canonical")
    return answer, scorer


def _tolerant_json_equal(
    actual: JsonValue,
    expected: JsonValue,
    scorer: JsonNumericToleranceScorer,
) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            _tolerant_json_equal(actual[key], expected[key], scorer) for key in expected
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _tolerant_json_equal(left, right, scorer)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if type(expected) is int:
        return type(actual) is int and actual == expected
    if type(expected) is float:
        if type(actual) not in (int, float):
            return False
        actual_number = cast(int | float, actual)
        try:
            return math.isfinite(actual_number) and math.isclose(
                actual_number,
                expected,
                rel_tol=scorer.relative_tolerance,
                abs_tol=scorer.absolute_tolerance,
            )
        except OverflowError:
            return False
    return type(actual) is type(expected) and actual == expected


def _canonical_pack(value: LoadedEvaluationPack | object) -> LoadedEvaluationPack:
    if isinstance(value, LoadedEvaluationPack):
        return LoadedEvaluationPack.model_validate_json(value.model_dump_json())
    return LoadedEvaluationPack.model_validate(value)


def _canonical_model_configuration(
    value: ModelConfiguration | object,
) -> ModelConfiguration:
    if isinstance(value, ModelConfiguration):
        return ModelConfiguration.model_validate_json(value.model_dump_json())
    return ModelConfiguration.model_validate(value)


def _canonical_policy(value: RunPolicy | object) -> RunPolicy:
    if isinstance(value, RunPolicy):
        return RunPolicy.model_validate_json(value.model_dump_json())
    return RunPolicy.model_validate(value)


def _verify_pack_database(pack: LoadedEvaluationPack) -> None:
    expected = pack.manifest.database
    try:
        with pack.database_path.open("rb") as source:
            if os.fstat(source.fileno()).st_size != expected.size_bytes:
                raise MlflowEvaluationError("evaluation_database_mismatch")
            digest = sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except MlflowEvaluationError:
        raise
    except OSError:
        raise MlflowEvaluationError("evaluation_database_unavailable") from None
    if digest.hexdigest() != expected.sha256:
        raise MlflowEvaluationError("evaluation_database_mismatch")


def _evaluation_configuration_failure() -> str | None:
    return mlflow_configuration_failure(os.environ)


def _canonical_run_tags(value: dict[str, str] | None) -> dict[str, str]:
    if value is None:
        return {}
    tags: dict[str, str] = {}
    safe_value_characters = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    for key in sorted(value):
        selected = value[key]
        if (
            not key.startswith("dsa.benchmark.")
            or len(key) > 128
            or not selected
            or len(selected) > 256
            or any(character not in safe_value_characters for character in selected)
        ):
            raise ValueError("evaluation run tags must be safe benchmark identities")
        tags[key] = selected
    return tags


@contextmanager
def _skip_mlflow_prediction_preflight() -> Generator[None]:
    """Prevent MLflow from executing a side-effecting prediction as validation."""
    key = "MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"
    original = os.environ.get(key)
    os.environ[key] = "true"
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


def _prediction(
    value: MlflowEvaluationPrediction | dict[str, JsonValue],
) -> MlflowEvaluationPrediction:
    data = (
        value.model_dump(mode="python")
        if isinstance(value, MlflowEvaluationPrediction)
        else value
    )
    return MlflowEvaluationPrediction.model_validate(data)


def _canonical_json(value: JsonValue) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _native_scorers(api: _EvaluationApi) -> list[object]:
    def score_end_to_end(
        outputs: dict[str, JsonValue], expectations: dict[str, JsonValue]
    ) -> bool:
        return end_to_end_exact_success(outputs, expectations)

    def score_conditional(
        outputs: dict[str, JsonValue], expectations: dict[str, JsonValue]
    ) -> object:
        value = conditional_exact_json(outputs, expectations)
        if value is None:
            return api.invalid_feedback(
                "conditional_exact_json",
                "The analysis did not produce an accepted answer",
            )
        return value

    def score_agent_failure(
        outputs: dict[str, JsonValue], expectations: dict[str, JsonValue]
    ) -> bool:
        return agent_failure(outputs, expectations)

    def score_end_to_end_policy(
        outputs: dict[str, JsonValue], expectations: dict[str, JsonValue]
    ) -> bool:
        return end_to_end_policy_success(outputs, expectations)

    def score_conditional_policy(
        outputs: dict[str, JsonValue], expectations: dict[str, JsonValue]
    ) -> object:
        value = conditional_policy_match(outputs, expectations)
        if value is None:
            return api.invalid_feedback(
                "conditional_policy_match",
                "The analysis did not produce an accepted answer",
            )
        return value

    def score_infrastructure_failure(outputs: dict[str, JsonValue]) -> bool:
        return infrastructure_failure(outputs)

    return [
        api.scorer(score_end_to_end, name="end_to_end_exact_success"),
        api.scorer(score_conditional, name="conditional_exact_json"),
        api.scorer(score_end_to_end_policy, name="end_to_end_policy_success"),
        api.scorer(score_conditional_policy, name="conditional_policy_match"),
        api.scorer(score_agent_failure, name="agent_failure"),
        api.scorer(score_infrastructure_failure, name="infrastructure_failure"),
    ]


class _MlflowEvaluationApi:
    def __init__(self) -> None:
        self.mlflow: Any = import_module("mlflow")
        self.datasets: Any = import_module("mlflow.genai.datasets")
        self.genai: Any = import_module("mlflow.genai")
        self.scorers: Any = import_module("mlflow.genai.scorers")
        entities: Any = import_module("mlflow.entities")
        self.feedback_type: Any = entities.Feedback

    def create_dataset(self, *, name: str, experiment_id: str) -> _NativeDataset:
        try:
            existing = self.datasets.get_dataset(name=name)
        except Exception as error:
            if not _dataset_is_missing(error):
                raise
        else:
            return cast(_NativeDataset, existing)
        try:
            created = self.datasets.create_dataset(
                name=name,
                experiment_id=experiment_id,
            )
        except Exception as create_error:
            try:
                created = self.datasets.get_dataset(name=name)
            except Exception:
                raise create_error from None
        return cast(_NativeDataset, created)

    def get_dataset(self, *, name: str) -> _NativeDataset:
        return cast(_NativeDataset, self.datasets.get_dataset(name=name))

    def scorer(self, function: Callable[..., object], *, name: str) -> object:
        return self.scorers.scorer(function, name=name)

    def invalid_feedback(self, name: str, rationale: str) -> object:
        return self.feedback_type(name=name, value=None, rationale=rationale, valid=False)

    def evaluation_context(
        self,
        tags: dict[str, str],
    ) -> AbstractContextManager[object]:
        return cast(AbstractContextManager[object], self.mlflow.start_run(tags=tags))

    def evaluate(
        self,
        *,
        data: _NativeDataset,
        predict_fn: Callable[..., object],
        scorers: list[object],
    ) -> object:
        return self.genai.evaluate(data=data, predict_fn=predict_fn, scorers=scorers)


def _dataset_is_missing(error: Exception) -> bool:
    if getattr(error, "error_code", None) == "RESOURCE_DOES_NOT_EXIST":
        return True
    try:
        not_found = import_module("databricks.sdk.errors").NotFound
    except (AttributeError, ModuleNotFoundError):
        return False
    return isinstance(error, not_found)
