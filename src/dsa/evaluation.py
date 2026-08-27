"""One small native MLflow evaluation path for canonical DSA runs."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import Field, JsonValue, field_validator, model_validator
from pydantic_ai.models import Model

from dsa.contract import ContractModel, RunRequest
from dsa.environment import PythonExecutor
from dsa.record import FailureStage, RunSuccess
from dsa.reporting import MlflowReporting
from dsa.runner import RunCompletion, run_analysis


class MlflowEvaluationCase(ContractModel):
    """A normal run request plus a host-only expected answer."""

    case_id: str = Field(pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    request: RunRequest
    expected_answer: JsonValue

    @field_validator("request", mode="before")
    @classmethod
    def revalidate_request(cls, value: object) -> object:
        if isinstance(value, RunRequest):
            return value.model_dump(mode="python", round_trip=True)
        return value

    @field_validator("expected_answer")
    @classmethod
    def snapshot_expected_answer(cls, value: JsonValue) -> JsonValue:
        _ensure_finite_json(value)
        return deepcopy(value)

    @model_validator(mode="after")
    def validate_expected_answer(self) -> MlflowEvaluationCase:
        validator = cast(
            Any,
            Draft202012Validator(self.request.answer_schema),
        )
        errors: list[JsonSchemaValidationError] = list(
            validator.iter_errors(self.expected_answer)
        )
        if errors:
            raise ValueError("expected answer must satisfy the request answer schema")
        return self

    def dataset_record(self) -> dict[str, JsonValue]:
        """Build one native row with expectations outside model inputs."""
        return {
            "inputs": {
                "case_id": self.case_id,
                "request": self.request.model_dump(mode="json"),
            },
            "expectations": {"answer": deepcopy(self.expected_answer)},
        }


class MlflowEvaluationPrediction(ContractModel):
    """Bounded scorer input projected from one completed DSA run."""

    case_id: str = Field(pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    run_id: str = Field(pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    terminal_sha256: str = Field(pattern=r"[0-9a-f]{64}")
    accepted: bool
    answer: JsonValue = None
    failure_stage: FailureStage | None = None
    failure_code: str | None = None
    reporting: MlflowReporting

    @model_validator(mode="after")
    def validate_outcome_projection(self) -> MlflowEvaluationPrediction:
        if self.accepted:
            if self.failure_stage is not None or self.failure_code is not None:
                raise ValueError("accepted predictions must not contain failure state")
        elif self.failure_stage is None or not self.failure_code or self.answer is not None:
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

    def scorer(self, function: Callable[..., object], *, name: str) -> object: ...

    def invalid_feedback(self, name: str, rationale: str) -> object: ...

    def evaluate(
        self,
        *,
        data: _NativeDataset,
        predict_fn: Callable[..., object],
        scorers: list[object],
    ) -> object: ...


ModelFactory = Callable[[MlflowEvaluationCase], Model | str | None]


def run_mlflow_evaluation(
    cases: Sequence[MlflowEvaluationCase | object],
    *,
    dataset_name: str,
    runs_directory: Path,
    model_factory: ModelFactory | None = None,
    python_executor: PythonExecutor | None = None,
    api: _EvaluationApi | None = None,
) -> MlflowEvaluationResult:
    """Create a native dataset and exact-score it through ordinary DSA runs."""
    canonical_cases = tuple(_canonical_case(case) for case in cases)
    if not canonical_cases:
        raise ValueError("at least one evaluation case is required")
    if len({case.case_id for case in canonical_cases}) != len(canonical_cases):
        raise ValueError("evaluation case IDs must be unique")
    if not dataset_name.strip():
        raise ValueError("dataset_name must not be blank")
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
        dataset = dataset.merge_records([case.dataset_record() for case in canonical_cases])
    except ModuleNotFoundError:
        raise MlflowEvaluationError("mlflow_dependency_missing") from None
    except Exception:
        raise MlflowEvaluationError("mlflow_dataset_failed") from None
    indexed = {case.case_id: case for case in canonical_cases}

    async def predict_fn(case_id: str, request: dict[str, JsonValue]) -> dict[str, JsonValue]:
        case = indexed.get(case_id)
        if case is None:
            raise ValueError("evaluation input contains an unknown case ID")
        canonical_request = RunRequest.model_validate_json(
            _canonical_json(request)
        ).model_copy(deep=True)
        if canonical_request != case.request:
            raise ValueError("evaluation input request does not match its retained case")
        selected_model = model_factory(case) if model_factory is not None else None
        completion = await run_analysis(
            canonical_request,
            runs_directory=runs_directory,
            model=selected_model,
            python_executor=python_executor,
            report_to_mlflow=True,
        )
        return evaluation_prediction(case_id, completion).model_dump(mode="json")

    scorers = _native_scorers(selected_api)
    try:
        with _skip_mlflow_prediction_preflight():
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
    return MlflowEvaluationResult(
        dataset_id=dataset.dataset_id,
        dataset_digest=dataset.digest,
        evaluation_run_id=(
            evaluation_run_id if isinstance(evaluation_run_id, str) else None
        ),
        metrics=metrics,
        predictions=_result_predictions(raw_result),
    )


def evaluation_prediction(
    case_id: str,
    completion: RunCompletion,
) -> MlflowEvaluationPrediction:
    """Project canonical completion state into bounded evaluation output."""
    if isinstance(completion.outcome, RunSuccess):
        return MlflowEvaluationPrediction(
            case_id=case_id,
            run_id=completion.record.run_id,
            terminal_sha256=completion.retained_record.sha256,
            accepted=True,
            answer=deepcopy(completion.outcome.answer),
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


def end_to_end_exact_success(
    outputs: MlflowEvaluationPrediction | dict[str, JsonValue],
    expectations: dict[str, JsonValue],
) -> bool:
    prediction = _prediction(outputs)
    return prediction.accepted and exact_json_equal(
        prediction.answer,
        expectations.get("answer"),
    ) and prediction.reporting.status == "reported"


def conditional_exact_json(
    outputs: MlflowEvaluationPrediction | dict[str, JsonValue],
    expectations: dict[str, JsonValue],
) -> bool | None:
    prediction = _prediction(outputs)
    if not prediction.accepted:
        return None
    return exact_json_equal(prediction.answer, expectations.get("answer"))


def agent_failure(
    outputs: MlflowEvaluationPrediction | dict[str, JsonValue],
    expectations: dict[str, JsonValue],
) -> bool:
    prediction = _prediction(outputs)
    if infrastructure_failure(prediction):
        return False
    if not prediction.accepted:
        return True
    return not exact_json_equal(prediction.answer, expectations.get("answer"))


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


def _canonical_case(value: MlflowEvaluationCase | object) -> MlflowEvaluationCase:
    data = (
        value.model_dump(mode="python", round_trip=True)
        if isinstance(value, MlflowEvaluationCase)
        else value
    )
    return MlflowEvaluationCase.model_validate(data).model_copy(deep=True)


def _evaluation_configuration_failure() -> str | None:
    if os.environ.get("MLFLOW_TRACKING_URI") != "databricks":
        return "mlflow_tracking_uri_invalid"
    required = ("MLFLOW_EXPERIMENT_ID", "DATABRICKS_HOST", "DATABRICKS_TOKEN")
    if any(not os.environ.get(key, "").strip() for key in required):
        return "mlflow_configuration_missing"
    return None


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


def _result_predictions(raw_result: object) -> tuple[MlflowEvaluationPrediction, ...]:
    result_frame = getattr(raw_result, "result_df", None)
    if result_frame is None or "outputs" not in result_frame:
        return ()
    predictions: list[MlflowEvaluationPrediction] = []
    for raw in result_frame["outputs"].tolist():
        value = json.loads(raw) if isinstance(raw, str) else raw
        predictions.append(MlflowEvaluationPrediction.model_validate(value))
    return tuple(predictions)


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


def _ensure_finite_json(value: JsonValue) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("expected answer must contain only finite JSON")
    if isinstance(value, dict):
        for child in value.values():
            _ensure_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _ensure_finite_json(child)


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

    def score_infrastructure_failure(outputs: dict[str, JsonValue]) -> bool:
        return infrastructure_failure(outputs)

    return [
        api.scorer(score_end_to_end, name="end_to_end_exact_success"),
        api.scorer(score_conditional, name="conditional_exact_json"),
        api.scorer(score_agent_failure, name="agent_failure"),
        api.scorer(score_infrastructure_failure, name="infrastructure_failure"),
    ]


class _MlflowEvaluationApi:
    def __init__(self) -> None:
        self.datasets: Any = import_module("mlflow.genai.datasets")
        self.genai: Any = import_module("mlflow.genai")
        self.scorers: Any = import_module("mlflow.genai.scorers")
        entities: Any = import_module("mlflow.entities")
        self.feedback_type: Any = entities.Feedback

    def create_dataset(self, *, name: str, experiment_id: str) -> _NativeDataset:
        return cast(
            _NativeDataset,
            self.datasets.create_dataset(name=name, experiment_id=experiment_id),
        )

    def scorer(self, function: Callable[..., object], *, name: str) -> object:
        return self.scorers.scorer(function, name=name)

    def invalid_feedback(self, name: str, rationale: str) -> object:
        return self.feedback_type(name=name, value=None, rationale=rationale, valid=False)

    def evaluate(
        self,
        *,
        data: _NativeDataset,
        predict_fn: Callable[..., object],
        scorers: list[object],
    ) -> object:
        return self.genai.evaluate(data=data, predict_fn=predict_fn, scorers=scorers)
