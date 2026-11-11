"""Bounded local evaluation of canonical DSA runs."""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import cast

from pydantic import Field, JsonValue, TypeAdapter, model_validator
from pydantic_ai.models import Model

from dsa.contract import (
    ContractModel,
    ModelConfiguration,
    RunPolicy,
    RunRequest,
)
from dsa.environment import PythonExecutor
from dsa.pack import (
    EvaluationPackCase,
    ExactJsonScorer,
    JsonNumericToleranceScorer,
    LoadedEvaluationPack,
    PackScorer,
)
from dsa.record import FailureStage, RunSuccess
from dsa.reporting import LogfireReporting
from dsa.runner import RunCompletion, run_analysis


class EvaluationPrediction(ContractModel):
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
    reporting: LogfireReporting

    @model_validator(mode="after")
    def validate_outcome_projection(self) -> EvaluationPrediction:
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


class EvaluationResult(ContractModel):
    """One complete set of predictions in pack order."""

    predictions: tuple[EvaluationPrediction, ...]


class EvaluationError(RuntimeError):
    """Stable infrastructure failure at the local evaluation boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(f"Evaluation failed ({code})")
        self.code = code


ModelFactory = Callable[[EvaluationPackCase], Model | str | None]
_PACK_SCORER_ADAPTER: TypeAdapter[PackScorer] = TypeAdapter(
    ExactJsonScorer | JsonNumericToleranceScorer
)


async def run_evaluation(
    pack: LoadedEvaluationPack | object,
    *,
    runs_directory: Path,
    model_configuration: ModelConfiguration | object,
    policy: RunPolicy | object,
    case_workers: int = 1,
    model_factory: ModelFactory | None = None,
    python_executor: PythonExecutor | None = None,
    report_to_logfire: bool = False,
) -> EvaluationResult:
    """Run each verified pack case once with bounded concurrency."""
    if type(case_workers) is not int or not 1 <= case_workers <= 64:
        raise ValueError("case_workers must be an integer between 1 and 64")
    if type(report_to_logfire) is not bool:
        raise TypeError("report_to_logfire must be a boolean")
    canonical_pack = _canonical_pack(pack)
    canonical_model = _canonical_model_configuration(model_configuration)
    canonical_policy = _canonical_policy(policy)
    _verify_pack_database(canonical_pack)
    semaphore = asyncio.Semaphore(case_workers)

    async def evaluate_case(case: EvaluationPackCase) -> EvaluationPrediction:
        async with semaphore:
            request = RunRequest(
                database_path=canonical_pack.database_path,
                question=case.question,
                answer_schema=case.answer_schema,
                derivation=case.derivation,
                model=canonical_model,
                policy=canonical_policy,
            )
            completion = await run_analysis(
                request,
                runs_directory=runs_directory,
                model=model_factory(case) if model_factory is not None else None,
                python_executor=python_executor,
                report_to_logfire=report_to_logfire,
            )
            return evaluation_prediction(case.case_id, completion)

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(evaluate_case(case)) for case in canonical_pack.cases]
    return EvaluationResult(predictions=tuple(task.result() for task in tasks))


def evaluation_prediction(
    case_id: str,
    completion: RunCompletion,
) -> EvaluationPrediction:
    """Project canonical completion state into bounded evaluation output."""
    if isinstance(completion.outcome, RunSuccess):
        verification = completion.outcome.derivation_verification
        return EvaluationPrediction(
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
    return EvaluationPrediction(
        case_id=case_id,
        run_id=completion.record.run_id,
        terminal_sha256=completion.retained_record.sha256,
        accepted=False,
        failure_stage=completion.outcome.failure.stage,
        failure_code=completion.outcome.failure.code,
        reporting=completion.reporting,
    )


def end_to_end_exact_success(
    outputs: EvaluationPrediction | dict[str, JsonValue],
    expectations: dict[str, JsonValue],
) -> bool:
    prediction = _prediction(outputs)
    expected, _scorer = _decode_scoring_expectations(expectations)
    return (
        prediction.accepted
        and exact_json_equal(prediction.answer, expected)
    )


def conditional_exact_json(
    outputs: EvaluationPrediction | dict[str, JsonValue],
    expectations: dict[str, JsonValue],
) -> bool | None:
    prediction = _prediction(outputs)
    if not prediction.accepted:
        return None
    expected, _scorer = _decode_scoring_expectations(expectations)
    return exact_json_equal(prediction.answer, expected)


def end_to_end_policy_success(
    outputs: EvaluationPrediction | dict[str, JsonValue],
    expectations: dict[str, JsonValue],
) -> bool:
    """Score accepted answers under the pack comparison policy."""
    prediction = _prediction(outputs)
    return (
        prediction.accepted
        and json_answers_equal(prediction.answer, expectations)
    )


def conditional_policy_match(
    outputs: EvaluationPrediction | dict[str, JsonValue],
    expectations: dict[str, JsonValue],
) -> bool | None:
    """Score accepted answers under the pack policy without failures in the denominator."""
    prediction = _prediction(outputs)
    if not prediction.accepted:
        return None
    return json_answers_equal(prediction.answer, expectations)


def agent_failure(
    outputs: EvaluationPrediction | dict[str, JsonValue],
    expectations: dict[str, JsonValue],
) -> bool:
    prediction = _prediction(outputs)
    if infrastructure_failure(prediction):
        return False
    if not prediction.accepted:
        return True
    return not json_answers_equal(prediction.answer, expectations)


def infrastructure_failure(
    outputs: EvaluationPrediction | dict[str, JsonValue],
) -> bool:
    prediction = _prediction(outputs)
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
    """Compare one answer under the canonical policy retained in the pack."""
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
                raise EvaluationError("evaluation_database_mismatch")
            digest = sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except EvaluationError:
        raise
    except OSError:
        raise EvaluationError("evaluation_database_unavailable") from None
    if digest.hexdigest() != expected.sha256:
        raise EvaluationError("evaluation_database_mismatch")


def _prediction(
    value: EvaluationPrediction | dict[str, JsonValue],
) -> EvaluationPrediction:
    data = (
        value.model_dump(mode="python")
        if isinstance(value, EvaluationPrediction)
        else value
    )
    return EvaluationPrediction.model_validate(data)


def _canonical_json(value: JsonValue) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
