"""Deterministic Pydantic AI episode and terminalization boundary."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import JsonValue
from pydantic_ai import (
    Agent,
    ModelAPIError,
    ModelHTTPError,
    ModelRetry,
    StructuredDict,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_graph import End

from dsa.contract import ContractModel, RunRequest
from dsa.record import (
    Failure,
    RetainedTerminalRecord,
    RunFailure,
    RunOutcome,
    RunSuccess,
    TerminalRecord,
    write_terminal_record,
)

_INSTRUCTIONS = """You are executing one completely specified data science task.
Return only the requested answer through the provided structured output boundary.
The answer must satisfy the caller's JSON Schema exactly.
"""


class _AnswerValidator(Protocol):
    def iter_errors(self, instance: JsonValue) -> Iterable[JsonSchemaValidationError]: ...


class RunCompletion(ContractModel):
    """A terminal run plus the integrity reference to its retained bytes."""

    record: TerminalRecord
    retained_record: RetainedTerminalRecord

    @property
    def outcome(self) -> RunOutcome:
        return self.record.outcome


async def run_analysis(
    request: RunRequest | object,
    *,
    runs_directory: Path,
    model: Model | str | None = None,
    identity_factory: Callable[[], str] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> RunCompletion:
    """Validate, execute and retain exactly one analysis run."""
    canonical_request = RunRequest.model_validate(request).model_copy(deep=True)
    identity_source = identity_factory or (lambda: f"run-{uuid4().hex}")
    now = clock or (lambda: datetime.now(UTC))
    run_id = identity_source()
    _validate_run_id(run_id)
    started_at = _aware_time(now())

    runs_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    run_directory = runs_directory / run_id
    run_directory.mkdir(mode=0o700)

    if not canonical_request.database_path.is_file():
        outcome = RunFailure(
            failure=Failure(
                stage="analysis_environment",
                code="source_database_not_file",
                message="The source database is not an available regular file",
            )
        )
        return _retain_completion(
            canonical_request,
            run_id,
            started_at,
            _aware_time(now()),
            (),
            {},
            outcome,
            run_directory,
        )

    messages: tuple[dict[str, JsonValue], ...] = ()
    usage: dict[str, JsonValue] = {}
    validation_failures = 0
    running: Any | None = None

    caller_schema = canonical_request.answer_schema
    wrapped = caller_schema.get("type") != "object"
    framework_schema = _framework_schema(caller_schema, wrapped)
    validator = cast(_AnswerValidator, Draft202012Validator(caller_schema))

    async def validate_answer(proposal: dict[str, Any]) -> dict[str, Any]:
        nonlocal validation_failures
        answer = proposal.get("value") if wrapped else proposal
        errors = sorted(
            validator.iter_errors(cast(JsonValue, answer)),
            key=_validation_error_key,
        )
        if errors:
            validation_failures += 1
            raise ModelRetry(_validation_feedback(errors))
        try:
            json.dumps(answer, allow_nan=False)
        except (TypeError, ValueError) as error:
            validation_failures += 1
            raise ModelRetry(
                '{"error":"answer_schema_validation_failed",'
                '"issues":[{"keyword":"json","message":"answer must be finite JSON"}]}'
            ) from error
        return proposal

    try:
        selected_model: Model | str = model or canonical_request.model.name
        agent = Agent(
            selected_model,
            output_type=StructuredDict(
                framework_schema,
                name="final_answer",
                description="Submit the exact caller-requested answer.",
            ),
            instructions=_INSTRUCTIONS,
            name="dsa",
            retries={"output": canonical_request.policy.max_validation_attempts - 1},
            end_strategy="early",
        )
        agent.output_validator(validate_answer)
        limits = UsageLimits(
            request_limit=canonical_request.policy.max_model_requests,
            tool_calls_limit=canonical_request.policy.max_tool_calls,
            total_tokens_limit=canonical_request.policy.max_total_tokens,
        )
        model_settings = _bounded_model_settings(canonical_request)
        async with asyncio.timeout(canonical_request.policy.max_run_seconds):
            async with agent.iter(
                canonical_request.question,
                model_settings=model_settings,
                usage_limits=limits,
            ) as running:
                next_node = running.next_node
                while not isinstance(next_node, End):
                    next_node = await running.next(next_node)
                    messages, usage = _snapshot_run(running)
                messages, usage = _snapshot_run(running)
                if running.result is None:
                    raise RuntimeError("Pydantic AI completed without a result")
                proposal = running.result.output
                answer = proposal["value"] if wrapped else proposal
                outcome: RunOutcome = RunSuccess(answer=cast(JsonValue, answer))
    except asyncio.CancelledError as error:
        if running is not None:
            messages, usage = _snapshot_run(running)
        outcome = RunFailure(
            failure=Failure(
                stage="cancelled",
                code="cancelled_by_caller",
                message="The caller cancelled the analysis run",
            )
        )
        completion = _retain_completion(
            canonical_request,
            run_id,
            started_at,
            _aware_time(now()),
            messages,
            usage,
            outcome,
            run_directory,
        )
        retained_error = cast(Any, error)
        retained_error.terminal_record = completion.record
        retained_error.retained_record = completion.retained_record
        raise
    except Exception as error:
        if running is not None:
            messages, usage = _snapshot_run(running)
        outcome = _failure_outcome(error, validation_failures)

    return _retain_completion(
        canonical_request,
        run_id,
        started_at,
        _aware_time(now()),
        messages,
        usage,
        outcome,
        run_directory,
    )


def _framework_schema(
    caller_schema: dict[str, JsonValue],
    wrapped: bool,
) -> dict[str, Any]:
    if not wrapped:
        return deepcopy(caller_schema)
    return {
        "type": "object",
        "properties": {"value": deepcopy(caller_schema)},
        "required": ["value"],
        "additionalProperties": False,
    }


def _bounded_model_settings(request: RunRequest) -> ModelSettings:
    settings: dict[str, Any] = dict(request.model.settings)
    configured = settings.get("max_tokens")
    if type(configured) is int:
        settings["max_tokens"] = min(configured, request.policy.max_model_output_tokens)
    else:
        settings["max_tokens"] = request.policy.max_model_output_tokens
    return cast(ModelSettings, settings)


def _snapshot_run(running: Any) -> tuple[tuple[dict[str, JsonValue], ...], dict[str, JsonValue]]:
    raw_messages = ModelMessagesTypeAdapter.dump_python(running.all_messages(), mode="json")
    messages = tuple(cast(dict[str, JsonValue], item) for item in raw_messages)
    native_usage = cast(RunUsage, running.usage)
    raw_usage = asdict(native_usage)
    raw_usage["total_tokens"] = native_usage.total_tokens
    return messages, cast(dict[str, JsonValue], raw_usage)


def _failure_outcome(error: Exception, validation_failures: int) -> RunFailure:
    if isinstance(error, ModelHTTPError):
        return RunFailure(
            failure=Failure(
                stage="model",
                code="model_http_error",
                message="The model provider returned an HTTP error",
                diagnostics={
                    "status_code": error.status_code,
                    "model_name": error.model_name,
                    "exception_type": type(error).__name__,
                },
            )
        )
    if isinstance(error, ModelAPIError):
        return RunFailure(
            failure=Failure(
                stage="model",
                code="model_api_error",
                message="The model provider request failed",
                diagnostics={
                    "model_name": error.model_name,
                    "exception_type": type(error).__name__,
                },
            )
        )
    if isinstance(error, UnexpectedModelBehavior):
        if validation_failures:
            return RunFailure(
                failure=Failure(
                    stage="answer_validation",
                    code="attempts_exhausted",
                    message="The model did not produce an answer satisfying the caller schema",
                    diagnostics={"attempts": validation_failures},
                )
            )
        return RunFailure(
            failure=Failure(
                stage="model",
                code="unexpected_model_behavior",
                message="The model response did not satisfy the framework protocol",
                diagnostics={
                    "exception_type": type(error).__name__,
                    "detail": _bounded_text(str(error)),
                },
            )
        )
    if isinstance(error, UsageLimitExceeded):
        return RunFailure(
            failure=Failure(
                stage="orchestration",
                code="usage_limit_exceeded",
                message="The run exceeded a host-enforced model usage limit",
                diagnostics={"detail": _bounded_text(str(error))},
            )
        )
    if isinstance(error, TimeoutError):
        return RunFailure(
            failure=Failure(
                stage="orchestration",
                code="run_timeout",
                message="The run exceeded its elapsed-time limit",
            )
        )
    return RunFailure(
        failure=Failure(
            stage="orchestration",
            code="internal_error",
            message="The analysis run failed inside the host orchestration boundary",
            diagnostics={"exception_type": type(error).__name__},
        )
    )


def _retain_completion(
    request: RunRequest,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    messages: tuple[dict[str, JsonValue], ...],
    usage: dict[str, JsonValue],
    outcome: RunOutcome,
    run_directory: Path,
) -> RunCompletion:
    record = TerminalRecord(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        request=request,
        messages=messages,
        usage=usage,
        outcome=outcome,
    )
    retained = write_terminal_record(record, run_directory)
    return RunCompletion(record=record, retained_record=retained)


def _validation_error_key(error: JsonSchemaValidationError) -> tuple[str, str]:
    return ("/".join(map(str, error.absolute_path)), "/".join(map(str, error.absolute_schema_path)))


def _validation_feedback(errors: list[JsonSchemaValidationError]) -> str:
    issues = [
        {
            "result_path": list(error.absolute_path),
            "schema_path": list(error.absolute_schema_path),
            "keyword": str(error.validator),
            "message": error.message,
        }
        for error in errors
    ]
    return json.dumps(
        {"error": "answer_schema_validation_failed", "issues": issues},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_run_id(run_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise ValueError("identity factory returned an unsafe run identity")


def _aware_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("clock must return timezone-aware timestamps")
    return value


def _bounded_text(value: str, limit: int = 2_000) -> str:
    return value if len(value) <= limit else value[:limit] + "…"
