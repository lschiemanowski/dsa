"""Deterministic Pydantic AI episode and terminalization boundary."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import stat
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

import duckdb
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import Field, JsonValue
from pydantic import ValidationError as PydanticValidationError
from pydantic_ai import (
    Agent,
    ModelAPIError,
    ModelHTTPError,
    ModelRequestNode,
    ModelRetry,
    RunContext,
    StructuredDict,
    Tool,
    ToolOutput,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_graph import End

from dsa.contract import ContractModel, Derivation, RunRequest
from dsa.derivation import DerivationError, VerifiedDerivation, verify_derivation
from dsa.environment import (
    AnalysisEnvironment,
    ArtifactError,
    PythonExecutor,
    ToolResultLimitExceeded,
)
from dsa.record import (
    ArtifactRecord,
    DatabaseRecord,
    Failure,
    RetainedDerivationNotebook,
    RetainedTerminalRecord,
    RunFailure,
    RunOutcome,
    RunSuccess,
    TerminalRecord,
    write_terminal_record,
)
from dsa.reporting import MlflowReporting, run_with_mlflow_reporting

_INSTRUCTIONS = """You are executing one completely specified data science task.
Use the database tools to inspect schema and run bounded read-only SQL.
Small complete query results are inline. Larger complete results are retained automatically.
Artifact previews are incomplete orientation only. Use their managed paths from Python.
Return only the requested answer through a provided structured output boundary.
The answer must satisfy the caller's JSON Schema exactly.
"""
_DERIVATION_INSTRUCTIONS = """\
Alongside the answer, submit a concise dsa-derivation/v1 for a human verifier.
This is a curated reproducibility document, not a transcript of your work.
Omit exploration, false starts, and computations that are unnecessary to verify the answer.
Begin with a brief Markdown explanation. Use plain Python code cells with brief explanations.
Code cells share one namespace and receive database_path bound to the pristine source database.
Do not depend on retained artifacts or prior database mutations.
The final code cell must assign the exact JSON answer to result.
The derivation is replayed independently and must reproduce the submitted answer exactly.
"""


class _AnswerValidator(Protocol):
    def iter_errors(self, instance: JsonValue) -> Iterable[JsonSchemaValidationError]: ...


class _ValidationAttemptsExceeded(Exception):
    pass


class _DerivationAttemptsExceeded(Exception):
    pass


class _SourceDatabaseError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class RunCompletion(ContractModel):
    """A terminal run plus the integrity reference to its retained bytes."""

    record: TerminalRecord
    retained_record: RetainedTerminalRecord
    retained_notebook: RetainedDerivationNotebook | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    reporting: MlflowReporting = MlflowReporting()

    @property
    def outcome(self) -> RunOutcome:
        return self.record.outcome

    def with_reporting(self, reporting: MlflowReporting) -> RunCompletion:
        """Return an operator projection without changing canonical run state."""
        return self.model_copy(update={"reporting": reporting})


async def run_analysis(
    request: RunRequest | object,
    *,
    runs_directory: Path,
    model: Model | str | None = None,
    python_executor: PythonExecutor | None = None,
    identity_factory: Callable[[], str] | None = None,
    clock: Callable[[], datetime] | None = None,
    keep_workdir: bool = False,
    report_to_mlflow: bool = False,
) -> RunCompletion:
    """Validate, execute and retain exactly one analysis run."""
    if type(keep_workdir) is not bool:
        raise TypeError("keep_workdir must be a boolean")
    if type(report_to_mlflow) is not bool:
        raise TypeError("report_to_mlflow must be a boolean")
    request_data = (
        request.model_dump(mode="python", round_trip=True)
        if isinstance(request, RunRequest)
        else request
    )
    canonical_request = RunRequest.model_validate(request_data).model_copy(deep=True)
    identity_source = identity_factory or (lambda: f"run-{uuid4().hex}")
    now = clock or (lambda: datetime.now(UTC))
    run_id = identity_source()
    _validate_run_id(run_id)
    started_at = _aware_time(now())

    async def operation() -> RunCompletion:
        return await _run_canonical_analysis(
            canonical_request,
            run_id=run_id,
            started_at=started_at,
            runs_directory=runs_directory,
            model=model,
            python_executor=python_executor,
            now=now,
            keep_workdir=keep_workdir,
        )

    return await run_with_mlflow_reporting(
        enabled=report_to_mlflow,
        run_id=run_id,
        request=canonical_request,
        operation=operation,
    )


async def _run_canonical_analysis(
    canonical_request: RunRequest,
    *,
    run_id: str,
    started_at: datetime,
    runs_directory: Path,
    model: Model | str | None,
    python_executor: PythonExecutor | None,
    now: Callable[[], datetime],
    keep_workdir: bool,
) -> RunCompletion:
    """Execute a validated request under its already-owned run identity."""

    runs_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    run_directory = runs_directory / run_id
    run_directory.mkdir(mode=0o700)

    try:
        working_database, source_database_sha256, pristine_database = (
            _prepare_working_database(
                canonical_request.database_path,
                run_directory,
                preserve_pristine=canonical_request.derivation is not None,
            )
        )
    except _SourceDatabaseError as error:
        outcome = RunFailure(
            failure=Failure(
                stage="analysis_environment",
                code=error.code,
                message=error.message,
            )
        )
        return _retain_completion(
            canonical_request,
            run_id,
            started_at,
            _aware_time(now()),
            (),
            {},
            (),
            outcome,
            run_directory,
        )

    environment = AnalysisEnvironment(
        database_path=working_database,
        run_directory=run_directory,
        policy=canonical_request.policy,
        python_executor=python_executor,
    )
    try:
        await environment.check_database()
    except asyncio.CancelledError as error:
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
            (),
            {},
            (),
            outcome,
            run_directory,
            source_database_sha256=source_database_sha256,
            working_database=working_database,
            keep_workdir=keep_workdir,
        )
        retained_error = cast(Any, error)
        retained_error.terminal_record = completion.record
        retained_error.retained_record = completion.retained_record
        raise
    except duckdb.Error:
        outcome = RunFailure(
            failure=Failure(
                stage="analysis_environment",
                code="source_database_invalid",
                message="The source database could not be opened as a read-only DuckDB database",
            )
        )
        return _retain_completion(
            canonical_request,
            run_id,
            started_at,
            _aware_time(now()),
            (),
            {},
            (),
            outcome,
            run_directory,
            source_database_sha256=source_database_sha256,
            working_database=working_database,
            keep_workdir=keep_workdir,
        )
    except TimeoutError:
        outcome = RunFailure(
            failure=Failure(
                stage="analysis_environment",
                code="source_database_timeout",
                message="Opening the source database exceeded its elapsed-time limit",
            )
        )
        return _retain_completion(
            canonical_request,
            run_id,
            started_at,
            _aware_time(now()),
            (),
            {},
            (),
            outcome,
            run_directory,
            source_database_sha256=source_database_sha256,
            working_database=working_database,
            keep_workdir=keep_workdir,
        )
    messages: tuple[dict[str, JsonValue], ...] = ()
    usage: dict[str, JsonValue] = {}
    validation_failures = 0
    derivation_failures = 0
    verified_derivation: VerifiedDerivation | None = None
    running: Any | None = None

    caller_schema = canonical_request.answer_schema
    derivation_requested = canonical_request.derivation is not None
    wrapped = not derivation_requested and caller_schema.get("type") != "object"
    framework_schema = _framework_schema(
        caller_schema,
        wrapped,
        derivation_requested=derivation_requested,
    )
    validator = cast(_AnswerValidator, Draft202012Validator(caller_schema))

    async def validate_answer(proposal: Any) -> Any:
        nonlocal derivation_failures, validation_failures, verified_derivation
        answer = (
            proposal.get("answer")
            if derivation_requested
            else proposal.get("value")
            if wrapped
            else proposal
        )
        errors = sorted(
            validator.iter_errors(cast(JsonValue, answer)),
            key=_validation_error_key,
        )
        if errors:
            validation_failures += 1
            if validation_failures >= canonical_request.policy.max_validation_attempts:
                raise _ValidationAttemptsExceeded
            raise ModelRetry(
                _validation_feedback(
                    errors,
                    canonical_request.policy.max_tool_result_bytes,
                )
            )
        try:
            json.dumps(answer, allow_nan=False)
        except (TypeError, ValueError) as error:
            validation_failures += 1
            if validation_failures >= canonical_request.policy.max_validation_attempts:
                raise _ValidationAttemptsExceeded from error
            raise ModelRetry(
                _bounded_retry_feedback(
                    [
                        {
                            "error": "answer_schema_validation_failed",
                            "issues": [
                                {
                                    "keyword": "json",
                                    "message": "answer must be finite JSON",
                                }
                            ],
                        },
                        {"error": "answer_schema_validation_failed"},
                    ],
                    canonical_request.policy.max_tool_result_bytes,
                )
            ) from error
        if derivation_requested:
            try:
                try:
                    derivation = Derivation.model_validate(proposal.get("derivation"))
                except PydanticValidationError as error:
                    raise DerivationError(
                        "derivation_invalid",
                        "The derivation does not satisfy its cell contract",
                    ) from error
                if pristine_database is None:
                    raise DerivationError(
                        "derivation_source_unavailable",
                        "The private source database is unavailable for derivation replay",
                        infrastructure=True,
                    )
                verified_derivation = await verify_derivation(
                    derivation,
                    cast(JsonValue, answer),
                    question=canonical_request.question,
                    source_database=pristine_database,
                    source_database_sha256=source_database_sha256,
                    run_directory=run_directory,
                    policy=canonical_request.policy,
                    python_executor=python_executor,
                )
            except DerivationError as error:
                if error.infrastructure:
                    raise
                derivation_failures += 1
                if derivation_failures >= canonical_request.policy.max_validation_attempts:
                    raise _DerivationAttemptsExceeded from error
                raise ModelRetry(
                    _bounded_retry_feedback(
                        [
                            {"error": error.code, "message": error.message},
                            {"error": error.code},
                        ],
                        canonical_request.policy.max_tool_result_bytes,
                    )
                ) from error
        return proposal

    async def answer_from_artifact(handle: str) -> JsonValue:
        """Submit a retained same-run JSON artifact as the final answer."""
        nonlocal validation_failures
        try:
            answer = environment.load_json_artifact(handle)
        except ArtifactError as error:
            validation_failures += 1
            if validation_failures >= canonical_request.policy.max_validation_attempts:
                raise _ValidationAttemptsExceeded from error
            raise ModelRetry(
                _bounded_retry_feedback(
                    [
                        {"error": error.code, "message": error.message},
                        {"error": error.code},
                    ],
                    canonical_request.policy.max_tool_result_bytes,
                )
            ) from error
        return {"value": answer} if wrapped else answer

    async def answer_and_derivation_from_artifact(
        handle: str,
        derivation: Derivation,
    ) -> dict[str, JsonValue]:
        """Submit one retained answer with its concise replayable derivation."""
        answer = await answer_from_artifact(handle)
        return {
            "answer": answer,
            "derivation": cast(JsonValue, derivation.model_dump(mode="json")),
        }

    async def inspect_database(
        context: RunContext[None],
        relation: str | None = None,
    ) -> str:
        """List canonical quoted relations, or inspect one returned relation name."""
        return await environment.inspect_database(
            relation,
            tool_call_id=_tool_call_id(context),
        )

    async def query_database(context: RunContext[None], sql: str) -> str:
        """Run one bounded read-only DuckDB query with automatic artifact routing."""
        return await environment.query_database(sql, tool_call_id=_tool_call_id(context))

    async def run_python(
        context: RunContext[None],
        source: str,
        inputs: list[str],
        expected_outputs: list[str],
    ) -> str:
        """Run Python in the configured executor using managed artifact paths."""
        return await environment.run_python(
            source,
            inputs,
            expected_outputs,
            tool_call_id=_tool_call_id(context),
        )

    try:
        selected_model: Model | str = model or canonical_request.model.name
        tools: list[Tool[None]] = [
            Tool(
                inspect_database,
                takes_ctx=True,
                name="inspect_database",
                description=(
                    "List sorted canonical quoted database relation names when relation is "
                    "omitted, or return ordered column names and DuckDB types for one exact "
                    "returned relation name. This returns schema only, never row data."
                ),
                sequential=True,
            ),
            Tool(
                query_database,
                takes_ctx=True,
                name="query_database",
                description=(
                    "Execute exactly one read-only SELECT, WITH, or VALUES statement. "
                    "Small complete tables are returned inline. Larger complete tables are "
                    "retained as Parquet with a managed path and at most five preview rows. "
                    "A preview is never the complete table."
                ),
                sequential=True,
            ),
        ]
        if python_executor is not None:
            tools.append(
                Tool(
                    run_python,
                    takes_ctx=True,
                    name="run_python",
                    description=(
                        "Execute source only through the configured isolated executor. "
                        "Name retained artifact handles in inputs. DSAGENT_DATABASE, "
                        "DSAGENT_INPUTS, and DSAGENT_OUTPUTS are environment-variable "
                        "names: resolve their paths with os.environ. Read only selected "
                        "artifacts from the inputs directory. Write exactly the declared "
                        ".json or .parquet files to the outputs directory. expected_outputs "
                        "may be empty for a database-only exploratory or mutating call."
                    ),
                    sequential=True,
                )
            )
        artifact_output = (
            ToolOutput(
                answer_and_derivation_from_artifact,
                name="answer_from_artifact",
                description=(
                    "Submit a same-run retained JSON answer with its concise replayable "
                    "human-verification derivation."
                ),
            )
            if derivation_requested
            else ToolOutput(
                answer_from_artifact,
                name="answer_from_artifact",
                description="Submit a same-run retained JSON artifact as the final answer.",
            )
        )
        agent = Agent(
            selected_model,
            output_type=[
                ToolOutput(
                    StructuredDict(
                        framework_schema,
                        name="final_answer",
                        description="Submit the exact caller-requested answer.",
                    ),
                    name="final_answer",
                    description="Submit the exact caller-requested answer directly.",
                ),
                artifact_output,
            ],
            instructions=(
                _INSTRUCTIONS + _DERIVATION_INSTRUCTIONS
                if derivation_requested
                else _INSTRUCTIONS
            ),
            name="dsa",
            retries={"output": canonical_request.policy.max_validation_attempts - 1},
            end_strategy="early",
            tools=tools,
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
                    messages = _include_pending_model_request(messages, next_node)
                    result_sizes = _model_visible_tool_result_sizes(messages)
                    if any(
                        size > canonical_request.policy.max_tool_result_bytes
                        for size in result_sizes
                    ):
                        raise ToolResultLimitExceeded(
                            "a model-visible tool result exceeded its per-result limit"
                        )
                    if sum(result_sizes) > canonical_request.policy.max_total_tool_result_bytes:
                        raise ToolResultLimitExceeded(
                            "cumulative model-visible tool results exceeded the run limit"
                        )
                messages, usage = _snapshot_run(running)
                if running.result is None:
                    raise RuntimeError("Pydantic AI completed without a result")
                proposal = running.result.output
                answer = (
                    proposal["answer"]
                    if derivation_requested
                    else proposal["value"]
                    if wrapped
                    else proposal
                )
                if derivation_requested:
                    if verified_derivation is None:
                        raise RuntimeError("verified derivation state is unavailable")
                    outcome = RunSuccess(
                        answer=cast(JsonValue, answer),
                        derivation=verified_derivation.derivation,
                        derivation_verification=verified_derivation.verification,
                    )
                else:
                    outcome = RunSuccess(answer=cast(JsonValue, answer))
    except asyncio.CancelledError as error:
        if running is not None:
            snapshot_messages, usage = _snapshot_run(running)
            if len(snapshot_messages) > len(messages):
                messages = snapshot_messages
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
            environment.artifact_records,
            outcome,
            run_directory,
            source_database_sha256=source_database_sha256,
            working_database=working_database,
            keep_workdir=keep_workdir,
            retained_notebook=(
                verified_derivation.notebook
                if verified_derivation is not None
                else None
            ),
        )
        retained_error = cast(Any, error)
        retained_error.terminal_record = completion.record
        retained_error.retained_record = completion.retained_record
        raise
    except Exception as error:
        if running is not None:
            snapshot_messages, usage = _snapshot_run(running)
            if len(snapshot_messages) > len(messages):
                messages = snapshot_messages
        outcome = _failure_outcome(
            error,
            validation_failures,
            derivation_failures,
        )

    return _retain_completion(
        canonical_request,
        run_id,
        started_at,
        _aware_time(now()),
        messages,
        usage,
        environment.artifact_records,
        outcome,
        run_directory,
        source_database_sha256=source_database_sha256,
        working_database=working_database,
        keep_workdir=keep_workdir,
        retained_notebook=(
            verified_derivation.notebook if verified_derivation is not None else None
        ),
    )


def _framework_schema(
    caller_schema: dict[str, JsonValue],
    wrapped: bool,
    *,
    derivation_requested: bool,
) -> dict[str, Any]:
    if derivation_requested:
        return {
            "type": "object",
            "properties": {
                "answer": deepcopy(caller_schema),
                "derivation": _derivation_schema(),
            },
            "required": ["answer", "derivation"],
            "additionalProperties": False,
        }
    if not wrapped:
        return deepcopy(caller_schema)
    return {
        "type": "object",
        "properties": {"value": deepcopy(caller_schema)},
        "required": ["value"],
        "additionalProperties": False,
    }


def _derivation_schema() -> dict[str, Any]:
    source = {"type": "string", "minLength": 1, "maxLength": 32 * 1024}
    return {
        "type": "object",
        "properties": {
            "format": {"const": "dsa-derivation/v1"},
            "cells": {
                "type": "array",
                "minItems": 2,
                "maxItems": 24,
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": {"const": "markdown"},
                                "source": deepcopy(source),
                            },
                            "required": ["type", "source"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": {"const": "code"},
                                "source": deepcopy(source),
                            },
                            "required": ["type", "source"],
                            "additionalProperties": False,
                        },
                    ]
                },
            },
            "required": ["format", "cells"],
            "additionalProperties": False,
        },
        "required": ["format", "cells"],
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


def _failure_outcome(
    error: Exception,
    validation_failures: int,
    derivation_failures: int,
) -> RunFailure:
    if isinstance(error, _ValidationAttemptsExceeded):
        return RunFailure(
            failure=Failure(
                stage="answer_validation",
                code="attempts_exhausted",
                message="The model did not produce an answer satisfying the caller schema",
                diagnostics={"attempts": validation_failures},
            )
        )
    if isinstance(error, _DerivationAttemptsExceeded):
        return RunFailure(
            failure=Failure(
                stage="derivation_validation",
                code="attempts_exhausted",
                message=(
                    "The model did not produce a derivation that reproduced its answer"
                ),
                diagnostics={"attempts": derivation_failures},
            )
        )
    if isinstance(error, DerivationError):
        return RunFailure(
            failure=Failure(
                stage=("analysis_environment" if error.infrastructure else "derivation_validation"),
                code=error.code,
                message=error.message,
            )
        )
    if isinstance(error, ToolResultLimitExceeded):
        return RunFailure(
            failure=Failure(
                stage="orchestration",
                code="tool_result_limit_exceeded",
                message="The run exceeded a host-enforced model-visible tool result limit",
            )
        )
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
        if derivation_failures:
            return RunFailure(
                failure=Failure(
                    stage="derivation_validation",
                    code="attempts_exhausted",
                    message=(
                        "The model did not produce a derivation that reproduced its answer"
                    ),
                    diagnostics={"attempts": derivation_failures},
                )
            )
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
    artifacts: tuple[ArtifactRecord, ...],
    outcome: RunOutcome,
    run_directory: Path,
    *,
    source_database_sha256: str | None = None,
    working_database: Path | None = None,
    keep_workdir: bool = False,
    retained_notebook: RetainedDerivationNotebook | None = None,
) -> RunCompletion:
    database: DatabaseRecord | None = None
    if source_database_sha256 is not None and working_database is not None:
        try:
            database = DatabaseRecord(
                source_sha256=source_database_sha256,
                final_sha256=_sha256_file(working_database),
            )
        except OSError:
            outcome = RunFailure(
                failure=Failure(
                    stage="analysis_environment",
                    code="working_database_unavailable",
                    message="The final private database could not be retained safely",
                )
            )
    if working_database is not None and not keep_workdir:
        try:
            shutil.rmtree(working_database.parent)
        except OSError:
            outcome = RunFailure(
                failure=Failure(
                    stage="analysis_environment",
                    code="workspace_cleanup_failed",
                    message="The private database workspace could not be removed",
                )
            )
    if not isinstance(outcome, RunSuccess) and retained_notebook is not None:
        with suppress(OSError):
            retained_notebook.path.unlink(missing_ok=True)
        retained_notebook = None
    record = TerminalRecord(
        schema_version=("2" if request.derivation is not None else "1"),
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        request=request,
        messages=messages,
        usage=usage,
        artifacts=artifacts,
        database=database,
        outcome=outcome,
    )
    retained = write_terminal_record(record, run_directory)
    return RunCompletion(
        record=record,
        retained_record=retained,
        retained_notebook=retained_notebook,
    )


def _prepare_working_database(
    source: Path,
    run_directory: Path,
    *,
    preserve_pristine: bool,
) -> tuple[Path, str, Path | None]:
    wal = Path(f"{source}.wal")
    if os.path.lexists(wal):
        raise _SourceDatabaseError(
            "source_database_wal",
            "The source database has an active write-ahead log",
        )

    work_directory = run_directory / "work"
    temporary = work_directory / f".database.{uuid4().hex}.tmp"
    destination = work_directory / "database.duckdb"
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    digest = sha256()
    copied = 0
    try:
        work_directory.mkdir(mode=0o700)
        (work_directory / "attempts").mkdir(mode=0o700)
        try:
            source_descriptor = os.open(
                source,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as error:
            raise _SourceDatabaseError(
                "source_database_not_file",
                "The source database is not an available regular file",
            ) from error
        initial = os.fstat(source_descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise _SourceDatabaseError(
                "source_database_not_file",
                "The source database is not an available regular file",
            )
        destination_descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(destination_descriptor, "wb") as target:
            destination_descriptor = None
            with os.fdopen(source_descriptor, "rb") as source_handle:
                source_descriptor = None
                while chunk := source_handle.read(1024 * 1024):
                    copied += len(chunk)
                    digest.update(chunk)
                    target.write(chunk)
                final = os.fstat(source_handle.fileno())
            target.flush()
            os.fsync(target.fileno())
        if (
            copied != initial.st_size
            or final.st_size != initial.st_size
            or final.st_mtime_ns != initial.st_mtime_ns
            or final.st_ino != initial.st_ino
            or final.st_dev != initial.st_dev
            or os.path.lexists(wal)
        ):
            raise _SourceDatabaseError(
                "source_database_changed",
                "The source database changed while its private copy was created",
            )
        os.replace(temporary, destination)
        pristine: Path | None = None
        if preserve_pristine:
            pristine = work_directory / "source.duckdb"
            os.link(destination, pristine)
        _fsync_directory(work_directory)
        return destination, digest.hexdigest(), pristine
    except _SourceDatabaseError:
        shutil.rmtree(work_directory, ignore_errors=True)
        raise
    except OSError as error:
        shutil.rmtree(work_directory, ignore_errors=True)
        raise _SourceDatabaseError(
            "source_database_copy_failed",
            "The source database could not be copied into the private run workspace",
        ) from error
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = sha256()
    with os.fdopen(descriptor, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validation_error_key(error: JsonSchemaValidationError) -> tuple[str, str]:
    return ("/".join(map(str, error.absolute_path)), "/".join(map(str, error.absolute_schema_path)))


def _validation_feedback(
    errors: list[JsonSchemaValidationError],
    max_bytes: int,
) -> str:
    issues: list[dict[str, JsonValue]] = [
        {
            "result_path": list(error.absolute_path),
            "schema_path": list(error.absolute_schema_path),
            "keyword": str(error.validator),
            "message": error.message,
        }
        for error in errors
    ]
    reduced_issues: list[dict[str, JsonValue]] = [
        {
            "result_path": issue["result_path"],
            "keyword": issue["keyword"],
        }
        for issue in issues
    ]
    return _bounded_retry_feedback(
        [
            {"error": "answer_schema_validation_failed", "issues": issues},
            {
                "error": "answer_schema_validation_failed",
                "issues": reduced_issues,
            },
            {"error": "answer_schema_validation_failed"},
        ],
        max_bytes,
    )


def _bounded_retry_feedback(
    candidates: Sequence[object],
    max_bytes: int,
) -> str:
    for candidate in candidates:
        encoded = json.dumps(
            candidate,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(encoded.encode("utf-8")) <= max_bytes:
            return encoded
    raise ToolResultLimitExceeded("retry feedback exceeds the per-result byte limit")


def _validate_run_id(run_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise ValueError("identity factory returned an unsafe run identity")


def _aware_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("clock must return timezone-aware timestamps")
    return value


def _bounded_text(value: str, limit: int = 2_000) -> str:
    return value if len(value) <= limit else value[:limit] + "…"


def _tool_call_id(context: RunContext[None]) -> str:
    tool_call_id = context.tool_call_id
    if tool_call_id is None:
        raise RuntimeError("Pydantic AI invoked a tool without a tool call identity")
    return tool_call_id


def _model_visible_tool_result_sizes(
    messages: tuple[dict[str, JsonValue], ...],
) -> list[int]:
    sizes: list[int] = []
    for message in messages:
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict) or part.get("part_kind") not in {
                "tool-return",
                "retry-prompt",
            }:
                continue
            content = part.get("content")
            if isinstance(content, str):
                sizes.append(len(content.encode("utf-8")))
            else:
                sizes.append(
                    len(
                    json.dumps(
                        content,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    )
                )
    return sizes


def _include_pending_model_request(
    messages: tuple[dict[str, JsonValue], ...],
    next_node: Any,
) -> tuple[dict[str, JsonValue], ...]:
    if not isinstance(next_node, ModelRequestNode):
        return messages
    dumped = ModelMessagesTypeAdapter.dump_python([next_node.request], mode="json")
    pending = cast(dict[str, JsonValue], dumped[0])
    if messages and messages[-1] == pending:
        return messages
    return (*messages, pending)
