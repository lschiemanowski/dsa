"""Optional Databricks MLflow projection of canonical DSA runs."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from hashlib import sha256
from importlib import import_module
from threading import Lock
from typing import Any, Literal, Protocol, Self, cast

from pydantic import JsonValue, field_validator, model_validator

from dsa.contract import ContractModel, RunRequest
from dsa.record import RetainedTerminalRecord, RunFailure, TerminalRecord


class MlflowReporting(ContractModel):
    """Safe operator-visible result of the optional reporting projection."""

    status: Literal["disabled", "reported", "failed"] = "disabled"
    tracking_run_id: str | None = None
    trace_id: str | None = None
    failure_code: str | None = None

    @field_validator("tracking_run_id", "trace_id")
    @classmethod
    def validate_remote_id(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", value
        ) is None:
            raise ValueError("remote identifiers must use the safe identifier alphabet")
        return value

    @field_validator("failure_code")
    @classmethod
    def validate_failure_code(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", value) is None:
            raise ValueError("failure code must be a stable safe identifier")
        return value

    @model_validator(mode="after")
    def validate_state(self) -> MlflowReporting:
        if self.status == "disabled":
            if any((self.tracking_run_id, self.trace_id, self.failure_code)):
                raise ValueError("disabled reporting must not contain remote state")
        elif self.status == "reported":
            if not self.tracking_run_id or not self.trace_id or self.failure_code is not None:
                raise ValueError("reported status requires both remote identifiers")
        elif not self.failure_code:
            raise ValueError("failed reporting requires a stable failure code")
        return self


class _ReportableCompletion(Protocol):
    record: TerminalRecord
    retained_record: RetainedTerminalRecord

    def with_reporting(self, reporting: MlflowReporting) -> Self: ...


class _ReportingBackend(Protocol):
    async def run[CompletionT: _ReportableCompletion](
        self,
        *,
        enabled: bool,
        run_id: str,
        request: RunRequest,
        operation: Callable[[], Awaitable[CompletionT]],
    ) -> CompletionT: ...


_AUTOLOG_LOCK = Lock()
_autolog_initialized = False


async def run_with_mlflow_reporting[CompletionT: _ReportableCompletion](
    *,
    enabled: bool,
    run_id: str,
    request: RunRequest,
    operation: Callable[[], Awaitable[CompletionT]],
    backend: _ReportingBackend | None = None,
) -> CompletionT:
    """Execute once and attach a non-canonical reporting projection."""
    selected = backend
    dependency_failure: str | None = None
    if selected is None:
        try:
            selected = _MlflowBackend()
        except ModuleNotFoundError:
            dependency_failure = "mlflow_dependency_missing"
        except Exception:
            dependency_failure = "mlflow_dependency_unavailable"

    if selected is not None:
        return await selected.run(
            enabled=enabled,
            run_id=run_id,
            request=request,
            operation=operation,
        )

    try:
        completion = await operation()
    except asyncio.CancelledError as error:
        failure_code = _configuration_failure() or dependency_failure
        cast(Any, error).reporting = (
            MlflowReporting(
                status="failed",
                failure_code=failure_code,
            )
            if enabled
            else MlflowReporting()
        )
        raise
    if not enabled:
        return completion.with_reporting(MlflowReporting())
    failure_code = _configuration_failure() or dependency_failure
    return completion.with_reporting(
        MlflowReporting(status="failed", failure_code=failure_code)
    )


class _MlflowBackend:
    """Lazy MLflow adapter; importing dsa never requires the optional extra."""

    def __init__(self) -> None:
        from mlflow import MlflowClient
        from mlflow.entities import Metric, Param, RunTag, SpanType
        from mlflow.entities.trace_location import MlflowExperimentLocation

        self.mlflow: Any = import_module("mlflow")
        self.pydantic_ai: Any = import_module("mlflow.pydantic_ai")
        self.client_type = MlflowClient
        self.metric_type = Metric
        self.param_type = Param
        self.tag_type = RunTag
        self.span_type = SpanType
        self.location_type = MlflowExperimentLocation

    async def run[CompletionT: _ReportableCompletion](
        self,
        *,
        enabled: bool,
        run_id: str,
        request: RunRequest,
        operation: Callable[[], Awaitable[CompletionT]],
    ) -> CompletionT:
        if not enabled:
            try:
                completion = await self._execute_with_tracing_disabled(operation)
            except asyncio.CancelledError as error:
                cast(Any, error).reporting = MlflowReporting()
                raise
            return completion.with_reporting(MlflowReporting())

        failure_code = _configuration_failure()
        if failure_code is not None:
            return await self._run_without_export(operation, failure_code)

        try:
            _ensure_autolog(self.pydantic_ai)
            experiment_id = os.environ["MLFLOW_EXPERIMENT_ID"]
            destination = self.location_type(experiment_id=experiment_id)
        except Exception:
            return await self._run_without_export(operation, "mlflow_setup_failed")

        completion: CompletionT | None = None
        trace_id: str | None = None
        operation_started = False
        try:
            with self.mlflow.tracing.context(
                enabled=True,
                tags={"dsa.run_id": run_id, "dsa.component": "analysis"},
            ), self.mlflow.start_span(
                name="dsa.run",
                span_type=self.span_type.AGENT,
                attributes={"dsa.run_id": run_id, "dsa.schema_version": "1"},
                trace_destination=destination,
            ) as span:
                trace_id = _safe_remote_id(span.trace_id)
                if trace_id is None:
                    raise ValueError("MLflow returned an unsafe trace identifier")
                span.set_inputs(
                    {
                        "run_id": run_id,
                        "model_name": request.model.name,
                        "answer_schema": request.answer_schema,
                    }
                )
                operation_started = True
                completion = await operation()
                span.set_outputs(_safe_trace_output(completion.record))
        except asyncio.CancelledError as error:
            reporting = MlflowReporting(
                status="failed",
                trace_id=trace_id,
                failure_code="mlflow_cancelled",
            )
            retained_error = cast(Any, error)
            if completion is not None:
                retained_error.terminal_record = completion.record
                retained_error.retained_record = completion.retained_record
            retained_error.reporting = reporting
            raise
        except BaseException:
            if completion is None:
                if not operation_started:
                    return await self._run_without_export(
                        operation,
                        "mlflow_trace_setup_failed",
                        trace_id=trace_id,
                    )
                raise
            return completion.with_reporting(
                MlflowReporting(
                    status="failed",
                    trace_id=trace_id,
                    failure_code="mlflow_trace_export_failed",
                )
            )

        assert completion is not None
        assert trace_id is not None
        try:
            reporting = await self._export_completion(
                experiment_id,
                trace_id,
                completion,
            )
        except asyncio.CancelledError as error:
            retained_error = cast(Any, error)
            retained_error.terminal_record = completion.record
            retained_error.retained_record = completion.retained_record
            retained_error.reporting = MlflowReporting(
                status="failed",
                trace_id=trace_id,
                failure_code="mlflow_cancelled",
            )
            raise
        return completion.with_reporting(reporting)

    async def _run_without_export[CompletionT: _ReportableCompletion](
        self,
        operation: Callable[[], Awaitable[CompletionT]],
        failure_code: str,
        *,
        tracking_run_id: str | None = None,
        trace_id: str | None = None,
    ) -> CompletionT:
        reporting = MlflowReporting(
            status="failed",
            tracking_run_id=tracking_run_id,
            trace_id=trace_id,
            failure_code=failure_code,
        )
        try:
            completion = await self._execute_with_tracing_disabled(operation)
        except asyncio.CancelledError as error:
            cast(Any, error).reporting = reporting
            raise
        return completion.with_reporting(
            reporting
        )

    async def _execute_with_tracing_disabled[CompletionT: _ReportableCompletion](
        self,
        operation: Callable[[], Awaitable[CompletionT]],
    ) -> CompletionT:
        completion: CompletionT | None = None
        operation_started = False
        try:
            with self.mlflow.tracing.context(enabled=False):
                operation_started = True
                completion = await operation()
        except asyncio.CancelledError:
            raise
        except Exception:
            if completion is not None:
                return completion
            if not operation_started:
                return await operation()
            raise
        assert completion is not None
        return completion

    async def _export_completion(
        self,
        experiment_id: str,
        trace_id: str,
        completion: _ReportableCompletion,
        *,
        terminal_status: str = "FINISHED",
    ) -> MlflowReporting:
        try:
            terminal_text = _verified_terminal_text(completion.retained_record)
            manifest_text = _artifact_manifest_text(completion.record)
            metrics, params, tags = self._run_metadata(completion.record)
        except Exception:
            return MlflowReporting(
                status="failed",
                trace_id=trace_id,
                failure_code="mlflow_export_failed",
            )

        def export() -> MlflowReporting:
            client: Any | None = None
            tracking_run_id: str | None = None
            created_run_id: str | None = None
            try:
                client = self.client_type(tracking_uri="databricks")
                tracking_run = client.create_run(
                    experiment_id,
                    tags={
                        "dsa.run_id": completion.record.run_id,
                        "dsa.component": "analysis",
                    },
                    run_name=f"dsa-{completion.record.run_id}",
                )
                raw_tracking_run_id = cast(object, tracking_run.info.run_id)
                if isinstance(raw_tracking_run_id, str):
                    created_run_id = raw_tracking_run_id
                tracking_run_id = _safe_remote_id(raw_tracking_run_id)
                if tracking_run_id is None:
                    raise ValueError("MLflow returned an unsafe tracking run identifier")
                self.mlflow.flush_trace_async_logging()
                client.log_text(tracking_run_id, terminal_text, "dsa/terminal.json")
                client.log_text(tracking_run_id, manifest_text, "dsa/artifacts.json")
                client.log_batch(
                    tracking_run_id,
                    metrics=metrics,
                    params=params,
                    tags=tags,
                    synchronous=True,
                )
                client.link_traces_to_run([trace_id], tracking_run_id)
                client.set_terminated(tracking_run_id, terminal_status)
            except Exception:
                if client is not None and created_run_id is not None:
                    _best_effort_terminate(client, created_run_id, "FAILED")
                return MlflowReporting(
                    status="failed",
                    tracking_run_id=tracking_run_id,
                    trace_id=trace_id,
                    failure_code="mlflow_export_failed",
                )
            return MlflowReporting(
                status="reported",
                tracking_run_id=tracking_run_id,
                trace_id=trace_id,
            )

        try:
            return await asyncio.to_thread(export)
        except asyncio.CancelledError:
            raise
        except Exception:
            return MlflowReporting(
                status="failed",
                trace_id=trace_id,
                failure_code="mlflow_export_failed",
            )

    def _run_metadata(
        self,
        record: TerminalRecord,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        timestamp = int(record.finished_at.timestamp() * 1000)
        elapsed = (record.finished_at - record.started_at).total_seconds()
        metrics = [
            self.metric_type("dsa.elapsed_seconds", elapsed, timestamp, 0),
            self.metric_type("dsa.artifact_count", len(record.artifacts), timestamp, 0),
            self.metric_type(
                "dsa.artifact_bytes",
                sum(artifact.size_bytes for artifact in record.artifacts),
                timestamp,
                0,
            ),
            self.metric_type("dsa.message_count", len(record.messages), timestamp, 0),
        ]
        for key in ("requests", "input_tokens", "output_tokens", "total_tokens"):
            value = record.usage.get(key)
            if type(value) in (int, float):
                metrics.append(self.metric_type(f"dsa.usage.{key}", value, timestamp, 0))
        params = [
            self.param_type("dsa.model_name", record.request.model.name),
            self.param_type("dsa.schema_version", record.schema_version),
        ]
        tags = [
            self.tag_type("dsa.run_id", record.run_id),
            self.tag_type("dsa.outcome", record.outcome.status),
        ]
        if isinstance(record.outcome, RunFailure):
            tags.extend(
                [
                    self.tag_type("dsa.failure_stage", record.outcome.failure.stage),
                    self.tag_type("dsa.failure_code", record.outcome.failure.code),
                ]
            )
        return metrics, params, tags


def _ensure_autolog(pydantic_ai_integration: Any) -> None:
    global _autolog_initialized
    with _AUTOLOG_LOCK:
        if _autolog_initialized:
            return
        pydantic_ai_integration.autolog(log_traces=True, silent=True)
        _autolog_initialized = True


def _configuration_failure() -> str | None:
    if os.environ.get("MLFLOW_TRACKING_URI") != "databricks":
        return "mlflow_tracking_uri_invalid"
    required = ("MLFLOW_EXPERIMENT_ID", "DATABRICKS_HOST", "DATABRICKS_TOKEN")
    if any(not os.environ.get(key, "").strip() for key in required):
        return "mlflow_configuration_missing"
    return None


def _verified_terminal_text(retained: RetainedTerminalRecord) -> str:
    content = retained.path.read_bytes()
    if len(content) != retained.byte_length or sha256(content).hexdigest() != retained.sha256:
        raise ValueError("terminal integrity check failed")
    return content.decode("utf-8")


def _artifact_manifest_text(record: TerminalRecord) -> str:
    manifest: dict[str, JsonValue] = {
        "schema_version": record.schema_version,
        "run_id": record.run_id,
        "artifacts": [artifact.model_dump(mode="json") for artifact in record.artifacts],
    }
    return json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _safe_trace_output(record: TerminalRecord) -> dict[str, JsonValue]:
    output: dict[str, JsonValue] = {
        "run_id": record.run_id,
        "status": record.outcome.status,
        "artifact_count": len(record.artifacts),
    }
    if isinstance(record.outcome, RunFailure):
        output["failure_stage"] = record.outcome.failure.stage
        output["failure_code"] = record.outcome.failure.code
    return output


def _safe_remote_id(value: object) -> str | None:
    if isinstance(value, str) and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", value
    ):
        return value
    return None


def _best_effort_terminate(client: Any, run_id: str, status: str) -> None:
    with suppress(BaseException):
        client.set_terminated(run_id, status)
