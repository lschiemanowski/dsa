"""Optional Logfire telemetry, separate from canonical DSA evidence."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from contextlib import ExitStack, suppress
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, cast

from pydantic import Field, model_validator
from pydantic_ai.capabilities import AbstractCapability

from dsa.contract import ContractModel
from dsa.cost import provider_cost_from_messages
from dsa.record import RunFailure, TerminalRecord

if TYPE_CHECKING:
    from logfire import Logfire
    from pydantic_ai import RunContext
    from pydantic_ai.capabilities.abstract import (
        ValidatedToolArgs,
        WrapModelRequestHandler,
        WrapToolExecuteHandler,
    )
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import ToolDefinition


class LogfireReporting(ContractModel):
    """Local telemetry status; recorded does not acknowledge remote ingestion."""

    status: Literal["disabled", "recorded", "failed"] = "disabled"
    trace_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    failure_code: str | None = Field(default=None, pattern=r"^logfire_[a-z0-9_]{1,120}$")

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if (
            (self.status == "disabled" and (self.trace_id or self.failure_code))
            or (self.status == "recorded" and (not self.trace_id or self.failure_code))
            or (self.status == "failed" and not self.failure_code)
        ):
            raise ValueError("invalid Logfire reporting state")
        return self


class _ReportableCompletion(Protocol):
    record: TerminalRecord

    def with_reporting(self, reporting: LogfireReporting) -> Self: ...


class LogfireConfigurationMissing(ValueError):
    """No explicit project write token was supplied."""


def configure_logfire(*, local: bool) -> Logfire:
    token = os.environ.get("LOGFIRE_TOKEN", "").strip()
    if not token:
        raise LogfireConfigurationMissing
    import logfire

    return logfire.configure(
        local=local,
        token=token,
        send_to_logfire=True,
        service_name="dsa",
        console=False,
        metrics=False,
        inspect_arguments=False,
        add_baggage_to_attributes=False,
        distributed_tracing=False,
    )


@cache
def _analysis_logfire() -> Logfire:
    # Per-instance configuration does not install instrumentation globally.
    return configure_logfire(local=True)



def flush_analysis_telemetry() -> bool:
    """Flush the shared analysis exporter at the cell worker boundary."""
    if not _analysis_logfire.cache_info().currsize:
        return True
    return _analysis_logfire().force_flush()


class AnalysisTelemetry(AbstractCapability[None]):
    """Trace Pydantic AI calls without capturing payloads or exception messages."""

    def __init__(self, sdk: Logfire) -> None:
        self.sdk = sdk
        self.failed = False

    async def _observe[T](
        self,
        name: str,
        action: Callable[[], Awaitable[T]],
        *,
        tool: str | None = None,
    ) -> T:
        stack = ExitStack()
        try:
            span = stack.enter_context(self.sdk.span(name, tool=tool))
        except Exception:
            self.failed = True
            with suppress(Exception):
                stack.close()
            return await action()
        try:
            return await action()
        except BaseException:
            try:
                span.set_attribute("dsa.call_failed", True)
            except Exception:
                self.failed = True
            raise
        finally:
            try:
                stack.close()
            except Exception:
                self.failed = True

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        return await self._observe("dsa.model.request", lambda: handler(request_context))

    async def wrap_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        return await self._observe("dsa.tool.execute", lambda: handler(args), tool=tool_def.name)


async def run_with_logfire_reporting[CompletionT: _ReportableCompletion](
    *,
    enabled: bool,
    run_id: str,
    operation: Callable[[AnalysisTelemetry | None], Awaitable[CompletionT]],
) -> CompletionT:
    """Run the operation exactly once, even if telemetry setup or recording fails."""
    if not enabled:
        return await operation(None)

    span_stack = ExitStack()
    try:
        from opentelemetry.trace import get_current_span

        sdk = _analysis_logfire()
        span = span_stack.enter_context(sdk.span("dsa.run"))
        span.set_attribute("dsa.run_id", run_id)
        context = get_current_span().get_span_context()
        if not context.is_valid:
            raise RuntimeError("missing trace context")
        trace_id = f"{context.trace_id:032x}"
    except Exception as error:
        with suppress(Exception):
            span_stack.close()
        code = (
            "logfire_token_missing"
            if isinstance(error, LogfireConfigurationMissing)
            else "logfire_dependency_missing"
            if isinstance(error, ModuleNotFoundError)
            else "logfire_setup_failed"
        )
        return (await operation(None)).with_reporting(
            LogfireReporting(status="failed", failure_code=code)
        )

    failure_code: str | None = None

    telemetry = AnalysisTelemetry(sdk)
    try:
        completion = await operation(telemetry)
        try:
            span.set_attributes(_terminal_attributes(completion.record))
        except Exception:
            failure_code = "logfire_recording_failed"
    except asyncio.CancelledError as error:
        cast(Any, error).reporting = LogfireReporting(
            status="failed", trace_id=trace_id, failure_code="logfire_cancelled"
        )
        raise
    finally:
        try:
            # No exception values or traceback locals from the analysis belong in telemetry.
            span_stack.close()
        except Exception:
            failure_code = "logfire_recording_failed"

    if telemetry.failed:
        failure_code = "logfire_recording_failed"
    return completion.with_reporting(
        LogfireReporting(status="failed", trace_id=trace_id, failure_code=failure_code)
        if failure_code
        else LogfireReporting(status="recorded", trace_id=trace_id)
    )


def _terminal_attributes(record: TerminalRecord) -> dict[str, str | int | float]:
    attributes: dict[str, str | int | float] = {
        "dsa.model_name": record.request.model.name,
        "dsa.outcome": record.outcome.status,
        "dsa.elapsed_seconds": (record.finished_at - record.started_at).total_seconds(),
        "dsa.artifact_count": len(record.artifacts),
        "dsa.artifact_bytes": sum(artifact.size_bytes for artifact in record.artifacts),
    }
    if isinstance(record.outcome, RunFailure):
        attributes["dsa.failure_stage"] = record.outcome.failure.stage
        attributes["dsa.failure_code"] = record.outcome.failure.code
    for key in ("requests", "input_tokens", "output_tokens", "total_tokens", "tool_calls"):
        value = record.usage.get(key)
        if type(value) is int and value >= 0:
            attributes[f"dsa.usage.{key}"] = value
    cost = provider_cost_from_messages(record.messages)
    attributes["dsa.provider_cost_status"] = cost.status
    if cost.status == "observed" and cost.amount_decimal is not None:
        attributes["dsa.provider_cost_usd"] = float(cost.amount_decimal)
    return attributes
