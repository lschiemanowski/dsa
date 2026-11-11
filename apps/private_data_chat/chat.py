"""Framework-neutral clarify, confirm, execute, and stop conversation flow."""

from __future__ import annotations

import asyncio
import json
import re
import tomllib
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, Protocol

import tomli_w
from pydantic import JsonValue

from apps.private_data_chat.broker import BrokerError, InMemoryProposalStore, PrivateDataBroker
from apps.private_data_chat.clarifier import (
    ConversationMessage,
    append_history,
)
from apps.private_data_chat.contracts import (
    AnalysisRequest,
    AnalysisResult,
    ArtifactIdentity,
    ClarifierTurn,
    MockDatabaseContext,
    ProposalRecord,
    ProposalStatus,
    proposal_payload_json,
)
from apps.private_data_chat.presentation import escape_markdown_text
from dsa.plots import VerifiedPlot, notebook_plots

Confirmation = Callable[[ProposalRecord], Awaitable[bool]]
ProposalDecision = Literal["approve", "revise"]
ChatStatus = Literal["open", "clarifying", "awaiting_decision", "running", "succeeded", "failed"]
_NOT_RUN = "The analysis was not run. You can continue refining the question."
_BUSY = "Another turn is already in progress in this conversation."


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    text: str
    proposal: ProposalRecord | None = None


@dataclass(frozen=True)
class ChatSnapshot:
    status: ChatStatus
    messages: tuple[ChatMessage, ...]
    pending_proposal: ProposalRecord | None
    result: AnalysisResult | None
    failure_code: str | None = None


class Clarifier(Protocol):
    """Untrusted, tool-free clarification boundary."""

    async def clarify(self, history: Sequence[ConversationMessage]) -> ClarifierTurn: ...


class SessionExecutor(Protocol):
    """Trusted execution and verified-notebook boundary used by one session."""

    async def execute(self, request: AnalysisRequest) -> AnalysisResult: ...

    def read_notebook(self, identity: ArtifactIdentity) -> bytes: ...


@dataclass(frozen=True)
class NotebookDownload:
    """Verified notebook bytes ready for a UI download element."""

    content: bytes
    sha256: str


@dataclass(frozen=True)
class ChatResponse:
    """One UI-independent response from the private-data conversation."""

    content: str
    terminal: bool = False
    notebook: NotebookDownload | None = None
    plots: tuple[VerifiedPlot, ...] = ()


class PrivateDataChatSession:
    """Own clarification, proposal review, and at most one analysis run."""

    def __init__(
        self,
        *,
        user_id: str,
        conversation_id: str,
        context: MockDatabaseContext,
        clarifier: Clarifier,
        executor: SessionExecutor,
        enable_analysis_guidance: bool = False,
    ) -> None:
        self._user_id = _bounded_identity(user_id, "user identity")
        self._conversation_id = _bounded_identity(conversation_id, "conversation identity")
        self._context = context.model_copy(deep=True)
        self._clarifier = clarifier
        self._executor = executor
        self._enable_analysis_guidance = enable_analysis_guidance
        self._broker = PrivateDataBroker(
            store=InMemoryProposalStore(),
            executor=executor,
        )
        self._history: tuple[ConversationMessage, ...] = ()
        self._state_lock = asyncio.Lock()
        self._clarifying = False
        self._messages: list[ChatMessage] = []
        self._pending: ProposalRecord | None = None
        self._approved: ProposalRecord | None = None
        self._run: asyncio.Task[AnalysisResult] | None = None

    async def submit_message(self, message: str) -> ChatMessage:
        """Clarify one message or publish a proposal for a separate decision."""
        async with self._state_lock:
            if self._run is not None:
                raise BrokerError("conversation_closed")
            await self._refresh_pending()
            if self._pending is not None:
                raise BrokerError("proposal_pending")
            if self._clarifying:
                raise BrokerError("conversation_busy")
            history = append_history(self._history, "user", message)
            self._messages.append(ChatMessage("user", message))
            self._clarifying = True

        try:
            return await self._clarify(history)
        finally:
            self._clarifying = False

    async def _clarify(self, history: Sequence[ConversationMessage]) -> ChatMessage:
        try:
            turn = await self._clarifier.clarify(history)
        except Exception:
            return self._reply(
                "The clarification step is unavailable. Check the application configuration."
            )

        if turn.kind == "clarification":
            self._history = append_history(history, "assistant", turn.message)
            return self._reply(turn.message)

        assert turn.proposal is not None
        try:
            if turn.proposal.analysis_guidance is not None and not self._enable_analysis_guidance:
                raise BrokerError("analysis_guidance_disabled")
            proposal = await self._broker.propose(
                user_id=self._user_id,
                conversation_id=self._conversation_id,
                data_source_id=self._context.data_source_id,
                payload=turn.proposal,
            )
        except Exception:
            return self._reply(
                "The proposed analysis could not be prepared safely. Refine the question "
                "and try again."
            )

        proposal_text = json.dumps(
            proposal_payload_json(proposal.payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        self._history = append_history(
            history, "assistant", f"{turn.message}\n\nUnapproved proposal: {proposal_text}"
        )
        self._pending = proposal
        return self._reply(turn.message, proposal)

    async def decide_proposal(
        self,
        proposal_id: str,
        digest: str,
        decision: ProposalDecision,
        *,
        run_slots: asyncio.BoundedSemaphore | None = None,
    ) -> bool:
        """Record a decision and return whether it scheduled a new run."""
        if decision not in ("approve", "revise"):
            raise BrokerError("invalid_decision")
        async with self._state_lock:
            proposal = self._approved if self._run is not None else self._pending
            if proposal is None or proposal.proposal_id != proposal_id:
                raise BrokerError("proposal_not_pending")
            if digest != proposal.proposal_sha256:
                raise BrokerError("proposal_digest_mismatch")
            if self._run is not None:
                if decision == "approve":
                    return False
                raise BrokerError("conversation_closed")

            if decision == "revise":
                if await self._refresh_pending():
                    raise BrokerError("proposal_expired")
                self._pending = None
                self._reply(_NOT_RUN)
                return False

            if run_slots is not None:
                if run_slots.locked():
                    if await self._refresh_pending():
                        raise BrokerError("proposal_expired")
                    raise BrokerError("analysis_capacity_reached")
                await run_slots.acquire()
            try:
                run = await self._broker.approve_and_start(
                    proposal_id=proposal_id,
                    proposal_sha256=digest,
                    user_id=self._user_id,
                    conversation_id=self._conversation_id,
                )
            except BaseException as error:
                if run_slots is not None:
                    run_slots.release()
                if isinstance(error, BrokerError) and error.code == "proposal_expired":
                    await self._refresh_pending()
                raise

            if run_slots is not None:
                slots = run_slots
                run.add_done_callback(lambda _: slots.release())
            self._approved = proposal
            self._pending = None
            self._run = run
            return True

    @property
    def is_busy(self) -> bool:
        return self._clarifying or (self._run is not None and not self._run.done())

    async def snapshot(self) -> ChatSnapshot:
        """Read the current conversation state, including any completed run."""
        async with self._state_lock:
            await self._refresh_pending()
            run = self._run
            result = None
            failure = None
            status: ChatStatus = "open"
            if run is not None:
                if not run.done():
                    status = "running"
                elif run.cancelled():
                    status = "failed"
                    failure = "analysis_cancelled"
                else:
                    try:
                        result = run.result()
                        status = result.status
                        failure = result.failure_code
                    except BrokerError as error:
                        status = "failed"
                        failure = error.code
                    except Exception:
                        status = "failed"
                        failure = "analysis_orchestration_failed"
            elif self._pending is not None:
                status = "awaiting_decision"
            elif self._clarifying:
                status = "clarifying"

            return ChatSnapshot(status, tuple(self._messages), self._pending, result, failure)

    def read_notebook(self, identity: ArtifactIdentity) -> bytes:
        return self._executor.read_notebook(identity)

    async def cancel_run(self) -> None:
        if self._run is not None and not self._run.done():
            self._run.cancel()
            with suppress(asyncio.CancelledError):
                await self._run

    async def _refresh_pending(self) -> bool:
        if self._pending is None:
            return False
        record = await self._broker.get(
            proposal_id=self._pending.proposal_id,
            user_id=self._user_id,
            conversation_id=self._conversation_id,
        )
        if record.status is not ProposalStatus.EXPIRED:
            return False
        self._pending = None
        self._reply("This proposal expired. Send another message to continue refining.")
        return True

    def _reply(self, text: str, proposal: ProposalRecord | None = None) -> ChatMessage:
        message = ChatMessage("assistant", text, proposal)
        self._messages.append(message)
        return message

    async def handle(self, message: str, confirm: Confirmation) -> ChatResponse:
        """Adapt the two-step session to Chainlit's inline approval prompt."""
        try:
            reply = await self.submit_message(message)
        except BrokerError as error:
            if error.code == "conversation_closed":
                return ChatResponse(_closed_message(), terminal=True)
            if error.code in {"conversation_busy", "proposal_pending"}:
                return ChatResponse(_BUSY)
            raise

        proposal = reply.proposal
        if proposal is None:
            return ChatResponse(reply.text)
        approved = False
        try:
            with suppress(Exception):
                approved = await confirm(proposal.model_copy(deep=True))
        finally:
            if not approved:
                with suppress(BrokerError):
                    await self.decide_proposal(
                        proposal.proposal_id, proposal.proposal_sha256, "revise"
                    )
        if not approved:
            return ChatResponse(_NOT_RUN)

        try:
            await self.decide_proposal(proposal.proposal_id, proposal.proposal_sha256, "approve")
        except BrokerError as error:
            if error.code in {"proposal_expired", "proposal_not_pending"}:
                return ChatResponse(_NOT_RUN)
            raise
        assert self._run is not None
        try:
            result = await self._run
        except BrokerError as error:
            return ChatResponse(_terminal_failure(error.code), terminal=True)
        except Exception:
            return ChatResponse(_terminal_failure("analysis_orchestration_failed"), terminal=True)
        return self._terminal_response(result)

    def _terminal_response(self, result: AnalysisResult) -> ChatResponse:
        if result.status == "failed":
            return ChatResponse(
                _terminal_failure(result.failure_code or "dsa_run_failed"),
                terminal=True,
            )
        notebook = None
        plots: tuple[VerifiedPlot, ...] = ()
        notebook_note = "No verified derivation notebook was produced."
        if result.notebook is not None:
            try:
                notebook = NotebookDownload(
                    content=self._executor.read_notebook(result.notebook),
                    sha256=result.notebook.sha256,
                )
                notebook_note = "Download the verified derivation notebook below."
            except Exception:
                notebook_note = (
                    "A verified notebook was retained, but its download could not be attached."
                )
        if notebook is not None:
            try:
                plots = notebook_plots(notebook.content)
            except Exception:
                notebook_note += " Plot previews could not be attached."
        return ChatResponse(
            "## Result\n\n"
            f"{_json_fence(result.answer)}\n\n"
            f"{notebook_note}\n\n"
            "This analysis is complete. Start a new conversation for another question.",
            terminal=True,
            notebook=notebook,
            plots=plots,
        )


def render_proposal(proposal: ProposalRecord) -> str:
    """Render the validated content; the approval digest stays internal."""
    payload = proposal_payload_json(proposal.payload)
    guidance = proposal.payload.analysis_guidance
    guidance_section = (
        "## Proposed analysis guidance\n\n"
        "This was produced from synthetic data only. The trusted model may correct or "
        "ignore it.\n\n"
        f"{_markdown_quote(guidance)}\n\n"
        if guidance is not None
        else ""
    )
    return (
        "The mock data was used only to clarify the request. Running the analysis sends "
        "the approved content below to the trusted DSA boundary.\n\n"
        f"{guidance_section}"
        "## Exact approved proposal\n\n"
        f"{render_proposal_toml(payload)}\n\n"
        "### Answer schema\n\n"
        f"{_json_fence(payload['answer_schema'])}"
    )


def _terminal_failure(code: str) -> str:
    return (
        f"The approved DSA analysis ended without an answer (`{code}`).\n\n"
        "This analysis is complete. Start a new conversation to try another question."
    )


def _closed_message() -> str:
    return "This analysis is already complete. Start a new conversation for another question."


def _json_fence(value: JsonValue | object) -> str:
    rendered = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    return _code_fence(rendered, "json")


def render_proposal_toml(payload: dict[str, JsonValue]) -> str:
    """Render request fields as TOML; the schema is displayed separately as JSON."""
    displayed = dict(payload)
    displayed.pop("answer_schema")
    interpretation = displayed.get("interpretation")
    if isinstance(interpretation, dict):
        # TOML has no null. These optional fields default to None when omitted;
        # omit them only in the display, without mutating the approved payload.
        displayed["interpretation"] = {
            key: value for key, value in interpretation.items()
            if key not in {"time_window", "units"} or value is not None
        }
    rendered = tomli_w.dumps(displayed, multiline_strings=True).rstrip()
    # TOML multiline strings normalize CRLF. Keep the approved text exact.
    if tomllib.loads(rendered) != displayed:
        rendered = tomli_w.dumps(displayed, multiline_strings=False).rstrip()
    return _code_fence(rendered, "toml")


def _code_fence(rendered: str, language: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", rendered)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{rendered}\n{fence}"


def _markdown_quote(value: str) -> str:
    """Keep every untrusted line visually inside one labeled block quote."""
    escaped = escape_markdown_text(value)
    return "\n".join(f"> {line}" if line else ">" for line in escaped.splitlines())


def _bounded_identity(value: str, label: str) -> str:
    if not value.strip() or len(value) > 256:
        raise ValueError(f"{label} is invalid")
    return value
