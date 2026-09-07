"""Framework-neutral clarify, confirm, execute, and stop conversation flow."""

from __future__ import annotations

import asyncio
import json
import re
import tomllib
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

import tomli_w
from pydantic import JsonValue

from apps.private_data_chat.broker import InMemoryProposalStore, PrivateDataBroker
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
    proposal_payload_json,
)
from apps.private_data_chat.presentation import escape_markdown_text
from dsa.plots import VerifiedPlot, notebook_plots

Confirmation = Callable[[ProposalRecord], Awaitable[bool]]


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
    """Own the small amount of state required for one Chainlit chat session."""

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
        self._active = False
        self._terminal = False

    async def handle(self, message: str, confirm: Confirmation) -> ChatResponse:
        """Handle one complete turn, including confirmation and trusted execution."""
        reservation = await self._reserve()
        if reservation == "terminal":
            return ChatResponse(_closed_message(), terminal=True)
        if reservation == "active":
            return ChatResponse("Another turn is already in progress in this conversation.")

        try:
            try:
                history = append_history(self._history, "user", message)
                turn = await self._clarifier.clarify(history)
            except Exception:
                return ChatResponse(
                    "The clarification step is unavailable. Check the application configuration."
                )

            if turn.kind == "clarification":
                self._history = append_history(history, "assistant", turn.message)
                return ChatResponse(turn.message)

            assert turn.proposal is not None
            if turn.proposal.analysis_guidance is not None and not self._enable_analysis_guidance:
                return ChatResponse(
                    "The proposed analysis could not be prepared safely. Refine the question "
                    "and try again."
                )
            try:
                proposal = await self._broker.propose(
                    user_id=self._user_id,
                    conversation_id=self._conversation_id,
                    data_source_id=self._context.data_source_id,
                    payload=turn.proposal,
                )
            except Exception:
                return ChatResponse(
                    "The proposed analysis could not be prepared safely. Refine the question "
                    "and try again."
                )

            approved = False
            with suppress(Exception):
                approved = await confirm(proposal)
            if not approved:
                proposal_text = json.dumps(
                    proposal_payload_json(proposal.payload),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
                self._history = append_history(
                    history,
                    "assistant",
                    f"{turn.message}\n\nUnapproved proposal: {proposal_text}",
                )
                return ChatResponse(
                    "The analysis was not run. You can continue refining the question."
                )

            await self._mark_terminal()
            try:
                await self._broker.approve(
                    proposal_id=proposal.proposal_id,
                    proposal_sha256=proposal.proposal_sha256,
                    user_id=self._user_id,
                    conversation_id=self._conversation_id,
                )
                result = await self._broker.execute(
                    proposal_id=proposal.proposal_id,
                    user_id=self._user_id,
                    conversation_id=self._conversation_id,
                )
                return self._terminal_response(result)
            except asyncio.CancelledError:
                raise
            except Exception:
                return ChatResponse(
                    _terminal_failure("analysis_orchestration_failed"),
                    terminal=True,
                )
        finally:
            await self._release()

    async def _reserve(self) -> str:
        async with self._state_lock:
            if self._terminal:
                return "terminal"
            if self._active:
                return "active"
            self._active = True
            return "reserved"

    async def _mark_terminal(self) -> None:
        async with self._state_lock:
            self._terminal = True

    async def _release(self) -> None:
        async with self._state_lock:
            self._active = False

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
