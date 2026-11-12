"""Framework-neutral clarify, confirm, execute, and stop conversation flow."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, Protocol

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

ProposalDecision = Literal["approve", "revise"]
ChatStatus = Literal["open", "clarifying", "awaiting_decision", "running", "succeeded", "failed"]
_NOT_RUN = "The analysis was not run. You can continue refining the question."


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


def _bounded_identity(value: str, label: str) -> str:
    if not value.strip() or len(value) > 256:
        raise ValueError(f"{label} is invalid")
    return value
