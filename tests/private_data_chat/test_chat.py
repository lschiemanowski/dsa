from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from apps.private_data_chat.broker import BrokerError
from apps.private_data_chat.chat import ChatSnapshot, PrivateDataChatSession, ProposalDecision
from apps.private_data_chat.clarifier import ConversationMessage
from apps.private_data_chat.contracts import (
    AnalysisRequest,
    AnalysisResult,
    ArtifactIdentity,
    ClarifierTurn,
    MockDatabaseContext,
    MockRelation,
    ProposalPayload,
)

from .test_contracts import artifact, proposal_payload


class StubClarifier:
    def __init__(self, *turns: ClarifierTurn, gate: asyncio.Event | None = None) -> None:
        self._turns = list(turns)
        self.gate = gate
        self.calls: list[tuple[ConversationMessage, ...]] = []

    async def clarify(self, history: Sequence[ConversationMessage]) -> ClarifierTurn:
        self.calls.append(tuple(history))
        if self.gate is not None:
            await self.gate.wait()
        if not self._turns:
            raise RuntimeError("no response")
        return self._turns.pop(0)


class StubExecutor:
    def __init__(
        self,
        *,
        notebook: bytes | None = None,
        fail: bool = False,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.notebook = notebook
        self.fail = fail
        self.gate = gate
        self.requests: list[AnalysisRequest] = []

    async def execute(self, request: AnalysisRequest) -> AnalysisResult:
        self.requests.append(request.model_copy(deep=True))
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            return AnalysisResult(
                run_id=request.run_id,
                proposal_id=request.proposal_id,
                status="failed",
                failure_code="dsa_run_failed",
            )
        return AnalysisResult(
            run_id=request.run_id,
            proposal_id=request.proposal_id,
            status="succeeded",
            answer={"total": 10.0, "months": ["2011-01"]},
            terminal=artifact(),
            notebook=artifact("notebook") if self.notebook is not None else None,
        )

    def read_notebook(self, identity: ArtifactIdentity) -> bytes:
        assert identity == artifact("notebook")
        if self.notebook is None:
            raise AssertionError("no notebook configured")
        return self.notebook


def clarification(message: str = "Which calendar year should I use?") -> ClarifierTurn:
    return ClarifierTurn(kind="clarification", message=message)


def proposal() -> ClarifierTurn:
    return ClarifierTurn(
        kind="proposal",
        message="The quantitative question is ready.",
        proposal=proposal_payload(),
    )


def proposal_with_guidance() -> ClarifierTurn:
    payload = ProposalPayload.model_validate(
        {
            **proposal_payload().model_dump(mode="python"),
            "analysis_guidance": (
                "Filter to 2011, compute quantity * unit_price_gbp, and aggregate by month."
            ),
        }
    )
    return ClarifierTurn(
        kind="proposal",
        message="The quantitative question and analysis guidance are ready.",
        proposal=payload,
    )


def context() -> MockDatabaseContext:
    return MockDatabaseContext(
        data_source_id="retail",
        display_name="Synthetic retail example",
        relations=(
            MockRelation(
                name="invoice_lines",
                columns=("invoice_id", "quantity"),
                sample_rows=({"invoice_id": "FAKE-001", "quantity": 2},),
            ),
        ),
    )


def session(
    clarifier: StubClarifier,
    executor: StubExecutor,
    *,
    enable_analysis_guidance: bool = False,
) -> PrivateDataChatSession:
    return PrivateDataChatSession(
        user_id="user-1",
        conversation_id="chat-1",
        context=context(),
        clarifier=clarifier,
        executor=executor,
        enable_analysis_guidance=enable_analysis_guidance,
    )


async def decide(
    chat: PrivateDataChatSession, text: str, decision: ProposalDecision = "approve"
) -> ChatSnapshot:
    reply = await chat.submit_message(text)
    proposal = reply.proposal
    assert proposal is not None
    await chat.decide_proposal(proposal.proposal_id, proposal.proposal_sha256, decision)
    async with asyncio.timeout(2):
        state = await chat.snapshot()
        while state.status == "running":
            await asyncio.sleep(0)
            state = await chat.snapshot()
        return state


async def test_clarification_retains_only_user_visible_conversation_text() -> None:
    clarifier = StubClarifier(clarification(), clarification("Should returns be excluded?"))
    chat = session(clarifier, StubExecutor())

    first = await chat.submit_message("Show monthly sales.")
    second = await chat.submit_message("Use calendar year 2011.")

    assert first.text == "Which calendar year should I use?"
    assert second.text == "Should returns be excluded?"
    assert clarifier.calls[1] == (
        ConversationMessage(role="user", content="Show monthly sales."),
        ConversationMessage(role="assistant", content=first.text),
        ConversationMessage(role="user", content="Use calendar year 2011."),
    )


async def test_exact_confirmation_runs_once_and_then_closes_conversation() -> None:
    clarifier = StubClarifier(proposal())
    executor = StubExecutor()
    chat = session(clarifier, executor)

    state = await decide(chat, "Use 2011 and report monthly net sales.")

    assert state.status == "succeeded"
    assert state.result is not None
    assert state.result.answer == {"total": 10.0, "months": ["2011-01"]}
    assert len(executor.requests) == 1
    assert executor.requests[0].data_source_id == "retail"
    assert executor.requests[0].request_derivation is True

    with pytest.raises(BrokerError, match="conversation_closed"):
        await chat.submit_message("Now break it down by country.")
    assert len(clarifier.calls) == 1
    assert len(executor.requests) == 1


@pytest.mark.parametrize("fields", [
    {}, {"time_window": None}, {"units": None},
    {"time_window": None, "units": None},
    {"time_window": "Calendar year 2024", "units": None},
    {"time_window": None, "units": "GWh"},
])
async def test_nullable_interpretation_can_be_approved(fields: dict[str, str | None]) -> None:
    raw = proposal_payload().model_dump(mode="json")
    raw["interpretation"].pop("time_window")
    raw["interpretation"].pop("units")
    raw["interpretation"].update(fields)
    payload = ProposalPayload.model_validate(raw)
    executor = StubExecutor()
    chat = session(
        StubClarifier(ClarifierTurn(kind="proposal", message="Ready", proposal=payload)), executor,
    )

    state = await decide(chat, "Analyze it.")

    assert state.status == "succeeded"
    approved = state.messages[1].proposal
    assert approved is not None
    assert approved.payload == payload
    assert len(executor.requests) == 1


async def test_declined_confirmation_does_not_run_and_allows_refinement() -> None:
    clarifier = StubClarifier(proposal(), clarification())
    executor = StubExecutor()
    chat = session(clarifier, executor)

    declined = await decide(chat, "Use 2011.", "revise")
    continued = await chat.submit_message("Actually use fiscal year 2011.")

    assert declined.status == "open"
    assert "not run" in declined.messages[-1].text
    assert continued.text == "Which calendar year should I use?"
    assert executor.requests == []
    assert len(clarifier.calls) == 2


async def test_guidance_is_rejected_when_disabled_and_forwarded_exactly_when_enabled() -> None:
    disabled_executor = StubExecutor()
    disabled = session(StubClarifier(proposal_with_guidance()), disabled_executor)
    rejected = await disabled.submit_message("Use the proposed plan.")
    assert "could not be prepared safely" in rejected.text
    assert disabled_executor.requests == []

    enabled_executor = StubExecutor()
    enabled = session(
        StubClarifier(proposal_with_guidance()),
        enabled_executor,
        enable_analysis_guidance=True,
    )
    result = await decide(enabled, "Use the proposed plan.")
    assert result.status == "succeeded"
    assert enabled_executor.requests[0].analysis_guidance == (
        "Filter to 2011, compute quantity * unit_price_gbp, and aggregate by month."
    )


async def test_approved_plot_permission_and_notebook_reach_trusted_executor() -> None:
    payload = proposal_payload().model_copy(update={"allow_plots": True})
    clarifier = StubClarifier(ClarifierTurn(kind="proposal", proposal=payload, message="Ready"))
    notebook = b'{"cells":[],"nbformat":4,"nbformat_minor":5}\n'
    executor = StubExecutor(notebook=notebook)
    chat = session(clarifier, executor)

    state = await decide(chat, "Show a chart")

    assert state.status == "succeeded"
    assert executor.requests[0].allow_plots
    assert state.result is not None and state.result.notebook is not None
    assert chat.read_notebook(state.result.notebook) == notebook
    assert len(clarifier.calls) == 1


async def test_dsa_failure_is_terminal() -> None:
    clarifier = StubClarifier(proposal())
    executor = StubExecutor(fail=True)
    chat = session(clarifier, executor)

    state = await decide(chat, "Use 2011.")

    assert state.status == "failed"
    assert state.failure_code == "dsa_run_failed"
    with pytest.raises(BrokerError, match="conversation_closed"):
        await chat.submit_message("Try again.")
    assert len(clarifier.calls) == 1
    assert len(executor.requests) == 1


async def test_overlapping_turns_are_rejected_and_confirmation_makes_terminal() -> None:
    execution_allowed = asyncio.Event()
    clarifier = StubClarifier(proposal())
    executor = StubExecutor(gate=execution_allowed)
    chat = session(clarifier, executor)

    reply = await chat.submit_message("Use 2011.")
    assert reply.proposal is not None
    with pytest.raises(BrokerError, match="proposal_pending"):
        await chat.submit_message("Run another version.")
    assert executor.requests == []

    await chat.decide_proposal(
        reply.proposal.proposal_id, reply.proposal.proposal_sha256, "approve"
    )
    try:
        assert (await chat.snapshot()).status == "running"
        with pytest.raises(BrokerError, match="conversation_closed"):
            await chat.submit_message("Try while DSA runs.")
    finally:
        execution_allowed.set()
    async with asyncio.timeout(2):
        while (await chat.snapshot()).status == "running":
            await asyncio.sleep(0)
    assert len(clarifier.calls) == 1
    assert len(executor.requests) == 1


async def test_clarifier_failure_is_safe_and_releases_reservation() -> None:
    clarifier = StubClarifier()
    chat = session(clarifier, StubExecutor())

    first = await chat.submit_message("Analyze it.")
    second = await chat.submit_message("Try once more.")

    assert "unavailable" in first.text
    assert "unavailable" in second.text
    assert len(clarifier.calls) == 2
