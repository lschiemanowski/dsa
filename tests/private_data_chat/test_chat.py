from __future__ import annotations

import asyncio
import base64
import json
import tomllib
from collections.abc import Sequence

import pytest
from pydantic import JsonValue

from apps.private_data_chat.chat import (
    PrivateDataChatSession,
    render_proposal,
    render_proposal_toml,
)
from apps.private_data_chat.clarifier import ConversationMessage
from apps.private_data_chat.contracts import (
    AnalysisRequest,
    AnalysisResult,
    ArtifactIdentity,
    ClarifierTurn,
    MockDatabaseContext,
    MockRelation,
    ProposalPayload,
    ProposalRecord,
)
from tests.test_derivation_plots import png

from .test_contracts import artifact, proposal_payload


def test_toml_projection_preserves_text_and_cannot_escape_code_fence() -> None:
    payload: dict[str, JsonValue] = {
        "question": 'Quotes " and backslashes \\ and Unicode £',
        "analysis_guidance": '1. First\r\n2. ```\n![pixel](https://attacker.example)\n"""',
        "answer_schema": {
            "type": "object",
            "properties": {"value": {"enum": [None, 1, "null"]}},
        },
    }
    rendered = render_proposal_toml(payload)
    opening, body = rendered.split("\n", 1)
    fence = opening.removesuffix("toml")
    assert len(fence) > 3
    assert body.endswith("\n" + fence)
    decoded = tomllib.loads(body.removesuffix("\n" + fence))
    assert decoded == {key: value for key, value in payload.items() if key != "answer_schema"}


async def test_approved_plot_permission_and_images_stay_on_trusted_side() -> None:
    payload = proposal_payload().model_copy(update={"allow_plots": True})
    clarifier = StubClarifier(ClarifierTurn(kind="proposal", proposal=payload, message="Ready"))
    content = png()
    notebook = json.dumps(
        {
            "cells": [
                {
                    "metadata": {"dsa_plot": {"filename": "counts.png", "title": "Count"}},
                    "outputs": [{"data": {"image/png": base64.b64encode(content).decode()}}],
                }
            ]
        }
    ).encode()
    executor = StubExecutor(notebook=notebook)
    chat = session(clarifier, executor)
    approved: list[ProposalRecord] = []

    async def confirm(record: ProposalRecord) -> bool:
        approved.append(record)
        return True

    response = await chat.handle("Show a chart", confirm)
    assert approved[0].payload.allow_plots
    assert "allow_plots = true" in render_proposal(approved[0])
    assert executor.requests[0].allow_plots
    assert response.plots[0].content == content
    assert response.notebook is not None
    assert response.terminal
    await chat.handle("Explain this plot", confirm)
    assert len(clarifier.calls) == 1


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


async def approve(_proposal: object) -> bool:
    return True


async def reject(_proposal: object) -> bool:
    return False


async def test_clarification_retains_only_user_visible_conversation_text() -> None:
    clarifier = StubClarifier(
        clarification(),
        clarification("Should returns be excluded?"),
    )
    chat = session(clarifier, StubExecutor())

    first = await chat.handle("Show monthly sales.", approve)
    second = await chat.handle("Use calendar year 2011.", approve)

    assert first.content == "Which calendar year should I use?"
    assert second.content == "Should returns be excluded?"
    assert clarifier.calls[1] == (
        ConversationMessage(role="user", content="Show monthly sales."),
        ConversationMessage(role="assistant", content=first.content),
        ConversationMessage(role="user", content="Use calendar year 2011."),
    )


async def test_exact_confirmation_runs_once_and_then_closes_conversation() -> None:
    clarifier = StubClarifier(proposal())
    executor = StubExecutor()
    chat = session(clarifier, executor)
    seen: list[object] = []

    async def capture(record: object) -> bool:
        seen.append(record)
        return True

    result = await chat.handle("Use 2011 and report monthly net sales.", capture)

    assert result.terminal is True
    assert '"total": 10.0' in result.content
    assert "Start a new conversation" in result.content
    assert len(seen) == 1
    assert len(executor.requests) == 1
    assert executor.requests[0].data_source_id == "retail"
    assert executor.requests[0].request_derivation is True

    followup = await chat.handle("Now break it down by country.", approve)
    assert followup.terminal is True
    assert "already complete" in followup.content
    assert len(clarifier.calls) == 1
    assert len(executor.requests) == 1


@pytest.mark.parametrize("fields", [
    {}, {"time_window": None}, {"units": None},
    {"time_window": None, "units": None},
    {"time_window": "Calendar year 2024", "units": None},
    {"time_window": None, "units": "GWh"},
])
async def test_nullable_interpretation_renders_and_can_be_approved(
    fields: dict[str, str | None],
) -> None:
    raw = proposal_payload().model_dump(mode="json")
    raw["interpretation"].pop("time_window")
    raw["interpretation"].pop("units")
    raw["interpretation"].update(fields)
    payload = ProposalPayload.model_validate(raw)
    before = payload.model_dump_json()
    executor = StubExecutor()
    seen: list[ProposalRecord] = []

    async def confirm(record: ProposalRecord) -> bool:
        rendered = render_proposal(record)
        toml = rendered.split("```toml\n", 1)[1].split("\n```", 1)[0]
        decoded = tomllib.loads(toml)
        schema = rendered.split("```json\n", 1)[1].split("\n```", 1)[0]
        decoded["answer_schema"] = json.loads(schema)
        assert ProposalPayload.model_validate(decoded) == payload
        for field in ("time_window", "units"):
            if fields.get(field) is None:
                assert field not in decoded["interpretation"]
            else:
                assert decoded["interpretation"][field] == fields[field]
        seen.append(record)
        return True

    chat = session(
        StubClarifier(ClarifierTurn(kind="proposal", message="Ready", proposal=payload)), executor,
    )
    result = await chat.handle("Analyze it.", confirm)
    assert len(seen) == 1
    assert len(executor.requests) == 1
    assert result.terminal
    assert payload.model_dump_json() == before


async def test_declined_confirmation_does_not_run_and_allows_refinement() -> None:
    clarifier = StubClarifier(proposal(), clarification())
    executor = StubExecutor()
    chat = session(clarifier, executor)

    declined = await chat.handle("Use 2011.", reject)
    continued = await chat.handle("Actually use fiscal year 2011.", approve)

    assert declined.terminal is False
    assert "not run" in declined.content
    assert continued.content == "Which calendar year should I use?"
    assert executor.requests == []
    assert len(clarifier.calls) == 2


async def test_guidance_is_rejected_when_disabled_and_forwarded_exactly_when_enabled() -> None:
    disabled_executor = StubExecutor()
    disabled = session(StubClarifier(proposal_with_guidance()), disabled_executor)
    rejected = await disabled.handle("Use the proposed plan.", approve)
    assert "could not be prepared safely" in rejected.content
    assert disabled_executor.requests == []

    enabled_executor = StubExecutor()
    enabled = session(
        StubClarifier(proposal_with_guidance()),
        enabled_executor,
        enable_analysis_guidance=True,
    )
    result = await enabled.handle("Use the proposed plan.", approve)
    assert result.terminal is True
    assert enabled_executor.requests[0].analysis_guidance == (
        "Filter to 2011, compute quantity * unit_price_gbp, and aggregate by month."
    )


async def test_verified_notebook_is_returned_as_exact_bytes_without_path() -> None:
    notebook = b'{"cells":[],"nbformat":4,"nbformat_minor":5}\n'
    chat = session(StubClarifier(proposal()), StubExecutor(notebook=notebook))

    result = await chat.handle("Use 2011.", approve)

    assert result.notebook is not None
    assert result.notebook.content == notebook
    assert result.notebook.sha256 == artifact("notebook").sha256
    assert "/private/" not in result.content


async def test_dsa_failure_is_terminal() -> None:
    clarifier = StubClarifier(proposal())
    executor = StubExecutor(fail=True)
    chat = session(clarifier, executor)

    result = await chat.handle("Use 2011.", approve)
    followup = await chat.handle("Try again.", approve)

    assert result.terminal is True and "dsa_run_failed" in result.content
    assert followup.terminal is True and "already complete" in followup.content
    assert len(clarifier.calls) == 1
    assert len(executor.requests) == 1


async def test_overlapping_turns_are_rejected_and_confirmation_makes_terminal() -> None:
    confirmation_entered = asyncio.Event()
    confirmation_allowed = asyncio.Event()
    execution_allowed = asyncio.Event()
    clarifier = StubClarifier(proposal())
    executor = StubExecutor(gate=execution_allowed)
    chat = session(clarifier, executor)

    async def delayed_confirmation(_proposal: object) -> bool:
        confirmation_entered.set()
        await confirmation_allowed.wait()
        return True

    first = asyncio.create_task(chat.handle("Use 2011.", delayed_confirmation))
    await confirmation_entered.wait()

    overlapping = await chat.handle("Run another version.", approve)
    assert "already in progress" in overlapping.content
    assert len(clarifier.calls) == 1
    assert executor.requests == []

    confirmation_allowed.set()
    while not executor.requests:
        await asyncio.sleep(0)
    terminal_overlap = await chat.handle("Try while DSA runs.", approve)
    assert terminal_overlap.terminal is True
    assert "already complete" in terminal_overlap.content

    execution_allowed.set()
    await first
    assert len(executor.requests) == 1


async def test_clarifier_failure_is_safe_and_releases_reservation() -> None:
    clarifier = StubClarifier()
    chat = session(clarifier, StubExecutor())

    first = await chat.handle("Analyze it.", approve)
    second = await chat.handle("Try once more.", approve)

    assert "unavailable" in first.content
    assert "unavailable" in second.content
    assert len(clarifier.calls) == 2


async def test_confirmation_exception_is_a_decline() -> None:
    executor = StubExecutor()
    chat = session(StubClarifier(proposal()), executor)

    async def broken_confirmation(_proposal: object) -> bool:
        raise RuntimeError("secret")

    result = await chat.handle("Analyze it.", broken_confirmation)
    assert "not run" in result.content
    assert executor.requests == []


async def test_proposal_render_contains_exact_payload_and_digest_without_privileged_values() -> (
    None
):
    # The broker-created record is covered in the async flow; this assertion targets rendering.
    captured = ""

    async def inspect(record: ProposalRecord) -> bool:
        nonlocal captured
        captured = render_proposal(record)
        return False

    await session(StubClarifier(proposal()), StubExecutor()).handle("Use 2011.", inspect)
    rendered = captured
    assert proposal_payload().question in rendered
    assert "Proposal digest:" not in rendered
    toml = rendered.split("```toml\n", 1)[1].split("\n```", 1)[0]
    decoded = tomllib.loads(toml)
    schema_json = rendered.split("```json\n", 1)[1].split("\n```", 1)[0]
    decoded["answer_schema"] = json.loads(schema_json)
    assert ProposalPayload.model_validate(decoded) == proposal_payload()
    assert "### Answer schema" in rendered
    assert '"properties": {' in schema_json
    assert "answer_schema_json" not in rendered
    assert "The answer schema is embedded" not in rendered
    assert "/private/" not in rendered


async def test_proposal_render_shows_exact_untrusted_guidance_and_bound_digest() -> None:
    captured = ""

    async def inspect(record: ProposalRecord) -> bool:
        nonlocal captured
        captured = render_proposal(record)
        return False

    await session(
        StubClarifier(proposal_with_guidance()),
        StubExecutor(),
        enable_analysis_guidance=True,
    ).handle("Use 2011.", inspect)

    assert "analysis_guidance" in captured
    assert "quantity * unit_price_gbp" in captured
    assert "## Proposed analysis guidance\n" in captured
    assert "Proposed analysis guidance (untrusted)" not in captured
    assert "> Filter to 2011" in captured
    assert captured.index("## Proposed analysis guidance") < captured.index("```toml")
    assert "Proposal digest:" not in captured


async def test_proposal_render_neutralizes_markdown_from_untrusted_guidance() -> None:
    dangerous = (
        "1. ![tracking pixel](https://attacker.example/pixel)\n"
        "2. [Run analysis](https://attacker.example/deceptive)"
    )
    payload = ProposalPayload.model_validate(
        {
            **proposal_payload().model_dump(mode="python"),
            "analysis_guidance": dangerous,
        }
    )
    turn = ClarifierTurn(
        kind="proposal",
        message="The proposal is ready.",
        proposal=payload,
    )
    captured = ""

    async def inspect(record: ProposalRecord) -> bool:
        nonlocal captured
        captured = render_proposal(record)
        return False

    await session(
        StubClarifier(turn),
        StubExecutor(),
        enable_analysis_guidance=True,
    ).handle("Use the proposal.", inspect)

    guidance = captured.partition("## Exact approved proposal")[0]
    assert "![tracking pixel](" not in guidance
    assert "[Run analysis](" not in guidance
    assert "\\!\\[tracking pixel\\]\\(https\\:\\/\\/attacker\\.example\\/pixel\\)" in guidance
