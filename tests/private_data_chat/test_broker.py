from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from apps.private_data_chat import (
    AnalysisRequest,
    AnalysisResult,
    ArtifactIdentity,
    BrokerError,
    InMemoryProposalStore,
    PrivateDataBroker,
    ProposalStatus,
)
from apps.private_data_chat.testing import FakeAnalysisExecutor

from .test_contracts import proposal_payload


class ManualClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 9, 3, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


def terminal() -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_id="artifact-terminal-0123456789abcdef",
        sha256="b" * 64,
        byte_length=456,
        media_type="application/json",
    )


def successful_outcome() -> AnalysisResult:
    return AnalysisResult(
        run_id="analysis-placeholder0123456789",
        proposal_id="proposal-placeholder0123456789",
        status="succeeded",
        answer={"total": 10.0, "months": ["2026-01"]},
        terminal=terminal(),
    )


def broker(
    executor: FakeAnalysisExecutor,
    clock: ManualClock,
    *,
    ttl: timedelta = timedelta(minutes=30),
) -> PrivateDataBroker:
    sequence = iter(("proposalid0123456789", "analysisid0123456789"))
    return PrivateDataBroker(
        store=InMemoryProposalStore(),
        executor=executor,
        proposal_ttl=ttl,
        clock=clock,
        token_factory=lambda: next(sequence),
    )


async def proposed(broker_instance: PrivateDataBroker):
    return await broker_instance.propose(
        user_id="user-1",
        conversation_id="chat-1",
        data_source_id="retail",
        payload=proposal_payload(),
    )


async def test_requires_exact_bound_approval_before_one_execution() -> None:
    clock = ManualClock()
    executor = FakeAnalysisExecutor(successful_outcome())
    service = broker(executor, clock)
    proposal = await proposed(service)

    with pytest.raises(BrokerError, match="proposal_not_approved"):
        await service.execute(
            proposal_id=proposal.proposal_id,
            user_id="user-1",
            conversation_id="chat-1",
        )
    with pytest.raises(BrokerError, match="proposal_digest_mismatch"):
        await service.approve(
            proposal_id=proposal.proposal_id,
            proposal_sha256="0" * 64,
            user_id="user-1",
            conversation_id="chat-1",
        )

    approved = await service.approve(
        proposal_id=proposal.proposal_id,
        proposal_sha256=proposal.proposal_sha256,
        user_id="user-1",
        conversation_id="chat-1",
    )
    assert approved.status is ProposalStatus.APPROVED
    assert await service.approve(
        proposal_id=proposal.proposal_id,
        proposal_sha256=proposal.proposal_sha256,
        user_id="user-1",
        conversation_id="chat-1",
    ) == approved

    first = await service.execute(
        proposal_id=proposal.proposal_id,
        user_id="user-1",
        conversation_id="chat-1",
    )
    second = await service.execute(
        proposal_id=proposal.proposal_id,
        user_id="user-1",
        conversation_id="chat-1",
    )

    assert first == second
    assert len(executor.requests) == 1
    request = executor.requests[0]
    assert request.data_source_id == "retail"
    assert request.request_derivation is True
    assert request.model_dump().keys().isdisjoint(
        {"database_path", "model", "credentials", "endpoint", "policy", "runs_directory"}
    )
    complete = await service.get(
        proposal_id=proposal.proposal_id,
        user_id="user-1",
        conversation_id="chat-1",
    )
    assert complete.status is ProposalStatus.SUCCEEDED


async def test_identity_binding_hides_other_users_proposals() -> None:
    clock = ManualClock()
    service = broker(FakeAnalysisExecutor(successful_outcome()), clock)
    proposal = await proposed(service)

    with pytest.raises(BrokerError, match="proposal_not_found"):
        await service.approve(
            proposal_id=proposal.proposal_id,
            proposal_sha256=proposal.proposal_sha256,
            user_id="user-2",
            conversation_id="chat-1",
        )


async def test_proposal_boundary_revalidates_typed_nested_values_and_owns_a_snapshot() -> None:
    clock = ManualClock()
    service = broker(FakeAnalysisExecutor(successful_outcome()), clock)
    payload = proposal_payload()
    payload.answer_schema["$ref"] = "file:///private/database"

    with pytest.raises(ValidationError, match="unsupported keywords"):
        await service.propose(
            user_id="user-1",
            conversation_id="chat-1",
            data_source_id="retail",
            payload=payload,
        )

    proposal = await proposed(service)
    proposal.payload.answer_schema["description"] = "caller mutation"
    retained = await service.get(
        proposal_id=proposal.proposal_id,
        user_id="user-1",
        conversation_id="chat-1",
    )
    assert "description" not in retained.payload.answer_schema
    with pytest.raises(BrokerError, match="proposal_not_found"):
        await service.get(
            proposal_id=proposal.proposal_id,
            user_id="user-1",
            conversation_id="chat-2",
        )


async def test_expiry_is_terminal_and_prevents_execution() -> None:
    clock = ManualClock()
    service = broker(
        FakeAnalysisExecutor(successful_outcome()),
        clock,
        ttl=timedelta(seconds=1),
    )
    proposal = await proposed(service)
    clock.value += timedelta(seconds=1)

    with pytest.raises(BrokerError, match="proposal_expired"):
        await service.approve(
            proposal_id=proposal.proposal_id,
            proposal_sha256=proposal.proposal_sha256,
            user_id="user-1",
            conversation_id="chat-1",
        )
    expired = await service.get(
        proposal_id=proposal.proposal_id,
        user_id="user-1",
        conversation_id="chat-1",
    )
    assert expired.status is ProposalStatus.EXPIRED


async def test_approved_proposal_must_execute_before_expiry() -> None:
    clock = ManualClock()
    executor = FakeAnalysisExecutor(successful_outcome())
    service = broker(executor, clock, ttl=timedelta(seconds=1))
    proposal = await proposed(service)
    await service.approve(
        proposal_id=proposal.proposal_id,
        proposal_sha256=proposal.proposal_sha256,
        user_id="user-1",
        conversation_id="chat-1",
    )
    clock.value += timedelta(seconds=1)

    with pytest.raises(BrokerError, match="proposal_expired"):
        await service.execute(
            proposal_id=proposal.proposal_id,
            user_id="user-1",
            conversation_id="chat-1",
        )
    assert executor.requests == []


async def test_concurrent_execute_does_not_duplicate_analysis() -> None:
    clock = ManualClock()
    gate = asyncio.Event()
    executor = FakeAnalysisExecutor(successful_outcome(), gate=gate)
    service = broker(executor, clock)
    proposal = await proposed(service)
    await service.approve(
        proposal_id=proposal.proposal_id,
        proposal_sha256=proposal.proposal_sha256,
        user_id="user-1",
        conversation_id="chat-1",
    )

    first = asyncio.create_task(
        service.execute(
            proposal_id=proposal.proposal_id,
            user_id="user-1",
            conversation_id="chat-1",
        )
    )
    while not executor.requests:
        await asyncio.sleep(0)
    with pytest.raises(BrokerError, match="proposal_execution_in_progress"):
        await service.execute(
            proposal_id=proposal.proposal_id,
            user_id="user-1",
            conversation_id="chat-1",
        )
    gate.set()
    await first
    assert len(executor.requests) == 1


async def test_executor_exception_is_sanitized_and_not_retried() -> None:
    clock = ManualClock()
    executor = FakeAnalysisExecutor(RuntimeError("secret endpoint and credential"))
    service = broker(executor, clock)
    proposal = await proposed(service)
    await service.approve(
        proposal_id=proposal.proposal_id,
        proposal_sha256=proposal.proposal_sha256,
        user_id="user-1",
        conversation_id="chat-1",
    )

    result = await service.execute(
        proposal_id=proposal.proposal_id,
        user_id="user-1",
        conversation_id="chat-1",
    )
    assert result.status == "failed"
    assert result.failure_code == "analysis_executor_failed"
    assert result.terminal is None
    assert "secret" not in result.model_dump_json()
    assert len(executor.requests) == 1


async def test_executor_success_must_match_the_approved_answer_schema() -> None:
    clock = ManualClock()
    invalid = successful_outcome().model_copy(update={"answer": {"wrong": 10.0}}, deep=True)
    executor = FakeAnalysisExecutor(invalid)
    service = broker(executor, clock)
    proposal = await proposed(service)
    await service.approve(
        proposal_id=proposal.proposal_id,
        proposal_sha256=proposal.proposal_sha256,
        user_id="user-1",
        conversation_id="chat-1",
    )

    result = await service.execute(
        proposal_id=proposal.proposal_id,
        user_id="user-1",
        conversation_id="chat-1",
    )
    assert result.status == "failed"
    assert result.failure_code == "analysis_executor_failed"


async def test_executor_cannot_change_the_brokers_approved_schema_snapshot() -> None:
    class MutatingExecutor:
        async def execute(self, request: AnalysisRequest) -> AnalysisResult:
            request.answer_schema["properties"] = {
                "wrong": {"type": "number"},
            }
            request.answer_schema["required"] = ["wrong"]
            return successful_outcome().model_copy(
                update={
                    "run_id": request.run_id,
                    "proposal_id": request.proposal_id,
                    "answer": {"wrong": 10.0},
                },
                deep=True,
            )

    clock = ManualClock()
    service = PrivateDataBroker(
        store=InMemoryProposalStore(),
        executor=MutatingExecutor(),
        clock=clock,
        token_factory=iter(("proposalid0123456789", "analysisid0123456789")).__next__,
    )
    proposal = await proposed(service)
    await service.approve(
        proposal_id=proposal.proposal_id,
        proposal_sha256=proposal.proposal_sha256,
        user_id="user-1",
        conversation_id="chat-1",
    )

    result = await service.execute(
        proposal_id=proposal.proposal_id,
        user_id="user-1",
        conversation_id="chat-1",
    )
    assert result.status == "failed"
    assert result.failure_code == "analysis_executor_failed"


async def test_cancellation_is_recorded_without_becoming_a_second_execution() -> None:
    clock = ManualClock()
    gate = asyncio.Event()
    executor = FakeAnalysisExecutor(successful_outcome(), gate=gate)
    service = broker(executor, clock)
    proposal = await proposed(service)
    await service.approve(
        proposal_id=proposal.proposal_id,
        proposal_sha256=proposal.proposal_sha256,
        user_id="user-1",
        conversation_id="chat-1",
    )
    task = asyncio.create_task(
        service.execute(
            proposal_id=proposal.proposal_id,
            user_id="user-1",
            conversation_id="chat-1",
        )
    )
    while not executor.requests:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    result = await service.execute(
        proposal_id=proposal.proposal_id,
        user_id="user-1",
        conversation_id="chat-1",
    )
    assert result.failure_code == "analysis_cancelled"
    assert len(executor.requests) == 1
