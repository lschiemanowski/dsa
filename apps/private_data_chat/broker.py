"""Deterministic approval and execution state machine for Private Data Chat."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Callable, Iterable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import JsonValue

from apps.private_data_chat.contracts import (
    AnalysisRequest,
    AnalysisResult,
    ProposalBinding,
    ProposalPayload,
    ProposalRecord,
    ProposalStatus,
    proposal_digest,
)


class BrokerError(Exception):
    """Stable application error suitable for an API response."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ProposalStore(Protocol):
    """Persistence boundary; production implementations need atomic transactions."""

    async def insert(self, record: ProposalRecord) -> None: ...

    async def get(self, proposal_id: str) -> ProposalRecord | None: ...

    async def replace(self, record: ProposalRecord) -> None: ...


class AnalysisExecutor(Protocol):
    """Trusted adapter that resolves source/model/policy and invokes DSA."""

    async def execute(self, request: AnalysisRequest) -> AnalysisResult: ...


class _AnswerValidator(Protocol):
    def iter_errors(self, instance: object) -> Iterable[JsonSchemaValidationError]: ...


class InMemoryProposalStore:
    """Single-process development store; deliberately not a production database."""

    def __init__(self) -> None:
        self._records: dict[str, ProposalRecord] = {}

    async def insert(self, record: ProposalRecord) -> None:
        if record.proposal_id in self._records:
            raise BrokerError("proposal_id_conflict")
        self._records[record.proposal_id] = _snapshot_record(record)

    async def get(self, proposal_id: str) -> ProposalRecord | None:
        record = self._records.get(proposal_id)
        return None if record is None else record.model_copy(deep=True)

    async def replace(self, record: ProposalRecord) -> None:
        if record.proposal_id not in self._records:
            raise BrokerError("proposal_not_found")
        self._records[record.proposal_id] = _snapshot_record(record)


class PrivateDataBroker:
    """Own proposal approval and guarantee at-most-once execution per process."""

    def __init__(
        self,
        *,
        store: ProposalStore,
        executor: AnalysisExecutor,
        proposal_ttl: timedelta = timedelta(minutes=30),
        clock: Callable[[], datetime] | None = None,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if proposal_ttl <= timedelta(0):
            raise ValueError("proposal_ttl must be positive")
        self._store = store
        self._executor = executor
        self._proposal_ttl = proposal_ttl
        self._clock = clock or (lambda: datetime.now(UTC))
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(24))
        self._lock = asyncio.Lock()

    async def propose(
        self,
        *,
        user_id: str,
        conversation_id: str,
        data_source_id: str,
        payload: ProposalPayload | object,
    ) -> ProposalRecord:
        canonical_payload = _revalidate(ProposalPayload, payload)
        binding = ProposalBinding(
            user_id=user_id,
            conversation_id=conversation_id,
            data_source_id=data_source_id,
        )
        now = _aware_time(self._clock())
        record = ProposalRecord(
            proposal_id=f"proposal-{self._token_factory()}",
            proposal_sha256=proposal_digest(binding, canonical_payload),
            payload=canonical_payload,
            binding=binding,
            status=ProposalStatus.PROPOSED,
            created_at=now,
            expires_at=now + self._proposal_ttl,
        )
        async with self._lock:
            await self._store.insert(record)
        return record

    async def get(
        self,
        *,
        proposal_id: str,
        user_id: str,
        conversation_id: str,
    ) -> ProposalRecord:
        async with self._lock:
            record = await self._required_record(proposal_id)
            self._authorize(record, user_id, conversation_id)
            return await self._expire_if_needed(record, _aware_time(self._clock()))

    async def approve(
        self,
        *,
        proposal_id: str,
        proposal_sha256: str,
        user_id: str,
        conversation_id: str,
    ) -> ProposalRecord:
        async with self._lock:
            record = await self._required_record(proposal_id)
            self._authorize(record, user_id, conversation_id)
            now = _aware_time(self._clock())
            record = await self._expire_if_needed(record, now)
            if record.status is ProposalStatus.EXPIRED:
                raise BrokerError("proposal_expired")
            if proposal_sha256 != record.proposal_sha256:
                raise BrokerError("proposal_digest_mismatch")
            if record.status is ProposalStatus.PROPOSED:
                record = ProposalRecord.model_validate(
                    {
                        **record.model_dump(mode="python"),
                        "status": ProposalStatus.APPROVED,
                        "approved_at": now,
                    }
                )
                await self._store.replace(record)
            return record

    async def execute(
        self,
        *,
        proposal_id: str,
        user_id: str,
        conversation_id: str,
    ) -> AnalysisResult:
        async with self._lock:
            record = await self._required_record(proposal_id)
            self._authorize(record, user_id, conversation_id)
            record = await self._expire_if_needed(record, _aware_time(self._clock()))
            if record.status is ProposalStatus.EXPIRED:
                raise BrokerError("proposal_expired")
            if record.result is not None:
                return record.result
            if record.status is ProposalStatus.PROPOSED:
                raise BrokerError("proposal_not_approved")
            if record.status is ProposalStatus.RUNNING:
                raise BrokerError("proposal_execution_in_progress")
            if record.status is not ProposalStatus.APPROVED:
                raise BrokerError("proposal_state_invalid")
            run_id = f"analysis-{self._token_factory()}"
            request = AnalysisRequest(
                run_id=run_id,
                proposal_id=record.proposal_id,
                proposal_sha256=record.proposal_sha256,
                data_source_id=record.binding.data_source_id,
                question=record.payload.question,
                answer_schema=record.payload.answer_schema,
                analysis_guidance=record.payload.analysis_guidance,
            )
            running = ProposalRecord.model_validate(
                {
                    **record.model_dump(mode="python"),
                    "status": ProposalStatus.RUNNING,
                    "run_id": run_id,
                }
            )
            await self._store.replace(running)

        try:
            executor_request = request.model_copy(deep=True)
            raw_result = await self._executor.execute(executor_request)
            result = AnalysisResult.model_validate(
                raw_result.model_dump(mode="python", round_trip=True)
            ).model_copy(deep=True)
            identity_mismatch = result.run_id != run_id or result.proposal_id != proposal_id
            answer_mismatch = result.status == "succeeded" and not _answer_matches(
                request.answer_schema,
                result.answer,
            )
            if identity_mismatch or answer_mismatch:
                result = _executor_failure(request)
        except asyncio.CancelledError:
            with suppress(Exception):
                await asyncio.shield(self._finish(request, _cancelled_result(request)))
            raise
        except Exception:
            result = _executor_failure(request)
        return await self._finish(request, result)

    async def _finish(self, request: AnalysisRequest, result: AnalysisResult) -> AnalysisResult:
        async with self._lock:
            record = await self._required_record(request.proposal_id)
            if record.status is not ProposalStatus.RUNNING or record.run_id != request.run_id:
                raise BrokerError("proposal_state_conflict")
            terminal_status = (
                ProposalStatus.SUCCEEDED
                if result.status == "succeeded"
                else ProposalStatus.FAILED
            )
            completed = ProposalRecord.model_validate(
                {
                    **record.model_dump(mode="python"),
                    "status": terminal_status,
                    "result": result,
                }
            )
            await self._store.replace(completed)
        return result

    async def _required_record(self, proposal_id: str) -> ProposalRecord:
        record = await self._store.get(proposal_id)
        if record is None:
            raise BrokerError("proposal_not_found")
        return record

    async def _expire_if_needed(
        self,
        record: ProposalRecord,
        now: datetime,
    ) -> ProposalRecord:
        if (
            record.status in {ProposalStatus.PROPOSED, ProposalStatus.APPROVED}
            and now >= record.expires_at
        ):
            expired = ProposalRecord.model_validate(
                {**record.model_dump(mode="python"), "status": ProposalStatus.EXPIRED}
            )
            await self._store.replace(expired)
            return expired
        return record

    @staticmethod
    def _authorize(record: ProposalRecord, user_id: str, conversation_id: str) -> None:
        if record.binding.user_id != user_id or record.binding.conversation_id != conversation_id:
            raise BrokerError("proposal_not_found")


def _revalidate(
    model_type: type[ProposalPayload],
    value: ProposalPayload | object,
) -> ProposalPayload:
    raw = (
        value.model_dump(mode="python", round_trip=True)
        if isinstance(value, model_type)
        else value
    )
    return model_type.model_validate(raw).model_copy(deep=True)


def _snapshot_record(record: ProposalRecord) -> ProposalRecord:
    return ProposalRecord.model_validate(
        record.model_dump(mode="python", round_trip=True)
    ).model_copy(deep=True)


def _aware_time(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value


def _answer_matches(schema: dict[str, JsonValue], answer: object) -> bool:
    validator = cast(_AnswerValidator, Draft202012Validator(schema))
    return next(iter(validator.iter_errors(answer)), None) is None


def _executor_failure(request: AnalysisRequest) -> AnalysisResult:
    return AnalysisResult(
        run_id=request.run_id,
        proposal_id=request.proposal_id,
        status="failed",
        failure_code="analysis_executor_failed",
    )


def _cancelled_result(request: AnalysisRequest) -> AnalysisResult:
    return AnalysisResult(
        run_id=request.run_id,
        proposal_id=request.proposal_id,
        status="failed",
        failure_code="analysis_cancelled",
    )
