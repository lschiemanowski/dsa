"""Deterministic test doubles for the Private Data Chat application."""

from __future__ import annotations

import asyncio

from apps.private_data_chat.contracts import AnalysisRequest, AnalysisResult


class FakeAnalysisExecutor:
    """Record requests and return one configured result or exception."""

    def __init__(
        self,
        outcome: AnalysisResult | Exception,
        *,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.outcome = outcome
        self.gate = gate
        self.requests: list[AnalysisRequest] = []

    async def execute(self, request: AnalysisRequest) -> AnalysisResult:
        self.requests.append(request.model_copy(deep=True))
        if self.gate is not None:
            await self.gate.wait()
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome.model_copy(
            update={"run_id": request.run_id, "proposal_id": request.proposal_id},
            deep=True,
        )
