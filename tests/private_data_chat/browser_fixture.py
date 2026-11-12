"""Deterministic FastAPI fixture for the assistant-ui browser test."""

from __future__ import annotations

import asyncio
import json
from hashlib import sha256

from fastapi import FastAPI
from pydantic import JsonValue

from apps.private_data_chat.chat import PrivateDataChatSession
from apps.private_data_chat.contracts import (
    AnalysisRequest,
    AnalysisResult,
    ArtifactIdentity,
    ClarifierTurn,
    ProposalPayload,
)
from apps.private_data_chat.http_api import create_app
from tests.private_data_chat.test_chat import StubClarifier, artifact, clarification, context

_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l1sAAAAASUVORK5CYII="
)
_NOTEBOOK = json.dumps(
    {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Verified derivation\n",
                    "![remote](https://example.invalid/track.png)\n",
                    "[remote link](https://example.invalid/link)",
                ],
            },
            {
                "cell_type": "code",
                "source": ["total = sum(row['sales_pence'] for row in rows)\n", "print(total)"],
                "metadata": {
                    "dsa_plot": {"title": "Monthly sales", "filename": "monthly-sales.png"}
                },
                "outputs": [
                    {"output_type": "stream", "text": "9007199254740993\n"},
                    {"output_type": "error", "traceback": ["\x1b[31mValueError: example\x1b[0m"]},
                    {"output_type": "display_data", "data": {"image/png": _PNG}},
                    {"output_type": "display_data", "data": {"text/html": "<b>Active output</b>"}},
                ],
            },
        ],
    }
).encode()
_IDENTITY = ArtifactIdentity(
    artifact_id="artifact-notebook-preview",
    sha256=sha256(_NOTEBOOK).hexdigest(),
    byte_length=len(_NOTEBOOK),
    media_type="application/x-ipynb+json",
)


def _proposal() -> ClarifierTurn:
    payload = ProposalPayload.model_validate(
        {
            "question": "What were net sales by month in 2011?",
            "interpretation": {
                "measure": "Net sales",
                "population": "Retail transactions",
                "time_window": "Calendar year 2011",
                "group_by": ["Month"],
                "filters": ["Exclude returns"],
                "units": "GBP pence",
            },
            "allow_plots": True,
            "analysis_guidance": "Sum line totals by month.\nExclude returned orders.",
            "answer_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "total_pence": {"type": "integer", "title": "Total net sales (pence)"},
                    "monthly": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "month": {"type": "string"},
                                "sales_pence": {"type": "integer", "title": "Sales (pence)"},
                            },
                            "required": ["month", "sales_pence"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["total_pence", "monthly"],
                "additionalProperties": False,
            },
        }
    )
    return ClarifierTurn(kind="proposal", message="Ready for review.", proposal=payload)


class BrowserExecutor:
    def __init__(self, gate: asyncio.Event, *, include_notebook: bool) -> None:
        self._gate = gate
        self._notebook = _IDENTITY if include_notebook else None

    async def execute(self, request: AnalysisRequest) -> AnalysisResult:
        await self._gate.wait()
        answer: JsonValue = {
            "total_pence": 9007199254740993,
            "monthly": [{"month": "2011-01", "sales_pence": 9007199254740993}],
        }
        return AnalysisResult(
            run_id=request.run_id,
            proposal_id=request.proposal_id,
            status="succeeded",
            answer=answer,
            terminal=artifact(),
            notebook=self._notebook,
        )

    def read_notebook(self, identity: ArtifactIdentity) -> bytes:
        if identity != self._notebook:
            raise ValueError("Notebook not retained")
        return _NOTEBOOK


def make_app() -> FastAPI:
    gate = asyncio.Event()
    created = 0

    def make_session(owner: str, conversation: str) -> PrivateDataChatSession:
        nonlocal created
        created += 1
        first = created == 1
        return PrivateDataChatSession(
            user_id=owner,
            conversation_id=conversation,
            context=context(),
            clarifier=(
                StubClarifier(clarification(), _proposal(), _proposal())
                if first
                else StubClarifier(_proposal())
            ),
            executor=BrowserExecutor(gate, include_notebook=first),
            enable_analysis_guidance=True,
        )

    async def release() -> None:
        gate.set()

    app = create_app(
        session_factory=make_session,
        welcome="## Instructions\n\nAsk a question about the sample retail database.",
    )
    app.add_api_route("/__test/release", release, methods=["POST"])
    return app
