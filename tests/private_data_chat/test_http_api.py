"""Exercise the same-origin HTTP workflow without providers, DuckDB or Docker."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from time import monotonic

import pytest
from httpx import ASGITransport, AsyncClient

from apps.private_data_chat import chat
from apps.private_data_chat.broker import PrivateDataBroker
from apps.private_data_chat.chat import PrivateDataChatSession
from tests.private_data_chat.test_chat import StubClarifier, StubExecutor, clarification, context
from tests.private_data_chat.test_chat import proposal as proposed_turn
from tests.private_data_chat.test_cli import write_configuration

pytest.importorskip("fastapi")

from fastapi import FastAPI

from apps.private_data_chat import http_api
from apps.private_data_chat.http_api import ConversationView, create_app


def make_app(
    clarifier: StubClarifier, executor: StubExecutor, *, clock: Callable[[], float] = monotonic
) -> FastAPI:
    def create_session(owner: str, conversation: str) -> PrivateDataChatSession:
        return PrivateDataChatSession(
            user_id=owner,
            conversation_id=conversation,
            context=context(),
            clarifier=clarifier,
            executor=executor,
        )

    return create_app(session_factory=create_session, clock=clock)


def client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost")


async def test_clarify_revise_approve_and_download() -> None:
    clarifier = StubClarifier(clarification(), proposed_turn(), proposed_turn())
    gate = asyncio.Event()
    notebook = b'{"cells":[],"nbformat":4,"nbformat_minor":5}\n'
    executor = StubExecutor(gate=gate, notebook=notebook)
    app = make_app(clarifier, executor)
    async with client(app) as browser, client(app) as stranger:
        created = await browser.post("/api/conversations")
        assert created.status_code == 201
        assert "HttpOnly" in created.headers["set-cookie"]
        assert "SameSite=strict" in created.headers["set-cookie"]
        path = f"/api/conversations/{created.json()['id']}"

        followup = await browser.post(f"{path}/messages", json={"text": "Show monthly sales."})
        assert followup.json()["status"] == "open"
        assert followup.json()["messages"][-1]["text"] == "Which calendar year should I use?"
        pending = await browser.post(f"{path}/messages", json={"text": "Use 2011."})
        proposal = pending.json()["messages"][-1]["proposal"]
        assert pending.json()["pending_proposal_id"] == proposal["id"]
        assert pending.json()["status"] == "awaiting_decision"
        assert json.loads(proposal["request_json"]) == proposal["payload"]
        assert "binding" not in proposal
        blocked_message = await browser.post(f"{path}/messages", json={"text": "Wait."})
        assert blocked_message.status_code == 409
        assert blocked_message.json()["detail"] == "proposal_pending"

        decision = f"{path}/proposals/{proposal['id']}/decision"
        approval = {"decision": "approve", "digest": proposal["digest"]}
        invalid = await browser.post(decision, json={**approval, "digest": "0" * 64})
        assert invalid.status_code == 409
        assert invalid.json()["detail"] == "proposal_digest_mismatch"
        assert (await stranger.post("/api/conversations")).status_code == 201
        assert (await stranger.post(decision, json=approval)).status_code == 404
        assert (await stranger.get(path)).status_code == 404
        assert (await browser.get(f"{path}/notebook")).status_code == 404
        assert executor.requests == []

        revised = await browser.post(decision, json={**approval, "decision": "revise"})
        assert revised.json()["status"] == "open"
        assert revised.json()["pending_proposal_id"] is None
        assert (await browser.post(decision, json=approval)).status_code == 409

        refined = await browser.post(f"{path}/messages", json={"text": "Use the fiscal year."})
        proposal = refined.json()["messages"][-1]["proposal"]
        assert "Unapproved proposal" in clarifier.calls[-1][-2].content
        decision = f"{path}/proposals/{proposal['id']}/decision"
        approval = {"decision": "approve", "digest": proposal["digest"]}
        try:
            approved = await asyncio.gather(
                browser.post(decision, json=approval),
                browser.post(decision, json=approval),
            )
            assert [response.status_code for response in approved] == [202, 202]
            assert (await browser.get(path)).json()["status"] == "running"
        finally:
            gate.set()

        completed = await completed_run(browser, path)
        assert completed.status == "succeeded"
        result = completed.result
        assert result is not None and result.response_json is not None
        assert json.loads(result.response_json) == result.answer
        assert result.answer == {"total": 10.0, "months": ["2011-01"]}
        assert len(executor.requests) == 1
        assert executor.requests[0].proposal_sha256 == proposal["digest"]
        duplicate = await browser.post(decision, json=approval)
        assert duplicate.status_code == 200
        assert ConversationView.model_validate(duplicate.json()).result == result
        closed = await browser.post(f"{path}/messages", json={"text": "Run again."})
        assert closed.status_code == 409
        assert len(clarifier.calls) == 3

        assert result.notebook is not None
        download = await browser.get(result.notebook.url)
        assert download.content == notebook
        assert download.headers["cache-control"] == "no-store"
        assert download.headers["content-type"] == "application/x-ipynb+json"
        assert download.headers["content-disposition"] == (
            'attachment; filename="dsa-derivation.ipynb"'
        )
        assert (await stranger.get(result.notebook.url)).status_code == 404


async def test_expired_proposal_returns_to_clarification(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(chat, "PrivateDataBroker", partial(PrivateDataBroker, clock=lambda: now))
    executor = StubExecutor()
    app = make_app(StubClarifier(proposed_turn(), clarification()), executor)
    async with client(app) as browser:
        created = await browser.post("/api/conversations")
        path = f"/api/conversations/{created.json()['id']}"
        pending = await browser.post(f"{path}/messages", json={"text": "Use 2011."})
        proposal = pending.json()["messages"][-1]["proposal"]
        now += timedelta(hours=1)
        expired = await browser.post(
            f"{path}/proposals/{proposal['id']}/decision",
            json={"decision": "approve", "digest": proposal["digest"]},
        )
        assert expired.status_code == 409
        assert expired.json()["detail"] == "proposal_expired"
        assert (await browser.get(path)).json()["status"] == "open"
        continued = await browser.post(f"{path}/messages", json={"text": "Try 2012."})
        assert continued.json()["messages"][-1]["text"] == "Which calendar year should I use?"
        assert executor.requests == []


async def test_cross_origin_and_invalid_inputs_do_not_run_the_clarifier() -> None:
    clarifier = StubClarifier()
    async with client(make_app(clarifier, StubExecutor())) as browser:
        created = await browser.post("/api/conversations", headers={"origin": "http://localhost"})
        path = f"/api/conversations/{created.json()['id']}/messages"
        cross_origin = await browser.post(
            path, headers={"origin": "https://different.example"}, json={"text": "Run it."}
        )
        assert cross_origin.status_code == 403
        invalid = await browser.post(path, json={"text": " ", "unexpected": "/private/path"})
        assert (await browser.post(path, headers={"host": "different.example"})).status_code == 400
        assert invalid.status_code == 422
        assert invalid.json() == {"detail": "invalid_request"}
        assert clarifier.calls == []


@pytest.mark.parametrize("fail", [False, True])
async def test_completed_run_without_notebook(fail: bool) -> None:
    app = make_app(StubClarifier(proposed_turn()), StubExecutor(fail=fail))
    async with client(app) as browser:
        created = await browser.post("/api/conversations")
        path = f"/api/conversations/{created.json()['id']}"
        pending = await browser.post(f"{path}/messages", json={"text": "Use 2011."})
        proposal = pending.json()["messages"][-1]["proposal"]
        approved = await browser.post(
            f"{path}/proposals/{proposal['id']}/decision",
            json={"decision": "approve", "digest": proposal["digest"]},
        )
        assert approved.status_code == 202
        completed = await completed_run(browser, path)
        assert completed.result is not None
        if fail:
            assert completed.status == "failed"
            assert completed.failure_code == "dsa_run_failed"
            assert completed.result.answer is None
        else:
            assert completed.status == "succeeded"
            assert completed.failure_code is None
            assert completed.result.answer is not None
        assert completed.result.notebook is None
        assert (await browser.get(f"{path}/notebook")).status_code == 404


async def test_ipv6_loopback_uses_the_same_origin_policy() -> None:
    app = make_app(StubClarifier(), StubExecutor())
    origin = "http://[::1]:8001"
    async with AsyncClient(transport=ASGITransport(app=app), base_url=origin) as browser:
        allowed = await browser.post("/api/conversations", headers={"origin": origin})
        assert allowed.status_code == 201
        rejected = await browser.post(
            "/api/conversations", headers={"origin": "http://localhost:8001"}
        )
        assert rejected.status_code == 403
        for host in ("[::1].example.test", "[::1"):
            invalid_host = await browser.post("/api/conversations", headers={"host": host})
            assert invalid_host.status_code == 400


async def test_conversation_limits_and_idle_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_api, "_MAX_CONVERSATIONS_PER_BROWSER", 1)
    monkeypatch.setattr(http_api, "_MAX_CONVERSATIONS", 2)
    now = 0.0
    app = make_app(StubClarifier(), StubExecutor(), clock=lambda: now)
    async with client(app) as first, client(app) as second, client(app) as third:
        conversation = await first.post("/api/conversations")
        assert conversation.status_code == 201
        path = f"/api/conversations/{conversation.json()['id']}"
        browser_limit = await first.post("/api/conversations")
        assert browser_limit.status_code == 429
        assert browser_limit.json()["detail"] == "browser_conversation_limit"
        assert (await second.post("/api/conversations")).status_code == 201
        process_limit = await third.post("/api/conversations")
        assert process_limit.status_code == 429
        assert process_limit.json()["detail"] == "conversation_capacity_reached"

        now += 3600
        assert (await first.get(path)).status_code == 404
        assert (await first.post("/api/conversations")).status_code == 201
        assert (await third.post("/api/conversations")).status_code == 201


async def test_running_limit_allows_retry_and_preserves_active_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(http_api, "_MAX_RUNNING_ANALYSES", 1)
    gate = asyncio.Event()
    executor = StubExecutor(gate=gate)
    now = 0.0
    app = make_app(StubClarifier(proposed_turn(), proposed_turn()), executor, clock=lambda: now)
    async with client(app) as browser:
        first_path, first_decision, first_body = await pending_proposal(browser)
        try:
            assert (await browser.post(first_decision, json=first_body)).status_code == 202

            now += 3600
            assert (await browser.get(first_path)).json()["status"] == "running"
            second_path, second_decision, second_body = await pending_proposal(browser)
            limited = await browser.post(second_decision, json=second_body)
            assert limited.status_code == 429
            assert limited.json()["detail"] == "analysis_capacity_reached"
            assert (await browser.get(second_path)).json()["status"] == "awaiting_decision"
            assert (await browser.post(first_decision, json=first_body)).status_code == 202
        finally:
            gate.set()

        assert (await completed_run(browser, first_path)).status == "succeeded"
        assert (await browser.post(second_decision, json=second_body)).status_code == 202
        assert (await completed_run(browser, second_path)).status == "succeeded"
        assert len(executor.requests) == 2

        now += 3600
        assert (await browser.get(first_path)).status_code == 404
        assert (await browser.get(second_path)).status_code == 404


async def pending_proposal(browser: AsyncClient) -> tuple[str, str, dict[str, str]]:
    created = await browser.post("/api/conversations")
    path = f"/api/conversations/{created.json()['id']}"
    reply = await browser.post(f"{path}/messages", json={"text": "Use 2011."})
    proposal = reply.json()["messages"][-1]["proposal"]
    return (
        path,
        f"{path}/proposals/{proposal['id']}/decision",
        {"decision": "approve", "digest": proposal["digest"]},
    )


async def test_configured_factory_builds_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DSA_CHAT_CONFIG_PATH", str(write_configuration(tmp_path)))
    async with client(create_app()) as browser:
        response = await browser.post("/api/conversations")
        assert response.status_code == 201
        state = ConversationView.model_validate(response.json())
        assert "**untrusted**" in state.welcome
        assert state.messages == []


async def completed_run(browser: AsyncClient, path: str) -> ConversationView:
    async with asyncio.timeout(3):
        while True:
            state = ConversationView.model_validate((await browser.get(path)).json())
            if state.status in ("succeeded", "failed"):
                return state
            await asyncio.sleep(0.01)
