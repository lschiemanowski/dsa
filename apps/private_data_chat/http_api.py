"""Same-origin HTTP adapter for the private-data conversation workflow."""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, JsonValue
from starlette.concurrency import run_in_threadpool

from apps.private_data_chat.broker import BrokerError
from apps.private_data_chat.chat import ChatStatus, PrivateDataChatSession, ProposalDecision
from apps.private_data_chat.contracts import AppContract, ProposalRecord, proposal_payload_json
from apps.private_data_chat.service import load_chat_service, model_display_name, welcome_message
from apps.private_data_chat.settings import CHAT_WEB_ROOT

SessionFactory = Callable[[str, str], PrivateDataChatSession]
_COOKIE = "dsa_chat_session"
_CONVERSATIONS = "/api/conversations"
_NOTEBOOK_FILENAME = "dsa-derivation.ipynb"
_LOOPBACK_HOST = re.compile(r"(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]{1,5})?", re.IGNORECASE)
_FINGERPRINTED_ASSET = re.compile(r"/assets/[^/]+-[A-Za-z0-9_-]{8,}\.(?:js|css)")
_UI_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
    "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
)
_MAX_CONVERSATIONS_PER_BROWSER = 8
_MAX_CONVERSATIONS = 64
_MAX_RUNNING_ANALYSES = 2
_CONVERSATION_IDLE_SECONDS = 60 * 60
_SWEEP_SECONDS = 60


class MessageInput(AppContract):
    text: str = Field(min_length=1, max_length=8192, pattern=r"\S")


class DecisionInput(AppContract):
    decision: ProposalDecision
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProposalView(BaseModel):
    id: str
    digest: str
    expires_at: datetime
    payload: dict[str, JsonValue]
    request_json: str


class MessageView(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    proposal: ProposalView | None = None


class NotebookView(BaseModel):
    url: str
    sha256: str
    byte_length: int


class ResultView(BaseModel):
    run_id: str
    proposal_id: str
    answer: JsonValue | None = None
    response_json: str | None = None
    notebook: NotebookView | None = None


class ConversationView(BaseModel):
    id: str
    welcome: str
    status: ChatStatus
    messages: list[MessageView]
    pending_proposal_id: str | None = None
    result: ResultView | None = None
    failure_code: str | None = None


@dataclass
class _Conversation:
    session: PrivateDataChatSession
    last_seen: float
    messages: list[MessageView] = field(default_factory=list[MessageView])


def create_app(
    *,
    session_factory: SessionFactory | None = None,
    welcome: str = "",
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Create the local single-process API, with injectable sessions for tests."""
    if session_factory is None:
        service = load_chat_service()
        session_factory = service.create_session
        welcome = welcome_message(
            service.database_card,
            model_display_name(service.configuration.clarifier_model.name),
        )
    new_session = session_factory
    browser_sessions: dict[str, dict[str, _Conversation]] = {}
    run_slots = asyncio.BoundedSemaphore(_MAX_RUNNING_ANALYSES)

    def purge_idle_conversations() -> None:
        now = clock()
        for owner, conversations in list(browser_sessions.items()):
            for conversation_id, entry in list(conversations.items()):
                if entry.session.is_busy:
                    entry.last_seen = now
                elif now - entry.last_seen >= _CONVERSATION_IDLE_SECONDS:
                    del conversations[conversation_id]
            if not conversations:
                del browser_sessions[owner]

    async def reap_idle_conversations() -> None:
        while True:
            await asyncio.sleep(_SWEEP_SECONDS)
            purge_idle_conversations()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        reaper = asyncio.create_task(reap_idle_conversations())
        try:
            yield
        finally:
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)
            await asyncio.gather(
                *(
                    entry.session.cancel_run()
                    for conversations in browser_sessions.values()
                    for entry in conversations.values()
                )
            )
            browser_sessions.clear()

    app = FastAPI(title="DSA Private Data Chat", lifespan=lifespan)

    async def protect_browser_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        host = request.headers.get("host", "")
        origin = request.headers.get("origin")
        if not _LOOPBACK_HOST.fullmatch(host):
            response: Response = JSONResponse({"detail": "invalid_host"}, status_code=400)
        elif request.method == "POST" and (
            (origin is not None and origin != f"{request.url.scheme}://{host.lower()}")
            or request.headers.get("sec-fetch-site") == "cross-site"
        ):
            response = JSONResponse({"detail": "invalid_origin"}, status_code=403)
        else:
            response = await call_next(request)
        path = request.scope["path"]
        if response.status_code in {200, 206, 304} and _FINGERPRINTED_ASSET.fullmatch(path):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-store"
        if path in {"/", "/index.html"} and response.headers.get("Content-Type", "").startswith(
            "text/html"
        ):
            response.headers["Content-Security-Policy"] = _UI_CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    async def invalid_request(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse({"detail": "invalid_request"}, status_code=422)

    def owned_conversation(request: Request, conversation_id: str) -> _Conversation:
        purge_idle_conversations()
        owner = request.cookies.get(_COOKIE, "")
        entry = browser_sessions.get(owner, {}).get(conversation_id)
        if entry is None:
            raise HTTPException(404, "conversation_not_found")
        entry.last_seen = clock()
        return entry

    async def view(conversation_id: str, entry: _Conversation) -> ConversationView:
        snapshot = await entry.session.snapshot()
        for message in snapshot.messages[len(entry.messages) :]:
            entry.messages.append(
                MessageView(
                    role=message.role,
                    text=message.text,
                    proposal=(
                        _proposal_view(message.proposal) if message.proposal is not None else None
                    ),
                )
            )
        pending = snapshot.pending_proposal
        result = snapshot.result
        outcome = None
        if result is not None:
            artifact = result.notebook
            notebook = (
                NotebookView(
                    url=f"{_CONVERSATIONS}/{conversation_id}/notebook",
                    sha256=artifact.sha256,
                    byte_length=artifact.byte_length,
                )
                if artifact is not None
                else None
            )
            outcome = ResultView(
                run_id=result.run_id,
                proposal_id=result.proposal_id,
                answer=result.answer,
                response_json=_pretty_json(result.answer) if result.status == "succeeded" else None,
                notebook=notebook,
            )
        return ConversationView(
            id=conversation_id,
            welcome=welcome,
            status=snapshot.status,
            messages=entry.messages,
            pending_proposal_id=pending.proposal_id if pending is not None else None,
            result=outcome,
            failure_code=snapshot.failure_code,
        )

    async def create_conversation(request: Request, response: Response) -> ConversationView:
        purge_idle_conversations()
        owner = request.cookies.get(_COOKIE, "")
        conversations = browser_sessions.get(owner)
        if conversations is not None and len(conversations) >= _MAX_CONVERSATIONS_PER_BROWSER:
            raise HTTPException(429, "browser_conversation_limit")
        if sum(map(len, browser_sessions.values())) >= _MAX_CONVERSATIONS:
            raise HTTPException(429, "conversation_capacity_reached")
        if conversations is None:
            owner = secrets.token_urlsafe(32)
            conversations = {}
            response.set_cookie(
                _COOKIE,
                owner,
                path="/api",
                httponly=True,
                samesite="strict",
                secure=request.url.scheme == "https",
            )
        conversation_id = secrets.token_urlsafe(24)
        entry = _Conversation(new_session(owner, conversation_id), clock())
        conversations[conversation_id] = entry
        browser_sessions[owner] = conversations
        return await view(conversation_id, entry)

    async def get_conversation(conversation_id: str, request: Request) -> ConversationView:
        return await view(conversation_id, owned_conversation(request, conversation_id))

    async def submit_message(
        conversation_id: str, body: MessageInput, request: Request
    ) -> ConversationView:
        entry = owned_conversation(request, conversation_id)
        try:
            await entry.session.submit_message(body.text)
        except BrokerError as error:
            raise HTTPException(409, error.code) from error
        return await view(conversation_id, entry)

    async def decide_proposal(
        conversation_id: str,
        proposal_id: str,
        body: DecisionInput,
        request: Request,
        response: Response,
    ) -> ConversationView:
        entry = owned_conversation(request, conversation_id)
        try:
            started = await entry.session.decide_proposal(
                proposal_id, body.digest, body.decision, run_slots=run_slots
            )
        except BrokerError as error:
            status = 429 if error.code == "analysis_capacity_reached" else 409
            raise HTTPException(status, error.code) from error
        state = await view(conversation_id, entry)
        if started or state.status == "running":
            response.status_code = 202
        return state

    async def get_notebook(conversation_id: str, request: Request) -> Response:
        entry = owned_conversation(request, conversation_id)
        snapshot = await entry.session.snapshot()
        identity = snapshot.result.notebook if snapshot.result is not None else None
        if identity is None:
            raise HTTPException(404, "notebook_unavailable")
        try:
            content = await run_in_threadpool(entry.session.read_notebook, identity)
        except Exception as error:
            raise HTTPException(404, "notebook_unavailable") from error
        return Response(
            content,
            media_type=identity.media_type,
            headers={"Content-Disposition": f'attachment; filename="{_NOTEBOOK_FILENAME}"'},
        )

    app.middleware("http")(protect_browser_requests)
    app.add_exception_handler(RequestValidationError, invalid_request)
    app.add_api_route(_CONVERSATIONS, create_conversation, methods=["POST"], status_code=201)
    conversation_path = f"{_CONVERSATIONS}/{{conversation_id}}"
    app.add_api_route(conversation_path, get_conversation, methods=["GET"])
    app.add_api_route(f"{conversation_path}/messages", submit_message, methods=["POST"])
    app.add_api_route(
        f"{conversation_path}/proposals/{{proposal_id}}/decision",
        decide_proposal,
        methods=["POST"],
        responses={202: {"model": ConversationView, "description": "Analysis accepted"}},
    )
    app.add_api_route(
        f"{conversation_path}/notebook", get_notebook, methods=["GET"], response_class=Response
    )
    return app


def create_chat_app() -> FastAPI:
    """Serve the built chat UI and its API from one origin."""
    app = create_app()
    app.mount("/", StaticFiles(directory=CHAT_WEB_ROOT, html=True), name="chat")
    return app


def _proposal_view(record: ProposalRecord) -> ProposalView:
    payload = proposal_payload_json(record.payload)
    return ProposalView(
        id=record.proposal_id,
        digest=record.proposal_sha256,
        expires_at=record.expires_at,
        payload=payload,
        request_json=_pretty_json(payload),
    )


def _pretty_json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
