"""Same-origin HTTP adapter for the private-data conversation workflow."""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, JsonValue
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from apps.private_data_chat.broker import BrokerError
from apps.private_data_chat.chat import ChatStatus, PrivateDataChatSession, ProposalDecision
from apps.private_data_chat.contracts import AppContract, ProposalRecord, proposal_payload_json
from apps.private_data_chat.service import load_chat_service, model_display_name, welcome_message

SessionFactory = Callable[[str, str], PrivateDataChatSession]
_COOKIE = "dsa_chat_session"
_CONVERSATIONS = "/api/conversations"
_NOTEBOOK_FILENAME = "dsa-derivation.ipynb"


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


def create_app(
    *,
    session_factory: SessionFactory | None = None,
    welcome: str = "",
) -> FastAPI:
    """Create the local single-process API, with injectable sessions for tests."""
    if session_factory is None:
        service = load_chat_service()
        session_factory = service.create_session
        welcome = welcome_message(
            service.context,
            service.database_card,
            model_display_name(service.configuration.clarifier_model.name),
        )
    new_session = session_factory
    browser_sessions: dict[str, dict[str, PrivateDataChatSession]] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            await asyncio.gather(
                *(
                    session.cancel_run()
                    for conversations in browser_sessions.values()
                    for session in conversations.values()
                )
            )
            browser_sessions.clear()

    app = FastAPI(title="DSA Private Data Chat", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    async def protect_browser_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        origin = request.headers.get("origin")
        expected_origin = f"{request.url.scheme}://{request.url.netloc}"
        if request.method == "POST" and (
            (origin is not None and origin != expected_origin)
            or request.headers.get("sec-fetch-site") == "cross-site"
        ):
            response: Response = JSONResponse({"detail": "invalid_origin"}, status_code=403)
        else:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    async def invalid_request(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse({"detail": "invalid_request"}, status_code=422)

    def owned_session(request: Request, conversation_id: str) -> PrivateDataChatSession:
        owner = request.cookies.get(_COOKIE, "")
        session = browser_sessions.get(owner, {}).get(conversation_id)
        if session is None:
            raise HTTPException(404, "conversation_not_found")
        return session

    async def view(conversation_id: str, session: PrivateDataChatSession) -> ConversationView:
        snapshot = await session.snapshot()
        messages = [
            MessageView(
                role=message.role,
                text=message.text,
                proposal=_proposal_view(message.proposal) if message.proposal is not None else None,
            )
            for message in snapshot.messages
        ]
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
            messages=messages,
            pending_proposal_id=pending.proposal_id if pending is not None else None,
            result=outcome,
            failure_code=snapshot.failure_code,
        )

    async def create_conversation(request: Request, response: Response) -> ConversationView:
        owner = request.cookies.get(_COOKIE, "")
        if owner not in browser_sessions:
            owner = secrets.token_urlsafe(32)
            browser_sessions[owner] = {}
            response.set_cookie(
                _COOKIE,
                owner,
                path="/api",
                httponly=True,
                samesite="strict",
                secure=request.url.scheme == "https",
            )
        conversation_id = secrets.token_urlsafe(24)
        session = new_session(owner, conversation_id)
        browser_sessions[owner][conversation_id] = session
        return await view(conversation_id, session)

    async def get_conversation(conversation_id: str, request: Request) -> ConversationView:
        return await view(conversation_id, owned_session(request, conversation_id))

    async def submit_message(
        conversation_id: str, body: MessageInput, request: Request
    ) -> ConversationView:
        session = owned_session(request, conversation_id)
        try:
            await session.submit_message(body.text)
        except BrokerError as error:
            raise HTTPException(409, error.code) from error
        return await view(conversation_id, session)

    async def decide_proposal(
        conversation_id: str,
        proposal_id: str,
        body: DecisionInput,
        request: Request,
        response: Response,
    ) -> ConversationView:
        session = owned_session(request, conversation_id)
        try:
            started = await session.decide_proposal(proposal_id, body.digest, body.decision)
        except BrokerError as error:
            raise HTTPException(409, error.code) from error
        state = await view(conversation_id, session)
        if started or state.status == "running":
            response.status_code = 202
        return state

    async def get_notebook(conversation_id: str, request: Request) -> Response:
        session = owned_session(request, conversation_id)
        snapshot = await session.snapshot()
        identity = snapshot.result.notebook if snapshot.result is not None else None
        if identity is None:
            raise HTTPException(404, "notebook_unavailable")
        try:
            content = await run_in_threadpool(session.read_notebook, identity)
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
