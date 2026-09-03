"""Minimal Open WebUI Pipe for clarify, confirm, execute, and stop."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import re
import stat
from collections.abc import Awaitable, Callable
from contextlib import suppress
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from pydantic import BaseModel, Field, JsonValue

from apps.private_data_chat.broker import InMemoryProposalStore, PrivateDataBroker
from apps.private_data_chat.contracts import (
    AnalysisResult,
    ClarifierTurn,
    MockDatabaseContext,
    ProposalPayload,
    canonical_json_bytes,
)
from apps.private_data_chat.dsa_adapter import DsaAnalysisExecutor, DsaRuntimeConfiguration

_TERMINAL_MARKER = "<!-- dsa-private-data-terminal/v1 -->"
_MAX_CONTEXT_BYTES = 64 * 1024
_MAX_HISTORY_MESSAGES = 30
_MAX_HISTORY_BYTES = 64 * 1024
_MAX_MESSAGE_BYTES = 8 * 1024


class EventEmitter(Protocol):
    def __call__(self, event: dict[str, object]) -> Awaitable[object]: ...


class EventCall(Protocol):
    def __call__(self, event: dict[str, object]) -> Awaitable[object]: ...


class Clarifier(Protocol):
    def __call__(
        self,
        body: dict[str, object],
        user: dict[str, object],
        request: object,
    ) -> Awaitable[object]: ...


ExecutorFactory = Callable[[DsaRuntimeConfiguration], DsaAnalysisExecutor]
ConversationKey = tuple[str, str]


class Pipe:
    """Open WebUI Function entrypoint with one intentionally terminal DSA result."""

    class Valves(BaseModel):
        CLARIFIER_MODEL_ID: str = Field(
            default="",
            description="Open WebUI model ID used only with the synthetic mock context.",
        )
        MOCK_CONTEXT_PATH: str = Field(
            default="/data/dsa/mock-database.json",
            description="Absolute path to the operator-authored synthetic context.",
        )
        DATA_SOURCE_ID: str = Field(default="private_data")
        DATABASE_PATH: str = Field(default="/data/dsa/private.duckdb")
        RUNS_DIRECTORY: str = Field(default="/data/dsa/runs")
        TRUSTED_MODEL_NAME: str = Field(default="")
        TRUSTED_MODEL_SETTINGS_JSON: str = Field(default="{}")
        DOCKER_IMAGE: str = Field(default="")
        REPORT_TO_MLFLOW: bool = Field(default=False)

    def __init__(
        self,
        *,
        clarifier: Clarifier | None = None,
        executor_factory: ExecutorFactory | None = None,
    ) -> None:
        self.id = "private_data_chat"
        self.name = "Private Data Chat"
        self.valves = self.Valves()
        self._clarifier = clarifier or _openwebui_clarifier
        self._executor_factory = executor_factory or DsaAnalysisExecutor
        self._conversation_lock = asyncio.Lock()
        self._active_conversations: set[ConversationKey] = set()
        self._terminal_conversations: set[ConversationKey] = set()

    async def pipe(
        self,
        body: dict[str, object],
        __user__: dict[str, object],
        __request__: object,
        __event_emitter__: EventEmitter | None = None,
        __event_call__: EventCall | None = None,
        __metadata__: dict[str, object] | None = None,
    ) -> str:
        """Run one safe turn; a DSA result permanently closes this conversation."""
        try:
            user_id = _bounded_identity(__user__.get("id"), "user identity")
            conversation_id = _conversation_id(body, __metadata__)
        except Exception:
            return "The conversation identity is unavailable. Please start a new conversation."
        key = (user_id, conversation_id)
        reservation = await self._reserve_conversation(key, _has_terminal_result(body))
        if reservation == "terminal":
            return _closed_message()
        if reservation == "active":
            return "Another turn is already in progress in this conversation."

        try:
            try:
                context = _load_mock_context(Path(self.valves.MOCK_CONTEXT_PATH))
                if context.data_source_id != self.valves.DATA_SOURCE_ID:
                    raise ValueError("mock context does not match the configured data source")
                clarification_body = _clarification_body(
                    body,
                    context,
                    self.valves.CLARIFIER_MODEL_ID,
                )
                raw_turn = await self._clarifier(clarification_body, __user__, __request__)
                turn = _parse_clarifier_turn(raw_turn)
            except Exception:
                await _emit_status(
                    __event_emitter__,
                    "Clarification is unavailable. Check the Pipe configuration.",
                    done=True,
                )
                return (
                    "The clarification step is unavailable. Please ask the operator to check "
                    "the Pipe configuration."
                )

            if turn.kind == "clarification":
                return _remove_terminal_marker(turn.message)

            assert turn.proposal is not None
            try:
                runtime = _runtime_configuration(self.valves)
                executor = self._executor_factory(runtime)
                broker = PrivateDataBroker(store=InMemoryProposalStore(), executor=executor)
                proposal = await broker.propose(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    data_source_id=context.data_source_id,
                    payload=turn.proposal,
                )
            except Exception:
                return (
                    "The proposed analysis could not be prepared safely. Please refine the "
                    "question and try again."
                )

            confirmed = await _confirm(
                __event_call__,
                proposal.payload,
                proposal.proposal_sha256,
            )
            if not confirmed:
                return (
                    "The analysis was not run. You can continue refining the question in this "
                    "conversation."
                )

            await self._mark_terminal(key)
            await _emit_status(__event_emitter__, "Running the approved analysis…", done=False)
            try:
                await broker.approve(
                    proposal_id=proposal.proposal_id,
                    proposal_sha256=proposal.proposal_sha256,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                result = await broker.execute(
                    proposal_id=proposal.proposal_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                response = await _terminal_response(result, executor, __event_emitter__)
            except Exception:
                response = _terminal_failure("analysis_orchestration_failed")
            finally:
                await _emit_status(__event_emitter__, "Analysis complete.", done=True)
            return response
        finally:
            await self._release_conversation(key)

    async def _reserve_conversation(
        self,
        key: ConversationKey,
        terminal_in_history: bool,
    ) -> str:
        async with self._conversation_lock:
            if terminal_in_history:
                self._terminal_conversations.add(key)
            if key in self._terminal_conversations:
                return "terminal"
            if key in self._active_conversations:
                return "active"
            self._active_conversations.add(key)
            return "reserved"

    async def _mark_terminal(self, key: ConversationKey) -> None:
        async with self._conversation_lock:
            self._terminal_conversations.add(key)

    async def _release_conversation(self, key: ConversationKey) -> None:
        async with self._conversation_lock:
            self._active_conversations.discard(key)


async def _openwebui_clarifier(
    body: dict[str, object],
    user_data: dict[str, object],
    request: object,
) -> object:
    users_module = import_module("open_webui.models.users")
    chat_module = import_module("open_webui.utils.chat")
    user_id = _bounded_identity(user_data.get("id"), "user identity")
    user = await users_module.Users.get_user_by_id(user_id)
    if user is None:
        raise ValueError("Open WebUI user was not found")
    return await chat_module.generate_chat_completion(request, body, user)


def _clarification_body(
    body: dict[str, object],
    context: MockDatabaseContext,
    model_id: str,
) -> dict[str, object]:
    if not model_id.strip():
        raise ValueError("clarifier model is not configured")
    prompt = _skill_text()
    context_json = canonical_json_bytes(context).decode("utf-8")
    system = f"{prompt}\n\nSynthetic database context:\n{context_json}"
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(_safe_history(body.get("messages")))
    return {
        "model": model_id,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "stream": False,
    }


def _safe_history(raw_messages: object) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        raise ValueError("chat messages are unavailable")
    retained: list[dict[str, str]] = []
    total = 0
    for raw_value in reversed(cast(list[object], raw_messages)):
        raw = cast(dict[str, object], raw_value) if isinstance(raw_value, dict) else None
        if raw is None or raw.get("role") not in {"user", "assistant"}:
            continue
        content = raw.get("content")
        if not isinstance(content, str):
            continue
        content = _remove_terminal_marker(content)
        encoded = content.encode("utf-8")
        if len(encoded) > _MAX_MESSAGE_BYTES:
            encoded = encoded[:_MAX_MESSAGE_BYTES]
            content = encoded.decode("utf-8", errors="ignore")
        if total + len(encoded) > _MAX_HISTORY_BYTES:
            break
        retained.append({"role": cast(str, raw["role"]), "content": content})
        total += len(encoded)
        if len(retained) == _MAX_HISTORY_MESSAGES:
            break
    retained.reverse()
    if not retained or retained[-1]["role"] != "user":
        raise ValueError("the current user message is unavailable")
    return retained


def _parse_clarifier_turn(raw: object) -> ClarifierTurn:
    response = raw
    body = getattr(response, "body", None)
    if isinstance(body, bytes):
        response = json.loads(body)
    if not isinstance(response, dict):
        raise ValueError("clarifier response is not an object")
    response_mapping = cast(dict[str, object], response)
    choices = response_mapping.get("choices")
    if not isinstance(choices, list):
        raise ValueError("clarifier response has no unique choice")
    choice_values = cast(list[object], choices)
    if len(choice_values) != 1 or not isinstance(choice_values[0], dict):
        raise ValueError("clarifier response has no unique choice")
    choice = cast(dict[str, object], choice_values[0])
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("clarifier response has no text content")
    message_mapping = cast(dict[str, object], message)
    content = message_mapping.get("content")
    if not isinstance(content, str):
        raise ValueError("clarifier response has no text content")
    payload = json.loads(content)
    return ClarifierTurn.model_validate(payload)


def _runtime_configuration(valves: Pipe.Valves) -> DsaRuntimeConfiguration:
    settings = json.loads(valves.TRUSTED_MODEL_SETTINGS_JSON)
    if not isinstance(settings, dict):
        raise ValueError("trusted model settings must be an object")
    return DsaRuntimeConfiguration(
        data_source_id=valves.DATA_SOURCE_ID,
        database_path=Path(valves.DATABASE_PATH),
        runs_directory=Path(valves.RUNS_DIRECTORY),
        trusted_model_name=valves.TRUSTED_MODEL_NAME,
        trusted_model_settings=cast(dict[str, JsonValue], settings),
        docker_image=valves.DOCKER_IMAGE,
        report_to_mlflow=valves.REPORT_TO_MLFLOW,
    )


def _load_mock_context(path: Path) -> MockDatabaseContext:
    if not path.is_absolute():
        raise ValueError("mock context path must be absolute")
    content = _read_regular_file(path, _MAX_CONTEXT_BYTES)
    return MockDatabaseContext.model_validate_json(content)


def _skill_text() -> str:
    path = Path(__file__).with_name("clarification-skill.md")
    return _read_regular_file(path, _MAX_CONTEXT_BYTES).decode("utf-8")


def _read_regular_file(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise ValueError("configured file is not a bounded regular file")
        content = b""
        while len(content) <= limit:
            chunk = os.read(descriptor, min(64 * 1024, limit + 1 - len(content)))
            if not chunk:
                break
            content += chunk
    finally:
        os.close(descriptor)
    if len(content) > limit:
        raise ValueError("configured file exceeds its byte limit")
    return content


async def _confirm(
    event_call: EventCall | None,
    payload: ProposalPayload,
    digest: str,
) -> bool:
    if event_call is None:
        return False
    proposal_json = json.dumps(
        payload.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    try:
        result = await event_call(
            {
                "type": "confirmation",
                "data": {
                    "title": "Run this analysis on the private database?",
                    "message": (
                        "The mock rows were used only to clarify the request. Confirming sends "
                        "the following exact request to the trusted DSA model.\n\n"
                        f"```json\n{proposal_json}\n```\n\nProposal digest: `{digest}`"
                    ),
                },
            }
        )
    except Exception:
        return False
    return result is True


async def _terminal_response(
    result: AnalysisResult,
    executor: DsaAnalysisExecutor,
    emitter: EventEmitter | None,
) -> str:
    if result.status == "failed":
        return _terminal_failure(result.failure_code or "dsa_run_failed")
    notebook_note = "No verified derivation notebook was produced."
    if result.notebook is not None:
        try:
            content = executor.read_notebook(result.notebook)
            await _emit_notebook(emitter, content, result.notebook.sha256)
            notebook_note = (
                "Use the download control below to save the verified derivation notebook."
            )
        except Exception:
            notebook_note = (
                "A verified notebook was retained, but its download could not be attached."
            )
    return (
        f"{_TERMINAL_MARKER}\n"
        "## Result\n\n"
        f"{_json_fence(result.answer)}\n\n"
        f"{notebook_note}\n\n"
        "This analysis is complete. Start a new conversation for another question."
    )


def _terminal_failure(code: str) -> str:
    return (
        f"{_TERMINAL_MARKER}\n"
        f"The approved DSA analysis ended without an answer (`{code}`).\n\n"
        "This analysis is complete. Start a new conversation to try another question."
    )


async def _emit_notebook(
    emitter: EventEmitter | None,
    content: bytes,
    digest: str,
) -> None:
    if emitter is None:
        return
    encoded = base64.b64encode(content).decode("ascii")
    html = (
        '<div style="font-family:system-ui;padding:12px">'
        '<a download="dsa-derivation.ipynb" '
        f'href="data:application/x-ipynb+json;base64,{encoded}" '
        'style="display:inline-block;padding:8px 12px;border-radius:6px;'
        'background:#2563eb;color:white;text-decoration:none">'
        "Download verified notebook</a>"
        f'<div style="margin-top:8px;font-size:12px;color:#666">SHA-256: {digest}</div>'
        "</div>"
    )
    await emitter({"type": "embeds", "data": {"embeds": [html], "replace": True}})


async def _emit_status(
    emitter: EventEmitter | None,
    description: str,
    *,
    done: bool,
) -> None:
    if emitter is None:
        return
    with suppress(Exception):
        result = emitter(
            {"type": "status", "data": {"description": description, "done": done}}
        )
        if inspect.isawaitable(result):
            await result


def _json_fence(value: JsonValue | None) -> str:
    rendered = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    longest = max((len(run) for run in re.findall(r"`+", rendered)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}json\n{rendered}\n{fence}"


def _has_terminal_result(body: dict[str, object]) -> bool:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    for value in cast(list[object], messages):
        if not isinstance(value, dict):
            continue
        message = cast(dict[str, object], value)
        content = message.get("content")
        if (
            message.get("role") == "assistant"
            and isinstance(content, str)
            and _TERMINAL_MARKER in content
        ):
            return True
    return False


def _conversation_id(
    body: dict[str, object],
    metadata: dict[str, object] | None,
) -> str:
    candidates = (
        None if metadata is None else metadata.get("chat_id"),
        body.get("chat_id"),
        body.get("id"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return _bounded_identity(candidate, "conversation identity")
    raise ValueError("conversation identity is unavailable")


def _bounded_identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{label} is invalid")
    return value


def _remove_terminal_marker(value: str) -> str:
    return value.replace(_TERMINAL_MARKER, "")


def _closed_message() -> str:
    return (
        f"{_TERMINAL_MARKER}\n"
        "This analysis is already complete. Start a new conversation for another question."
    )
