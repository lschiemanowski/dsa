"""Tool-free clarification against bounded text and a synthetic database context."""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from apps.private_data_chat.contracts import (
    ClarifierTurn,
    MockDatabaseContext,
    canonical_json_bytes,
)
from dsa import ModelConfiguration

_MAX_CONTEXT_BYTES = 64 * 1024
_MAX_HISTORY_MESSAGES = 30
_MAX_HISTORY_BYTES = 64 * 1024
_MAX_MESSAGE_BYTES = 8 * 1024
_MAX_CLARIFIER_TOKENS = 32_000
_MAX_CLARIFIER_SECONDS = 120


@dataclass(frozen=True)
class ConversationMessage:
    """One bounded user-visible message retained for clarification."""

    role: Literal["user", "assistant"]
    content: str


ClarifierRunner = Callable[
    [str, str, ModelConfiguration],
    Awaitable[ClarifierTurn | object],
]


class PydanticClarifier:
    """Run a structured Pydantic AI turn without registering any tools."""

    def __init__(
        self,
        configuration: ModelConfiguration,
        context: MockDatabaseContext,
        *,
        enable_analysis_guidance: bool = False,
        runner: ClarifierRunner | None = None,
    ) -> None:
        self.configuration = ModelConfiguration.model_validate_json(
            configuration.model_dump_json()
        )
        self._instructions = _clarifier_instructions(
            context,
            enable_analysis_guidance=enable_analysis_guidance,
        )
        self._runner = runner or _run_pydantic_clarifier

    async def clarify(self, history: Sequence[ConversationMessage]) -> ClarifierTurn:
        prompt = _conversation_prompt(history)
        raw = await self._runner(self._instructions, prompt, self.configuration)
        value = (
            raw.model_dump(mode="python", round_trip=True)
            if isinstance(raw, ClarifierTurn)
            else raw
        )
        return ClarifierTurn.model_validate(value).model_copy(deep=True)


def append_history(
    history: Sequence[ConversationMessage],
    role: Literal["user", "assistant"],
    content: str,
) -> tuple[ConversationMessage, ...]:
    """Append one message and retain only the newest bounded UTF-8 suffix."""
    bounded = ConversationMessage(role=role, content=_bounded_text(content, _MAX_MESSAGE_BYTES))
    candidates = [*history, bounded]
    retained: list[ConversationMessage] = []
    retained_bytes = 0
    for message in reversed(candidates):
        size = len(message.content.encode("utf-8"))
        if retained and retained_bytes + size > _MAX_HISTORY_BYTES:
            break
        retained.append(message)
        retained_bytes += size
        if len(retained) == _MAX_HISTORY_MESSAGES:
            break
    retained.reverse()
    return tuple(retained)


def load_mock_context(path: Path) -> MockDatabaseContext:
    """Read one bounded, regular, non-symlink synthetic context file."""
    if not path.is_absolute():
        raise ValueError("mock context path must be absolute")
    return MockDatabaseContext.model_validate_json(_read_regular_file(path, _MAX_CONTEXT_BYTES))


def _clarifier_instructions(
    context: MockDatabaseContext,
    *,
    enable_analysis_guidance: bool,
) -> str:
    skill_path = Path(__file__).with_name("clarification-skill.md")
    skill = _read_regular_file(skill_path, _MAX_CONTEXT_BYTES).decode("utf-8")
    context_json = canonical_json_bytes(context).decode("utf-8")
    guidance_policy = (
        "Analysis guidance is enabled. A proposal may include concise, ordered "
        "analysis_guidance for a human verifier, with optional SQL or Python snippets. "
        "It must not claim that mock-derived values answer the real question or that any "
        "suggested code has been executed."
        if enable_analysis_guidance
        else "Analysis guidance is disabled. Omit analysis_guidance from proposals."
    )
    return (
        f"{skill}\n\nHost analysis-guidance policy:\n{guidance_policy}\n\n"
        f"Synthetic database context:\n{context_json}"
    )


def _conversation_prompt(history: Sequence[ConversationMessage]) -> str:
    if not history or history[-1].role != "user":
        raise ValueError("clarification history must end with a user message")
    lines = ["Conversation to clarify:"]
    lines.extend(f"{message.role.upper()}: {message.content}" for message in history)
    return "\n\n".join(lines)


async def _run_pydantic_clarifier(
    instructions: str,
    prompt: str,
    configuration: ModelConfiguration,
) -> ClarifierTurn:
    settings = {**configuration.settings, "max_tokens": 4_096}
    agent = Agent(
        configuration.name,
        output_type=ClarifierTurn,
        instructions=instructions,
        retries=1,
        tools=[],
    )
    async with asyncio.timeout(_MAX_CLARIFIER_SECONDS):
        result = await agent.run(
            prompt,
            model_settings=cast(ModelSettings, settings),
            usage_limits=UsageLimits(
                request_limit=2,
                total_tokens_limit=_MAX_CLARIFIER_TOKENS,
            ),
        )
    return result.output


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


def _bounded_text(value: str, limit: int) -> str:
    if not value.strip():
        raise ValueError("chat message must not be blank")
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="ignore")
