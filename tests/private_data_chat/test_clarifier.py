from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from apps.private_data_chat.clarifier import (
    ConversationMessage,
    PydanticClarifier,
    append_history,
    load_mock_context,
)
from apps.private_data_chat.contracts import ClarifierTurn, MockDatabaseContext
from dsa import ModelConfiguration
from tests.private_data_chat.test_database_card import MODEL_ONLY_NOTE, card


def mock_context() -> MockDatabaseContext:
    path = Path(__file__).parents[2] / "apps/private_data_chat/mock-database.example.json"
    return MockDatabaseContext.model_validate_json(path.read_bytes())


async def test_clarifier_receives_only_skill_mock_context_history_and_safe_model() -> None:
    calls: list[tuple[str, str, ModelConfiguration]] = []

    async def runner(
        instructions: str,
        prompt: str,
        configuration: ModelConfiguration,
    ) -> ClarifierTurn:
        calls.append((instructions, prompt, configuration))
        return ClarifierTurn(kind="clarification", message="Which year?")

    clarifier = PydanticClarifier(
        ModelConfiguration(name="openai:untrusted", settings={"temperature": 0.2}),
        MockDatabaseContext.model_validate(
            {
                **mock_context().model_dump(mode="python"),
                "relations": [
                    {
                        "name": "analysis.lines",
                        "columns": ["invoice_id", "quantity"],
                        "sample_rows": [{"invoice_id": "FAKE-1001", "quantity": 2}],
                    }
                ],
            }
        ),
        database_card=card(),
        runner=runner,
    )
    result = await clarifier.clarify(
        (ConversationMessage(role="user", content="Show sales by month."),)
    )

    assert result.message == "Which year?"
    instructions, prompt, configuration = calls[0]
    assert "FAKE-1001" in instructions
    assert '"synthetic":true' in instructions
    assert "Show sales by month." in prompt
    combined = instructions + prompt + configuration.model_dump_json()
    assert "/private/real.duckdb" not in combined
    assert "SECRET" not in combined
    assert "Analysis guidance is disabled" in instructions
    assert "Dataset-owned database card:" in instructions
    assert '"primary_relation":"analysis.lines"' in instructions
    assert MODEL_ONLY_NOTE in instructions


async def test_clarifier_can_be_instructed_to_propose_untrusted_analysis_guidance() -> None:
    calls: list[str] = []

    async def runner(
        instructions: str,
        prompt: str,
        configuration: ModelConfiguration,
    ) -> ClarifierTurn:
        del prompt, configuration
        calls.append(instructions)
        return ClarifierTurn(kind="clarification", message="Which year?")

    clarifier = PydanticClarifier(
        ModelConfiguration(name="openai:untrusted"),
        mock_context(),
        enable_analysis_guidance=True,
        runner=runner,
    )
    await clarifier.clarify((ConversationMessage(role="user", content="Analyze sales."),))

    assert "Analysis guidance is enabled" in calls[0]
    assert "numbered list of 3 to 8 steps" in calls[0]
    assert "must not claim" in calls[0]


def test_clarifier_revalidates_a_mutated_typed_database_card() -> None:
    selected = card().model_copy(update={"relations": ()})

    with pytest.raises(ValidationError, match="at least 1 item"):
        PydanticClarifier(
            ModelConfiguration(name="openai:untrusted"),
            mock_context(),
            database_card=selected,
        )


async def test_clarifier_revalidates_untrusted_structured_output() -> None:
    async def runner(
        instructions: str,
        prompt: str,
        configuration: ModelConfiguration,
    ) -> object:
        del instructions, prompt, configuration
        return {"kind": "proposal", "message": "Ready", "proposal": None}

    clarifier = PydanticClarifier(
        ModelConfiguration(name="test"),
        mock_context(),
        runner=runner,
    )
    with pytest.raises(ValidationError, match="only proposal turns"):
        await clarifier.clarify((ConversationMessage(role="user", content="Analyze it."),))


def test_history_is_bounded_to_newest_messages_and_utf8_bytes() -> None:
    history = tuple(
        ConversationMessage(role="user", content=f"message-{index}") for index in range(40)
    )
    history = append_history(history, "assistant", "é" * 10_000)

    assert len(history) <= 30
    assert sum(len(message.content.encode("utf-8")) for message in history) <= 64 * 1024
    assert all(len(message.content.encode("utf-8")) <= 8 * 1024 for message in history)
    assert any(message.role == "assistant" and message.content.endswith("é") for message in history)
    assert all(message.content != "message-0" for message in history)


def test_history_rejects_blank_messages() -> None:
    with pytest.raises(ValueError, match="blank"):
        append_history((), "user", "  ")


def test_mock_context_loader_rejects_relative_symlink_and_oversized_files(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "mock.json"
    valid.write_text(json.dumps(mock_context().model_dump(mode="json")))
    assert load_mock_context(valid).data_source_id == "online_retail_ii"

    with pytest.raises(ValueError, match="absolute"):
        load_mock_context(Path("mock.json"))

    symlink = tmp_path / "mock-link.json"
    symlink.symlink_to(valid)
    with pytest.raises(OSError):
        load_mock_context(symlink)

    oversized = tmp_path / "large.json"
    oversized.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="bounded"):
        load_mock_context(oversized)
