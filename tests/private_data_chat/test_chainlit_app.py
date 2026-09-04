from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from apps.private_data_chat.contracts import (
    MockDatabaseContext,
    MockRelation,
    ProposalBinding,
    ProposalPayload,
    ProposalRecord,
    ProposalStatus,
    proposal_digest,
)
from tests.private_data_chat.test_database_card import MODEL_ONLY_NOTE, card

from .test_contracts import proposal_payload


def test_welcome_combines_fixed_trust_flow_with_dataset_context() -> None:
    pytest.importorskip("chainlit")
    application = importlib.import_module("apps.private_data_chat.chainlit_app")
    context = MockDatabaseContext(
        data_source_id="retail",
        display_name="Online Retail II",
        relations=(MockRelation(name="analysis.lines", columns=("value",)),),
    )

    message = application._welcome_message(
        context,
        card(),
        application._model_display_name("openrouter:example/remote-model"),
    )

    assert message.startswith("## Instructions")
    assert "**remote-model**" in message
    assert "openrouter" not in message
    assert "request for a subagent" in message
    assert "run by a trusted model" in message
    assert "ask a follow-up question" in message
    assert "Review the subagent request" in message
    assert "downloadable Jupyter notebook" in message
    assert "session ends" in message
    assert "## About Online Retail II" in message
    assert "The original first paragraph." in message
    assert "### Available data" in message
    assert "1,044,848" in message
    assert "- How did monthly sales change?" in message
    assert MODEL_ONLY_NOTE not in message


def test_model_display_name_omits_provider_and_routing_qualifiers() -> None:
    pytest.importorskip("chainlit")
    application = importlib.import_module("apps.private_data_chat.chainlit_app")

    assert (
        application._model_display_name("openrouter:deepseek/deepseek-v4-flash-0731")
        == "deepseek-v4-flash-0731"
    )
    assert application._model_display_name("openai-chat:local-model") == "local-model"


def test_chainlit_adapter_imports_and_strictly_checks_actions() -> None:
    pytest.importorskip("chainlit")
    application = importlib.import_module("apps.private_data_chat.chainlit_app")
    digest = "a" * 64

    assert application._approved_action(
        {
            "name": "dsa_approve",
            "payload": {"decision": "approve", "proposal_sha256": digest},
        },
        digest,
    )
    assert not application._approved_action(
        {
            "name": "dsa_approve",
            "payload": {"decision": "approve", "proposal_sha256": "b" * 64},
        },
        digest,
    )
    assert not application._approved_action(None, digest)


async def test_confirmation_keeps_proposal_persistent_and_uses_a_small_action_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("chainlit")
    application = importlib.import_module("apps.private_data_chat.chainlit_app")
    payload = ProposalPayload.model_validate(
        {
            **proposal_payload().model_dump(mode="python"),
            "analysis_guidance": "Filter to 2011, then aggregate net sales by month.",
        }
    )
    binding = ProposalBinding(
        user_id="user-1",
        conversation_id="chat-1",
        data_source_id="retail",
    )
    created_at = datetime(2026, 9, 4, 12, tzinfo=UTC)
    record = ProposalRecord(
        proposal_id="proposal-0123456789abcdef",
        proposal_sha256=proposal_digest(binding, payload),
        payload=payload,
        binding=binding,
        status=ProposalStatus.PROPOSED,
        created_at=created_at,
        expires_at=created_at + timedelta(minutes=30),
    )
    messages: list[str] = []
    prompts: list[str] = []

    class Message:
        def __init__(self, *, content: str) -> None:
            self.content = content

        async def send(self) -> None:
            messages.append(self.content)

    class AskActionMessage:
        def __init__(self, *, content: str, **kwargs: object) -> None:
            del kwargs
            self.content = content

        async def send(self) -> dict[str, object]:
            prompts.append(self.content)
            return {
                "name": "dsa_approve",
                "payload": {
                    "decision": "approve",
                    "proposal_sha256": record.proposal_sha256,
                },
            }

    class Action:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    monkeypatch.setattr(
        application,
        "cl",
        SimpleNamespace(Message=Message, AskActionMessage=AskActionMessage, Action=Action),
    )

    assert await application._confirm_proposal(record) is True
    assert "analysis_guidance" in messages[0]
    assert record.proposal_sha256 in messages[0]
    assert messages[1] == "Running the approved analysis…"
    assert prompts == [f"Run proposal `{record.proposal_sha256}` against the private database?"]
    assert payload.question not in prompts[0]
