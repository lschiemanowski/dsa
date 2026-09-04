from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from apps.private_data_chat.contracts import (
    ProposalBinding,
    ProposalPayload,
    ProposalRecord,
    ProposalStatus,
    proposal_digest,
)

from .test_contracts import proposal_payload


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
