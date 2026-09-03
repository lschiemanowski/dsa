from __future__ import annotations

import importlib

import pytest


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
