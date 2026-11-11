"""Thin Chainlit UI for the framework-neutral private-data chat flow."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from apps.private_data_chat.chat import ChatResponse, PrivateDataChatSession, render_proposal
from apps.private_data_chat.contracts import ProposalRecord
from apps.private_data_chat.presentation import escape_markdown_text
from apps.private_data_chat.service import load_chat_service
from apps.private_data_chat.service import model_display_name as _model_display_name
from apps.private_data_chat.service import welcome_message as _welcome_message

cl: Any = import_module("chainlit")

_SESSION_KEY = "dsa_private_data_chat_session"
_CONFIRM_SECONDS = 300


@cl.on_chat_start
async def on_chat_start() -> None:
    """Build one isolated application session from host-owned configuration."""
    try:
        service = load_chat_service()
        conversation_id = _session_identity(cl.user_session.get("id"), "conversation")
        user = cl.user_session.get("user")
        user_id = _session_identity(getattr(user, "identifier", None), conversation_id)
        session = service.create_session(user_id, conversation_id)
        cl.user_session.set(_SESSION_KEY, session)
    except Exception:
        cl.user_session.set(_SESSION_KEY, None)
        await cl.Message(
            content="Private Data Chat is unavailable. Check the host configuration."
        ).send()
        return

    await cl.Message(
        content=_welcome_message(
            service.context,
            service.database_card,
            _model_display_name(service.configuration.clarifier_model.name),
        )
    ).send()


@cl.on_message
async def on_message(message: Any) -> None:
    """Clarify one turn and, after native confirmation, run DSA exactly once."""
    session = cl.user_session.get(_SESSION_KEY)
    if not isinstance(session, PrivateDataChatSession):
        await cl.Message(
            content="Private Data Chat is unavailable. Start a new chat after configuration."
        ).send()
        return

    response = await session.handle(str(message.content), _confirm_proposal)
    await _send_response(response)


async def _send_response(response: ChatResponse) -> None:
    """Display trusted results; only verified bytes become image/file elements."""
    elements: list[object] = []
    if response.notebook is not None:
        elements.append(
            cl.File(
                name="dsa-derivation.ipynb",
                content=response.notebook.content,
                display="inline",
            )
        )
    await cl.Message(content=response.content, elements=elements).send()
    for plot in response.plots:
        await cl.Message(
            content=escape_markdown_text(plot.declaration.title)
            + (
                "\n\n" + escape_markdown_text(plot.declaration.caption)
                if plot.declaration.caption
                else ""
            ),
            elements=[
                cl.Image(name=plot.declaration.filename, content=plot.content, display="inline"),
                cl.File(name=plot.declaration.filename, content=plot.content, display="inline"),
            ],
        ).send()


async def _confirm_proposal(proposal: ProposalRecord) -> bool:
    """Persist the proposal before showing the transient native action prompt."""
    await cl.Message(content=render_proposal(proposal)).send()
    response = await cl.AskActionMessage(
        content="Run this proposal against the private database?",
        actions=[
            cl.Action(
                name="dsa_approve",
                payload={
                    "decision": "approve",
                    "proposal_sha256": proposal.proposal_sha256,
                },
                label="Run analysis",
            ),
            cl.Action(
                name="dsa_revise",
                payload={
                    "decision": "revise",
                    "proposal_sha256": proposal.proposal_sha256,
                },
                label="Keep refining",
            ),
        ],
        timeout=_CONFIRM_SECONDS,
        raise_on_timeout=False,
    ).send()
    approved = _approved_action(response, proposal.proposal_sha256)
    if approved:
        await cl.Message(content="Running the approved analysis…").send()
    return approved


def _approved_action(response: object, expected_digest: str) -> bool:
    """Accept only the exact host-rendered approval action and digest."""
    if not isinstance(response, dict):
        return False
    action = cast(dict[object, object], response)
    payload = action.get("payload")
    if not isinstance(payload, dict):
        return False
    values = cast(dict[object, object], payload)
    return bool(
        action.get("name") == "dsa_approve"
        and values.get("decision") == "approve"
        and values.get("proposal_sha256") == expected_digest
    )


def _session_identity(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip() and len(value) <= 240:
        return value
    return f"chainlit-{fallback}"[:256]
