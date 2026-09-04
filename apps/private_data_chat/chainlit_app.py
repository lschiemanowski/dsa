"""Thin Chainlit UI for the framework-neutral private-data chat flow."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from apps.private_data_chat.chat import PrivateDataChatSession, render_proposal
from apps.private_data_chat.clarifier import PydanticClarifier, load_mock_context
from apps.private_data_chat.contracts import ProposalRecord
from apps.private_data_chat.dsa_adapter import DsaAnalysisExecutor
from apps.private_data_chat.settings import load_configuration

cl: Any = import_module("chainlit")

_SESSION_KEY = "dsa_private_data_chat_session"
_CONFIRM_SECONDS = 300


@cl.on_chat_start
async def on_chat_start() -> None:
    """Build one isolated application session from host-owned configuration."""
    try:
        configuration = load_configuration()
        context = load_mock_context(configuration.mock_context_path)
        if context.data_source_id != configuration.data_source_id:
            raise ValueError("mock and configured data source IDs differ")
        conversation_id = _session_identity(cl.user_session.get("id"), "conversation")
        user = cl.user_session.get("user")
        user_id = _session_identity(getattr(user, "identifier", None), conversation_id)
        session = PrivateDataChatSession(
            user_id=user_id,
            conversation_id=conversation_id,
            context=context,
            clarifier=PydanticClarifier(
                configuration.clarifier_model,
                context,
                enable_analysis_guidance=configuration.enable_analysis_guidance,
            ),
            executor=DsaAnalysisExecutor(configuration.dsa),
            enable_analysis_guidance=configuration.enable_analysis_guidance,
        )
        cl.user_session.set(_SESSION_KEY, session)
    except Exception:
        cl.user_session.set(_SESSION_KEY, None)
        await cl.Message(
            content="Private Data Chat is unavailable. Check the host configuration."
        ).send()
        return

    await cl.Message(
        content=(
            f"Ask a quantitative question about **{context.display_name}**. I will help make "
            "the request precise using synthetic example data, then ask before DSA accesses "
            "the real database."
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


async def _confirm_proposal(proposal: ProposalRecord) -> bool:
    """Persist the proposal before showing the transient native action prompt."""
    await cl.Message(content=render_proposal(proposal)).send()
    response = await cl.AskActionMessage(
        content=f"Run proposal `{proposal.proposal_sha256}` against the private database?",
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
