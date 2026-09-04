"""Thin Chainlit UI for the framework-neutral private-data chat flow."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from apps.private_data_chat.chat import PrivateDataChatSession, render_proposal
from apps.private_data_chat.clarifier import PydanticClarifier, load_mock_context
from apps.private_data_chat.contracts import MockDatabaseContext, ProposalRecord
from apps.private_data_chat.dsa_adapter import DsaAnalysisExecutor
from apps.private_data_chat.public_description import load_database_description
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
        database_description = (
            load_database_description(configuration.database_description)
            if configuration.database_description is not None
            else None
        )
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
                database_description=database_description,
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

    await cl.Message(content=_welcome_message(context, database_description)).send()


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


def _welcome_message(
    context: MockDatabaseContext,
    database_description: str | None,
) -> str:
    """Combine the fixed trust flow with bounded dataset-owned public context."""
    sections = [
        "## How this works\n\n"
        "1. Ask a quantitative question in ordinary language.\n"
        "2. A capable but untrusted model helps make it precise. It sees only this chat, the "
        "configured public context below, and synthetic example rows—not the real data.\n"
        "3. You review the exact question, answer format, and any untrusted analysis guidance. "
        "Nothing accesses the real database until you approve that proposal.\n"
        "4. After approval, a separately configured trusted model runs DSA against the real "
        "DuckDB. You receive its structured answer and, when derivation replay succeeds, a "
        "downloadable Jupyter notebook.\n\n"
        "The DSA result ends this conversation; start a new chat for another analysis.",
        f"## About {context.display_name}",
    ]
    if database_description is not None:
        sections.append(database_description)
    else:
        sections.append(
            "This configured data source provides a synthetic schema-compatible sample for "
            "question clarification."
        )
    sections.append(f"Ask a question about **{context.display_name}** to begin.")
    return "\n\n".join(sections)
