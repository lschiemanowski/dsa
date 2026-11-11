"""Host-owned configuration shared by both private-data chat transports."""

from dataclasses import dataclass

from apps.private_data_chat.chat import PrivateDataChatSession
from apps.private_data_chat.clarifier import PydanticClarifier, load_mock_context
from apps.private_data_chat.contracts import MockDatabaseContext
from apps.private_data_chat.database_card import (
    DatabaseCard,
    load_database_card,
    render_database_overview,
)
from apps.private_data_chat.dsa_adapter import DsaAnalysisExecutor
from apps.private_data_chat.presentation import escape_markdown_inline
from apps.private_data_chat.settings import PrivateDataChatConfiguration, load_configuration


@dataclass(frozen=True)
class ChatService:
    configuration: PrivateDataChatConfiguration
    context: MockDatabaseContext
    database_card: DatabaseCard | None

    def create_session(self, user_id: str, conversation_id: str) -> PrivateDataChatSession:
        configuration = self.configuration
        return PrivateDataChatSession(
            user_id=user_id,
            conversation_id=conversation_id,
            context=self.context,
            clarifier=PydanticClarifier(
                configuration.clarifier_model,
                self.context,
                database_card=self.database_card,
                enable_analysis_guidance=configuration.enable_analysis_guidance,
            ),
            executor=DsaAnalysisExecutor(configuration.dsa),
            enable_analysis_guidance=configuration.enable_analysis_guidance,
        )


def load_chat_service() -> ChatService:
    configuration = load_configuration()
    context = load_mock_context(configuration.mock_context_path, configuration.synthetic_context)
    database_card = (
        load_database_card(configuration.database_card)
        if configuration.database_card is not None
        else None
    )
    if context.data_source_id != configuration.data_source_id:
        raise ValueError("mock and configured data source IDs differ")
    return ChatService(configuration, context, database_card)


def welcome_message(
    context: MockDatabaseContext,
    database_card: DatabaseCard | None,
    clarifier_model_name: str,
) -> str:
    """Combine the fixed trust flow with bounded dataset-owned public context."""
    title = escape_markdown_inline(database_card.title) if database_card else "this database"
    sections = [
        "## Instructions\n\n"
        "This chat interface allows you to ask quantitative questions about a DuckDB in plain "
        f"language. **{clarifier_model_name}** will turn your question into a request for a "
        "subagent, which is run by a trusted model. A typical session works as follows:\n\n"
        "1. Ask your question in ordinary language.\n"
        f"2. **{clarifier_model_name}** will either ask a follow-up question or directly prepare "
        "the request for the subagent.\n"
        "3. Review the subagent request. If you are happy with it, run the request.\n"
        "4. The agent returns the result. If it succeeds, you receive a structured answer and a "
        "downloadable Jupyter notebook containing the derivation.\n"
        "5. To prevent data from leaking to the untrusted model, the session ends after the "
        "result is returned. Start a new session to ask another question.",
        f"## About {title}",
    ]
    if database_card is not None:
        sections.append(render_database_overview(database_card))
    else:
        sections.append("Ask quantitative questions about the configured database.")
    sections.append(f"Ask a question about **{title}** to begin.")
    return "\n\n".join(sections)


def model_display_name(configured_name: str) -> str:
    """Project a provider-qualified model ID into a compact user-facing label."""
    unqualified = configured_name.partition(":")[2] or configured_name
    return unqualified.rsplit("/", maxsplit=1)[-1]
