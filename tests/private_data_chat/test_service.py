from apps.private_data_chat.service import model_display_name, welcome_message
from tests.private_data_chat.test_database_card import MODEL_ONLY_NOTE, card


def test_welcome_combines_fixed_trust_flow_with_dataset_context() -> None:
    message = welcome_message(card(), model_display_name("openrouter:example/remote-model"))

    assert message.startswith("## Instructions")
    assert "**remote-model**" in message
    assert "openrouter" not in message
    assert "request for a subagent" in message
    assert "run by a trusted model" in message
    assert "ask a follow-up question" in message
    assert "Review the subagent request" in message
    assert "downloadable Jupyter notebook" in message
    assert "session ends" in message
    assert "## About Retail data" in message
    assert "Ask a question about **Retail data** to begin." in message
    assert "synthetic" not in message
    assert r"The original first paragraph\." in message
    assert "### Available data" in message
    assert "1,044,848" in message
    assert r"- How did monthly sales change\?" in message
    assert MODEL_ONLY_NOTE not in message


def test_welcome_without_card_uses_generic_database_description() -> None:
    message = welcome_message(None, "model")
    assert "## About this database" in message
    assert "Ask quantitative questions about the configured database." in message


def test_welcome_escapes_database_title() -> None:
    title = "![pixel](https://example.org/image)"
    message = welcome_message(card().model_copy(update={"title": title}), "model")
    assert title not in message


def test_model_display_name_omits_provider_and_routing_qualifiers() -> None:
    assert model_display_name("openrouter:deepseek/deepseek-v4-flash-0731") == (
        "deepseek-v4-flash-0731"
    )
    assert model_display_name("openai-chat:local-model") == "local-model"
