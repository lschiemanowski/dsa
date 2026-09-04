from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from apps.private_data_chat.settings import load_configuration

IMAGE = f"dsa-python@sha256:{'a' * 64}"


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "DSA_CHAT_DATA_SOURCE_ID": "retail",
        "DSA_CHAT_MOCK_CONTEXT_PATH": str(tmp_path / "mock.json"),
        "DSA_CHAT_CLARIFIER_MODEL_NAME": "openai:untrusted",
        "DSA_CHAT_CLARIFIER_MODEL_SETTINGS_JSON": '{"temperature":0.2}',
        "DSA_CHAT_DATABASE_PATH": str(tmp_path / "private.duckdb"),
        "DSA_CHAT_RUNS_DIRECTORY": str(tmp_path / "runs"),
        "DSA_CHAT_TRUSTED_MODEL_NAME": "openai:trusted",
        "DSA_CHAT_TRUSTED_MODEL_SETTINGS_JSON": '{"temperature":0}',
        "DSA_CHAT_DOCKER_IMAGE": IMAGE,
        "DSA_CHAT_REPORT_TO_MLFLOW": "true",
    }


def test_configuration_keeps_clarifier_and_trusted_runtime_separate(tmp_path: Path) -> None:
    configuration = load_configuration(environment(tmp_path))

    assert configuration.enable_analysis_guidance is False
    assert configuration.clarifier_model.name == "openai:untrusted"
    assert configuration.clarifier_model.settings == {"temperature": 0.2}
    assert configuration.dsa.trusted_model_name == "openai:trusted"
    assert configuration.dsa.database_path == tmp_path / "private.duckdb"
    assert configuration.dsa.report_to_mlflow is True


def test_analysis_guidance_is_an_opt_in_host_setting(tmp_path: Path) -> None:
    values = environment(tmp_path)
    values["DSA_CHAT_ENABLE_ANALYSIS_GUIDANCE"] = "true"

    assert load_configuration(values).enable_analysis_guidance is True


def test_toml_configuration_resolves_paths_relative_to_itself(tmp_path: Path) -> None:
    path = tmp_path / "chat.toml"
    path.write_text(
        "\n".join(
            (
                'format = "dsa-private-data-chat-config/v1"',
                'data_source_id = "retail"',
                'mock_context_path = "mock.json"',
                'database_path = "data/private.duckdb"',
                'runs_directory = "state/runs"',
                f'docker_image = "{IMAGE}"',
                "enable_analysis_guidance = true",
                "report_to_mlflow = false",
                "",
                "[clarifier]",
                'model = "openai:untrusted"',
                "settings = { temperature = 0.2 }",
                "",
                "[trusted]",
                'model = "openai:trusted"',
                "settings = { temperature = 0 }",
                "",
            )
        )
    )

    configuration = load_configuration({}, config_path=path)

    assert configuration.data_source_id == "retail"
    assert configuration.mock_context_path == (tmp_path / "mock.json").resolve()
    assert configuration.clarifier_model.name == "openai:untrusted"
    assert configuration.enable_analysis_guidance is True
    assert configuration.dsa.database_path == (tmp_path / "data/private.duckdb").resolve()
    assert configuration.dsa.runs_directory == (tmp_path / "state/runs").resolve()
    assert configuration.dsa.trusted_model_name == "openai:trusted"


def test_environment_can_select_the_toml_configuration(tmp_path: Path) -> None:
    path = tmp_path / "chat.toml"
    path.write_text(
        "\n".join(
            (
                'format = "dsa-private-data-chat-config/v1"',
                'data_source_id = "retail"',
                'mock_context_path = "mock.json"',
                'database_path = "private.duckdb"',
                'runs_directory = "runs"',
                f'docker_image = "{IMAGE}"',
                "",
                "[clarifier]",
                'model = "openai:untrusted"',
                "",
                "[trusted]",
                'model = "openai:trusted"',
                "",
            )
        )
    )

    configuration = load_configuration({"DSA_CHAT_CONFIG_PATH": str(path)})

    assert configuration.data_source_id == "retail"


def test_toml_configuration_accepts_a_pinned_huggingface_description(tmp_path: Path) -> None:
    path = tmp_path / "chat.toml"
    path.write_text(
        "\n".join(
            (
                'format = "dsa-private-data-chat-config/v1"',
                'data_source_id = "retail"',
                'mock_context_path = "mock.json"',
                'database_path = "private.duckdb"',
                'runs_directory = "runs"',
                f'docker_image = "{IMAGE}"',
                "",
                "[description]",
                'format = "dsa-huggingface-database-description/v1"',
                'repo_id = "lschiemanowski/dsa-datasets"',
                f'revision = "{"a" * 40}"',
                'path = "online-retail-ii/1.0.0/DATABASE.md"',
                f'sha256 = "{"b" * 64}"',
                "",
                "[clarifier]",
                'model = "openai:untrusted"',
                "",
                "[trusted]",
                'model = "openai:trusted"',
                "",
            )
        )
    )

    configuration = load_configuration({}, config_path=path)

    assert configuration.mock_context_path == (tmp_path / "mock.json").resolve()
    assert configuration.database_description is not None
    assert configuration.database_description.revision == "a" * 40


@pytest.mark.parametrize(
    "extra",
    [
        'settings = { api_key = "SECRET" }',
        'settings = { base_url = "https://private" }',
    ],
)
def test_toml_configuration_rejects_credentials_and_endpoints(
    tmp_path: Path,
    extra: str,
) -> None:
    path = tmp_path / "chat.toml"
    path.write_text(
        "\n".join(
            (
                'format = "dsa-private-data-chat-config/v1"',
                'data_source_id = "retail"',
                'mock_context_path = "mock.json"',
                'database_path = "private.duckdb"',
                'runs_directory = "runs"',
                f'docker_image = "{IMAGE}"',
                "",
                "[clarifier]",
                'model = "openai:untrusted"',
                extra,
                "",
                "[trusted]",
                'model = "openai:trusted"',
                "",
            )
        )
    )

    with pytest.raises(ValidationError):
        load_configuration({}, config_path=path)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DSA_CHAT_CLARIFIER_MODEL_SETTINGS_JSON", '{"api_key":"SECRET"}'),
        ("DSA_CHAT_TRUSTED_MODEL_SETTINGS_JSON", '{"base_url":"https://private"}'),
    ],
)
def test_configuration_rejects_credentials_and_endpoints(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    values = environment(tmp_path)
    values[name] = value
    with pytest.raises(ValidationError, match="credentials or endpoints"):
        load_configuration(values)


def test_configuration_rejects_missing_invalid_or_relative_values(tmp_path: Path) -> None:
    missing = environment(tmp_path)
    del missing["DSA_CHAT_DATABASE_PATH"]
    with pytest.raises(ValueError, match="DSA_CHAT_DATABASE_PATH is required"):
        load_configuration(missing)

    invalid_json = environment(tmp_path)
    invalid_json["DSA_CHAT_CLARIFIER_MODEL_SETTINGS_JSON"] = "[]"
    with pytest.raises(ValueError, match="JSON object"):
        load_configuration(invalid_json)

    relative = environment(tmp_path)
    relative["DSA_CHAT_MOCK_CONTEXT_PATH"] = "mock.json"
    with pytest.raises(ValidationError, match="absolute"):
        load_configuration(relative)

    invalid_bool = environment(tmp_path)
    invalid_bool["DSA_CHAT_REPORT_TO_MLFLOW"] = "sometimes"
    with pytest.raises(ValueError, match="true or false"):
        load_configuration(invalid_bool)

    invalid_guidance = environment(tmp_path)
    invalid_guidance["DSA_CHAT_ENABLE_ANALYSIS_GUIDANCE"] = "sometimes"
    with pytest.raises(ValueError, match="DSA_CHAT_ENABLE_ANALYSIS_GUIDANCE"):
        load_configuration(invalid_guidance)
