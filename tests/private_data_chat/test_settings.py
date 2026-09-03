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

    assert configuration.clarifier_model.name == "openai:untrusted"
    assert configuration.clarifier_model.settings == {"temperature": 0.2}
    assert configuration.dsa.trusted_model_name == "openai:trusted"
    assert configuration.dsa.database_path == tmp_path / "private.duckdb"
    assert configuration.dsa.report_to_mlflow is True


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
