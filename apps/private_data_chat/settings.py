"""Host-owned configuration for the Chainlit private-data application."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from pydantic import Field, JsonValue, field_validator, model_validator

from apps.private_data_chat.contracts import AppContract
from apps.private_data_chat.dsa_adapter import DsaRuntimeConfiguration
from dsa import ModelConfiguration


class PrivateDataChatConfiguration(AppContract):
    """Separate untrusted clarification inputs from privileged DSA configuration."""

    data_source_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,127}$")
    mock_context_path: Path
    clarifier_model: ModelConfiguration
    dsa: DsaRuntimeConfiguration
    enable_analysis_guidance: bool = False

    @field_validator("mock_context_path")
    @classmethod
    def require_absolute_mock_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("mock context path must be absolute")
        return value

    @model_validator(mode="after")
    def require_one_data_source(self) -> PrivateDataChatConfiguration:
        if self.data_source_id != self.dsa.data_source_id:
            raise ValueError("chat and DSA data source IDs must match")
        return self


def load_configuration(
    environ: Mapping[str, str] | None = None,
) -> PrivateDataChatConfiguration:
    """Load strict configuration without accepting credentials in model settings."""
    values = os.environ if environ is None else environ
    data_source_id = _required(values, "DSA_CHAT_DATA_SOURCE_ID")
    return PrivateDataChatConfiguration(
        data_source_id=data_source_id,
        mock_context_path=Path(_required(values, "DSA_CHAT_MOCK_CONTEXT_PATH")),
        enable_analysis_guidance=_boolean(
            values.get("DSA_CHAT_ENABLE_ANALYSIS_GUIDANCE", "false"),
            "DSA_CHAT_ENABLE_ANALYSIS_GUIDANCE",
        ),
        clarifier_model=ModelConfiguration(
            name=_required(values, "DSA_CHAT_CLARIFIER_MODEL_NAME"),
            settings=_settings(values, "DSA_CHAT_CLARIFIER_MODEL_SETTINGS_JSON"),
        ),
        dsa=DsaRuntimeConfiguration(
            data_source_id=data_source_id,
            database_path=Path(_required(values, "DSA_CHAT_DATABASE_PATH")),
            runs_directory=Path(_required(values, "DSA_CHAT_RUNS_DIRECTORY")),
            trusted_model_name=_required(values, "DSA_CHAT_TRUSTED_MODEL_NAME"),
            trusted_model_settings=_settings(
                values,
                "DSA_CHAT_TRUSTED_MODEL_SETTINGS_JSON",
            ),
            docker_image=_required(values, "DSA_CHAT_DOCKER_IMAGE"),
            report_to_mlflow=_boolean(
                values.get("DSA_CHAT_REPORT_TO_MLFLOW", "false"),
                "DSA_CHAT_REPORT_TO_MLFLOW",
            ),
        ),
    )


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _settings(environ: Mapping[str, str], name: str) -> dict[str, JsonValue]:
    value = json.loads(environ.get(name, "{}"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return cast(dict[str, JsonValue], value)


def _boolean(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be true or false")
