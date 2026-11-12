"""Host-owned configuration for the private-data chat application."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, JsonValue, field_validator, model_validator

from apps.private_data_chat.contracts import AppContract
from apps.private_data_chat.database_card import HuggingFaceDatabaseCardReference
from apps.private_data_chat.dsa_adapter import DsaRuntimeConfiguration
from apps.private_data_chat.synthetic_context import HuggingFaceSyntheticContextReference
from dsa import ModelConfiguration
from dsa.cli import read_contract

CHAT_WEB_ROOT = Path(__file__).with_name("web")


class PrivateDataChatConfiguration(AppContract):
    """Separate untrusted clarification inputs from privileged DSA configuration."""

    data_source_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,127}$")
    mock_context_path: Path | None = None
    synthetic_context: HuggingFaceSyntheticContextReference | None = None
    database_card: HuggingFaceDatabaseCardReference | None = None
    clarifier_model: ModelConfiguration
    dsa: DsaRuntimeConfiguration
    enable_analysis_guidance: bool = False

    @field_validator("mock_context_path")
    @classmethod
    def require_absolute_mock_path(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("mock context path must be absolute")
        return value

    @model_validator(mode="after")
    def require_one_data_source(self) -> PrivateDataChatConfiguration:
        if (self.mock_context_path is None) == (self.synthetic_context is None):
            raise ValueError("configure exactly one local or Hugging Face synthetic context")
        if self.synthetic_context is not None and (
            self.database_card is None
            or (
                self.synthetic_context.repo_id,
                self.synthetic_context.revision,
            )
            != (self.database_card.repo_id, self.database_card.revision)
        ):
            raise ValueError("remote context and card must pin the same repository revision")
        if self.data_source_id != self.dsa.data_source_id:
            raise ValueError("chat and DSA data source IDs must match")
        return self


class _ConfiguredModel(AppContract):
    model: str = Field(min_length=1, max_length=512)
    settings: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def reject_blank_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value

    @model_validator(mode="after")
    def reject_privileged_settings(self) -> _ConfiguredModel:
        ModelConfiguration(name=self.model, settings=self.settings)
        return self


class LegacyChatReportingError(ValueError):
    """An obsolete setting requested an export to the old backend."""


class _FileConfiguration(AppContract):
    format: Literal["dsa-private-data-chat-config/v1"]
    data_source_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,127}$")
    mock_context_path: str | None = Field(default=None, min_length=1, max_length=4096)
    synthetic_context: HuggingFaceSyntheticContextReference | None = None
    database_card: HuggingFaceDatabaseCardReference | None = None
    database_path: str = Field(min_length=1, max_length=4096)
    runs_directory: str = Field(min_length=1, max_length=4096)
    docker_image: str
    enable_analysis_guidance: bool = False
    report_to_logfire: bool = False
    clarifier: _ConfiguredModel
    trusted: _ConfiguredModel

    @field_validator("mock_context_path", "database_path", "runs_directory")
    @classmethod
    def reject_blank_path(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("configuration paths must not be blank")
        return value


def load_configuration(
    environ: Mapping[str, str] | None = None,
    *,
    config_path: Path | None = None,
) -> PrivateDataChatConfiguration:
    """Load strict configuration without accepting credentials in model settings."""
    values = os.environ if environ is None else environ
    selected_path = config_path
    if selected_path is None and values.get("DSA_CHAT_CONFIG_PATH"):
        selected_path = Path(values["DSA_CHAT_CONFIG_PATH"])
    if selected_path is not None:
        return _load_file_configuration(selected_path)
    return _load_environment_configuration(values)


def _load_environment_configuration(
    values: Mapping[str, str],
) -> PrivateDataChatConfiguration:
    legacy_reporting = values.get("DSA_CHAT_REPORT_TO_MLFLOW")
    if legacy_reporting is not None and _boolean(legacy_reporting, "DSA_CHAT_REPORT_TO_MLFLOW"):
        raise LegacyChatReportingError(
            "Remove DSA_CHAT_REPORT_TO_MLFLOW and set DSA_CHAT_REPORT_TO_LOGFIRE=true "
            "if you want Logfire telemetry."
        )
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
            report_to_logfire=_boolean(
                values.get("DSA_CHAT_REPORT_TO_LOGFIRE", "false"),
                "DSA_CHAT_REPORT_TO_LOGFIRE",
            ),
        ),
    )


def _load_file_configuration(path: Path) -> PrivateDataChatConfiguration:
    absolute_path = path.absolute()
    content = read_contract(absolute_path)
    raw = tomllib.loads(content.decode("utf-8"))
    if "report_to_mlflow" in raw and raw.pop("report_to_mlflow") is not False:
        raise LegacyChatReportingError(
            "Remove report_to_mlflow and set report_to_logfire = true "
            "if you want Logfire telemetry."
        )
    configured = _FileConfiguration.model_validate(raw)
    base = absolute_path.parent
    return PrivateDataChatConfiguration(
        data_source_id=configured.data_source_id,
        mock_context_path=(
            _configured_path(base, configured.mock_context_path)
            if configured.mock_context_path is not None
            else None
        ),
        synthetic_context=configured.synthetic_context,
        database_card=configured.database_card,
        enable_analysis_guidance=configured.enable_analysis_guidance,
        clarifier_model=ModelConfiguration(
            name=configured.clarifier.model,
            settings=configured.clarifier.settings,
        ),
        dsa=DsaRuntimeConfiguration(
            data_source_id=configured.data_source_id,
            database_path=_configured_path(base, configured.database_path),
            runs_directory=_configured_path(base, configured.runs_directory),
            trusted_model_name=configured.trusted.model,
            trusted_model_settings=configured.trusted.settings,
            docker_image=configured.docker_image,
            report_to_logfire=configured.report_to_logfire,
        ),
    )


def _configured_path(base: Path, value: str) -> Path:
    selected = Path(value)
    return selected.absolute() if selected.is_absolute() else (base / selected).absolute()


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
