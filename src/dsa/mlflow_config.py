"""Validated operator-owned configuration for supported MLflow backends."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

_DATASET_COMPONENT = r"[a-z0-9_][a-z0-9_-]{0,127}"
SAFE_DATASET_NAME_PATTERN = rf"^{_DATASET_COMPONENT}(?:\.{_DATASET_COMPONENT}){{0,2}}$"
DATABRICKS_DATASET_NAME_PATTERN = (
    rf"^{_DATASET_COMPONENT}\.{_DATASET_COMPONENT}\.{_DATASET_COMPONENT}$"
)
_SAFE_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class MlflowConfiguration:
    """One validated MLflow destination without retained credentials."""

    tracking_uri: str
    experiment_id: str
    backend: Literal["databricks", "tracking_server"]


class MlflowConfigurationError(ValueError):
    """Stable configuration failure safe to project outside the host boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def load_mlflow_configuration(
    environment: Mapping[str, str],
) -> MlflowConfiguration:
    """Validate Databricks or HTTP(S) tracking-server configuration."""
    tracking_uri = environment.get("MLFLOW_TRACKING_URI", "")
    experiment_id = environment.get("MLFLOW_EXPERIMENT_ID", "")
    if not _tracking_uri_is_supported(tracking_uri):
        raise MlflowConfigurationError("mlflow_tracking_uri_invalid")
    if (
        not experiment_id
        or experiment_id != experiment_id.strip()
        or _SAFE_EXPERIMENT_ID.fullmatch(experiment_id) is None
    ):
        raise MlflowConfigurationError("mlflow_configuration_missing")
    if tracking_uri == "databricks":
        if any(
            not environment.get(key, "").strip()
            for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN")
        ):
            raise MlflowConfigurationError("mlflow_configuration_missing")
        backend: Literal["databricks", "tracking_server"] = "databricks"
    else:
        backend = "tracking_server"
    return MlflowConfiguration(
        tracking_uri=tracking_uri,
        experiment_id=experiment_id,
        backend=backend,
    )


def mlflow_configuration_failure(environment: Mapping[str, str]) -> str | None:
    """Return only a stable failure code for an invalid configuration."""
    try:
        load_mlflow_configuration(environment)
    except MlflowConfigurationError as error:
        return error.code
    return None


def dataset_name_matches_backend(
    dataset_name: str,
    configuration: MlflowConfiguration,
) -> bool:
    """Require Unity Catalog identity only when Databricks is selected."""
    pattern = (
        DATABRICKS_DATASET_NAME_PATTERN
        if configuration.backend == "databricks"
        else SAFE_DATASET_NAME_PATTERN
    )
    return re.fullmatch(pattern, dataset_name) is not None


def _tracking_uri_is_supported(value: str) -> bool:
    if value == "databricks":
        return True
    if (
        not value
        or value != value.strip()
        or len(value) > 2048
        or any(ord(character) <= 0x20 or ord(character) == 0x7F for character in value)
    ):
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        return False
    return parsed.scheme == "https" or parsed.hostname.lower() in _LOOPBACK_HOSTS
