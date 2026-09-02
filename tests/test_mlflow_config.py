"""Contracts for operator-selected MLflow destinations."""

from __future__ import annotations

import pytest

from dsa.mlflow_config import (
    MlflowConfigurationError,
    dataset_name_matches_backend,
    load_mlflow_configuration,
    load_mlflow_destination,
)


def test_databricks_requires_credentials_and_uses_unity_catalog_names() -> None:
    configuration = load_mlflow_configuration(
        {
            "MLFLOW_TRACKING_URI": "databricks",
            "MLFLOW_EXPERIMENT_ID": "123",
            "DATABRICKS_HOST": "https://workspace.example",
            "DATABRICKS_TOKEN": "token",
        }
    )

    assert configuration.backend == "databricks"
    assert dataset_name_matches_backend("catalog.schema.dataset", configuration)
    assert not dataset_name_matches_backend("dataset", configuration)


@pytest.mark.parametrize(
    "tracking_uri",
    [
        "http://127.0.0.1:5000",
        "http://localhost:5000/",
        "http://[::1]:5000",
        "https://mlflow.example/tracking",
    ],
)
def test_tracking_servers_do_not_require_databricks_credentials(
    tracking_uri: str,
) -> None:
    configuration = load_mlflow_configuration(
        {
            "MLFLOW_TRACKING_URI": tracking_uri,
            "MLFLOW_EXPERIMENT_ID": "1",
        }
    )

    assert configuration.backend == "tracking_server"
    assert dataset_name_matches_backend("local_dataset", configuration)
    assert dataset_name_matches_backend("catalog.schema.dataset", configuration)


@pytest.mark.parametrize(
    "tracking_uri",
    [
        "",
        "file:/tmp/mlruns",
        "sqlite:////tmp/mlflow.db",
        "http://tracking.example",
        "http://user:password@localhost:5000",
        "http://localhost:5000?token=secret",
        "http://localhost:5000#fragment",
        "http://local\nhost:5000",
        " http://localhost:5000",
    ],
)
def test_unsafe_or_unsupported_tracking_uris_are_rejected(tracking_uri: str) -> None:
    with pytest.raises(MlflowConfigurationError) as caught:
        load_mlflow_configuration(
            {
                "MLFLOW_TRACKING_URI": tracking_uri,
                "MLFLOW_EXPERIMENT_ID": "1",
            }
        )

    assert caught.value.code == "mlflow_tracking_uri_invalid"


def test_databricks_missing_credentials_is_classified_without_retaining_values() -> None:
    with pytest.raises(MlflowConfigurationError) as caught:
        load_mlflow_configuration(
            {
                "MLFLOW_TRACKING_URI": "databricks",
                "MLFLOW_EXPERIMENT_ID": "1",
                "DATABRICKS_HOST": "https://workspace.example/private",
            }
        )

    assert caught.value.code == "mlflow_configuration_missing"
    assert "workspace" not in str(caught.value)


@pytest.mark.parametrize(
    "environment, expected_backend",
    [
        (
            {"MLFLOW_TRACKING_URI": "http://127.0.0.1:5000"},
            "tracking_server",
        ),
        (
            {
                "MLFLOW_TRACKING_URI": "databricks",
                "DATABRICKS_HOST": "https://workspace.example",
                "DATABRICKS_TOKEN": "token",
            },
            "databricks",
        ),
    ],
)
def test_read_destination_does_not_require_an_experiment(
    environment: dict[str, str],
    expected_backend: str,
) -> None:
    destination = load_mlflow_destination(environment)

    assert destination.backend == expected_backend
