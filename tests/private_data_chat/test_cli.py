"""Tests for the optional Chainlit launcher boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from apps.private_data_chat.cli import main

IMAGE = f"dsa-python@sha256:{'a' * 64}"


def write_configuration(tmp_path: Path) -> Path:
    context = {
        "format": "dsa-mock-database/v1",
        "data_source_id": "retail",
        "display_name": "Synthetic retail",
        "relations": [
            {
                "name": "analysis.rows",
                "columns": ["value"],
                "sample_rows": [{"value": 1}],
            }
        ],
        "synthetic": True,
    }
    (tmp_path / "mock.json").write_text(json.dumps(context))
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
    return path


def test_launches_packaged_application_with_an_explicit_config(
    tmp_path: Path,
) -> None:
    config_path = write_configuration(tmp_path)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def launch(arguments: Sequence[str], environ: Mapping[str, str]) -> int:
        calls.append((list(arguments), dict(environ)))
        return 23

    status = main(
        [
            "--config",
            str(config_path),
            "--host",
            "0.0.0.0",
            "--port",
            "8123",
            "--headless",
            "--watch",
        ],
        launcher=launch,
        dependency_available=lambda: True,
        environ={"OPENAI_API_KEY": "SECRET"},
    )

    assert status == 23
    assert len(calls) == 1
    arguments, environment = calls[0]
    assert arguments[0] == "run"
    assert arguments[1].endswith("/apps/private_data_chat/chainlit_app.py")
    assert arguments[2:] == [
        "--host",
        "0.0.0.0",
        "--port",
        "8123",
        "--headless",
        "--watch",
    ]
    assert environment["DSA_CHAT_CONFIG_PATH"] == str(config_path.resolve())
    assert environment["OPENAI_API_KEY"] == "SECRET"


def test_check_validates_configuration_without_starting_chainlit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_configuration(tmp_path)

    def unexpected(arguments: Sequence[str], environ: Mapping[str, str]) -> int:
        del arguments, environ
        raise AssertionError("preflight must not launch Chainlit")

    assert (
        main(
            ["--config", str(config_path), "--check"],
            launcher=unexpected,
            dependency_available=lambda: True,
            environ={},
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "config_path": str(config_path.resolve()),
        "data_source_id": "retail",
        "status": "ready",
    }


def test_missing_optional_dependency_is_a_stable_preflight_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_configuration(tmp_path)

    assert (
        main(
            ["--config", str(config_path)],
            dependency_available=lambda: False,
            environ={},
        )
        == 2
    )
    assert capsys.readouterr().out == (
        '{"code":"chat_dependency_unavailable","status":"rejected"}\n'
    )


def test_missing_optional_dependency_is_reported_before_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([], dependency_available=lambda: False, environ={}) == 2
    assert capsys.readouterr().out == (
        '{"code":"chat_dependency_unavailable","status":"rejected"}\n'
    )


def test_mismatched_mock_context_is_rejected_before_launch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_configuration(tmp_path)
    context_path = tmp_path / "mock.json"
    context = json.loads(context_path.read_text())
    context["data_source_id"] = "other"
    context_path.write_text(json.dumps(context))

    assert (
        main(
            ["--config", str(config_path)],
            dependency_available=lambda: True,
            environ={},
        )
        == 2
    )
    assert capsys.readouterr().out == (
        '{"code":"chat_preflight_failed","status":"rejected"}\n'
    )
