"""Tests for the FastAPI launcher boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

import apps.private_data_chat.cli as cli_module
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
                "report_to_logfire = false",
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_configuration(tmp_path)
    (tmp_path / "index.html").write_text("<!doctype html>")
    monkeypatch.setattr(cli_module, "CHAT_WEB_ROOT", tmp_path)
    calls: list[tuple[list[str], dict[str, str], str | None]] = []

    def launch(arguments: Sequence[str], environ: Mapping[str, str], url: str | None) -> int:
        calls.append((list(arguments), dict(environ), url))
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
            "--debug",
        ],
        launcher=launch,
        dependency_available=lambda: True,
        environ={"OPENAI_API_KEY": "SECRET"},
    )

    assert status == 23
    assert len(calls) == 1
    arguments, environment, url = calls[0]
    assert arguments == [
        "apps.private_data_chat.http_api:create_chat_app",
        "--factory",
        "--host",
        "0.0.0.0",
        "--port",
        "8123",
        "--reload",
        "--log-level",
        "debug",
    ]
    assert environment["DSA_CHAT_CONFIG_PATH"] == str(config_path.resolve())
    assert environment["OPENAI_API_KEY"] == "SECRET"
    assert url is None


def test_check_validates_configuration_without_starting_server(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_configuration(tmp_path)
    monkeypatch.setattr(cli_module, "CHAT_WEB_ROOT", tmp_path / "missing")

    def unexpected(arguments: Sequence[str], environ: Mapping[str, str], url: str | None) -> int:
        raise AssertionError("preflight must not start the server")

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

    assert main(
        ["--config", str(config_path)], launcher=unexpected,
        dependency_available=lambda: True, environ={},
    ) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "chat_ui_unavailable"


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
    assert json.loads(capsys.readouterr().out) == {
        "code": "chat_preflight_failed",
        "stage": "context",
        "status": "rejected",
    }


def test_invalid_configuration_reports_only_safe_field_names(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_configuration(tmp_path)
    content = config_path.read_text().replace(IMAGE, "SECRET-invalid-image")
    config_path.write_text(content)

    assert (
        main(
            ["--config", str(config_path)],
            dependency_available=lambda: True,
            environ={},
        )
        == 2
    )
    output = capsys.readouterr().out
    assert "SECRET" not in output
    assert json.loads(output) == {
        "code": "chat_preflight_failed",
        "fields": ["docker_image"],
        "stage": "configuration",
        "status": "rejected",
    }


def test_invalid_arguments_are_distinguished_from_host_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--port", "70000"], environ={}) == 2
    assert json.loads(capsys.readouterr().out) == {
        "code": "chat_preflight_failed",
        "stage": "arguments",
        "status": "rejected",
    }


def test_database_card_download_failure_is_safely_classified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_configuration(tmp_path)
    content = config_path.read_text().replace(
        "[clarifier]",
        "\n".join(
            (
                "[database_card]",
                'format = "dsa-huggingface-database-card/v1"',
                'repo_id = "lschiemanowski/dsa-datasets"',
                f'revision = "{"a" * 40}"',
                'path = "online-retail-ii/1.0.0/database-card.json"',
                f'sha256 = "{"b" * 64}"',
                "",
                "[clarifier]",
            )
        ),
    )
    config_path.write_text(content)

    def fail(reference: object) -> str:
        del reference
        raise RuntimeError("SECRET provider detail")

    monkeypatch.setattr(cli_module, "load_database_card", fail)

    assert (
        main(
            ["--config", str(config_path)],
            dependency_available=lambda: True,
            environ={},
        )
        == 2
    )
    output = capsys.readouterr().out
    assert "SECRET" not in output
    assert json.loads(output) == {
        "code": "chat_preflight_failed",
        "stage": "database_card",
        "status": "rejected",
    }


@pytest.mark.parametrize("legacy_value", ["false", "true"])
def test_check_handles_legacy_mlflow_setting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], legacy_value: str
) -> None:
    config_path = write_configuration(tmp_path)
    config_path.write_text(
        config_path.read_text().replace(
            "report_to_logfire = false", f"report_to_mlflow = {legacy_value}"
        )
    )
    result = main(
        ["--config", str(config_path), "--check"],
        dependency_available=lambda: True,
        environ={},
    )
    response = json.loads(capsys.readouterr().out)
    if legacy_value == "false":
        assert result == 0
        assert response["status"] == "ready"
    else:
        assert result == 2
        assert response["code"] == "chat_reporting_setting_renamed"
        assert "report_to_logfire" in response["message"]
