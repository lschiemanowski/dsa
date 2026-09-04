"""Optional Chainlit launcher for the Private Data Chat application."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from apps.private_data_chat.clarifier import load_mock_context
from apps.private_data_chat.public_description import load_database_description
from apps.private_data_chat.settings import load_configuration
from dsa.cli import HelpRequested, Parser, emit

Launcher = Callable[[Sequence[str], Mapping[str, str]], int]
DependencyCheck = Callable[[], bool]
_SAFE_CONFIGURATION_FIELDS = frozenset(
    {
        "clarifier",
        "data_source_id",
        "database_path",
        "description",
        "docker_image",
        "enable_analysis_guidance",
        "format",
        "mock_context_path",
        "model",
        "path",
        "repo_id",
        "report_to_mlflow",
        "revision",
        "runs_directory",
        "settings",
        "sha256",
        "trusted",
    }
)


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "dsa-chat",
    launcher: Launcher | None = None,
    dependency_available: DependencyCheck | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Validate host configuration and start the packaged Chainlit application."""
    try:
        parsed = _parser(prog).parse_args(argv)
        values = dict(os.environ if environ is None else environ)
        config_path = Path(parsed.config).absolute() if parsed.config is not None else None
    except HelpRequested:
        return 0
    except Exception as error:
        _emit_preflight_failure("arguments", error)
        return 2

    check_dependency = dependency_available or _chainlit_available
    if not check_dependency():
        emit({"code": "chat_dependency_unavailable", "status": "rejected"})
        return 2

    try:
        configuration = load_configuration(values, config_path=config_path)
    except Exception as error:
        _emit_preflight_failure("configuration", error)
        return 2
    try:
        context = load_mock_context(configuration.mock_context_path)
        if context.data_source_id != configuration.data_source_id:
            raise ValueError("mock and configured data source IDs differ")
    except Exception as error:
        _emit_preflight_failure("context", error)
        return 2
    if configuration.database_description is not None:
        try:
            load_database_description(configuration.database_description)
        except Exception as error:
            _emit_preflight_failure("description", error)
            return 2

    if parsed.check:
        result = {
            "data_source_id": configuration.data_source_id,
            "status": "ready",
        }
        if config_path is not None:
            result["config_path"] = str(config_path)
        emit(result)
        return 0

    if config_path is not None:
        values["DSA_CHAT_CONFIG_PATH"] = str(config_path)
    arguments = [
        "run",
        str(Path(__file__).with_name("chainlit_app.py")),
        "--host",
        parsed.host,
        "--port",
        str(parsed.port),
    ]
    if parsed.headless:
        arguments.append("--headless")
    if parsed.watch:
        arguments.append("--watch")
    if parsed.debug:
        arguments.append("--debug")
    start = launcher or _launch_chainlit
    try:
        return start(arguments, values)
    except KeyboardInterrupt:
        return 130
    except Exception:
        emit({"code": "chat_launch_failed", "status": "failed"})
        return 1


def _parser(prog: str) -> Parser:
    parser = Parser(prog=prog, add_help=True)
    parser.add_argument("--config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8001)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65_535:
        raise ValueError("port is outside the valid range")
    return port


def _chainlit_available() -> bool:
    return importlib.util.find_spec("chainlit") is not None


def _launch_chainlit(arguments: Sequence[str], environ: Mapping[str, str]) -> int:
    completed = subprocess.run(
        (sys.executable, "-m", "chainlit", *arguments),
        env=dict(environ),
        check=False,
    )
    return completed.returncode


def _emit_preflight_failure(stage: str, error: Exception) -> None:
    result: dict[str, object] = {
        "code": "chat_preflight_failed",
        "stage": stage,
        "status": "rejected",
    }
    if stage == "configuration" and isinstance(error, ValidationError):
        fields = sorted(
            {
                field
                for issue in error.errors(
                    include_context=False,
                    include_input=False,
                    include_url=False,
                )
                if (field := _safe_field_location(issue.get("loc"))) is not None
            }
        )
        if fields:
            result["fields"] = fields[:8]
    emit(result)


def _safe_field_location(value: object) -> str | None:
    if not isinstance(value, tuple) or not value:
        return None
    components: list[str] = []
    for item in cast(tuple[object, ...], value):
        if not isinstance(item, str) or item not in _SAFE_CONFIGURATION_FIELDS:
            return None
        components.append(item)
    return ".".join(components)


if __name__ == "__main__":
    raise SystemExit(main())
