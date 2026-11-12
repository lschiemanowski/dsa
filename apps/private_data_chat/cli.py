"""FastAPI launcher for the Private Data Chat application."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from typing import cast
from urllib.parse import urlsplit

from pydantic import ValidationError

from apps.private_data_chat.clarifier import load_mock_context
from apps.private_data_chat.database_card import load_database_card
from apps.private_data_chat.settings import (
    CHAT_WEB_ROOT,
    LegacyChatReportingError,
    load_configuration,
)
from apps.private_data_chat.synthetic_context import validate_context_card
from dsa.cli import HelpRequested, Parser, emit

Launcher = Callable[[Sequence[str], Mapping[str, str], str | None], int]
DependencyCheck = Callable[[], bool]
_BIND_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0", "::")
_SAFE_CONFIGURATION_FIELDS = frozenset(
    {
        "clarifier",
        "data_source_id",
        "database_path",
        "database_card",
        "docker_image",
        "enable_analysis_guidance",
        "format",
        "mock_context_path",
        "model",
        "path",
        "repo_id",
        "report_to_logfire",
        "revision",
        "runs_directory",
        "settings",
        "synthetic_context",
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
    """Validate host configuration and start the packaged chat application."""
    try:
        parsed = _parser(prog).parse_args(argv)
        values = dict(os.environ if environ is None else environ)
        config_path = Path(parsed.config).absolute() if parsed.config is not None else None
    except HelpRequested:
        return 0
    except Exception as error:
        _emit_preflight_failure("arguments", error)
        return 2

    if parsed.host not in _BIND_HOSTS:
        emit(
            {
                "code": "chat_host_unsupported",
                "status": "rejected",
                "message": "Use --host with localhost, 127.0.0.1, ::1, 0.0.0.0, or ::. "
                "Open the chat through a loopback address.",
            }
        )
        return 2

    check_dependency = dependency_available or _chat_dependencies_available
    if not check_dependency():
        emit({"code": "chat_dependency_unavailable", "status": "rejected"})
        return 2

    try:
        configuration = load_configuration(values, config_path=config_path)
    except Exception as error:
        _emit_preflight_failure("configuration", error)
        return 2
    try:
        context = load_mock_context(
            configuration.mock_context_path, configuration.synthetic_context
        )
        if context.data_source_id != configuration.data_source_id:
            raise ValueError("mock and configured data source IDs differ")
    except Exception as error:
        _emit_preflight_failure("context", error)
        return 2
    if configuration.database_card is not None:
        try:
            card = load_database_card(configuration.database_card)
            validate_context_card(context, card)
        except Exception as error:
            _emit_preflight_failure("database_card", error)
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
    if not (CHAT_WEB_ROOT / "index.html").is_file():
        emit(
            {
                "code": "chat_ui_unavailable",
                "status": "rejected",
                "message": "Build the UI with npm ci and npm run build in ui/, "
                "or install the built DSA wheel.",
            }
        )
        return 2

    arguments = [
        "apps.private_data_chat.http_api:create_chat_app",
        "--factory",
        "--host",
        parsed.host,
        "--port",
        str(parsed.port),
    ]
    if parsed.watch:
        arguments.extend(("--reload", "--reload-dir", str(Path(__file__).resolve().parent)))
    if parsed.debug:
        arguments.extend(("--log-level", "debug"))
    url = None if parsed.headless else _browser_url(parsed.host, parsed.port)
    start = launcher or _launch_fastapi
    try:
        return start(arguments, values, url)
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


def _chat_dependencies_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("fastapi", "uvicorn"))


def _browser_url(host: str, port: int) -> str:
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    address = f"[{host}]" if ":" in host else host
    return f"http://{address}:{port}/"


def _launch_fastapi(
    arguments: Sequence[str], environ: Mapping[str, str], url: str | None
) -> int:
    with subprocess.Popen(
        (sys.executable, "-m", "uvicorn", *arguments),
        env=dict(environ),
    ) as process:
        if url is not None:
            Thread(target=_open_when_ready, args=(process, url), daemon=True).start()
        return process.wait()


def _open_when_ready(process: subprocess.Popen[bytes], url: str) -> None:
    address = urlsplit(url)
    assert address.hostname is not None
    notice_at = monotonic() + 15
    reported_delay = False
    while process.poll() is None:
        if not reported_delay and monotonic() >= notice_at:
            print(
                f"Chat is still starting. The browser will open when {url} is ready.",
                file=sys.stderr,
            )
            reported_delay = True
        connection = HTTPConnection(address.hostname, address.port, timeout=0.5)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            if response.status == 200 and b"<title>Private Data Chat" in response.read(4096):
                break
        except (OSError, HTTPException):
            pass
        finally:
            connection.close()
        sleep(0.25)
    if process.poll() is not None:
        return
    try:
        opened = webbrowser.open(url)
    except (OSError, webbrowser.Error):
        opened = False
    if not opened:
        print(f"Open {url} in your browser.", file=sys.stderr)


def _emit_preflight_failure(stage: str, error: Exception) -> None:
    result: dict[str, object] = {
        "code": "chat_preflight_failed",
        "stage": stage,
        "status": "rejected",
    }
    if isinstance(error, LegacyChatReportingError):
        result["code"] = "chat_reporting_setting_renamed"
        result["message"] = str(error)
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
