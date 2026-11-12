"""Smoke-test an installed chat wheel on Linux with only its venv on PATH."""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
from http.client import HTTPConnection, HTTPMessage
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

from tests.private_data_chat.test_cli import write_configuration


def request(port: int, method: str, path: str) -> tuple[int, bytes, HTTPMessage]:
    connection = HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, response.read(), response.headers
    finally:
        connection.close()


def smoke(dsa: Path, directory: Path) -> None:
    config = write_configuration(directory)
    environment = {"PATH": str(dsa.parent)}
    subprocess.run(
        [
            str(dsa.with_name("python")),
            "-I",
            "-c",
            "import importlib.util, shutil; "
            "assert shutil.which('node') is None; "
            "assert importlib.util.find_spec('chainlit') is None",
        ],
        cwd=directory,
        env=environment,
        check=True,
        timeout=10,
    )

    command = [str(dsa), "chat", "--config", str(config)]
    check = subprocess.run(
        [*command, "--check"],
        cwd=directory,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert json.loads(check.stdout) == {
        "config_path": str(config), "data_source_id": "retail", "status": "ready",
    }

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with subprocess.Popen(
        [*command, "--headless", "--port", str(port)],
        cwd=directory,
        env=environment,
        start_new_session=True,
    ) as process:
        try:
            deadline = monotonic() + 10
            while process.poll() is None and monotonic() < deadline:
                try:
                    status, page, headers = request(port, "GET", "/")
                    break
                except OSError:
                    sleep(0.1)
            else:
                raise AssertionError("dsa chat did not start")

            assert status == 200 and b"<title>Private Data Chat" in page
            assert headers["Cache-Control"] == "no-store"
            assert "script-src 'self'" in headers["Content-Security-Policy"]
            assert "img-src 'self' data:" in headers["Content-Security-Policy"]
            assets = re.findall(rb'(?:src|href)="(/assets/[^"]+)"', page)
            assert any(asset.endswith(b".js") for asset in assets)
            assert any(asset.endswith(b".css") for asset in assets)
            for asset in assets:
                status, body, headers = request(port, "GET", asset.decode())
                assert status == 200 and body
                assert headers["Cache-Control"] == "public, max-age=31536000, immutable"
            status, body, headers = request(port, "GET", "/dsa-mark.svg")
            assert status == 200 and body and headers["Cache-Control"] == "no-store"
            status, body, headers = request(port, "POST", "/api/conversations")
            assert status == 201 and json.loads(body)["status"] == "open"
            assert headers["Cache-Control"] == "no-store"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)


if __name__ == "__main__":
    with TemporaryDirectory() as temporary:
        smoke(Path(sys.argv[1]).absolute(), Path(temporary))
    print("Installed wheel passed dsa chat --check and served the UI and API without Node.")
