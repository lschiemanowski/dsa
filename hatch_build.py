"""Optionally compile the browser assets for a distribution wheel."""

from os import environ
from pathlib import Path
from subprocess import run
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version != "standard" or environ.get("DSA_BUILD_CHAT_UI") != "1":
            return

        root = Path(self.root)
        if (root / "apps/private_data_chat/web/index.html").is_file():
            return

        ui = root / "ui"
        run(["npm", "ci", "--no-audit", "--no-fund"], cwd=ui, check=True)
        run(["npm", "run", "build"], cwd=ui, check=True)
