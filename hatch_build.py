"""Compile the browser assets when building a distribution wheel."""

from pathlib import Path
from subprocess import run
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version != "standard":
            return

        ui = Path(self.root) / "ui"
        run(["npm", "ci"], cwd=ui, check=True)
        run(["npm", "run", "build"], cwd=ui, check=True)
