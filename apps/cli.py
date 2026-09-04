"""Product-level command dispatcher without application logic."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Protocol

from apps.private_data_chat.cli import main as chat_main
from dsa.benchmark_cli import main as benchmark_main
from dsa.cli import Parser, emit
from dsa.run_cli import main as run_main


class Command(Protocol):
    def __call__(
        self,
        argv: Sequence[str] | None,
        *,
        prog: str,
    ) -> int: ...


def main(
    argv: Sequence[str] | None = None,
    *,
    run_command: Command = run_main,
    benchmark_command: Command = benchmark_main,
    chat_command: Command = chat_main,
) -> int:
    """Route one command while keeping each implementation in its own package."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _parser().print_help()
        return 0
    command = arguments[0]
    if command not in {"run", "benchmark", "chat"}:
        emit({"code": "dsa_usage", "status": "rejected"})
        return 2

    commands = {
        "run": run_command,
        "benchmark": benchmark_command,
        "chat": chat_command,
    }
    selected = commands[command]
    return selected(arguments[1:], prog=f"dsa {command}")


def _parser() -> Parser:
    parser = Parser(
        prog="dsa",
        description="Run DSA tasks, benchmarks, and applications.",
        add_help=True,
    )
    parser.add_argument("command", choices=("run", "benchmark", "chat"))
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
