"""Tests for the product-level command dispatcher."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from apps.cli import main


def test_help_lists_commands_without_importing_a_ui(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"]) == 0

    output = capsys.readouterr()
    assert output.out.startswith("usage: dsa")
    assert "run" in output.out
    assert "benchmark" in output.out
    assert "chat" in output.out
    assert output.err == ""


@pytest.mark.parametrize("command", ["run", "benchmark", "chat"])
def test_dispatches_remaining_arguments_with_the_umbrella_program_name(
    command: str,
) -> None:
    calls: list[tuple[str, list[str], str]] = []

    def selected(argv: Sequence[str] | None, *, prog: str) -> int:
        calls.append((command, list(argv or ()), prog))
        return 17

    commands = {
        "run": selected if command == "run" else _unexpected,
        "benchmark": selected if command == "benchmark" else _unexpected,
        "chat": selected if command == "chat" else _unexpected,
    }

    assert (
        main(
            [command, "--example", "value"],
            run_command=commands["run"],
            benchmark_command=commands["benchmark"],
            chat_command=commands["chat"],
        )
        == 17
    )
    assert calls == [(command, ["--example", "value"], f"dsa {command}")]


def test_invalid_command_emits_a_stable_usage_rejection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["unknown"]) == 2
    assert capsys.readouterr().out == '{"code":"dsa_usage","status":"rejected"}\n'


@pytest.mark.parametrize(
    ("arguments", "usage"),
    [
        (["run", "--help"], "usage: dsa run"),
        (["benchmark", "--help"], "usage: dsa benchmark"),
        (["chat", "--help"], "usage: dsa chat"),
    ],
)
def test_subcommand_help_is_owned_by_the_selected_command(
    arguments: list[str],
    usage: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(arguments) == 0
    assert capsys.readouterr().out.startswith(usage)


def _unexpected(argv: Sequence[str] | None, *, prog: str) -> int:
    del argv, prog
    raise AssertionError("wrong command dispatched")
