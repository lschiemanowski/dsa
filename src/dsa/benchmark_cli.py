"""Thin command-line boundary for benchmark planning and execution."""

from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
from collections.abc import Callable, Coroutine, Sequence
from pathlib import Path
from typing import Any, Protocol

from dsa.benchmark import (
    BenchmarkCellRunResult,
    BenchmarkRuntime,
    BenchmarkStudy,
    PreparedBenchmark,
    benchmark_plan,
    prepare_benchmark,
    run_prepared_benchmark,
)
from dsa.benchmark_report import (
    BenchmarkReportConfigurationError,
    RetainedBenchmarkReport,
    create_benchmark_report,
)
from dsa.cli import HelpRequested, Parser, emit, read_contract

Preparer = Callable[..., Coroutine[Any, Any, PreparedBenchmark]]
Runner = Callable[
    ...,
    Coroutine[Any, Any, tuple[BenchmarkCellRunResult, ...]],
]


class Reporter(Protocol):
    def __call__(
        self,
        study: BenchmarkStudy | object,
        runtime: BenchmarkRuntime | object,
        *,
        output_directory: Path,
        reporter_revision: str,
    ) -> RetainedBenchmarkReport: ...


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "dsa-benchmark",
    current_revision: str | None = None,
    preparer: Preparer = prepare_benchmark,
    runner: Runner = run_prepared_benchmark,
    reporter: Reporter = create_benchmark_report,
) -> int:
    """Validate and execute one benchmark CLI operation."""
    try:
        parsed = _parser(prog).parse_args(argv)
        study = _load_study(Path(parsed.study))
        runtime = _load_runtime(Path(parsed.runtime))
        revision = current_revision or _current_git_revision()
    except HelpRequested:
        return 0
    except Exception:
        emit({"code": "benchmark_usage", "status": "rejected"})
        return 2
    if parsed.command == "report":
        try:
            retained = reporter(
                study,
                runtime,
                output_directory=Path(parsed.output).resolve(),
                reporter_revision=revision,
            )
        except KeyboardInterrupt:
            return 130
        except BenchmarkReportConfigurationError:
            emit(
                {
                    "code": "benchmark_report_preflight_failed",
                    "status": "rejected",
                }
            )
            return 2
        except Exception:
            emit({"code": "benchmark_report_failed", "status": "failed"})
            return 1
        emit(
            {
                "json_path": str(retained.json_path),
                "markdown_path": str(retained.markdown_path),
                "report_sha256": retained.report_sha256,
                "status": "completed",
                "study_sha256": study.sha256,
            }
        )
        return 0
    try:
        prepared = asyncio.run(
            preparer(
                study,
                runtime,
                current_revision=revision,
                resume=bool(getattr(parsed, "resume", False)),
            )
        )
    except KeyboardInterrupt:
        return 130
    except Exception:
        emit({"code": "benchmark_preflight_failed", "status": "rejected"})
        return 2
    if parsed.command == "plan":
        try:
            sys.stdout.write(benchmark_plan(prepared).canonical_json)
        except Exception:
            emit({"code": "benchmark_preflight_failed", "status": "rejected"})
            return 2
        return 0
    try:
        results = asyncio.run(runner(prepared, resume=parsed.resume))
    except KeyboardInterrupt:
        return 130
    except Exception:
        emit({"code": "benchmark_execution_failed", "status": "failed"})
        return 1
    complete = all(item.status in {"completed", "skipped"} for item in results)
    emit(
        {
            "cells": [item.model_dump(mode="json") for item in results],
            "status": "completed" if complete else "incomplete",
            "study_sha256": prepared.study.sha256,
        }
    )
    return 0 if complete else 1


def _parser(prog: str = "dsa-benchmark") -> Parser:
    parser = Parser(prog=prog, add_help=True)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", add_help=True)
    _common_arguments(plan)
    run = commands.add_parser("run", add_help=True)
    _common_arguments(run)
    run.add_argument("--resume", action="store_true")
    report = commands.add_parser("report", add_help=True)
    _common_arguments(report)
    report.add_argument("--output", required=True)
    return parser


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study", required=True)
    parser.add_argument("--runtime", required=True)


def _load_study(path: Path) -> BenchmarkStudy:
    content = read_contract(path)
    study = BenchmarkStudy.model_validate_json(content)
    if content != study.canonical_json.encode():
        raise ValueError("study must use canonical JSON")
    return study


def _load_runtime(path: Path) -> BenchmarkRuntime:
    content = read_contract(path)
    runtime = BenchmarkRuntime.model_validate_json(content)
    if content != runtime.canonical_json.encode():
        raise ValueError("runtime must use canonical JSON")
    root = runtime.workspace_root
    if not root.is_absolute():
        root = (path.resolve().parent / root).resolve()
    return BenchmarkRuntime.model_validate(
        {
            **runtime.model_dump(mode="python"),
            "workspace_root": root,
        }
    )


def _current_git_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("agent revision unavailable") from None
    revision = completed.stdout.decode(errors="replace").strip()
    if completed.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("agent revision unavailable")
    try:
        status = subprocess.run(
            (
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                "src/dsa",
                "pyproject.toml",
                "uv.lock",
            ),
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("agent revision unavailable") from None
    if status.returncode != 0 or status.stdout:
        raise ValueError("agent implementation does not match its Git revision")
    return revision


if __name__ == "__main__":
    raise SystemExit(main())
