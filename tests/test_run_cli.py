"""Tests for the standalone single-task command-line boundary."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from dsa import (
    Failure,
    MlflowReporting,
    PythonExecutionRequest,
    PythonExecutionResult,
    PythonExecutor,
    RetainedTerminalRecord,
    RunCompletion,
    RunFailure,
    RunRequest,
    RunSuccess,
    TerminalRecord,
)
from dsa.run_cli import main

from .test_contract import request_value

IMAGE = f"sha256:{'a' * 64}"


class StubExecutor:
    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        del request
        raise AssertionError("the CLI test runner must not invoke the executor")


def write_request(
    tmp_path: Path,
    *,
    database_path: Path | None = None,
    answer_schema: dict[str, object] | None = None,
) -> Path:
    raw = request_value(database_path or Path("data/source.duckdb"))
    if answer_schema is not None:
        raw["answer_schema"] = answer_schema
    request = RunRequest.model_validate(raw)
    path = tmp_path / "task.json"
    path.write_text(
        json.dumps(
            request.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return path


def completion(
    request: RunRequest,
    runs_directory: Path,
    *,
    outcome: RunSuccess | RunFailure,
    reporting: MlflowReporting | None = None,
) -> RunCompletion:
    run_directory = runs_directory / "run-001"
    retained = RetainedTerminalRecord(
        path=run_directory / "terminal.json",
        sha256="f" * 64,
        byte_length=123,
    )
    record = TerminalRecord(
        run_id="run-001",
        started_at=datetime(2026, 8, 31, 8, 0, tzinfo=UTC),
        finished_at=datetime(2026, 8, 31, 8, 1, tzinfo=UTC),
        request=request,
        outcome=outcome,
    )
    return RunCompletion(
        record=record,
        retained_record=retained,
        reporting=reporting or MlflowReporting(),
    )


def test_help_exits_zero_without_a_rejection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr()
    assert output.out.startswith("usage: dsa-run")
    assert "run_usage" not in output.out
    assert output.err == ""


def test_success_runs_one_resolved_request_with_the_hardened_executor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path = write_request(tmp_path)
    runs_directory = tmp_path / "runs"
    executor = StubExecutor()
    executor_images: list[str] = []
    calls: list[tuple[RunRequest, Path, PythonExecutor, bool]] = []

    def executor_factory(image: str) -> PythonExecutor:
        executor_images.append(image)
        return executor

    async def run(
        request: RunRequest,
        *,
        runs_directory: Path,
        python_executor: PythonExecutor,
        report_to_mlflow: bool,
    ) -> RunCompletion:
        calls.append((request, runs_directory, python_executor, report_to_mlflow))
        return completion(
            request,
            runs_directory,
            outcome=RunSuccess(answer={"count": 3}),
            reporting=MlflowReporting(
                status="reported",
                tracking_run_id="tracking-1",
                trace_id="trace-1",
            ),
        )

    status = main(
        [
            "--request",
            str(request_path),
            "--runs-directory",
            str(runs_directory),
            "--docker-image",
            IMAGE,
            "--report-to-mlflow",
        ],
        runner=run,
        executor_factory=executor_factory,
    )

    assert status == 0
    assert executor_images == [IMAGE]
    assert len(calls) == 1
    selected, selected_runs, selected_executor, report = calls[0]
    assert selected.database_path == (tmp_path / "data/source.duckdb").resolve()
    assert selected_runs == runs_directory.resolve()
    assert selected_executor is executor
    assert report is True
    assert json.loads(capsys.readouterr().out) == {
        "answer": {"count": 3},
        "answer_inline": True,
        "reporting": {
            "status": "reported",
            "trace_id": "trace-1",
            "tracking_run_id": "tracking-1",
        },
        "run_id": "run-001",
        "status": "succeeded",
        "terminal_record": {
            "byte_length": 123,
            "path": str(runs_directory / "run-001/terminal.json"),
            "sha256": "f" * 64,
        },
    }


def test_analysis_failure_returns_one_without_exposing_failure_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path = write_request(tmp_path)
    runs_directory = tmp_path / "runs"

    async def run(
        request: RunRequest,
        **kwargs: object,
    ) -> RunCompletion:
        del kwargs
        return completion(
            request,
            runs_directory,
            outcome=RunFailure(
                failure=Failure(
                    stage="model",
                    code="model_api_error",
                    message="provider said SECRET signed-url",
                    diagnostics={"private": "SECRET"},
                )
            ),
        )

    status = main(
        [
            "--request",
            str(request_path),
            "--runs-directory",
            str(runs_directory),
            "--docker-image",
            IMAGE,
        ],
        runner=run,
        executor_factory=lambda _image: StubExecutor(),
    )

    assert status == 1
    raw = capsys.readouterr().out
    assert "SECRET" not in raw
    assert json.loads(raw) == {
        "failure": {"code": "model_api_error", "stage": "model"},
        "reporting": {"status": "disabled"},
        "run_id": "run-001",
        "status": "failed",
        "terminal_record": {
            "byte_length": 123,
            "path": str(runs_directory / "run-001/terminal.json"),
            "sha256": "f" * 64,
        },
    }


def test_large_success_answer_remains_only_in_the_terminal_record(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path = write_request(
        tmp_path,
        answer_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "string",
        },
    )

    async def run(
        request: RunRequest,
        **kwargs: object,
    ) -> RunCompletion:
        del kwargs
        return completion(
            request,
            tmp_path / "runs",
            outcome=RunSuccess(answer="x" * (1024 * 1024 + 1)),
        )

    status = main(
        [
            "--request",
            str(request_path),
            "--runs-directory",
            str(tmp_path / "runs"),
            "--docker-image",
            IMAGE,
        ],
        runner=run,
        executor_factory=lambda _image: StubExecutor(),
    )

    assert status == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "succeeded"
    assert result["answer_inline"] is False
    assert "answer" not in result


def test_cancellation_emits_retained_terminal_identity_before_returning_130(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path = write_request(tmp_path)
    runs_directory = tmp_path / "runs"

    async def cancel(
        request: RunRequest,
        **kwargs: object,
    ) -> RunCompletion:
        del kwargs
        terminalized = completion(
            request,
            runs_directory,
            outcome=RunFailure(
                failure=Failure(
                    stage="cancelled",
                    code="cancelled_by_caller",
                    message="The caller cancelled the analysis run",
                )
            ),
            reporting=MlflowReporting(
                status="failed",
                failure_code="mlflow_cancelled",
            ),
        )
        error = asyncio.CancelledError()
        retained_error = cast(Any, error)
        retained_error.terminal_record = terminalized.record
        retained_error.retained_record = terminalized.retained_record
        retained_error.reporting = terminalized.reporting
        raise error

    status = main(
        [
            "--request",
            str(request_path),
            "--runs-directory",
            str(runs_directory),
            "--docker-image",
            IMAGE,
            "--report-to-mlflow",
        ],
        runner=cancel,
        executor_factory=lambda _image: StubExecutor(),
    )

    assert status == 130
    assert json.loads(capsys.readouterr().out) == {
        "failure": {"code": "cancelled_by_caller", "stage": "cancelled"},
        "reporting": {
            "failure_code": "mlflow_cancelled",
            "status": "failed",
        },
        "run_id": "run-001",
        "status": "failed",
        "terminal_record": {
            "byte_length": 123,
            "path": str(runs_directory / "run-001/terminal.json"),
            "sha256": "f" * 64,
        },
    }


@pytest.mark.parametrize("invalid_kind", ["noncanonical", "symlink"])
def test_invalid_request_is_rejected_before_executor_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    invalid_kind: str,
) -> None:
    request_path = write_request(tmp_path)
    if invalid_kind == "noncanonical":
        request_path.write_bytes(request_path.read_bytes() + b"\n")
        selected_path = request_path
    else:
        selected_path = tmp_path / "task-link.json"
        selected_path.symlink_to(request_path)
    created = False

    def create(_image: str) -> PythonExecutor:
        nonlocal created
        created = True
        return StubExecutor()

    status = main(
        [
            "--request",
            str(selected_path),
            "--runs-directory",
            str(tmp_path / "runs"),
            "--docker-image",
            IMAGE,
        ],
        executor_factory=create,
    )

    assert status == 2
    assert created is False
    assert json.loads(capsys.readouterr().out) == {
        "code": "run_usage",
        "status": "rejected",
    }


def test_setup_and_unexpected_execution_errors_have_stable_safe_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path = write_request(tmp_path)
    arguments = [
        "--request",
        str(request_path),
        "--runs-directory",
        str(tmp_path / "runs"),
        "--docker-image",
        IMAGE,
    ]

    def reject_executor(_image: str) -> PythonExecutor:
        raise RuntimeError("SECRET Docker setup detail")

    assert main(arguments, executor_factory=reject_executor) == 2
    assert json.loads(capsys.readouterr().out) == {
        "code": "run_usage",
        "status": "rejected",
    }

    async def fail(*_args: object, **_kwargs: object) -> RunCompletion:
        raise RuntimeError("SECRET provider diagnostic")

    assert main(arguments, runner=fail, executor_factory=lambda _image: StubExecutor()) == 1
    raw = capsys.readouterr().out
    assert "SECRET" not in raw
    assert json.loads(raw) == {"code": "run_execution_failed", "status": "failed"}
