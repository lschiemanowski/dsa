"""Tests for the thin benchmark command-line boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsa.benchmark import (
    BenchmarkCellRunResult,
    BenchmarkRuntime,
    PreparedBenchmark,
    benchmark_plan,
)
from dsa.benchmark_cli import main

from .test_benchmark import REVISION
from .test_benchmark_execution import prepared_benchmark


def write_contracts(
    tmp_path: Path,
    prepared: PreparedBenchmark,
    *,
    relative_workspace: bool = False,
) -> tuple[Path, Path]:
    study_path = tmp_path / "study.json"
    runtime_path = tmp_path / "runtime.json"
    runtime = (
        prepared.runtime.model_copy(update={"workspace_root": Path("workspace")})
        if relative_workspace
        else prepared.runtime
    )
    study_path.write_bytes(prepared.study.canonical_json.encode())
    runtime_path.write_bytes(runtime.canonical_json.encode())
    return study_path, runtime_path


def test_help_exits_zero_without_a_rejection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr()
    assert output.out.startswith("usage: dsa-benchmark")
    assert "benchmark_usage" not in output.out
    assert output.err == ""


def test_plan_preflights_and_emits_only_the_canonical_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = prepared_benchmark(tmp_path)
    study_path, runtime_path = write_contracts(
        tmp_path,
        prepared,
        relative_workspace=True,
    )
    calls: list[tuple[str, Path, bool]] = []

    async def prepare(
        study: object,
        runtime: BenchmarkRuntime,
        *,
        current_revision: str,
        resume: bool,
    ) -> PreparedBenchmark:
        del study
        calls.append((current_revision, runtime.workspace_root, resume))
        return prepared

    status = main(
        ["plan", "--study", str(study_path), "--runtime", str(runtime_path)],
        current_revision=REVISION,
        preparer=prepare,
    )

    assert status == 0
    assert calls == [(REVISION, (tmp_path / "workspace").resolve(), False)]
    captured = capsys.readouterr()
    assert captured.out == benchmark_plan(prepared).canonical_json
    assert captured.err == ""


def test_run_forwards_explicit_resume_and_reports_completed_cells(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = prepared_benchmark(tmp_path)
    study_path, runtime_path = write_contracts(tmp_path, prepared)
    prepared_resumes: list[bool] = []
    runner_resumes: list[bool] = []

    async def prepare(
        study: object,
        runtime: object,
        *,
        current_revision: str,
        resume: bool,
    ) -> PreparedBenchmark:
        del study, runtime, current_revision
        prepared_resumes.append(resume)
        return prepared

    async def run(
        selected: PreparedBenchmark,
        *,
        resume: bool,
    ) -> tuple[BenchmarkCellRunResult, ...]:
        assert selected is prepared
        runner_resumes.append(resume)
        return (
            BenchmarkCellRunResult(
                cell_id=prepared.cells[0].cell_id,
                status="skipped",
                receipt_sha256="f" * 64,
            ),
        )

    status = main(
        [
            "run",
            "--study",
            str(study_path),
            "--runtime",
            str(runtime_path),
            "--resume",
        ],
        current_revision=REVISION,
        preparer=prepare,
        runner=run,
    )

    assert status == 0
    assert prepared_resumes == [True]
    assert runner_resumes == [True]
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "completed"
    assert output["study_sha256"] == prepared.study.sha256
    assert output["cells"][0]["status"] == "skipped"


def test_run_returns_one_for_an_incomplete_matrix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = prepared_benchmark(tmp_path)
    study_path, runtime_path = write_contracts(tmp_path, prepared)

    async def prepare(*_args: object, **_kwargs: object) -> PreparedBenchmark:
        return prepared

    async def run(
        _prepared: PreparedBenchmark,
        *,
        resume: bool,
    ) -> tuple[BenchmarkCellRunResult, ...]:
        del resume
        return (
            BenchmarkCellRunResult(
                cell_id=prepared.cells[0].cell_id,
                status="failed",
                failure_code="cell_evaluation_failed",
            ),
        )

    status = main(
        ["run", "--study", str(study_path), "--runtime", str(runtime_path)],
        current_revision=REVISION,
        preparer=prepare,
        runner=run,
    )

    assert status == 1
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"


def test_preflight_rejection_returns_two_without_running_cells(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = prepared_benchmark(tmp_path)
    study_path, runtime_path = write_contracts(tmp_path, prepared)
    ran = False

    async def reject(*_args: object, **_kwargs: object) -> PreparedBenchmark:
        raise ValueError("preflight rejected")

    async def run(*_args: object, **_kwargs: object) -> tuple[BenchmarkCellRunResult, ...]:
        nonlocal ran
        ran = True
        return ()

    status = main(
        ["run", "--study", str(study_path), "--runtime", str(runtime_path)],
        current_revision=REVISION,
        preparer=reject,
        runner=run,
    )

    assert status == 2
    assert ran is False
    assert json.loads(capsys.readouterr().out) == {
        "code": "benchmark_preflight_failed",
        "status": "rejected",
    }


def test_noncanonical_or_symlinked_contract_is_rejected_before_preflight(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = prepared_benchmark(tmp_path)
    study_path, runtime_path = write_contracts(tmp_path, prepared)
    study_path.write_bytes(prepared.study.canonical_json.encode() + b"\n")
    calls = 0

    async def prepare(*_args: object, **_kwargs: object) -> PreparedBenchmark:
        nonlocal calls
        calls += 1
        return prepared

    status = main(
        ["plan", "--study", str(study_path), "--runtime", str(runtime_path)],
        current_revision=REVISION,
        preparer=prepare,
    )

    assert status == 2
    assert calls == 0
    assert json.loads(capsys.readouterr().out) == {
        "code": "benchmark_usage",
        "status": "rejected",
    }

    study_path.write_bytes(prepared.study.canonical_json.encode())
    link = tmp_path / "study-link.json"
    link.symlink_to(study_path)
    assert main(
        ["plan", "--study", str(link), "--runtime", str(runtime_path)],
        current_revision=REVISION,
        preparer=prepare,
    ) == 2
    assert calls == 0
