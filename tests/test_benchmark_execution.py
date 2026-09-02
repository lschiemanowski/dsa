"""Lifecycle tests for benchmark cell attempts and receipts."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from dsa.benchmark import (
    BenchmarkCellInvocation,
    BenchmarkCellReceipt,
    BenchmarkRuntime,
    BenchmarkStudy,
    BenchmarkWorkerAmbiguous,
    BenchmarkWorkerCompleted,
    BenchmarkWorkerFailed,
    PreparedBenchmark,
    execute_benchmark_cell,
    expand_benchmark_study,
    invoke_benchmark_cell,
    read_benchmark_cell_receipt,
    remove_benchmark_cell_containers,
    retain_benchmark_cell_receipt,
    run_prepared_benchmark,
    stop_benchmark_cell_worker,
)
from dsa.docker import DockerCommandResult, DockerPythonExecutor
from dsa.evaluation import (
    MlflowEvaluationError,
    MlflowEvaluationPrediction,
    MlflowEvaluationResult,
)
from dsa.pack import LoadedEvaluationPack
from dsa.reporting import MlflowReporting

from .test_benchmark import IMAGE, runtime_value, study_value
from .test_mlflow_evaluation import evaluation_pack


def prepared_benchmark(
    tmp_path: Path,
    *,
    cells: int = 1,
    cell_workers: int = 1,
    case_workers: int = 1,
    max_tool_calls: int = 40,
    pack: LoadedEvaluationPack | None = None,
) -> PreparedBenchmark:
    selected_evaluation_pack = pack or evaluation_pack(tmp_path)
    selected_pack = cast(dict[str, object], study_value()["packs"][0])
    value = study_value(
        packs=(
            {
                **selected_pack,
                "reference": selected_evaluation_pack.reference.model_dump(mode="json"),
            },
        ),
        models=tuple(study_value()["models"][:cells]),
        execution={
            "docker_image": IMAGE,
            "case_workers": case_workers,
            "cell_workers": cell_workers,
        },
        policy={"max_tool_calls": max_tool_calls},
        repetitions=1,
    )
    study = BenchmarkStudy.model_validate(value)
    runtime = BenchmarkRuntime.model_validate(
        runtime_value(
            tmp_path,
            datasets=(
                {"pack_id": study.packs[0].pack_id, "dataset_name": "catalog.schema.pack"},
            ),
        )
    )
    return PreparedBenchmark(
        study=study,
        runtime=runtime,
        cells=expand_benchmark_study(study),
        packs=((study.packs[0].pack_id, selected_evaluation_pack),),
    )


def receipt(invocation: BenchmarkCellInvocation) -> BenchmarkCellReceipt:
    return BenchmarkCellReceipt(
        study_sha256=invocation.cell.study_sha256,
        cell_id=invocation.cell.cell_id,
        pack_id=invocation.cell.pack_id,
        model_id=invocation.cell.model_id,
        repetition=invocation.cell.repetition,
        agent_revision=invocation.agent_revision,
        docker_image=invocation.docker_image,
        dataset_name=invocation.dataset_name,
        dataset_id="dataset-id",
        dataset_digest="dataset-digest",
        evaluation_run_id="evaluation-run",
        predictions=(
            MlflowEvaluationPrediction(
                case_id="tiny-count",
                run_id="run-id",
                terminal_sha256="f" * 64,
                accepted=True,
                answer={"count": 3},
                reporting=MlflowReporting(
                    status="reported",
                    tracking_run_id="tracking-run",
                    trace_id="trace-id",
                ),
            ),
        ),
    )


async def complete(invocation: BenchmarkCellInvocation) -> BenchmarkWorkerCompleted:
    retained = retain_benchmark_cell_receipt(receipt(invocation), invocation.receipt_path)
    return BenchmarkWorkerCompleted(
        cell_id=invocation.cell.cell_id,
        receipt_sha256=retained.sha256,
    )


def test_receipt_is_canonical_atomic_and_no_overwrite(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    selected = receipt(invocation)

    retained = retain_benchmark_cell_receipt(selected, invocation.receipt_path)

    assert retained.sha256 == selected.sha256
    assert retained.path == invocation.receipt_path
    assert invocation.receipt_path.read_text().endswith("\n")
    assert read_benchmark_cell_receipt(invocation.receipt_path) == selected
    assert retain_benchmark_cell_receipt(selected, invocation.receipt_path) == retained

    changed = selected.model_copy(update={"dataset_id": "different"})
    with pytest.raises(ValueError, match="conflict"):
        retain_benchmark_cell_receipt(changed, invocation.receipt_path)

    invocation.receipt_path.unlink()
    target = tmp_path / "outside.json"
    target.write_bytes(selected.canonical_json.encode())
    invocation.receipt_path.symlink_to(target)
    with pytest.raises(ValueError, match="unavailable"):
        read_benchmark_cell_receipt(invocation.receipt_path)


async def test_runner_bounds_cells_and_retains_completed_results(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path, cells=2)
    active = 0
    maximum = 0

    async def controlled(invocation: BenchmarkCellInvocation) -> BenchmarkWorkerCompleted:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0)
        result = await complete(invocation)
        active -= 1
        return result

    results = await run_prepared_benchmark(prepared, invoker=controlled)

    assert maximum == 1
    assert [result.status for result in results] == ["completed", "completed"]
    assert all(result.receipt_sha256 for result in results)


async def test_runner_uses_the_explicit_cell_worker_bound(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path, cells=2, cell_workers=2)
    active = 0
    maximum = 0
    release = asyncio.Event()

    async def controlled(invocation: BenchmarkCellInvocation) -> BenchmarkWorkerCompleted:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == 2:
            release.set()
        await release.wait()
        result = await complete(invocation)
        active -= 1
        return result

    results = await run_prepared_benchmark(prepared, invoker=controlled)

    assert maximum == 2
    assert all(result.status == "completed" for result in results)


async def test_resume_skips_only_a_verified_completed_cell(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path)
    first = await run_prepared_benchmark(prepared, invoker=complete)
    called = False

    async def unexpected(invocation: BenchmarkCellInvocation) -> BenchmarkWorkerCompleted:
        nonlocal called
        del invocation
        called = True
        raise AssertionError("verified completed cells must be skipped")

    resumed = await run_prepared_benchmark(prepared, resume=True, invoker=unexpected)

    assert first[0].status == "completed"
    assert resumed[0].status == "skipped"
    assert resumed[0].receipt_sha256 == first[0].receipt_sha256
    assert called is False


async def test_resume_rejects_unclassified_partial_attempts(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path)
    cell = prepared.cells[0]
    partial = prepared.runtime.workspace_root / cell.cell_id / "attempts" / "attempt-0001"
    partial.mkdir(parents=True)

    results = await run_prepared_benchmark(
        prepared,
        resume=True,
        invoker=complete,
    )

    assert results[0].status == "ambiguous"
    assert results[0].failure_code == "cell_attempt_ambiguous"


async def test_resume_retries_a_classified_worker_failure_in_a_new_attempt(
    tmp_path: Path,
) -> None:
    prepared = prepared_benchmark(tmp_path)

    async def fail(invocation: BenchmarkCellInvocation) -> BenchmarkWorkerFailed:
        return BenchmarkWorkerFailed(
            cell_id=invocation.cell.cell_id,
            failure_code="cell_evaluation_failed",
        )

    first = await run_prepared_benchmark(prepared, invoker=fail)
    resumed = await run_prepared_benchmark(prepared, resume=True, invoker=complete)

    attempts = prepared.runtime.workspace_root / prepared.cells[0].cell_id / "attempts"
    assert first[0].status == "failed"
    assert resumed[0].status == "completed"
    assert tuple(path.name for path in sorted(attempts.iterdir())) == (
        "attempt-0001",
        "attempt-0002",
    )


async def test_resume_never_retries_an_ambiguous_remote_evaluation(
    tmp_path: Path,
) -> None:
    prepared = prepared_benchmark(tmp_path)
    calls = 0

    async def ambiguous(invocation: BenchmarkCellInvocation) -> BenchmarkWorkerAmbiguous:
        nonlocal calls
        calls += 1
        return BenchmarkWorkerAmbiguous(
            cell_id=invocation.cell.cell_id,
            failure_code="cell_evaluation_ambiguous",
        )

    first = await run_prepared_benchmark(prepared, invoker=ambiguous)
    resumed = await run_prepared_benchmark(prepared, resume=True, invoker=ambiguous)

    assert first[0].status == "ambiguous"
    assert resumed[0].status == "ambiguous"
    assert resumed[0].failure_code == "cell_attempt_ambiguous"
    assert calls == 1


async def test_runner_revalidates_prepared_state_before_creating_a_workspace(
    tmp_path: Path,
) -> None:
    prepared = prepared_benchmark(tmp_path)
    prepared.study.models[0].configuration.settings["temperature"] = 1

    with pytest.raises(ValueError, match="cells do not match"):
        await run_prepared_benchmark(prepared, invoker=complete)

    assert not prepared.runtime.workspace_root.exists()


async def test_runner_preserves_cancellation_and_attempt_state(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path)
    started = asyncio.Event()

    async def blocked(invocation: BenchmarkCellInvocation) -> BenchmarkWorkerCompleted:
        del invocation
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    task = asyncio.create_task(run_prepared_benchmark(prepared, invoker=blocked))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    attempts = (
        prepared.runtime.workspace_root
        / prepared.cells[0].cell_id
        / "attempts"
        / "attempt-0001"
    )
    assert attempts.is_dir()
    assert not (prepared.runtime.workspace_root / prepared.cells[0].cell_id / "cell.json").exists()


def test_worker_composes_existing_evaluation_and_records_effective_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_benchmark(tmp_path, case_workers=3)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    invocation.attempt_directory.mkdir(parents=True)
    captured: dict[str, object] = {}
    original_workers = "17"
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", original_workers)

    def evaluate(pack: object, **kwargs: object) -> MlflowEvaluationResult:
        captured["pack"] = pack
        captured.update(kwargs)
        assert __import__("os").environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] == "3"
        assert __import__("os").environ["MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT"] == "930"
        return MlflowEvaluationResult(
            dataset_id="dataset-id",
            dataset_digest="dataset-digest",
            evaluation_run_id="evaluation-run",
            predictions=(
                MlflowEvaluationPrediction(
                    case_id="tiny-count",
                    run_id="run-id",
                    terminal_sha256="f" * 64,
                    accepted=True,
                    answer={"count": 3},
                    reporting=MlflowReporting(
                        status="reported",
                        tracking_run_id="tracking-run",
                        trace_id="trace-id",
                    ),
                ),
            ),
        )

    result = execute_benchmark_cell(
        invocation,
        pack_loader=lambda _reference: prepared.packs[0][1],
        evaluation_runner=evaluate,
    )

    assert isinstance(result, BenchmarkWorkerCompleted)
    assert __import__("os").environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] == original_workers
    assert "MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT" not in __import__("os").environ
    assert captured["dataset_name"] == invocation.dataset_name
    assert captured["model_configuration"] == invocation.model_configuration
    assert captured["policy"] == invocation.policy
    executor = cast(DockerPythonExecutor, captured["python_executor"])
    assert executor.container_labels == (
        ("dsa.benchmark.cleanup", invocation.cleanup_token),
    )
    assert captured["run_tags"] == {
        "dsa.benchmark.agent_revision": invocation.agent_revision,
        "dsa.benchmark.cell_id": invocation.cell.cell_id,
        "dsa.benchmark.model_id": invocation.cell.model_id,
        "dsa.benchmark.pack_id": invocation.cell.pack_id,
        "dsa.benchmark.repetition": "0",
        "dsa.benchmark.study_sha256": invocation.cell.study_sha256,
    }
    retained = read_benchmark_cell_receipt(invocation.receipt_path)
    assert retained.evaluation_run_id == "evaluation-run"
    assert retained.predictions[0].terminal_sha256 == "f" * 64


async def test_real_subprocess_uses_the_framed_failure_protocol(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)

    result = await invoke_benchmark_cell(invocation)

    assert isinstance(result, BenchmarkWorkerFailed)
    assert result.cell_id == invocation.cell.cell_id
    assert result.failure_code == "cell_setup_failed"


def test_worker_classifies_uncertain_mlflow_outcomes_as_ambiguous(
    tmp_path: Path,
) -> None:
    prepared = prepared_benchmark(tmp_path)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    invocation.attempt_directory.mkdir(parents=True)

    def uncertain(*_args: object, **_kwargs: object) -> MlflowEvaluationResult:
        raise RuntimeError("remote response was lost")

    result = execute_benchmark_cell(
        invocation,
        pack_loader=lambda _reference: prepared.packs[0][1],
        evaluation_runner=uncertain,
    )

    assert isinstance(result, BenchmarkWorkerAmbiguous)
    assert result.failure_code == "cell_evaluation_ambiguous"
    assert not invocation.receipt_path.exists()


def test_worker_retains_a_stable_mlflow_failure_phase(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    invocation.attempt_directory.mkdir(parents=True)

    def failed(*_args: object, **_kwargs: object) -> MlflowEvaluationResult:
        raise MlflowEvaluationError("mlflow_dataset_failed")

    result = execute_benchmark_cell(
        invocation,
        pack_loader=lambda _reference: prepared.packs[0][1],
        evaluation_runner=failed,
    )

    assert isinstance(result, BenchmarkWorkerAmbiguous)
    assert result.failure_code == "cell_mlflow_dataset_failed"
    assert not invocation.receipt_path.exists()


class CleanupDockerRunner:
    def __init__(self, results: tuple[DockerCommandResult, ...]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, ...]] = []
        self.output_limits: list[int] = []

    async def run(
        self,
        arguments: Sequence[str],
        *,
        input_bytes: bytes | None,
        timeout_seconds: float,
        output_limit: int,
    ) -> DockerCommandResult:
        del input_bytes, timeout_seconds
        self.calls.append(tuple(arguments))
        self.output_limits.append(output_limit)
        return self.results.pop(0)

    async def run_to_file(
        self,
        arguments: Sequence[str],
        *,
        destination: Path,
        timeout_seconds: float,
        output_limit: int,
    ) -> DockerCommandResult:
        del arguments, destination, timeout_seconds, output_limit
        raise AssertionError("container cleanup must not stream files")


def docker_result(
    *,
    returncode: int = 0,
    stdout: bytes = b"",
) -> DockerCommandResult:
    return DockerCommandResult(
        returncode=returncode,
        stdout=stdout,
        stderr=b"",
        stdout_truncated=False,
        stderr_truncated=False,
        timed_out=False,
    )


async def test_parent_cleanup_removes_only_containers_with_the_attempt_label(
    tmp_path: Path,
) -> None:
    prepared = prepared_benchmark(tmp_path, case_workers=2, max_tool_calls=1)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    first_id = "a" * 12
    second_id = "b" * 64
    runner = CleanupDockerRunner(
        (
            docker_result(stdout=f"{first_id}\n{second_id}\n".encode()),
            docker_result(),
        )
    )

    removed = await remove_benchmark_cell_containers(invocation, runner=runner)

    assert removed is True
    assert runner.output_limits == [8 * 65, 2 * 65]
    assert runner.calls[0][-2:] == (
        "--filter",
        f"label=dsa.benchmark.cleanup={invocation.cleanup_token}",
    )
    assert runner.calls[1] == (
        "docker",
        "rm",
        "--force",
        first_id,
        second_id,
    )


async def test_parent_cleanup_removes_a_large_valid_cell_in_bounded_batches(
    tmp_path: Path,
) -> None:
    prepared = prepared_benchmark(tmp_path, case_workers=2, max_tool_calls=65)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    container_ids = tuple(f"{index:012x}" for index in range(130))
    runner = CleanupDockerRunner(
        (
            docker_result(stdout=("\n".join(container_ids) + "\n").encode()),
            docker_result(),
            docker_result(),
        )
    )

    removed = await remove_benchmark_cell_containers(invocation, runner=runner)

    assert removed is True
    assert runner.output_limits == [136 * 65, 128 * 65, 2 * 65]
    assert runner.calls[1][:3] == ("docker", "rm", "--force")
    assert runner.calls[1][3:] == container_ids[:128]
    assert runner.calls[2][3:] == container_ids[128:]


async def test_parent_freezes_and_cleans_a_stuck_worker_before_killing_it(
    tmp_path: Path,
) -> None:
    prepared = prepared_benchmark(tmp_path)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)

    class StuckProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.pid = 123
            self.signals: list[int] = []
            self.exited = asyncio.Event()

        async def wait(self) -> int:
            await self.exited.wait()
            assert self.returncode is not None
            return self.returncode

        def send_signal(self, selected_signal: int) -> None:
            self.signals.append(selected_signal)
            if selected_signal == signal.SIGKILL:
                self.returncode = -signal.SIGKILL
                self.exited.set()

    process = StuckProcess()
    states: list[int | None] = []

    async def clean(_invocation: BenchmarkCellInvocation) -> bool:
        states.append(process.returncode)
        return True

    await stop_benchmark_cell_worker(
        cast(Any, process),
        invocation,
        cleaner=clean,
        grace_seconds=0.01,
        settle_seconds=0,
    )

    assert states == [None, -signal.SIGKILL]
    assert process.signals == [signal.SIGINT, signal.SIGSTOP, signal.SIGKILL]
