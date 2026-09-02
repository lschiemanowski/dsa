"""Immutable benchmark studies and isolated cell orchestration."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import stat
import sys
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, cast
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from dsa.contract import ContractModel, ModelConfiguration, RunPolicy
from dsa.docker import (
    AsyncSubprocessDockerRunner,
    DockerCommandRunner,
    DockerPythonExecutor,
    default_docker_configuration,
)
from dsa.evaluation import (
    MlflowEvaluationError,
    MlflowEvaluationPrediction,
    MlflowEvaluationResult,
    run_mlflow_evaluation,
)
from dsa.mlflow_config import (
    SAFE_DATASET_NAME_PATTERN,
    MlflowConfigurationError,
    dataset_name_matches_backend,
    load_mlflow_configuration,
)
from dsa.pack import (
    HuggingFacePackReference,
    LoadedEvaluationPack,
    load_huggingface_evaluation_pack,
)

SafeName = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"),
]
SafeDatasetName = Annotated[
    str,
    Field(pattern=SAFE_DATASET_NAME_PATTERN, max_length=386),
]
SafeRemoteId = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$"),
]
GitRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
ImmutableImage = Annotated[
    str,
    Field(pattern=r"^(?:sha256:|[^@\s]+@sha256:)[0-9a-f]{64}$"),
]
PositiveBounded = Annotated[int, Field(gt=0, le=64)]


class BenchmarkPack(ContractModel):
    """One exact evaluation pack selected by a portable study."""

    pack_id: SafeName
    reference: HuggingFacePackReference


class BenchmarkModel(ContractModel):
    """One safe model configuration selected by a portable study."""

    model_id: SafeName
    configuration: ModelConfiguration


class BenchmarkExecution(ContractModel):
    """Behavior-affecting execution controls retained in study identity."""

    docker_image: ImmutableImage
    case_workers: PositiveBounded = 1
    cell_workers: PositiveBounded = 1


class BenchmarkStudy(ContractModel):
    """One canonical content-addressed benchmark matrix."""

    format: Literal["dsa-benchmark-study/v1"]
    study_id: SafeName
    version: str = Field(
        pattern=(
            r"^(?:0|[1-9][0-9]*)\."
            r"(?:0|[1-9][0-9]*)\."
            r"(?:0|[1-9][0-9]*)$"
        )
    )
    agent_revision: GitRevision
    packs: tuple[BenchmarkPack, ...] = Field(min_length=1, max_length=64)
    models: tuple[BenchmarkModel, ...] = Field(min_length=1, max_length=64)
    policy: RunPolicy
    execution: BenchmarkExecution
    repetitions: Annotated[int, Field(gt=0, le=100)]

    @model_validator(mode="after")
    def cell_count_is_bounded(self) -> BenchmarkStudy:
        if len(self.packs) * len(self.models) * self.repetitions > 10_000:
            raise ValueError("benchmark study contains too many cells")
        if (
            self.execution.case_workers * self.policy.max_tool_calls
            > _MAX_CELL_CONTAINERS
        ):
            raise ValueError("benchmark cell container concurrency is too large")
        return self

    @field_validator("packs")
    @classmethod
    def unique_ordered_packs(
        cls,
        value: tuple[BenchmarkPack, ...],
    ) -> tuple[BenchmarkPack, ...]:
        ordered = tuple(sorted(value, key=lambda item: item.pack_id))
        identities = tuple(item.pack_id for item in ordered)
        if len(identities) != len(set(identities)):
            raise ValueError("benchmark pack identities must be unique")
        return ordered

    @field_validator("models")
    @classmethod
    def unique_ordered_models(
        cls,
        value: tuple[BenchmarkModel, ...],
    ) -> tuple[BenchmarkModel, ...]:
        ordered = tuple(sorted(value, key=lambda item: item.model_id))
        identities = tuple(item.model_id for item in ordered)
        if len(identities) != len(set(identities)):
            raise ValueError("benchmark model identities must be unique")
        return ordered

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json")) + "\n"

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json.encode()).hexdigest()


class BenchmarkDatasetBinding(ContractModel):
    """One host-selected MLflow dataset name for a study pack."""

    pack_id: SafeName
    dataset_name: SafeDatasetName


class BenchmarkRuntime(ContractModel):
    """Non-secret host bindings deliberately outside study identity."""

    format: Literal["dsa-benchmark-runtime/v1"]
    workspace_root: Path
    datasets: tuple[BenchmarkDatasetBinding, ...] = Field(min_length=1, max_length=64)

    @field_validator("datasets")
    @classmethod
    def unique_ordered_datasets(
        cls,
        value: tuple[BenchmarkDatasetBinding, ...],
    ) -> tuple[BenchmarkDatasetBinding, ...]:
        ordered = tuple(sorted(value, key=lambda item: item.pack_id))
        identities = tuple(item.pack_id for item in ordered)
        if len(identities) != len(set(identities)):
            raise ValueError("runtime dataset identities must be unique")
        dataset_names = tuple(item.dataset_name for item in ordered)
        if len(dataset_names) != len(set(dataset_names)):
            raise ValueError("runtime dataset names must be unique across packs")
        return ordered

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json")) + "\n"


class BenchmarkCell(ContractModel):
    """One stable pack-model-repetition cell."""

    cell_id: str = Field(pattern=r"^cell-[0-9a-f]{64}$")
    study_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pack_id: SafeName
    model_id: SafeName
    repetition: Annotated[int, Field(ge=0)]


class BenchmarkCellInvocation(ContractModel):
    """One complete secret-free payload accepted by an isolated worker."""

    format: Literal["dsa-benchmark-cell-invocation/v1"] = (
        "dsa-benchmark-cell-invocation/v1"
    )
    cell: BenchmarkCell
    pack_reference: HuggingFacePackReference
    dataset_name: SafeDatasetName
    model_configuration: ModelConfiguration
    policy: RunPolicy
    docker_image: ImmutableImage
    case_workers: PositiveBounded
    case_count: Annotated[int, Field(gt=0, le=10_000)]
    agent_revision: GitRevision
    workspace_root: Path
    attempt: Annotated[int, Field(gt=0)]
    cleanup_token: str = Field(pattern=r"^[0-9a-f]{32}$")

    @model_validator(mode="after")
    def cleanup_container_count_is_bounded(self) -> BenchmarkCellInvocation:
        if self.case_workers * self.policy.max_tool_calls > _MAX_CELL_CONTAINERS:
            raise ValueError("benchmark cell container concurrency is too large")
        return self

    @property
    def cell_directory(self) -> Path:
        return self.workspace_root / self.cell.cell_id

    @property
    def attempt_directory(self) -> Path:
        return self.cell_directory / "attempts" / f"attempt-{self.attempt:04d}"

    @property
    def receipt_path(self) -> Path:
        return self.cell_directory / "cell.json"

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json")) + "\n"

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedBenchmark,
        cell: BenchmarkCell,
        attempt: int,
    ) -> BenchmarkCellInvocation:
        selected = _canonical_prepared(prepared)
        canonical_cell = BenchmarkCell.model_validate_json(cell.model_dump_json())
        if canonical_cell not in selected.cells:
            raise ValueError("benchmark cell is not part of the prepared study")
        return _invocation_from_canonical_prepared(selected, canonical_cell, attempt)


class BenchmarkCellReceipt(ContractModel):
    """Canonical local correlation for one completed MLflow evaluation cell."""

    format: Literal["dsa-benchmark-cell/v1"] = "dsa-benchmark-cell/v1"
    study_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cell_id: str = Field(pattern=r"^cell-[0-9a-f]{64}$")
    pack_id: SafeName
    model_id: SafeName
    repetition: Annotated[int, Field(ge=0)]
    agent_revision: GitRevision
    docker_image: ImmutableImage
    dataset_name: SafeDatasetName
    dataset_id: SafeRemoteId
    dataset_digest: SafeRemoteId
    evaluation_run_id: SafeRemoteId
    predictions: tuple[MlflowEvaluationPrediction, ...]

    @field_validator("predictions")
    @classmethod
    def prediction_identities_are_unique(
        cls,
        value: tuple[MlflowEvaluationPrediction, ...],
    ) -> tuple[MlflowEvaluationPrediction, ...]:
        identities = tuple(item.case_id for item in value)
        if len(identities) != len(set(identities)):
            raise ValueError("receipt prediction identities must be unique")
        return value

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json")) + "\n"

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json.encode()).hexdigest()


class RetainedBenchmarkReceipt(ContractModel):
    """Exact local path and digest of one retained cell receipt."""

    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BenchmarkWorkerCompleted(ContractModel):
    status: Literal["completed"] = "completed"
    cell_id: str = Field(pattern=r"^cell-[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BenchmarkWorkerFailed(ContractModel):
    status: Literal["failed"] = "failed"
    cell_id: str = Field(pattern=r"^cell-[0-9a-f]{64}$")
    failure_code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")


class BenchmarkWorkerAmbiguous(ContractModel):
    status: Literal["ambiguous"] = "ambiguous"
    cell_id: str = Field(pattern=r"^cell-[0-9a-f]{64}$")
    failure_code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")


BenchmarkWorkerResult = Annotated[
    BenchmarkWorkerCompleted | BenchmarkWorkerFailed | BenchmarkWorkerAmbiguous,
    Field(discriminator="status"),
]


class BenchmarkCellRunResult(ContractModel):
    """Parent-visible terminal state for one selected cell."""

    cell_id: str = Field(pattern=r"^cell-[0-9a-f]{64}$")
    status: Literal["completed", "skipped", "failed", "ambiguous"]
    receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,127}$",
    )

    @model_validator(mode="after")
    def terminal_fields_match_status(self) -> BenchmarkCellRunResult:
        if self.status in {"completed", "skipped"}:
            if self.receipt_sha256 is None or self.failure_code is not None:
                raise ValueError("completed cells require only a receipt digest")
        elif self.failure_code is None or self.receipt_sha256 is not None:
            raise ValueError("incomplete cells require only a failure code")
        return self


class BenchmarkPlan(ContractModel):
    """Canonical side-effect-free description of one preflighted matrix."""

    format: Literal["dsa-benchmark-plan/v1"] = "dsa-benchmark-plan/v1"
    study_id: SafeName
    study_version: str
    study_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cells: tuple[BenchmarkCell, ...] = Field(min_length=1, max_length=10_000)

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json")) + "\n"


CellInvoker = Callable[
    [BenchmarkCellInvocation],
    Awaitable[BenchmarkWorkerResult],
]


PackLoader = Callable[[HuggingFacePackReference], LoadedEvaluationPack]
ImageVerifier = Callable[[str], Awaitable[None]]
EvaluationRunner = Callable[..., MlflowEvaluationResult]
ContainerCleaner = Callable[[BenchmarkCellInvocation], Awaitable[bool]]

_WORKER_ARGUMENT = "--cell-worker"
_WORKER_PREFIX = b"DSA_BENCHMARK_CELL_RESULT="
_WORKER_STDOUT_BYTES = 1024 * 1024
_WORKER_STDERR_BYTES = 16 * 1024
_WORKER_INPUT_BYTES = 1024 * 1024
_WORKER_STOP_SECONDS = 15
_RECEIPT_BYTES = 64 * 1024 * 1024
_DOCKER_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DOCKER_CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}\Z")
_BENCHMARK_CLEANUP_LABEL = "dsa.benchmark.cleanup"
_MAX_CELL_CONTAINERS = 10_000
_CONTAINER_ID_BYTES = 65
_CONTAINER_REMOVE_BATCH = 128


@dataclass(frozen=True)
class PreparedBenchmark:
    """A completely preflighted matrix and its verified pack content."""

    study: BenchmarkStudy
    runtime: BenchmarkRuntime
    cells: tuple[BenchmarkCell, ...]
    packs: tuple[tuple[str, LoadedEvaluationPack], ...]


def _invocation_from_canonical_prepared(
    prepared: PreparedBenchmark,
    cell: BenchmarkCell,
    attempt: int,
) -> BenchmarkCellInvocation:
    packs = {item.pack_id: item for item in prepared.study.packs}
    models = {item.model_id: item for item in prepared.study.models}
    datasets = {item.pack_id: item.dataset_name for item in prepared.runtime.datasets}
    loaded = dict(prepared.packs)
    selected_pack = packs[cell.pack_id]
    selected_model = models[cell.model_id]
    return BenchmarkCellInvocation(
        cell=cell,
        pack_reference=selected_pack.reference,
        dataset_name=datasets[cell.pack_id],
        model_configuration=selected_model.configuration,
        policy=prepared.study.policy,
        docker_image=prepared.study.execution.docker_image,
        case_workers=prepared.study.execution.case_workers,
        case_count=loaded[cell.pack_id].manifest.cases.case_count,
        agent_revision=prepared.study.agent_revision,
        workspace_root=prepared.runtime.workspace_root,
        attempt=attempt,
        cleanup_token=uuid4().hex,
    )


def retain_benchmark_cell_receipt(
    receipt: BenchmarkCellReceipt | object,
    path: Path,
) -> RetainedBenchmarkReceipt:
    """Atomically publish canonical receipt bytes without overwriting a conflict."""
    selected = _canonical_receipt(receipt)
    content = selected.canonical_json.encode()
    if len(content) > _RECEIPT_BYTES:
        raise ValueError("benchmark cell receipt exceeds its retained bound")
    digest = sha256(content).hexdigest()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.parent / f".cell.{uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = None
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            retained = read_benchmark_cell_receipt(path)
            if retained != selected:
                raise ValueError("benchmark cell receipt conflict") from None
        _fsync_directory(path.parent)
    except ValueError:
        raise
    except OSError:
        raise ValueError("benchmark cell receipt unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return RetainedBenchmarkReceipt(path=path, sha256=digest)


def read_benchmark_cell_receipt(path: Path) -> BenchmarkCellReceipt:
    """Read and revalidate one bounded canonical regular receipt file."""
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > _RECEIPT_BYTES
        ):
            raise ValueError("benchmark cell receipt unavailable")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = None
            content = source.read(_RECEIPT_BYTES + 1)
    except ValueError:
        raise
    except OSError:
        raise ValueError("benchmark cell receipt unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        receipt = BenchmarkCellReceipt.model_validate_json(content)
    except Exception:
        raise ValueError("benchmark cell receipt invalid") from None
    if content != receipt.canonical_json.encode():
        raise ValueError("benchmark cell receipt invalid")
    return receipt


async def run_prepared_benchmark(
    prepared: PreparedBenchmark,
    *,
    resume: bool = False,
    invoker: CellInvoker | None = None,
) -> tuple[BenchmarkCellRunResult, ...]:
    """Run independent cells under the study's explicit process bound."""
    selected = _canonical_prepared(prepared)
    root = selected.runtime.workspace_root
    if root.exists() and not resume:
        raise ValueError("benchmark workspace already exists")
    if root.is_symlink():
        raise ValueError("benchmark workspace must not be a symlink")
    root.mkdir(mode=0o700, parents=True, exist_ok=resume)
    selected_invoker = invoker or invoke_benchmark_cell
    semaphore = asyncio.Semaphore(selected.study.execution.cell_workers)
    results: list[BenchmarkCellRunResult | None] = [None] * len(selected.cells)

    async def run_cell(index: int, cell: BenchmarkCell) -> None:
        async with semaphore:
            results[index] = await _run_prepared_cell(
                selected,
                cell,
                resume=resume,
                invoker=selected_invoker,
            )

    async with asyncio.TaskGroup() as group:
        for index, cell in enumerate(selected.cells):
            group.create_task(run_cell(index, cell))
    if any(item is None for item in results):
        raise RuntimeError("benchmark runner lost a cell result")
    return tuple(item for item in results if item is not None)


async def _run_prepared_cell(
    prepared: PreparedBenchmark,
    cell: BenchmarkCell,
    *,
    resume: bool,
    invoker: CellInvoker,
) -> BenchmarkCellRunResult:
    cell_directory = prepared.runtime.workspace_root / cell.cell_id
    receipt_path = cell_directory / "cell.json"
    if os.path.lexists(receipt_path):
        if not resume:
            return BenchmarkCellRunResult(
                cell_id=cell.cell_id,
                status="ambiguous",
                failure_code="cell_receipt_unexpected",
            )
        try:
            receipt = read_benchmark_cell_receipt(receipt_path)
        except ValueError:
            return BenchmarkCellRunResult(
                cell_id=cell.cell_id,
                status="ambiguous",
                failure_code="cell_receipt_invalid",
            )
        if not _receipt_matches_cell(receipt, prepared, cell):
            return BenchmarkCellRunResult(
                cell_id=cell.cell_id,
                status="ambiguous",
                failure_code="cell_receipt_mismatch",
            )
        return BenchmarkCellRunResult(
            cell_id=cell.cell_id,
            status="skipped",
            receipt_sha256=receipt.sha256,
        )

    attempts = cell_directory / "attempts"
    existing = _existing_attempts(attempts)
    if existing and not all(
        _valid_attempt_failure(path / "failure.json", cell.cell_id) for path in existing
    ):
        return BenchmarkCellRunResult(
            cell_id=cell.cell_id,
            status="ambiguous",
            failure_code="cell_attempt_ambiguous",
        )
    attempt = len(existing) + 1
    invocation = _invocation_from_canonical_prepared(prepared, cell, attempt)
    invocation.attempt_directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        worker_result = await invoker(invocation)
    except asyncio.CancelledError:
        raise
    except Exception:
        return BenchmarkCellRunResult(
            cell_id=cell.cell_id,
            status="ambiguous",
            failure_code="cell_worker_unavailable",
        )
    if isinstance(worker_result, BenchmarkWorkerFailed):
        try:
            _retain_attempt_failure(invocation, worker_result.failure_code)
        except (OSError, ValueError):
            return BenchmarkCellRunResult(
                cell_id=cell.cell_id,
                status="ambiguous",
                failure_code="cell_failure_record_unavailable",
            )
        return BenchmarkCellRunResult(
            cell_id=cell.cell_id,
            status="failed",
            failure_code=worker_result.failure_code,
        )
    if isinstance(worker_result, BenchmarkWorkerAmbiguous):
        return BenchmarkCellRunResult(
            cell_id=cell.cell_id,
            status="ambiguous",
            failure_code=worker_result.failure_code,
        )
    try:
        receipt = read_benchmark_cell_receipt(invocation.receipt_path)
    except ValueError:
        return BenchmarkCellRunResult(
            cell_id=cell.cell_id,
            status="ambiguous",
            failure_code="cell_receipt_missing",
        )
    if (
        worker_result.cell_id != cell.cell_id
        or worker_result.receipt_sha256 != receipt.sha256
        or not _receipt_matches_cell(receipt, prepared, cell)
    ):
        return BenchmarkCellRunResult(
            cell_id=cell.cell_id,
            status="ambiguous",
            failure_code="cell_worker_result_mismatch",
        )
    return BenchmarkCellRunResult(
        cell_id=cell.cell_id,
        status="completed",
        receipt_sha256=receipt.sha256,
    )


def _existing_attempts(directory: Path) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    if directory.is_symlink() or not directory.is_dir():
        return (directory,)
    values = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    expected = tuple(f"attempt-{index:04d}" for index in range(1, len(values) + 1))
    if tuple(path.name for path in values) != expected or any(
        path.is_symlink() or not path.is_dir() for path in values
    ):
        return (directory,)
    return values


def _retain_attempt_failure(invocation: BenchmarkCellInvocation, code: str) -> None:
    failure = BenchmarkWorkerFailed(cell_id=invocation.cell.cell_id, failure_code=code)
    content = (_canonical_json(failure.model_dump(mode="json")) + "\n").encode()
    path = invocation.attempt_directory / "failure.json"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as destination:
        destination.write(content)
        destination.flush()
        os.fsync(destination.fileno())
    _fsync_directory(invocation.attempt_directory)


def _valid_attempt_failure(path: Path, cell_id: str) -> bool:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= 1024:
            return False
        with os.fdopen(descriptor, "rb") as source:
            descriptor = None
            content = source.read(1025)
        failure = BenchmarkWorkerFailed.model_validate_json(content)
        canonical = (_canonical_json(failure.model_dump(mode="json")) + "\n").encode()
        return content == canonical and failure.cell_id == cell_id
    except Exception:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _receipt_matches_cell(
    receipt: BenchmarkCellReceipt,
    prepared: PreparedBenchmark,
    cell: BenchmarkCell,
) -> bool:
    datasets = {item.pack_id: item.dataset_name for item in prepared.runtime.datasets}
    loaded = dict(prepared.packs)
    expected_cases = tuple(item.case_id for item in loaded[cell.pack_id].cases)
    return (
        receipt.study_sha256 == prepared.study.sha256
        and receipt.cell_id == cell.cell_id
        and receipt.pack_id == cell.pack_id
        and receipt.model_id == cell.model_id
        and receipt.repetition == cell.repetition
        and receipt.agent_revision == prepared.study.agent_revision
        and receipt.docker_image == prepared.study.execution.docker_image
        and receipt.dataset_name == datasets[cell.pack_id]
        and tuple(item.case_id for item in receipt.predictions) == expected_cases
    )


def _canonical_receipt(value: BenchmarkCellReceipt | object) -> BenchmarkCellReceipt:
    if isinstance(value, BenchmarkCellReceipt):
        return BenchmarkCellReceipt.model_validate_json(value.model_dump_json())
    return BenchmarkCellReceipt.model_validate(value)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


async def verify_docker_image(image: str) -> None:
    """Require an already-present immutable Docker image without pulling it."""
    configuration = default_docker_configuration(image)
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            configuration.docker_executable,
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            image,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=configuration.control_timeout_seconds,
        )
    except TimeoutError:
        if process is not None:
            await _stop_control_process(process)
        raise ValueError("benchmark Docker image is unavailable") from None
    except asyncio.CancelledError:
        if process is not None:
            await _stop_control_process(process)
        raise
    except OSError:
        raise ValueError("benchmark Docker image is unavailable") from None
    image_id = stdout.decode(errors="replace").strip()
    if process.returncode != 0 or _DOCKER_IMAGE_ID.fullmatch(image_id) is None:
        raise ValueError("benchmark Docker image is unavailable")
    if image.startswith("sha256:") and image != image_id:
        raise ValueError("benchmark Docker image identity does not match")


async def invoke_benchmark_cell(
    invocation: BenchmarkCellInvocation,
) -> BenchmarkWorkerResult:
    """Execute one cell in a fresh bounded and cancellable Python process."""
    selected = BenchmarkCellInvocation.model_validate_json(invocation.model_dump_json())
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "dsa.benchmark_worker",
        _WORKER_ARGUMENT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_task = asyncio.create_task(_read_stream_tail(process.stdout, _WORKER_STDOUT_BYTES))
    stderr_task = asyncio.create_task(_read_stream_tail(process.stderr, _WORKER_STDERR_BYTES))
    try:
        process.stdin.write(selected.canonical_json.encode())
        await process.stdin.drain()
        process.stdin.close()
        timeout = (
            selected.policy.max_run_seconds
            * ((selected.case_count + selected.case_workers - 1) // selected.case_workers)
            + 300
        )
        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=timeout)
        stdout, stdout_overflow = await stdout_task
        _stderr, _stderr_overflow = await stderr_task
    except BaseException:
        await stop_benchmark_cell_worker(process, selected)
        for task in (stdout_task, stderr_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise
    if process.returncode != 0 or stdout_overflow:
        raise RuntimeError("benchmark cell worker failed")
    framed = [
        line[len(_WORKER_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(_WORKER_PREFIX)
    ]
    if len(framed) != 1:
        raise RuntimeError("benchmark cell worker returned an invalid result")
    try:
        raw = json.loads(framed[0])
        if not isinstance(raw, dict):
            raise ValueError("worker result must be an object")
        mapping = cast(dict[str, object], raw)
        result: BenchmarkWorkerResult
        if mapping.get("status") == "completed":
            result = BenchmarkWorkerCompleted.model_validate(mapping)
        elif mapping.get("status") == "failed":
            result = BenchmarkWorkerFailed.model_validate(mapping)
        elif mapping.get("status") == "ambiguous":
            result = BenchmarkWorkerAmbiguous.model_validate(mapping)
        else:
            raise ValueError("worker result has an unknown status")
    except Exception:
        raise RuntimeError("benchmark cell worker returned an invalid result") from None
    if result.cell_id != selected.cell.cell_id:
        raise RuntimeError("benchmark cell worker returned an invalid result")
    return result


async def _read_stream_tail(
    stream: asyncio.StreamReader,
    limit: int,
) -> tuple[bytes, bool]:
    value = bytearray()
    total = 0
    while chunk := await stream.read(64 * 1024):
        total += len(chunk)
        value.extend(chunk)
        if len(value) > limit:
            del value[: len(value) - limit]
    return bytes(value), total > limit


async def _stop_control_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        await process.wait()
        return
    process.terminate()
    try:
        await asyncio.wait_for(
            asyncio.shield(process.wait()),
            timeout=_WORKER_STOP_SECONDS,
        )
    except TimeoutError:
        process.kill()
        await process.wait()


async def stop_benchmark_cell_worker(
    process: asyncio.subprocess.Process,
    invocation: BenchmarkCellInvocation,
    *,
    cleaner: ContainerCleaner | None = None,
    grace_seconds: float = _WORKER_STOP_SECONDS,
    settle_seconds: float | None = None,
) -> None:
    """Stop one worker and remove only its labeled Docker containers."""
    selected_cleaner = cleaner or remove_benchmark_cell_containers
    if process.returncode is not None:
        await process.wait()
        await _attempt_container_cleanup(selected_cleaner, invocation)
        return
    _signal_worker(process, signal.SIGINT)
    wait_task = asyncio.create_task(process.wait())
    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout=grace_seconds)
    except TimeoutError:
        _signal_worker(process, signal.SIGSTOP)
        await _attempt_container_cleanup(selected_cleaner, invocation)
        _signal_worker(process, signal.SIGKILL)
        await wait_task
        selected_settle_seconds = (
            default_docker_configuration(
                invocation.docker_image
            ).control_timeout_seconds
            if settle_seconds is None
            else settle_seconds
        )
        if selected_settle_seconds > 0:
            await asyncio.sleep(selected_settle_seconds)
    await _attempt_container_cleanup(selected_cleaner, invocation)


async def _attempt_container_cleanup(
    cleaner: ContainerCleaner,
    invocation: BenchmarkCellInvocation,
) -> bool:
    try:
        return await cleaner(invocation)
    except Exception:
        return False


def _signal_worker(process: asyncio.subprocess.Process, selected_signal: int) -> None:
    with suppress(ProcessLookupError):
        process.send_signal(selected_signal)


async def remove_benchmark_cell_containers(
    invocation: BenchmarkCellInvocation,
    *,
    runner: DockerCommandRunner | None = None,
) -> bool:
    """Force-remove the bounded set of containers labeled for one cell attempt."""
    configuration = default_docker_configuration(invocation.docker_image)
    selected_runner = runner or AsyncSubprocessDockerRunner()
    container_bound = invocation.case_workers * invocation.policy.max_tool_calls
    try:
        listed = await selected_runner.run(
            (
                configuration.docker_executable,
                "container",
                "ls",
                "--all",
                "--quiet",
                "--filter",
                f"label={_BENCHMARK_CLEANUP_LABEL}={invocation.cleanup_token}",
            ),
            input_bytes=None,
            timeout_seconds=configuration.control_timeout_seconds,
            output_limit=container_bound * _CONTAINER_ID_BYTES,
        )
    except OSError:
        return False
    if (
        listed.timed_out
        or listed.returncode != 0
        or listed.stdout_truncated
        or listed.stderr_truncated
    ):
        return False
    try:
        container_ids = tuple(line.decode("ascii") for line in listed.stdout.splitlines())
    except UnicodeDecodeError:
        return False
    if len(container_ids) > container_bound or any(
        _DOCKER_CONTAINER_ID.fullmatch(container_id) is None
        for container_id in container_ids
    ):
        return False
    if not container_ids:
        return True
    complete = True
    for start in range(0, len(container_ids), _CONTAINER_REMOVE_BATCH):
        batch = container_ids[start : start + _CONTAINER_REMOVE_BATCH]
        try:
            removed = await selected_runner.run(
                (
                    configuration.docker_executable,
                    "rm",
                    "--force",
                    *batch,
                ),
                input_bytes=None,
                timeout_seconds=configuration.control_timeout_seconds,
                output_limit=len(batch) * _CONTAINER_ID_BYTES,
            )
        except OSError:
            complete = False
            continue
        if removed.timed_out or removed.returncode != 0:
            complete = False
    return complete


def execute_benchmark_cell(
    invocation: BenchmarkCellInvocation | object,
    *,
    pack_loader: PackLoader = load_huggingface_evaluation_pack,
    evaluation_runner: EvaluationRunner = run_mlflow_evaluation,
) -> BenchmarkWorkerResult:
    """Run one preflighted cell and publish its correlation receipt."""
    try:
        selected = (
            BenchmarkCellInvocation.model_validate_json(invocation.model_dump_json())
            if isinstance(invocation, BenchmarkCellInvocation)
            else BenchmarkCellInvocation.model_validate(invocation)
        )
        _validate_worker_workspace(selected)
        loaded_pack = pack_loader(selected.pack_reference)
        pack = LoadedEvaluationPack.model_validate_json(loaded_pack.model_dump_json())
        if (
            pack.reference != selected.pack_reference
            or pack.manifest.cases.case_count != selected.case_count
        ):
            raise ValueError("benchmark pack changed after preflight")
        executor = DockerPythonExecutor(
            default_docker_configuration(selected.docker_image),
            container_labels={_BENCHMARK_CLEANUP_LABEL: selected.cleanup_token},
        )
        tags = {
            "dsa.benchmark.agent_revision": selected.agent_revision,
            "dsa.benchmark.cell_id": selected.cell.cell_id,
            "dsa.benchmark.model_id": selected.cell.model_id,
            "dsa.benchmark.pack_id": selected.cell.pack_id,
            "dsa.benchmark.repetition": str(selected.cell.repetition),
            "dsa.benchmark.study_sha256": selected.cell.study_sha256,
        }
    except Exception:
        return BenchmarkWorkerFailed(
            cell_id=_invocation_cell_id(invocation),
            failure_code="cell_setup_failed",
        )
    try:
        with _mlflow_worker_environment(selected):
            result = evaluation_runner(
                pack,
                dataset_name=selected.dataset_name,
                runs_directory=selected.attempt_directory / "runs",
                model_configuration=selected.model_configuration,
                policy=selected.policy,
                python_executor=executor,
                run_tags=tags,
            )
    except MlflowEvaluationError as error:
        return BenchmarkWorkerAmbiguous(
            cell_id=selected.cell.cell_id,
            failure_code=f"cell_{error.code}",
        )
    except Exception:
        return BenchmarkWorkerAmbiguous(
            cell_id=selected.cell.cell_id,
            failure_code="cell_evaluation_ambiguous",
        )
    try:
        expected_cases = tuple(item.case_id for item in pack.cases)
        returned_cases = tuple(item.case_id for item in result.predictions)
        if (
            result.evaluation_run_id is None
            or len(result.predictions) != selected.case_count
            or returned_cases != expected_cases
        ):
            raise ValueError("benchmark evaluation result is incomplete")
        retained = retain_benchmark_cell_receipt(
            BenchmarkCellReceipt(
                study_sha256=selected.cell.study_sha256,
                cell_id=selected.cell.cell_id,
                pack_id=selected.cell.pack_id,
                model_id=selected.cell.model_id,
                repetition=selected.cell.repetition,
                agent_revision=selected.agent_revision,
                docker_image=selected.docker_image,
                dataset_name=selected.dataset_name,
                dataset_id=result.dataset_id,
                dataset_digest=result.dataset_digest,
                evaluation_run_id=result.evaluation_run_id,
                predictions=result.predictions,
            ),
            selected.receipt_path,
        )
        return BenchmarkWorkerCompleted(
            cell_id=selected.cell.cell_id,
            receipt_sha256=retained.sha256,
        )
    except Exception:
        return BenchmarkWorkerAmbiguous(
            cell_id=selected.cell.cell_id,
            failure_code="cell_receipt_unavailable",
        )


def _invocation_cell_id(invocation: object) -> str:
    if isinstance(invocation, BenchmarkCellInvocation):
        return invocation.cell.cell_id
    return "cell-" + "0" * 64


def _validate_worker_workspace(invocation: BenchmarkCellInvocation) -> None:
    root = invocation.workspace_root
    expected = (
        root
        / invocation.cell.cell_id
        / "attempts"
        / f"attempt-{invocation.attempt:04d}"
    )
    if invocation.attempt_directory != expected:
        raise ValueError("benchmark attempt path is invalid")
    for path in (
        root,
        invocation.cell_directory,
        invocation.attempt_directory.parent,
        invocation.attempt_directory,
    ):
        if path.is_symlink() or not path.is_dir():
            raise ValueError("benchmark attempt workspace is unavailable")
    if os.path.lexists(invocation.receipt_path):
        raise ValueError("benchmark cell receipt already exists")


@contextmanager
def _mlflow_worker_environment(invocation: BenchmarkCellInvocation):
    values = {
        "MLFLOW_GENAI_EVAL_MAX_WORKERS": str(invocation.case_workers),
        "MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT": str(invocation.policy.max_run_seconds + 30),
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def benchmark_cell_worker_main() -> int:
    """Read one bounded cell invocation and emit its framed terminal result."""
    content = sys.stdin.buffer.read(_WORKER_INPUT_BYTES + 1)
    if not content or len(content) > _WORKER_INPUT_BYTES:
        return 1
    try:
        invocation = BenchmarkCellInvocation.model_validate_json(content)
    except Exception:
        return 1
    if content != invocation.canonical_json.encode():
        return 1
    result = execute_benchmark_cell(invocation)
    framed = _canonical_json(result.model_dump(mode="json"))
    print(f"{_WORKER_PREFIX.decode()}{framed}", flush=True)
    return 0
def expand_benchmark_study(
    study: BenchmarkStudy | object,
) -> tuple[BenchmarkCell, ...]:
    """Expand a revalidated study into its deterministic Cartesian product."""
    selected = _canonical_study(study)
    cells: list[BenchmarkCell] = []
    for pack in selected.packs:
        for model in selected.models:
            for repetition in range(selected.repetitions):
                identity = {
                    "model_id": model.model_id,
                    "pack_id": pack.pack_id,
                    "repetition": repetition,
                    "study_sha256": selected.sha256,
                }
                cell_id = f"cell-{sha256(_canonical_json(identity).encode()).hexdigest()}"
                cells.append(
                    BenchmarkCell(
                        cell_id=cell_id,
                        study_sha256=selected.sha256,
                        pack_id=pack.pack_id,
                        model_id=model.model_id,
                        repetition=repetition,
                    )
                )
    return tuple(cells)


async def prepare_benchmark(
    study: BenchmarkStudy | object,
    runtime: BenchmarkRuntime | object,
    *,
    current_revision: str,
    environment: Mapping[str, str] = os.environ,
    pack_loader: PackLoader = load_huggingface_evaluation_pack,
    image_verifier: ImageVerifier | None = None,
    resume: bool = False,
) -> PreparedBenchmark:
    """Validate every cheap boundary before resolving packs or inspecting Docker."""
    selected_study = _canonical_study(study)
    selected_runtime = _canonical_runtime(runtime)
    if current_revision != selected_study.agent_revision:
        raise ValueError("benchmark agent revision does not match the current revision")
    try:
        mlflow_configuration = load_mlflow_configuration(environment)
    except MlflowConfigurationError:
        raise ValueError("benchmark requires complete MLflow configuration") from None
    if any(
        not dataset_name_matches_backend(item.dataset_name, mlflow_configuration)
        for item in selected_runtime.datasets
    ):
        raise ValueError("benchmark dataset name does not match the MLflow backend")
    expected = tuple(item.pack_id for item in selected_study.packs)
    actual = tuple(item.pack_id for item in selected_runtime.datasets)
    if actual != expected:
        raise ValueError("runtime dataset bindings must exactly cover study packs")
    if not selected_runtime.workspace_root.is_absolute():
        raise ValueError("benchmark workspace must be absolute")
    if selected_runtime.workspace_root.exists() and not resume:
        raise ValueError("benchmark workspace already exists")
    if selected_runtime.workspace_root.is_symlink():
        raise ValueError("benchmark workspace must not be a symlink")

    loaded: list[tuple[str, LoadedEvaluationPack]] = []
    for item in selected_study.packs:
        pack = pack_loader(item.reference)
        selected_pack = LoadedEvaluationPack.model_validate_json(pack.model_dump_json())
        if selected_pack.reference != item.reference:
            raise ValueError("loaded benchmark pack does not match the study")
        loaded.append((item.pack_id, selected_pack))
    await (image_verifier or verify_docker_image)(
        selected_study.execution.docker_image
    )
    return PreparedBenchmark(
        study=selected_study,
        runtime=selected_runtime,
        cells=expand_benchmark_study(selected_study),
        packs=tuple(loaded),
    )


def benchmark_plan(prepared: PreparedBenchmark) -> BenchmarkPlan:
    """Project a preflighted matrix without retaining host runtime paths."""
    selected = _canonical_prepared(prepared)
    return BenchmarkPlan(
        study_id=selected.study.study_id,
        study_version=selected.study.version,
        study_sha256=selected.study.sha256,
        cells=selected.cells,
    )


def _canonical_prepared(value: object) -> PreparedBenchmark:
    if not isinstance(value, PreparedBenchmark):
        raise ValueError("benchmark was not preflighted")
    study = _canonical_study(value.study)
    runtime = _canonical_runtime(value.runtime)
    if not runtime.workspace_root.is_absolute():
        raise ValueError("benchmark workspace must be absolute")
    cells = tuple(
        BenchmarkCell.model_validate_json(item.model_dump_json()) for item in value.cells
    )
    if cells != expand_benchmark_study(study):
        raise ValueError("prepared benchmark cells do not match the study")
    expected_pack_ids = tuple(item.pack_id for item in study.packs)
    if tuple(item.pack_id for item in runtime.datasets) != expected_pack_ids:
        raise ValueError("prepared benchmark runtime does not match the study")
    loaded: list[tuple[str, LoadedEvaluationPack]] = []
    for pack_id, pack in value.packs:
        selected_pack = LoadedEvaluationPack.model_validate_json(pack.model_dump_json())
        loaded.append((pack_id, selected_pack))
    if tuple(pack_id for pack_id, _pack in loaded) != expected_pack_ids:
        raise ValueError("prepared benchmark packs do not match the study")
    expected_references = {item.pack_id: item.reference for item in study.packs}
    if any(pack.reference != expected_references[pack_id] for pack_id, pack in loaded):
        raise ValueError("prepared benchmark pack reference does not match the study")
    return PreparedBenchmark(
        study=study,
        runtime=runtime,
        cells=cells,
        packs=tuple(loaded),
    )


def _canonical_study(value: BenchmarkStudy | object) -> BenchmarkStudy:
    if isinstance(value, BenchmarkStudy):
        return BenchmarkStudy.model_validate_json(value.model_dump_json())
    return BenchmarkStudy.model_validate(value)


def _canonical_runtime(value: BenchmarkRuntime | object) -> BenchmarkRuntime:
    if isinstance(value, BenchmarkRuntime):
        return BenchmarkRuntime.model_validate_json(value.model_dump_json())
    return BenchmarkRuntime.model_validate(value)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
