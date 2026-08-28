"""Read-only immutable publication of completed benchmark studies."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, cast
from uuid import uuid4

from pydantic import Field, ValidationInfo, field_validator, model_validator

from dsa.benchmark import (
    BenchmarkCell,
    BenchmarkCellReceipt,
    BenchmarkRuntime,
    BenchmarkStudy,
    GitRevision,
    PackLoader,
    SafeDatasetName,
    SafeName,
    SafeRemoteId,
    expand_benchmark_study,
)
from dsa.contract import ContractModel
from dsa.evaluation import (
    MlflowEvaluationPrediction,
    agent_failure,
    conditional_exact_json,
    end_to_end_exact_success,
    exact_json_equal,
    infrastructure_failure,
)
from dsa.pack import LoadedEvaluationPack, load_huggingface_evaluation_pack
from dsa.record import FailureStage, RunFailure, RunSuccess, TerminalRecord
from dsa.reporting import MlflowReporting

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]

_EVALUATION_METRICS = (
    "agent_failure/mean",
    "conditional_exact_json/mean",
    "end_to_end_exact_success/mean",
    "infrastructure_failure/mean",
)
_ANALYSIS_METRICS = (
    "dsa.elapsed_seconds",
    "dsa.usage.input_tokens",
    "dsa.usage.output_tokens",
    "dsa.usage.requests",
    "dsa.usage.total_tokens",
)
_RUN_PARAMS = ("dsa.model_name",)
_RUN_TAGS = (
    "dsa.component",
    "dsa.benchmark.agent_revision",
    "dsa.benchmark.cell_id",
    "dsa.benchmark.model_id",
    "dsa.benchmark.pack_id",
    "dsa.benchmark.repetition",
    "dsa.benchmark.study_sha256",
    "dsa.outcome",
    "dsa.run_id",
)
_REPORT_JSON_BYTES = 256 * 1024 * 1024
_REPORT_MARKDOWN_BYTES = 64 * 1024 * 1024
_MAX_REPORT_CASE_EXECUTIONS = 100_000
_RECEIPT_BYTES = 64 * 1024 * 1024
_TERMINAL_BYTES = 64 * 1024 * 1024
_MAX_CELL_ATTEMPTS = 10_000


class BenchmarkReportConfigurationError(ValueError):
    """A rejected local report invocation before evidence publication."""


class BenchmarkRate(ContractModel):
    """One exact count ratio without a fabricated zero-denominator value."""

    numerator: NonNegativeInt
    denominator: NonNegativeInt
    rate: NonNegativeFinite | None

    @model_validator(mode="after")
    def rate_matches_counts(self) -> BenchmarkRate:
        if self.numerator > self.denominator:
            raise ValueError("metric numerator exceeds its denominator")
        if self.denominator == 0:
            if self.numerator != 0 or self.rate is not None:
                raise ValueError("zero-denominator metrics must have no rate")
            return self
        expected = self.numerator / self.denominator
        if self.rate is None or not math.isclose(
            self.rate,
            expected,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("metric rate does not match its counts")
        return self


class BenchmarkMetrics(ContractModel):
    """The five deterministic study metrics with explicit denominators."""

    end_to_end_exact_success: BenchmarkRate
    completion_rate: BenchmarkRate
    conditional_exact_accuracy: BenchmarkRate
    agent_failure: BenchmarkRate
    infrastructure_failure: BenchmarkRate


class BenchmarkObservedSummary(ContractModel):
    """Coverage and aggregate for one optional observed MLflow metric."""

    observed_count: NonNegativeInt
    unavailable_count: NonNegativeInt
    total: NonNegativeFinite | None
    mean: NonNegativeFinite | None

    @model_validator(mode="after")
    def values_match_coverage(self) -> BenchmarkObservedSummary:
        if self.observed_count == 0:
            if self.total is not None or self.mean is not None:
                raise ValueError("unobserved metrics must not contain values")
            return self
        if self.total is None or self.mean is None:
            raise ValueError("observed metrics require total and mean")
        if not math.isclose(
            self.mean,
            self.total / self.observed_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("observed metric mean does not match its total")
        return self


class BenchmarkObservations(ContractModel):
    """Observed latency and model usage, each with explicit coverage."""

    elapsed_seconds: BenchmarkObservedSummary
    requests: BenchmarkObservedSummary
    input_tokens: BenchmarkObservedSummary
    output_tokens: BenchmarkObservedSummary
    total_tokens: BenchmarkObservedSummary


class BenchmarkAggregate(ContractModel):
    """Metrics and observations over an exact set of case executions."""

    case_count: NonNegativeInt
    metrics: BenchmarkMetrics
    observations: BenchmarkObservations

    @model_validator(mode="after")
    def denominators_match_case_count(self) -> BenchmarkAggregate:
        all_case_rates = (
            self.metrics.end_to_end_exact_success,
            self.metrics.completion_rate,
            self.metrics.agent_failure,
            self.metrics.infrastructure_failure,
        )
        if any(item.denominator != self.case_count for item in all_case_rates):
            raise ValueError("all-case metric denominator does not match case count")
        if (
            self.metrics.conditional_exact_accuracy.denominator
            != self.metrics.completion_rate.numerator
        ):
            raise ValueError("conditional metric denominator does not match completions")
        for item in self.observations.model_dump(mode="python").values():
            summary = BenchmarkObservedSummary.model_validate(item)
            if summary.observed_count + summary.unavailable_count != self.case_count:
                raise ValueError("observation coverage does not match case count")
        return self


class BenchmarkCaseOutcome(ContractModel):
    """Safe per-case classification without questions, answers, or expectations."""

    case_id: SafeName
    run_id: SafeRemoteId
    terminal_sha256: Sha256
    accepted: bool
    failure_stage: FailureStage | None = None
    failure_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,127}$",
    )
    reporting: MlflowReporting
    end_to_end_exact_success: bool
    conditional_exact_json: bool | None
    agent_failure: bool
    infrastructure_failure: bool

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> BenchmarkCaseOutcome:
        if self.accepted:
            if self.failure_stage is not None or self.failure_code is not None:
                raise ValueError("accepted case outcome contains failure state")
            if self.conditional_exact_json is None:
                raise ValueError("accepted case outcome requires conditional exactness")
        elif (
            self.failure_stage is None
            or self.failure_code is None
            or self.conditional_exact_json is not None
        ):
            raise ValueError("failed case outcome has invalid classification state")
        projection = MlflowEvaluationPrediction(
            case_id=self.case_id,
            run_id=self.run_id,
            terminal_sha256=self.terminal_sha256,
            accepted=self.accepted,
            answer=None,
            failure_stage=self.failure_stage,
            failure_code=self.failure_code,
            reporting=self.reporting,
        )
        expected_infrastructure = infrastructure_failure(projection)
        expected_agent = not expected_infrastructure and (
            not self.accepted or self.conditional_exact_json is False
        )
        expected_end_to_end = (
            self.accepted
            and self.conditional_exact_json is True
            and self.reporting.status == "reported"
        )
        if (
            self.infrastructure_failure != expected_infrastructure
            or self.agent_failure != expected_agent
            or self.end_to_end_exact_success != expected_end_to_end
        ):
            raise ValueError("case scorer outcomes are inconsistent")
        return self


class BenchmarkPackEvidence(ContractModel):
    """Verified portable pack identities needed to interpret one report."""

    pack_id: SafeName
    manifest_sha256: Sha256
    database_id: SafeName
    database_sha256: Sha256
    cases_sha256: Sha256
    case_count: Annotated[int, Field(gt=0, le=10_000)]


class BenchmarkDatasetEvidence(ContractModel):
    """Exact runtime and remote dataset identities for one pack."""

    pack_id: SafeName
    dataset_name: SafeDatasetName
    dataset_id: SafeRemoteId
    dataset_digest: SafeRemoteId


class BenchmarkCellReport(ContractModel):
    """One cell's immutable receipt correlation and derived result."""

    cell_id: str = Field(pattern=r"^cell-[0-9a-f]{64}$")
    pack_id: SafeName
    model_id: SafeName
    repetition: NonNegativeInt
    receipt_sha256: Sha256
    evaluation_run_id: SafeRemoteId
    case_count: NonNegativeInt
    cases: tuple[BenchmarkCaseOutcome, ...]
    metrics: BenchmarkMetrics
    observations: BenchmarkObservations

    @model_validator(mode="after")
    def cases_match_cell_summary(self) -> BenchmarkCellReport:
        if len(self.cases) != self.case_count:
            raise ValueError("cell case count does not match its outcomes")
        if len({item.case_id for item in self.cases}) != len(self.cases):
            raise ValueError("cell case identities must be unique")
        BenchmarkAggregate(
            case_count=self.case_count,
            metrics=self.metrics,
            observations=self.observations,
        )
        expected = _metrics_from_outcomes(self.cases)
        if self.metrics != expected:
            raise ValueError("cell metrics do not match its case outcomes")
        return self


class BenchmarkModelAggregate(ContractModel):
    model_id: SafeName
    result: BenchmarkAggregate


class BenchmarkPackModelAggregate(ContractModel):
    pack_id: SafeName
    model_id: SafeName
    result: BenchmarkAggregate


class BenchmarkUnavailableCost(ContractModel):
    status: Literal["unavailable"] = "unavailable"


class BenchmarkReport(ContractModel):
    """Canonical immutable report for one complete benchmark study."""

    format: Literal["dsa-benchmark-report/v1"] = "dsa-benchmark-report/v1"
    reporter_revision: GitRevision
    study_sha256: Sha256
    study: BenchmarkStudy
    packs: tuple[BenchmarkPackEvidence, ...]
    datasets: tuple[BenchmarkDatasetEvidence, ...]
    cells: tuple[BenchmarkCellReport, ...]
    overall: BenchmarkAggregate
    models: tuple[BenchmarkModelAggregate, ...]
    pack_models: tuple[BenchmarkPackModelAggregate, ...]
    provider_cost: BenchmarkUnavailableCost = BenchmarkUnavailableCost()

    @model_validator(mode="after")
    def identities_match_study(self) -> BenchmarkReport:
        if self.study_sha256 != self.study.sha256:
            raise ValueError("report study digest does not match its study")
        pack_ids = tuple(item.pack_id for item in self.study.packs)
        model_ids = tuple(item.model_id for item in self.study.models)
        if tuple(item.pack_id for item in self.packs) != pack_ids:
            raise ValueError("report pack evidence does not match its study")
        if any(
            evidence.manifest_sha256 != study_pack.reference.manifest_sha256
            for evidence, study_pack in zip(self.packs, self.study.packs, strict=True)
        ):
            raise ValueError("report pack manifest evidence does not match its study")
        if tuple(item.pack_id for item in self.datasets) != pack_ids:
            raise ValueError("report dataset evidence does not match its study")
        expected_cells = expand_benchmark_study(self.study)
        if tuple(item.cell_id for item in self.cells) != tuple(
            item.cell_id for item in expected_cells
        ):
            raise ValueError("report cells do not match its study")
        if any(
            (
                report.pack_id,
                report.model_id,
                report.repetition,
            )
            != (cell.pack_id, cell.model_id, cell.repetition)
            for report, cell in zip(self.cells, expected_cells, strict=True)
        ):
            raise ValueError("report cell identities do not match its study")
        for pack in self.packs:
            selected_cells = tuple(
                item for item in self.cells if item.pack_id == pack.pack_id
            )
            if any(item.case_count != pack.case_count for item in selected_cells):
                raise ValueError("report pack case count does not match its cells")
            case_orders = {
                tuple(case.case_id for case in item.cases) for item in selected_cells
            }
            if len(case_orders) != 1:
                raise ValueError("report pack case order conflicts across cells")
        evaluation_ids = tuple(item.evaluation_run_id for item in self.cells)
        if len(evaluation_ids) != len(set(evaluation_ids)):
            raise ValueError("report evaluation run identities must be unique")
        case_outcomes = tuple(case for cell in self.cells for case in cell.cases)
        run_ids = tuple(item.run_id for item in case_outcomes)
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("report analysis run identities must be unique")
        tracking_ids = tuple(
            item.reporting.tracking_run_id
            for item in case_outcomes
            if item.reporting.tracking_run_id is not None
        )
        if len(tracking_ids) != len(set(tracking_ids)):
            raise ValueError("report tracking run identities must be unique")
        trace_ids = tuple(
            item.reporting.trace_id
            for item in case_outcomes
            if item.reporting.trace_id is not None
        )
        if len(trace_ids) != len(set(trace_ids)):
            raise ValueError("report trace identities must be unique")
        if tuple(item.model_id for item in self.models) != model_ids:
            raise ValueError("report model aggregates do not match its study")
        expected_pairs = tuple(
            (pack_id, model_id) for pack_id in pack_ids for model_id in model_ids
        )
        if tuple((item.pack_id, item.model_id) for item in self.pack_models) != expected_pairs:
            raise ValueError("report pack-model aggregates do not match its study")
        if self.overall.case_count != sum(item.case_count for item in self.cells):
            raise ValueError("report overall case count does not match its cells")
        cell_aggregates = tuple(_cell_aggregate(item) for item in self.cells)
        if not _aggregates_equivalent(
            self.overall,
            _combine_aggregates(cell_aggregates),
        ):
            raise ValueError("report overall aggregate does not match its cells")
        for model in self.models:
            expected = _combine_aggregates(
                aggregate
                for cell, aggregate in zip(self.cells, cell_aggregates, strict=True)
                if cell.model_id == model.model_id
            )
            if not _aggregates_equivalent(model.result, expected):
                raise ValueError("report model aggregate does not match its cells")
        for pair in self.pack_models:
            expected = _combine_aggregates(
                aggregate
                for cell, aggregate in zip(self.cells, cell_aggregates, strict=True)
                if cell.pack_id == pair.pack_id and cell.model_id == pair.model_id
            )
            if not _aggregates_equivalent(pair.result, expected):
                raise ValueError("report pack-model aggregate does not match its cells")
        return self

    @property
    def canonical_json(self) -> str:
        selected = BenchmarkReport.model_validate_json(self.model_dump_json())
        return _canonical_json(selected.model_dump(mode="json")) + "\n"

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json.encode()).hexdigest()


class BenchmarkMlflowRun(ContractModel):
    """Narrow safe projection of one exact MLflow run."""

    run_id: SafeRemoteId
    status: str = Field(pattern=r"^[A-Z][A-Z_]{0,31}$")
    lifecycle_stage: Literal["active", "deleted"]
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, str] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("metrics")
    @classmethod
    def finite_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        allowed = set((*_EVALUATION_METRICS, *_ANALYSIS_METRICS))
        if not set(value).issubset(allowed):
            raise ValueError("MLflow run contains unsupported metrics")
        if any(not math.isfinite(item) for item in value.values()):
            raise ValueError("MLflow run metrics must be finite")
        return {key: value[key] for key in sorted(value)}

    @field_validator("params", "tags")
    @classmethod
    def ordered_strings(
        cls,
        value: dict[str, str],
        info: ValidationInfo,
    ) -> dict[str, str]:
        allowed = set(_RUN_PARAMS if info.field_name == "params" else _RUN_TAGS)
        if not set(value).issubset(allowed):
            raise ValueError("MLflow run contains unsupported metadata")
        if any(len(item) > 4096 for item in value.values()):
            raise ValueError("MLflow run metadata exceeds its retained bound")
        return {key: value[key] for key in sorted(value)}


class BenchmarkEvidenceReader(Protocol):
    def get_run(self, run_id: str) -> BenchmarkMlflowRun: ...


class RetainedBenchmarkReport(ContractModel):
    """Exact local identities of both published report representations."""

    report_sha256: Sha256
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class _CaseComputation:
    outcome: BenchmarkCaseOutcome
    observations: dict[str, float]


def build_benchmark_report(
    study: BenchmarkStudy | object,
    runtime: BenchmarkRuntime | object,
    *,
    reporter_revision: str,
    pack_loader: PackLoader = load_huggingface_evaluation_pack,
    evidence_reader: BenchmarkEvidenceReader | None = None,
    environment: Mapping[str, str] = os.environ,
) -> BenchmarkReport:
    """Verify one complete study and derive its immutable report without writes."""
    selected_study = _canonical_study(study)
    selected_runtime = _canonical_runtime(runtime)
    selected_revision = _validate_revision(reporter_revision)
    _validate_runtime(selected_study, selected_runtime)
    reader = evidence_reader or _default_evidence_reader(environment)
    packs: list[tuple[str, LoadedEvaluationPack]] = []
    for selected_pack in selected_study.packs:
        loaded = pack_loader(selected_pack.reference)
        canonical = LoadedEvaluationPack.model_validate_json(loaded.model_dump_json())
        if canonical.reference != selected_pack.reference:
            raise ValueError("loaded report pack does not match the study")
        _verify_pack_database(canonical)
        packs.append((selected_pack.pack_id, canonical))
    cells = expand_benchmark_study(selected_study)
    pack_map = dict(packs)
    total_cases = sum(pack_map[cell.pack_id].manifest.cases.case_count for cell in cells)
    if total_cases > _MAX_REPORT_CASE_EXECUTIONS:
        raise ValueError("benchmark report contains too many case executions")
    receipts = _read_complete_receipts(
        selected_study,
        selected_runtime,
        cells,
        pack_map,
    )
    datasets = _dataset_evidence(selected_study, selected_runtime, receipts)
    model_names = {
        item.model_id: item.configuration.name for item in selected_study.models
    }
    all_computations: list[_CaseComputation] = []
    cell_reports: list[BenchmarkCellReport] = []
    for cell, receipt in zip(cells, receipts, strict=True):
        pack = pack_map[cell.pack_id]
        computations = _verify_and_compute_cell(
            cell,
            receipt,
            pack,
            model_name=model_names[cell.model_id],
            evidence_reader=reader,
        )
        aggregate = _aggregate(computations)
        all_computations.extend(computations)
        cell_reports.append(
            BenchmarkCellReport(
                cell_id=cell.cell_id,
                pack_id=cell.pack_id,
                model_id=cell.model_id,
                repetition=cell.repetition,
                receipt_sha256=receipt.sha256,
                evaluation_run_id=receipt.evaluation_run_id,
                case_count=aggregate.case_count,
                cases=tuple(item.outcome for item in computations),
                metrics=aggregate.metrics,
                observations=aggregate.observations,
            )
        )
    pack_evidence = tuple(
        BenchmarkPackEvidence(
            pack_id=pack_id,
            manifest_sha256=pack.reference.manifest_sha256,
            database_id=pack.manifest.database.id,
            database_sha256=pack.manifest.database.sha256,
            cases_sha256=pack.manifest.cases.sha256,
            case_count=pack.manifest.cases.case_count,
        )
        for pack_id, pack in packs
    )
    cell_computations = _cell_computations(cell_reports, all_computations)
    model_aggregates = tuple(
        BenchmarkModelAggregate(
            model_id=model.model_id,
            result=_aggregate(
                computation
                for report, computations in zip(
                    cell_reports,
                    cell_computations,
                    strict=True,
                )
                if report.model_id == model.model_id
                for computation in computations
            ),
        )
        for model in selected_study.models
    )
    pack_model_aggregates = tuple(
        BenchmarkPackModelAggregate(
            pack_id=pack.pack_id,
            model_id=model.model_id,
            result=_aggregate(
                computation
                for report, computations in zip(
                    cell_reports,
                    cell_computations,
                    strict=True,
                )
                if report.pack_id == pack.pack_id and report.model_id == model.model_id
                for computation in computations
            ),
        )
        for pack in selected_study.packs
        for model in selected_study.models
    )
    return BenchmarkReport(
        reporter_revision=selected_revision,
        study_sha256=selected_study.sha256,
        study=selected_study,
        packs=pack_evidence,
        datasets=datasets,
        cells=tuple(cell_reports),
        overall=_aggregate(all_computations),
        models=model_aggregates,
        pack_models=pack_model_aggregates,
    )


def publish_benchmark_report(
    report: BenchmarkReport | object,
    output_directory: Path,
) -> RetainedBenchmarkReport:
    """Publish exact JSON and Markdown bytes idempotently without overwriting."""
    selected = _canonical_report(report)
    if not output_directory.is_absolute():
        raise ValueError("benchmark report output directory must be absolute")
    try:
        output_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        raise ValueError("benchmark report output directory is unavailable") from None
    basename = f"{selected.study.study_id}-{selected.study_sha256}"
    json_name = f"{basename}.report.json"
    markdown_name = f"{basename}.report.md"
    json_path = output_directory / json_name
    markdown_path = output_directory / markdown_name
    json_bytes = selected.canonical_json.encode()
    if len(json_bytes) > _REPORT_JSON_BYTES:
        raise ValueError("benchmark report JSON exceeds its retained bound")
    report_digest = sha256(json_bytes).hexdigest()
    markdown_bytes = benchmark_report_markdown(selected, report_digest).encode()
    if len(markdown_bytes) > _REPORT_MARKDOWN_BYTES:
        raise ValueError("benchmark report Markdown exceeds its retained bound")
    directory_descriptor: int | None = None
    try:
        directory_descriptor = os.open(
            output_directory,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        if not stat.S_ISDIR(os.fstat(directory_descriptor).st_mode):
            raise ValueError("benchmark report output directory is unavailable")
        _publish_exact_bytes(
            directory_descriptor,
            json_name,
            json_bytes,
            _REPORT_JSON_BYTES,
        )
        _publish_exact_bytes(
            directory_descriptor,
            markdown_name,
            markdown_bytes,
            _REPORT_MARKDOWN_BYTES,
        )
    except ValueError:
        raise
    except OSError:
        raise ValueError("benchmark report output directory is unavailable") from None
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    return RetainedBenchmarkReport(
        report_sha256=report_digest,
        json_path=json_path,
        markdown_path=markdown_path,
    )


def benchmark_report_markdown(
    report: BenchmarkReport | object,
    report_sha256: str | None = None,
) -> str:
    """Render the deterministic compact human projection of one report."""
    selected = _canonical_report(report)
    digest = report_sha256 or selected.sha256
    if digest != selected.sha256:
        raise ValueError("report JSON digest does not match the report")
    lines = [
        f"# Benchmark report: {selected.study.study_id}",
        "",
        f"Study version: `{selected.study.version}`  ",
        f"Study SHA-256: `{selected.study_sha256}`  ",
        f"Report JSON SHA-256: `{digest}`  ",
        f"Agent revision: `{selected.study.agent_revision}`  ",
        f"Reporter revision: `{selected.reporter_revision}`  ",
        f"Packs: {', '.join(f'`{item.pack_id}`' for item in selected.packs)}  ",
        f"Models: {', '.join(f'`{item.model_id}`' for item in selected.models)}  ",
        f"Repetitions: {selected.study.repetitions}  ",
        "Execution policy: `"
        + _canonical_json(selected.study.policy.model_dump(mode="json"))
        + "`",
        "",
        "## Complete study",
        "",
        _metric_table_header(),
        _metric_row("All cases", selected.overall),
        "",
        "## Models",
        "",
        _metric_table_header(),
        *(
            _metric_row(item.model_id, item.result) for item in selected.models
        ),
        "",
        "## Pack-model pairs",
        "",
        _metric_table_header(),
        *(
            _metric_row(f"{item.pack_id} / {item.model_id}", item.result)
            for item in selected.pack_models
        ),
        "",
        "## Cells",
        "",
        "| Cell | Pack | Model | Repetition | E2E exact | Completion | "
        "Conditional exact | Agent failure | Infrastructure failure | MLflow run |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        *(_cell_markdown_row(item) for item in selected.cells),
        "",
        "## Observed runtime",
        "",
        _observation_line("Elapsed seconds", selected.overall.observations.elapsed_seconds),
        _observation_line("Requests", selected.overall.observations.requests),
        _observation_line("Input tokens", selected.overall.observations.input_tokens),
        _observation_line("Output tokens", selected.overall.observations.output_tokens),
        _observation_line("Total tokens", selected.overall.observations.total_tokens),
        "Provider cost: unavailable",
        "",
    ]
    return "\n".join(lines)


def create_benchmark_report(
    study: BenchmarkStudy | object,
    runtime: BenchmarkRuntime | object,
    *,
    output_directory: Path,
    reporter_revision: str,
) -> RetainedBenchmarkReport:
    """Production composition for the read-only report CLI."""
    selected_runtime = _canonical_runtime(runtime)
    try:
        workspace = selected_runtime.workspace_root.resolve(strict=False)
        output = output_directory.resolve(strict=False)
    except OSError:
        raise BenchmarkReportConfigurationError(
            "benchmark report paths are unavailable"
        ) from None
    if output == workspace or output.is_relative_to(workspace):
        raise BenchmarkReportConfigurationError(
            "benchmark report output must be outside the benchmark workspace"
        )
    report = build_benchmark_report(
        study,
        selected_runtime,
        reporter_revision=reporter_revision,
    )
    return publish_benchmark_report(report, output_directory)


def _verify_and_compute_cell(
    cell: BenchmarkCell,
    receipt: BenchmarkCellReceipt,
    pack: LoadedEvaluationPack,
    *,
    model_name: str,
    evidence_reader: BenchmarkEvidenceReader,
) -> tuple[_CaseComputation, ...]:
    evaluation_run = _read_remote_run(evidence_reader, receipt.evaluation_run_id)
    _require_completed_run(evaluation_run, "evaluation")
    expected_tags = {
        "dsa.benchmark.agent_revision": receipt.agent_revision,
        "dsa.benchmark.cell_id": receipt.cell_id,
        "dsa.benchmark.model_id": receipt.model_id,
        "dsa.benchmark.pack_id": receipt.pack_id,
        "dsa.benchmark.repetition": str(receipt.repetition),
        "dsa.benchmark.study_sha256": receipt.study_sha256,
    }
    if any(evaluation_run.tags.get(key) != value for key, value in expected_tags.items()):
        raise ValueError("MLflow evaluation identity does not match its receipt")
    cases = {item.case_id: item for item in pack.cases}
    computations: list[_CaseComputation] = []
    for prediction in receipt.predictions:
        case = cases[prediction.case_id]
        expectations = {"answer": case.expected_answer}
        exact = end_to_end_exact_success(prediction, expectations)
        conditional = conditional_exact_json(prediction, expectations)
        agent = agent_failure(prediction, expectations)
        infrastructure = infrastructure_failure(prediction)
        observations: dict[str, float] = {}
        if prediction.reporting.status == "reported":
            tracking_run_id = prediction.reporting.tracking_run_id
            if tracking_run_id is None:
                raise ValueError("reported prediction has no tracking run identity")
            tracking = _read_remote_run(
                evidence_reader,
                tracking_run_id,
            )
            _require_completed_run(tracking, "analysis")
            expected_outcome = "succeeded" if prediction.accepted else "failed"
            if (
                tracking.tags.get("dsa.run_id") != prediction.run_id
                or tracking.tags.get("dsa.outcome") != expected_outcome
                or tracking.tags.get("dsa.component") != "analysis"
                or tracking.params.get("dsa.model_name") != model_name
            ):
                raise ValueError("MLflow analysis identity does not match its receipt")
            for key in _ANALYSIS_METRICS:
                if key not in tracking.metrics:
                    continue
                value = tracking.metrics[key]
                if value < 0:
                    raise ValueError("MLflow analysis metrics must be nonnegative")
                if key != "dsa.elapsed_seconds" and not value.is_integer():
                    raise ValueError("MLflow usage metrics must be integral")
                observations[key] = value
        computations.append(
            _CaseComputation(
                outcome=BenchmarkCaseOutcome(
                    case_id=prediction.case_id,
                    run_id=prediction.run_id,
                    terminal_sha256=prediction.terminal_sha256,
                    accepted=prediction.accepted,
                    failure_stage=prediction.failure_stage,
                    failure_code=prediction.failure_code,
                    reporting=prediction.reporting,
                    end_to_end_exact_success=exact,
                    conditional_exact_json=conditional,
                    agent_failure=agent,
                    infrastructure_failure=infrastructure,
                ),
                observations=observations,
            )
        )
    aggregate = _aggregate(computations)
    expected_metrics = {
        "agent_failure/mean": aggregate.metrics.agent_failure.rate,
        "conditional_exact_json/mean": (
            aggregate.metrics.conditional_exact_accuracy.rate
        ),
        "end_to_end_exact_success/mean": (
            aggregate.metrics.end_to_end_exact_success.rate
        ),
        "infrastructure_failure/mean": (
            aggregate.metrics.infrastructure_failure.rate
        ),
    }
    for name, expected in expected_metrics.items():
        actual = evaluation_run.metrics.get(name)
        if expected is None:
            if actual is not None:
                raise ValueError("MLflow evaluation metrics contradict local scoring")
        elif actual is None or not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("MLflow evaluation metrics contradict local scoring")
    return tuple(computations)


def _aggregate(values: Iterable[_CaseComputation]) -> BenchmarkAggregate:
    selected = tuple(values)
    case_count = len(selected)
    metrics = _metrics_from_outcomes(tuple(item.outcome for item in selected))
    return BenchmarkAggregate(
        case_count=case_count,
        metrics=metrics,
        observations=BenchmarkObservations(
            elapsed_seconds=_observed(selected, "dsa.elapsed_seconds"),
            requests=_observed(selected, "dsa.usage.requests"),
            input_tokens=_observed(selected, "dsa.usage.input_tokens"),
            output_tokens=_observed(selected, "dsa.usage.output_tokens"),
            total_tokens=_observed(selected, "dsa.usage.total_tokens"),
        ),
    )


def _metrics_from_outcomes(
    outcomes: tuple[BenchmarkCaseOutcome, ...],
) -> BenchmarkMetrics:
    case_count = len(outcomes)
    accepted = sum(item.accepted for item in outcomes)
    conditional_exact = sum(item.conditional_exact_json is True for item in outcomes)
    return BenchmarkMetrics(
        end_to_end_exact_success=_rate(
            sum(item.end_to_end_exact_success for item in outcomes),
            case_count,
        ),
        completion_rate=_rate(accepted, case_count),
        conditional_exact_accuracy=_rate(conditional_exact, accepted),
        agent_failure=_rate(
            sum(item.agent_failure for item in outcomes),
            case_count,
        ),
        infrastructure_failure=_rate(
            sum(item.infrastructure_failure for item in outcomes),
            case_count,
        ),
    )


def _cell_aggregate(cell: BenchmarkCellReport) -> BenchmarkAggregate:
    return BenchmarkAggregate(
        case_count=cell.case_count,
        metrics=cell.metrics,
        observations=cell.observations,
    )


def _combine_aggregates(
    values: Iterable[BenchmarkAggregate],
) -> BenchmarkAggregate:
    selected = tuple(values)
    case_count = sum(item.case_count for item in selected)

    def combined_rate(name: str) -> BenchmarkRate:
        rates = tuple(
            getattr(item.metrics, name)
            for item in selected
        )
        return _rate(
            sum(item.numerator for item in rates),
            sum(item.denominator for item in rates),
        )

    return BenchmarkAggregate(
        case_count=case_count,
        metrics=BenchmarkMetrics(
            end_to_end_exact_success=combined_rate("end_to_end_exact_success"),
            completion_rate=combined_rate("completion_rate"),
            conditional_exact_accuracy=combined_rate("conditional_exact_accuracy"),
            agent_failure=combined_rate("agent_failure"),
            infrastructure_failure=combined_rate("infrastructure_failure"),
        ),
        observations=BenchmarkObservations(
            elapsed_seconds=_combine_observed(
                item.observations.elapsed_seconds for item in selected
            ),
            requests=_combine_observed(item.observations.requests for item in selected),
            input_tokens=_combine_observed(
                item.observations.input_tokens for item in selected
            ),
            output_tokens=_combine_observed(
                item.observations.output_tokens for item in selected
            ),
            total_tokens=_combine_observed(
                item.observations.total_tokens for item in selected
            ),
        ),
    )


def _combine_observed(
    values: Iterable[BenchmarkObservedSummary],
) -> BenchmarkObservedSummary:
    selected = tuple(values)
    observed_count = sum(item.observed_count for item in selected)
    unavailable_count = sum(item.unavailable_count for item in selected)
    total = math.fsum(
        item.total for item in selected if item.total is not None
    ) if observed_count else None
    return BenchmarkObservedSummary(
        observed_count=observed_count,
        unavailable_count=unavailable_count,
        total=total,
        mean=total / observed_count if total is not None else None,
    )


def _aggregates_equivalent(
    left: BenchmarkAggregate,
    right: BenchmarkAggregate,
) -> bool:
    if left.case_count != right.case_count or left.metrics != right.metrics:
        return False
    pairs = (
        (left.observations.elapsed_seconds, right.observations.elapsed_seconds),
        (left.observations.requests, right.observations.requests),
        (left.observations.input_tokens, right.observations.input_tokens),
        (left.observations.output_tokens, right.observations.output_tokens),
        (left.observations.total_tokens, right.observations.total_tokens),
    )
    for first, second in pairs:
        if (
            first.observed_count != second.observed_count
            or first.unavailable_count != second.unavailable_count
            or not _optional_float_equal(first.total, second.total)
            or not _optional_float_equal(first.mean, second.mean)
        ):
            return False
    return True


def _optional_float_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _rate(numerator: int, denominator: int) -> BenchmarkRate:
    return BenchmarkRate(
        numerator=numerator,
        denominator=denominator,
        rate=numerator / denominator if denominator else None,
    )


def _observed(
    computations: tuple[_CaseComputation, ...],
    name: str,
) -> BenchmarkObservedSummary:
    values = tuple(
        item.observations[name] for item in computations if name in item.observations
    )
    total = math.fsum(values) if values else None
    return BenchmarkObservedSummary(
        observed_count=len(values),
        unavailable_count=len(computations) - len(values),
        total=total,
        mean=total / len(values) if total is not None else None,
    )


def _cell_computations(
    reports: list[BenchmarkCellReport],
    computations: list[_CaseComputation],
) -> tuple[tuple[_CaseComputation, ...], ...]:
    result: list[tuple[_CaseComputation, ...]] = []
    offset = 0
    for report in reports:
        end = offset + report.case_count
        result.append(tuple(computations[offset:end]))
        offset = end
    if offset != len(computations):
        raise ValueError("report computation partition is inconsistent")
    return tuple(result)


def _read_complete_receipts(
    study: BenchmarkStudy,
    runtime: BenchmarkRuntime,
    cells: tuple[BenchmarkCell, ...],
    packs: dict[str, LoadedEvaluationPack],
) -> tuple[BenchmarkCellReceipt, ...]:
    root = runtime.workspace_root
    expected_names = {cell.cell_id for cell in cells}
    root_descriptor: int | None = None
    try:
        root_descriptor = os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        if not stat.S_ISDIR(os.fstat(root_descriptor).st_mode):
            raise ValueError("benchmark report requires complete cell receipts")
        with os.scandir(root_descriptor) as entries:
            actual_names: set[str] = set()
            for entry in entries:
                if len(actual_names) >= len(expected_names) + 1:
                    raise ValueError("benchmark report requires exact cell directories")
                if entry.name not in expected_names or not entry.is_dir(follow_symlinks=False):
                    raise ValueError("benchmark report requires exact cell directories")
                actual_names.add(entry.name)
        if actual_names != expected_names:
            raise ValueError("benchmark report requires complete cell receipts")
        dataset_names = {item.pack_id: item.dataset_name for item in runtime.datasets}
        receipts: list[BenchmarkCellReceipt] = []
        for cell in cells:
            receipt = _read_receipt_at(root_descriptor, cell.cell_id)
            expected_cases = tuple(item.case_id for item in packs[cell.pack_id].cases)
            if (
                receipt.study_sha256 != study.sha256
                or receipt.cell_id != cell.cell_id
                or receipt.pack_id != cell.pack_id
                or receipt.model_id != cell.model_id
                or receipt.repetition != cell.repetition
                or receipt.agent_revision != study.agent_revision
                or receipt.docker_image != study.execution.docker_image
                or receipt.dataset_name != dataset_names[cell.pack_id]
                or tuple(item.case_id for item in receipt.predictions) != expected_cases
            ):
                raise ValueError("benchmark report cell receipt does not match the study")
            receipts.append(receipt)
        return tuple(receipts)
    except ValueError:
        raise
    except OSError:
        raise ValueError("benchmark report requires complete cell receipts") from None
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)


def _read_receipt_at(root_descriptor: int, cell_id: str) -> BenchmarkCellReceipt:
    cell_descriptor: int | None = None
    receipt_descriptor: int | None = None
    try:
        cell_descriptor = os.open(
            cell_id,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
        receipt_descriptor = os.open(
            "cell.json",
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=cell_descriptor,
        )
        metadata = os.fstat(receipt_descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > _RECEIPT_BYTES
        ):
            raise ValueError("benchmark report requires complete cell receipts")
        with os.fdopen(receipt_descriptor, "rb") as source:
            receipt_descriptor = None
            content = source.read(_RECEIPT_BYTES + 1)
        receipt = BenchmarkCellReceipt.model_validate_json(content)
        if content != receipt.canonical_json.encode():
            raise ValueError("benchmark report requires canonical cell receipts")
        _verify_terminal_records_at(cell_descriptor, receipt)
        return receipt
    except ValueError:
        raise
    except Exception:
        raise ValueError("benchmark report requires complete cell receipts") from None
    finally:
        if receipt_descriptor is not None:
            os.close(receipt_descriptor)
        if cell_descriptor is not None:
            os.close(cell_descriptor)


def _verify_terminal_records_at(
    cell_descriptor: int,
    receipt: BenchmarkCellReceipt,
) -> None:
    attempts_descriptor: int | None = None
    attempt_descriptor: int | None = None
    runs_descriptor: int | None = None
    try:
        attempts_descriptor = os.open(
            "attempts",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=cell_descriptor,
        )
        attempt_names: list[str] = []
        with os.scandir(attempts_descriptor) as entries:
            for entry in entries:
                if len(attempt_names) >= _MAX_CELL_ATTEMPTS:
                    raise ValueError("benchmark report has too many cell attempts")
                if not entry.is_dir(follow_symlinks=False):
                    raise ValueError("benchmark report requires managed cell attempts")
                attempt_names.append(entry.name)
        expected_names = [
            f"attempt-{index:04d}" for index in range(1, len(attempt_names) + 1)
        ]
        if not attempt_names or sorted(attempt_names) != expected_names:
            raise ValueError("benchmark report requires sequential cell attempts")
        attempt_descriptor = os.open(
            expected_names[-1],
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=attempts_descriptor,
        )
        runs_descriptor = os.open(
            "runs",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=attempt_descriptor,
        )
        expected_run_ids = {prediction.run_id for prediction in receipt.predictions}
        if len(expected_run_ids) != len(receipt.predictions):
            raise ValueError("benchmark report requires unique terminal run identities")
        actual_run_ids: set[str] = set()
        with os.scandir(runs_descriptor) as entries:
            for entry in entries:
                if len(actual_run_ids) >= len(expected_run_ids) + 1:
                    raise ValueError("benchmark report requires exact terminal records")
                if (
                    entry.name not in expected_run_ids
                    or not entry.is_dir(follow_symlinks=False)
                ):
                    raise ValueError("benchmark report requires exact terminal records")
                actual_run_ids.add(entry.name)
        if actual_run_ids != expected_run_ids:
            raise ValueError("benchmark report requires exact terminal records")
        for prediction in receipt.predictions:
            _verify_terminal_record_at(runs_descriptor, prediction)
    except ValueError:
        raise
    except Exception:
        raise ValueError("benchmark report requires exact terminal records") from None
    finally:
        if runs_descriptor is not None:
            os.close(runs_descriptor)
        if attempt_descriptor is not None:
            os.close(attempt_descriptor)
        if attempts_descriptor is not None:
            os.close(attempts_descriptor)


def _verify_terminal_record_at(
    runs_descriptor: int,
    prediction: MlflowEvaluationPrediction,
) -> None:
    run_descriptor: int | None = None
    terminal_descriptor: int | None = None
    try:
        run_descriptor = os.open(
            prediction.run_id,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=runs_descriptor,
        )
        terminal_descriptor = os.open(
            "terminal.json",
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=run_descriptor,
        )
        metadata = os.fstat(terminal_descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > _TERMINAL_BYTES
        ):
            raise ValueError("benchmark report requires bounded terminal records")
        with os.fdopen(terminal_descriptor, "rb") as source:
            terminal_descriptor = None
            content = source.read(_TERMINAL_BYTES + 1)
        if sha256(content).hexdigest() != prediction.terminal_sha256:
            raise ValueError("benchmark report terminal digest does not match its receipt")
        terminal = TerminalRecord.model_validate_json(content)
        canonical = _canonical_json(terminal.model_dump(mode="json")).encode() + b"\n"
        if content != canonical:
            raise ValueError("benchmark report requires canonical terminal records")
        if terminal.run_id != prediction.run_id:
            raise ValueError("benchmark report terminal run identity does not match its receipt")
        if prediction.accepted:
            if not isinstance(terminal.outcome, RunSuccess) or not exact_json_equal(
                terminal.outcome.answer,
                prediction.answer,
            ):
                raise ValueError("benchmark report terminal outcome does not match its receipt")
        elif not isinstance(terminal.outcome, RunFailure) or (
            terminal.outcome.failure.stage != prediction.failure_stage
            or terminal.outcome.failure.code != prediction.failure_code
        ):
            raise ValueError("benchmark report terminal outcome does not match its receipt")
    except ValueError:
        raise
    except Exception:
        raise ValueError("benchmark report requires exact terminal records") from None
    finally:
        if terminal_descriptor is not None:
            os.close(terminal_descriptor)
        if run_descriptor is not None:
            os.close(run_descriptor)


def _dataset_evidence(
    study: BenchmarkStudy,
    runtime: BenchmarkRuntime,
    receipts: tuple[BenchmarkCellReceipt, ...],
) -> tuple[BenchmarkDatasetEvidence, ...]:
    bindings = {item.pack_id: item.dataset_name for item in runtime.datasets}
    result: list[BenchmarkDatasetEvidence] = []
    for pack in study.packs:
        selected = tuple(item for item in receipts if item.pack_id == pack.pack_id)
        identities = {(item.dataset_id, item.dataset_digest) for item in selected}
        if len(identities) != 1:
            raise ValueError("benchmark report dataset identities conflict across cells")
        dataset_id, dataset_digest = next(iter(identities))
        result.append(
            BenchmarkDatasetEvidence(
                pack_id=pack.pack_id,
                dataset_name=bindings[pack.pack_id],
                dataset_id=dataset_id,
                dataset_digest=dataset_digest,
            )
        )
    return tuple(result)


def _validate_runtime(study: BenchmarkStudy, runtime: BenchmarkRuntime) -> None:
    if not runtime.workspace_root.is_absolute():
        raise BenchmarkReportConfigurationError(
            "benchmark report workspace must be absolute"
        )
    if tuple(item.pack_id for item in runtime.datasets) != tuple(
        item.pack_id for item in study.packs
    ):
        raise BenchmarkReportConfigurationError(
            "report runtime must exactly cover study packs"
        )


def _verify_pack_database(pack: LoadedEvaluationPack) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            pack.database_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != pack.manifest.database.size_bytes
        ):
            raise ValueError("benchmark report pack database is invalid")
        digest = sha256()
        with os.fdopen(descriptor, "rb") as source:
            descriptor = None
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != pack.manifest.database.sha256:
            raise ValueError("benchmark report pack database is invalid")
    except ValueError:
        raise
    except OSError:
        raise ValueError("benchmark report pack database is unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_revision(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise BenchmarkReportConfigurationError(
            "benchmark reporter revision is invalid"
        )
    return value


def _canonical_study(value: BenchmarkStudy | object) -> BenchmarkStudy:
    if isinstance(value, BenchmarkStudy):
        return BenchmarkStudy.model_validate_json(value.model_dump_json())
    return BenchmarkStudy.model_validate(value)


def _canonical_runtime(value: BenchmarkRuntime | object) -> BenchmarkRuntime:
    if isinstance(value, BenchmarkRuntime):
        return BenchmarkRuntime.model_validate_json(value.model_dump_json())
    return BenchmarkRuntime.model_validate(value)


def _canonical_report(value: BenchmarkReport | object) -> BenchmarkReport:
    if isinstance(value, BenchmarkReport):
        return BenchmarkReport.model_validate_json(value.model_dump_json())
    return BenchmarkReport.model_validate(value)


def _default_evidence_reader(
    environment: Mapping[str, str],
) -> BenchmarkEvidenceReader:
    if environment.get("MLFLOW_TRACKING_URI") != "databricks" or any(
        not environment.get(key, "").strip()
        for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN")
    ):
        raise BenchmarkReportConfigurationError(
            "benchmark report requires Databricks MLflow configuration"
        )
    try:
        tracking = import_module("mlflow.tracking")
        client_type = tracking.MlflowClient
        client = client_type(tracking_uri="databricks")
    except Exception:
        raise BenchmarkReportConfigurationError(
            "benchmark report MLflow client is unavailable"
        ) from None
    return MlflowBenchmarkEvidenceReader(client)


class MlflowBenchmarkEvidenceReader:
    """Read only the bounded MLflow fields needed by benchmark publication."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def get_run(self, run_id: str) -> BenchmarkMlflowRun:
        try:
            raw = self.client.get_run(run_id)
            raw_run_id = raw.info.run_id
            if raw_run_id != run_id:
                raise ValueError("MLflow returned a foreign run")
            raw_metrics = raw.data.metrics
            raw_params = raw.data.params
            raw_tags = raw.data.tags
            metrics = {
                name: float(raw_metrics[name])
                for name in (*_EVALUATION_METRICS, *_ANALYSIS_METRICS)
                if name in raw_metrics
            }
            params = {
                name: str(raw_params[name]) for name in _RUN_PARAMS if name in raw_params
            }
            tags = {name: str(raw_tags[name]) for name in _RUN_TAGS if name in raw_tags}
            return BenchmarkMlflowRun(
                run_id=run_id,
                status=str(raw.info.status),
                lifecycle_stage=cast(
                    Literal["active", "deleted"],
                    str(raw.info.lifecycle_stage),
                ),
                metrics=metrics,
                params=params,
                tags=tags,
            )
        except Exception:
            raise ValueError("required MLflow run is unavailable") from None


def _read_remote_run(
    reader: BenchmarkEvidenceReader,
    run_id: str,
) -> BenchmarkMlflowRun:
    try:
        value = reader.get_run(run_id)
        selected = BenchmarkMlflowRun.model_validate_json(value.model_dump_json())
    except Exception:
        raise ValueError("required MLflow run is unavailable") from None
    if selected.run_id != run_id:
        raise ValueError("MLflow returned a foreign run identity")
    return selected


def _require_completed_run(run: BenchmarkMlflowRun, kind: str) -> None:
    if run.status != "FINISHED" or run.lifecycle_stage != "active":
        raise ValueError(f"MLflow {kind} run is not completed active evidence")


def _metric_table_header() -> str:
    return (
        "| Scope | Cases | E2E exact | Completion | Conditional exact | "
        "Agent failure | Infrastructure failure |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )


def _metric_row(scope: str, result: BenchmarkAggregate) -> str:
    metrics = result.metrics
    return (
        f"| {scope} | {result.case_count} | "
        f"{_format_rate(metrics.end_to_end_exact_success)} | "
        f"{_format_rate(metrics.completion_rate)} | "
        f"{_format_rate(metrics.conditional_exact_accuracy)} | "
        f"{_format_rate(metrics.agent_failure)} | "
        f"{_format_rate(metrics.infrastructure_failure)} |"
    )


def _cell_markdown_row(cell: BenchmarkCellReport) -> str:
    metrics = cell.metrics
    return (
        f"| `{cell.cell_id}` | `{cell.pack_id}` | `{cell.model_id}` | "
        f"{cell.repetition} | {_format_rate(metrics.end_to_end_exact_success)} | "
        f"{_format_rate(metrics.completion_rate)} | "
        f"{_format_rate(metrics.conditional_exact_accuracy)} | "
        f"{_format_rate(metrics.agent_failure)} | "
        f"{_format_rate(metrics.infrastructure_failure)} | "
        f"`{cell.evaluation_run_id}` |"
    )


def _format_rate(value: BenchmarkRate) -> str:
    if value.rate is None:
        return "unavailable"
    return f"{value.numerator}/{value.denominator} ({value.rate:.6f})"


def _observation_line(name: str, value: BenchmarkObservedSummary) -> str:
    if value.mean is None or value.total is None:
        return f"{name}: unavailable (0/{value.unavailable_count} observed)  "
    denominator = value.observed_count + value.unavailable_count
    return (
        f"{name}: total {value.total:.6f}, mean {value.mean:.6f} "
        f"({value.observed_count}/{denominator} observed)  "
    )


def _publish_exact_bytes(
    directory_descriptor: int,
    name: str,
    content: bytes,
    maximum: int,
) -> None:
    temporary = f".{name}.{uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = None
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_exact_bytes(directory_descriptor, name, maximum) != content:
                raise ValueError("benchmark report publication conflict") from None
        os.fsync(directory_descriptor)
    except ValueError:
        raise
    except OSError:
        raise ValueError("benchmark report publication unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(OSError):
            os.unlink(temporary, dir_fd=directory_descriptor)


def _read_exact_bytes(
    directory_descriptor: int,
    name: str,
    maximum: int,
) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum:
            raise ValueError("benchmark report publication conflict")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = None
            return source.read(maximum + 1)
    except ValueError:
        raise
    except OSError:
        raise ValueError("benchmark report publication conflict") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
