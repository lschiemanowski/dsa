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
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from dsa.benchmark import (
    BenchmarkCell,
    BenchmarkCellReceipt,
    BenchmarkRuntime,
    BenchmarkStudy,
    GitRevision,
    PackLoader,
    SafeName,
    expand_benchmark_study,
)
from dsa.contract import ContractModel, ModelConfiguration, RunPolicy
from dsa.cost import (
    ProviderCostSummary,
    combine_provider_costs,
    provider_cost_from_messages,
    safe_openrouter_response_ids,
)
from dsa.evaluation import EvaluationPrediction, exact_json_equal, score_outcome, score_prediction
from dsa.pack import EvaluationPackCase, LoadedEvaluationPack, load_huggingface_evaluation_pack
from dsa.plots import MAX_NOTEBOOK_BYTES
from dsa.record import (
    DerivationVerification,
    FailureStage,
    RunFailure,
    RunSuccess,
    TerminalRecord,
)
from dsa.reporting import LogfireReporting

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]

_REPORT_JSON_BYTES = 256 * 1024 * 1024
_REPORT_MARKDOWN_BYTES = 64 * 1024 * 1024
_MAX_REPORT_CASE_EXECUTIONS = 100_000
_RECEIPT_BYTES = 64 * 1024 * 1024
_TERMINAL_BYTES = 64 * 1024 * 1024
_DERIVATION_NOTEBOOK_BYTES = MAX_NOTEBOOK_BYTES
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
    """The exact, policy, completion, and failure metrics with explicit denominators."""

    end_to_end_exact_success: BenchmarkRate
    end_to_end_policy_success: BenchmarkRate
    completion_rate: BenchmarkRate
    conditional_exact_accuracy: BenchmarkRate
    conditional_policy_accuracy: BenchmarkRate
    agent_failure: BenchmarkRate
    infrastructure_failure: BenchmarkRate


class BenchmarkObservedSummary(ContractModel):
    """Coverage and aggregate for one optional observed terminal metric."""

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
    provider_cost: ProviderCostSummary

    @model_validator(mode="after")
    def denominators_match_case_count(self) -> BenchmarkAggregate:
        all_case_rates = (
            self.metrics.end_to_end_exact_success,
            self.metrics.end_to_end_policy_success,
            self.metrics.completion_rate,
            self.metrics.agent_failure,
            self.metrics.infrastructure_failure,
        )
        if any(item.denominator != self.case_count for item in all_case_rates):
            raise ValueError("all-case metric denominator does not match case count")
        if (
            self.metrics.conditional_exact_accuracy.denominator
            != self.metrics.completion_rate.numerator
            or self.metrics.conditional_policy_accuracy.denominator
            != self.metrics.completion_rate.numerator
        ):
            raise ValueError("conditional metric denominator does not match completions")
        for item in self.observations.model_dump(mode="python").values():
            summary = BenchmarkObservedSummary.model_validate(item)
            if summary.observed_count + summary.unavailable_count != self.case_count:
                raise ValueError("observation coverage does not match case count")
        if (
            self.provider_cost.complete_case_count
            + self.provider_cost.incomplete_case_count
            != self.case_count
        ):
            raise ValueError("provider cost coverage does not match case count")
        return self


class BenchmarkCaseOutcome(ContractModel):
    """Safe per-case classification without questions, answers, or expectations."""

    case_id: SafeName
    run_id: SafeName
    terminal_sha256: Sha256
    accepted: bool
    failure_stage: FailureStage | None = None
    failure_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,127}$",
    )
    reporting: LogfireReporting
    end_to_end_exact_success: bool
    end_to_end_policy_success: bool
    conditional_exact_json: bool | None
    conditional_policy_match: bool | None
    agent_failure: bool
    infrastructure_failure: bool
    provider_cost: ProviderCostSummary

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> BenchmarkCaseOutcome:
        if (
            self.provider_cost.complete_case_count
            + self.provider_cost.incomplete_case_count
            != 1
        ):
            raise ValueError("case provider cost must cover exactly one execution")
        if self.accepted:
            if self.failure_stage is not None or self.failure_code is not None:
                raise ValueError("accepted case outcome contains failure state")
            if (
                self.conditional_exact_json is None
                or self.conditional_policy_match is None
            ):
                raise ValueError("accepted case outcome requires conditional scores")
        elif (
            self.failure_stage is None
            or self.failure_code is None
            or self.conditional_exact_json is not None
            or self.conditional_policy_match is not None
        ):
            raise ValueError("failed case outcome has invalid classification state")
        if (
            self.conditional_exact_json is True
            and self.conditional_policy_match is not True
        ):
            raise ValueError("exact match must satisfy the pack policy")
        scores = score_outcome(
            self.accepted,
            self.failure_stage,
            self.failure_code,
            self.conditional_exact_json,
            self.conditional_policy_match,
        )
        if any(getattr(self, name) != value for name, value in scores.items()):
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


class BenchmarkCellReport(ContractModel):
    """One cell's immutable receipt correlation and derived result."""

    cell_id: str = Field(pattern=r"^cell-[0-9a-f]{64}$")
    pack_id: SafeName
    model_id: SafeName
    repetition: NonNegativeInt
    receipt_sha256: Sha256
    case_count: NonNegativeInt
    cases: tuple[BenchmarkCaseOutcome, ...]
    metrics: BenchmarkMetrics
    observations: BenchmarkObservations
    provider_cost: ProviderCostSummary

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
            provider_cost=self.provider_cost,
        )
        expected = _metrics_from_outcomes(self.cases)
        if self.metrics != expected:
            raise ValueError("cell metrics do not match its case outcomes")
        expected_cost = combine_provider_costs(
            item.provider_cost for item in self.cases
        )
        if self.provider_cost != expected_cost:
            raise ValueError("cell provider cost does not match its case outcomes")
        return self


class BenchmarkModelAggregate(ContractModel):
    model_id: SafeName
    result: BenchmarkAggregate


class BenchmarkPackModelAggregate(ContractModel):
    pack_id: SafeName
    model_id: SafeName
    result: BenchmarkAggregate


class BenchmarkReport(ContractModel):
    """Canonical immutable report for one complete benchmark study."""

    format: Literal["dsa-benchmark-report/v4"] = "dsa-benchmark-report/v4"
    reporter_revision: GitRevision
    study_sha256: Sha256
    study: BenchmarkStudy
    packs: tuple[BenchmarkPackEvidence, ...]
    cells: tuple[BenchmarkCellReport, ...]
    overall: BenchmarkAggregate
    models: tuple[BenchmarkModelAggregate, ...]
    pack_models: tuple[BenchmarkPackModelAggregate, ...]

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
        case_outcomes = tuple(case for cell in self.cells for case in cell.cases)
        run_ids = tuple(item.run_id for item in case_outcomes)
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("report analysis run identities must be unique")
        trace_ids = [
            outcome.reporting.trace_id
            for outcome in case_outcomes
            if outcome.reporting.trace_id is not None
        ]
        if len(trace_ids) != len(set(trace_ids)):
            raise ValueError("report analysis trace identities must be unique")
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


class RetainedBenchmarkReport(ContractModel):
    """Exact local identities of both published report representations."""

    report_sha256: Sha256
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class _CaseComputation:
    outcome: BenchmarkCaseOutcome
    observations: dict[str, float]


@dataclass(frozen=True)
class _TerminalEvidence:
    provider_cost: ProviderCostSummary
    observations: dict[str, float]


@dataclass(frozen=True)
class _VerifiedCellEvidence:
    receipt: BenchmarkCellReceipt
    terminals: dict[str, _TerminalEvidence]
    provider_response_ids: frozenset[str]


def build_benchmark_report(
    study: BenchmarkStudy | object,
    runtime: BenchmarkRuntime | object,
    *,
    reporter_revision: str,
    pack_loader: PackLoader = load_huggingface_evaluation_pack,
) -> BenchmarkReport:
    """Verify one complete study and derive its immutable report without writes."""
    selected_study = _canonical_study(study)
    selected_runtime = _canonical_runtime(runtime)
    selected_revision = _validate_revision(reporter_revision)
    _validate_runtime(selected_runtime)
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
    cell_evidence = _read_complete_cell_evidence(
        selected_study,
        selected_runtime,
        cells,
        pack_map,
    )
    provider_response_ids: set[str] = set()
    for evidence in cell_evidence:
        if provider_response_ids.intersection(evidence.provider_response_ids):
            raise ValueError(
                "benchmark report provider response identities must be unique"
            )
        provider_response_ids.update(evidence.provider_response_ids)
    all_computations: list[_CaseComputation] = []
    cell_reports: list[BenchmarkCellReport] = []
    for cell, evidence in zip(cells, cell_evidence, strict=True):
        receipt = evidence.receipt
        pack = pack_map[cell.pack_id]
        computations = _verify_and_compute_cell(
            receipt,
            pack,
            evidence.terminals,
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
                case_count=aggregate.case_count,
                cases=tuple(item.outcome for item in computations),
                metrics=aggregate.metrics,
                observations=aggregate.observations,
                provider_cost=aggregate.provider_cost,
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
    json_name = f"{basename}.report-v4.json"
    markdown_name = f"{basename}.report-v4.md"
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
        "Conditional exact | E2E policy | Conditional policy | Agent failure | "
        "Infrastructure failure | Provider cost |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *(_cell_markdown_row(item) for item in selected.cells),
        "",
        "## Observed runtime",
        "",
        _observation_line("Elapsed seconds", selected.overall.observations.elapsed_seconds),
        _observation_line("Requests", selected.overall.observations.requests),
        _observation_line("Input tokens", selected.overall.observations.input_tokens),
        _observation_line("Output tokens", selected.overall.observations.output_tokens),
        _observation_line("Total tokens", selected.overall.observations.total_tokens),
        _provider_cost_line(selected.overall.provider_cost),
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
    receipt: BenchmarkCellReceipt,
    pack: LoadedEvaluationPack,
    terminals: Mapping[str, _TerminalEvidence],
) -> tuple[_CaseComputation, ...]:
    cases = {item.case_id: item for item in pack.cases}
    computations: list[_CaseComputation] = []
    for prediction in receipt.predictions:
        terminal = terminals[prediction.run_id]
        case = cases[prediction.case_id]
        scores = score_prediction(prediction, case.expected_answer, pack.manifest.scorer)
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
                    provider_cost=terminal.provider_cost,
                    **scores,
                ),
                observations=terminal.observations,
            )
        )
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
        provider_cost=combine_provider_costs(
            item.outcome.provider_cost for item in selected
        ),
    )


def _metrics_from_outcomes(
    outcomes: tuple[BenchmarkCaseOutcome, ...],
) -> BenchmarkMetrics:
    case_count = len(outcomes)
    accepted = sum(item.accepted for item in outcomes)
    conditional_exact = sum(item.conditional_exact_json is True for item in outcomes)
    conditional_policy = sum(item.conditional_policy_match is True for item in outcomes)
    return BenchmarkMetrics(
        end_to_end_exact_success=_rate(
            sum(item.end_to_end_exact_success for item in outcomes),
            case_count,
        ),
        end_to_end_policy_success=_rate(
            sum(item.end_to_end_policy_success for item in outcomes),
            case_count,
        ),
        completion_rate=_rate(accepted, case_count),
        conditional_exact_accuracy=_rate(conditional_exact, accepted),
        conditional_policy_accuracy=_rate(conditional_policy, accepted),
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
        provider_cost=cell.provider_cost,
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
            end_to_end_policy_success=combined_rate("end_to_end_policy_success"),
            completion_rate=combined_rate("completion_rate"),
            conditional_exact_accuracy=combined_rate("conditional_exact_accuracy"),
            conditional_policy_accuracy=combined_rate("conditional_policy_accuracy"),
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
        provider_cost=combine_provider_costs(
            item.provider_cost for item in selected
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
    if (
        left.case_count != right.case_count
        or left.metrics != right.metrics
        or left.provider_cost != right.provider_cost
    ):
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


def _read_complete_cell_evidence(
    study: BenchmarkStudy,
    runtime: BenchmarkRuntime,
    cells: tuple[BenchmarkCell, ...],
    packs: dict[str, LoadedEvaluationPack],
) -> tuple[_VerifiedCellEvidence, ...]:
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
        models = {item.model_id: item.configuration for item in study.models}
        evidence: list[_VerifiedCellEvidence] = []
        for cell in cells:
            pack = packs[cell.pack_id]
            verified = _read_receipt_at(
                root_descriptor, cell.cell_id, pack, models[cell.model_id], study.policy
            )
            receipt = verified.receipt
            expected_cases = tuple(item.case_id for item in packs[cell.pack_id].cases)
            if (
                receipt.study_sha256 != study.sha256
                or receipt.cell_id != cell.cell_id
                or receipt.pack_id != cell.pack_id
                or receipt.model_id != cell.model_id
                or receipt.repetition != cell.repetition
                or receipt.agent_revision != study.agent_revision
                or receipt.docker_image != study.execution.docker_image
                or receipt.pack_manifest_sha256 != pack.reference.manifest_sha256
                or tuple(item.case_id for item in receipt.predictions) != expected_cases
            ):
                raise ValueError("benchmark report cell receipt does not match the study")
            evidence.append(verified)
        return tuple(evidence)
    except ValueError:
        raise
    except OSError:
        raise ValueError("benchmark report requires complete cell receipts") from None
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)


def _read_receipt_at(
    root_descriptor: int,
    cell_id: str,
    pack: LoadedEvaluationPack,
    model: ModelConfiguration,
    policy: RunPolicy,
) -> _VerifiedCellEvidence:
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
        terminals, response_ids = _verify_terminal_records_at(
            cell_descriptor, receipt, pack, model, policy
        )
        return _VerifiedCellEvidence(
            receipt=receipt,
            terminals=terminals,
            provider_response_ids=response_ids,
        )
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
    pack: LoadedEvaluationPack,
    model: ModelConfiguration,
    policy: RunPolicy,
) -> tuple[dict[str, _TerminalEvidence], frozenset[str]]:
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
        terminals: dict[str, _TerminalEvidence] = {}
        cases = {case.case_id: case for case in pack.cases}
        response_ids: set[str] = set()
        for prediction in receipt.predictions:
            case = cases.get(prediction.case_id)
            if case is None:
                raise ValueError("benchmark report cell receipt does not match the study")
            terminal, case_response_ids = _verify_terminal_record_at(
                runs_descriptor, prediction, case, pack.manifest.database.sha256, model, policy
            )
            selected_ids = set(case_response_ids)
            if response_ids.intersection(selected_ids):
                raise ValueError(
                    "benchmark report provider response identities must be unique"
                )
            response_ids.update(selected_ids)
            terminals[prediction.run_id] = terminal
        return terminals, frozenset(response_ids)
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
    prediction: EvaluationPrediction,
    case: EvaluationPackCase,
    database_sha256: str,
    model: ModelConfiguration,
    policy: RunPolicy,
) -> tuple[_TerminalEvidence, tuple[str, ...]]:
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
        if (
            terminal.request.question != case.question
            or terminal.request.answer_schema != case.answer_schema
            or terminal.request.derivation != case.derivation
            or terminal.request.model != model
            or terminal.request.policy != policy
            or (
                terminal.database is not None
                and terminal.database.source_sha256 != database_sha256
            )
            or (prediction.accepted and terminal.database is None)
        ):
            raise ValueError("benchmark report terminal request does not match the study case")
        if prediction.accepted:
            if not isinstance(terminal.outcome, RunSuccess) or not exact_json_equal(
                terminal.outcome.answer,
                prediction.answer,
            ):
                raise ValueError("benchmark report terminal outcome does not match its receipt")
            verification = terminal.outcome.derivation_verification
            terminal_derivation_sha256 = (
                verification.derivation_sha256 if verification is not None else None
            )
            if terminal_derivation_sha256 != prediction.derivation_sha256:
                raise ValueError(
                    "benchmark report derivation identity does not match its receipt"
                )
            if verification is not None:
                _verify_derivation_notebook_at(run_descriptor, verification)
        elif not isinstance(terminal.outcome, RunFailure) or (
            terminal.outcome.failure.stage != prediction.failure_stage
            or terminal.outcome.failure.code != prediction.failure_code
        ):
            raise ValueError("benchmark report terminal outcome does not match its receipt")
        observations = {
            "dsa.elapsed_seconds": (terminal.finished_at - terminal.started_at).total_seconds()
        }
        for key in ("requests", "input_tokens", "output_tokens", "total_tokens"):
            value = terminal.usage.get(key)
            if type(value) is int and value >= 0:
                observations[f"dsa.usage.{key}"] = float(value)
        return (
            _TerminalEvidence(provider_cost_from_messages(terminal.messages), observations),
            safe_openrouter_response_ids(terminal.messages),
        )
    except ValueError:
        raise
    except Exception:
        raise ValueError("benchmark report requires exact terminal records") from None
    finally:
        if terminal_descriptor is not None:
            os.close(terminal_descriptor)
        if run_descriptor is not None:
            os.close(run_descriptor)


def _verify_derivation_notebook_at(
    run_descriptor: int,
    verification: DerivationVerification,
) -> None:
    notebook_descriptor: int | None = None
    try:
        notebook_descriptor = os.open(
            verification.notebook_relative_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=run_descriptor,
        )
        metadata = os.fstat(notebook_descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != verification.notebook_byte_length
            or metadata.st_size <= 0
            or metadata.st_size > _DERIVATION_NOTEBOOK_BYTES
        ):
            raise ValueError("benchmark report requires the exact derivation notebook")
        with os.fdopen(notebook_descriptor, "rb") as source:
            notebook_descriptor = None
            digest = sha256()
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != verification.notebook_sha256:
            raise ValueError("benchmark report derivation notebook digest does not match")
    finally:
        if notebook_descriptor is not None:
            os.close(notebook_descriptor)


def _validate_runtime(runtime: BenchmarkRuntime) -> None:
    if not runtime.workspace_root.is_absolute():
        raise BenchmarkReportConfigurationError(
            "benchmark report workspace must be absolute"
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


def _metric_table_header() -> str:
    return (
        "| Scope | Cases | E2E exact | Completion | Conditional exact | E2E policy | "
        "Conditional policy | Agent failure | Infrastructure failure | Provider cost |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )


def _metric_row(scope: str, result: BenchmarkAggregate) -> str:
    metrics = result.metrics
    return (
        f"| {scope} | {result.case_count} | "
        f"{_format_rate(metrics.end_to_end_exact_success)} | "
        f"{_format_rate(metrics.completion_rate)} | "
        f"{_format_rate(metrics.conditional_exact_accuracy)} | "
        f"{_format_rate(metrics.end_to_end_policy_success)} | "
        f"{_format_rate(metrics.conditional_policy_accuracy)} | "
        f"{_format_rate(metrics.agent_failure)} | "
        f"{_format_rate(metrics.infrastructure_failure)} | "
        f"{_format_provider_cost(result.provider_cost, result.case_count)} |"
    )


def _cell_markdown_row(cell: BenchmarkCellReport) -> str:
    metrics = cell.metrics
    return (
        f"| `{cell.cell_id}` | `{cell.pack_id}` | `{cell.model_id}` | "
        f"{cell.repetition} | {_format_rate(metrics.end_to_end_exact_success)} | "
        f"{_format_rate(metrics.completion_rate)} | "
        f"{_format_rate(metrics.conditional_exact_accuracy)} | "
        f"{_format_rate(metrics.end_to_end_policy_success)} | "
        f"{_format_rate(metrics.conditional_policy_accuracy)} | "
        f"{_format_rate(metrics.agent_failure)} | "
        f"{_format_rate(metrics.infrastructure_failure)} | "
        f"{_format_provider_cost(cell.provider_cost, cell.case_count)} |"
    )


def _format_rate(value: BenchmarkRate) -> str:
    if value.rate is None:
        return "unavailable"
    return f"{value.numerator}/{value.denominator} ({value.rate:.6f})"


def _format_provider_cost(value: ProviderCostSummary, case_count: int) -> str:
    amount = value.amount_decimal
    if amount is None:
        return "unavailable"
    generation_count = (
        value.observed_generation_count + value.unavailable_generation_count
    )
    if value.status != "observed":
        return (
            f"partial ${value.observed_amount} observed; per-case mean unavailable; "
            f"{value.observed_generation_count}/{generation_count} generations"
        )
    mean = amount / case_count if case_count else amount
    mean_text = format(mean, ".18g")
    return (
        f"${value.observed_amount}; ${mean_text}/case; "
        f"{value.observed_generation_count}/{generation_count} generations"
    )


def _provider_cost_line(value: ProviderCostSummary) -> str:
    case_count = value.complete_case_count + value.incomplete_case_count
    return (
        "Provider inference cost (OpenRouter usage, USD): "
        f"{_format_provider_cost(value, case_count)} "
        f"({value.complete_case_count}/{case_count} cases complete)  "
    )


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
