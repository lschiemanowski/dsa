"""Read-only immutable benchmark report contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from dsa.benchmark import (
    BenchmarkCellInvocation,
    BenchmarkCellReceipt,
    PreparedBenchmark,
    retain_benchmark_cell_receipt,
)
from dsa.benchmark_report import (
    BenchmarkMlflowRun,
    BenchmarkReport,
    BenchmarkReportConfigurationError,
    MlflowBenchmarkEvidenceReader,
    build_benchmark_report,
    create_benchmark_report,
    publish_benchmark_report,
)
from dsa.contract import RunRequest
from dsa.evaluation import MlflowEvaluationPrediction
from dsa.pack import EvaluationCaseMetadata, EvaluationPackCase
from dsa.record import Failure, RunFailure, RunSuccess, TerminalRecord, write_terminal_record
from dsa.reporting import MlflowReporting

from .test_benchmark_execution import prepared_benchmark, receipt
from .test_episode import valid_request
from .test_mlflow_evaluation import evaluation_pack

REPORTER_REVISION = "e" * 40


class EvidenceReader:
    def __init__(self, runs: dict[str, BenchmarkMlflowRun]) -> None:
        self.runs = runs
        self.requested: list[str] = []

    def get_run(self, run_id: str) -> BenchmarkMlflowRun:
        self.requested.append(run_id)
        try:
            return self.runs[run_id]
        except KeyError:
            raise ValueError("missing run") from None


def evaluation_tags(invocation: BenchmarkCellInvocation) -> dict[str, str]:
    return {
        "dsa.benchmark.agent_revision": invocation.agent_revision,
        "dsa.benchmark.cell_id": invocation.cell.cell_id,
        "dsa.benchmark.model_id": invocation.cell.model_id,
        "dsa.benchmark.pack_id": invocation.cell.pack_id,
        "dsa.benchmark.repetition": str(invocation.cell.repetition),
        "dsa.benchmark.study_sha256": invocation.cell.study_sha256,
    }


def successful_evaluation_metrics(
    *,
    exact: bool,
    policy: bool | None = None,
) -> dict[str, float]:
    selected_policy = exact if policy is None else policy
    return {
        "agent_failure/mean": 0.0 if selected_policy else 1.0,
        "conditional_exact_json/mean": 1.0 if exact else 0.0,
        "conditional_policy_match/mean": 1.0 if selected_policy else 0.0,
        "end_to_end_exact_success/mean": 1.0 if exact else 0.0,
        "end_to_end_policy_success/mean": 1.0 if selected_policy else 0.0,
        "infrastructure_failure/mean": 0.0,
    }


def retain_receipt_with_terminals(
    prepared: PreparedBenchmark,
    invocation: BenchmarkCellInvocation,
    selected_receipt: BenchmarkCellReceipt,
    *,
    terminal_run_id: str | None = None,
    terminal_sha256: str | None = None,
) -> BenchmarkCellReceipt:
    selected = BenchmarkCellReceipt.model_validate_json(selected_receipt.model_dump_json())
    cases = {
        case.case_id: case
        for pack_id, pack in prepared.packs
        if pack_id == selected.pack_id
        for case in pack.cases
    }
    predictions: list[MlflowEvaluationPrediction] = []
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    for prediction in selected.predictions:
        case = cases[prediction.case_id]
        run_directory = invocation.attempt_directory / "runs" / prediction.run_id
        run_directory.mkdir(parents=True)
        if prediction.accepted:
            outcome = RunSuccess(answer=prediction.answer)
        else:
            if prediction.failure_stage is None or prediction.failure_code is None:
                raise AssertionError("failed test prediction must be classified")
            outcome = RunFailure(
                failure=Failure(
                    stage=prediction.failure_stage,
                    code=prediction.failure_code,
                    message="classified benchmark failure",
                )
            )
        terminal = TerminalRecord(
            run_id=terminal_run_id or prediction.run_id,
            started_at=started_at,
            finished_at=started_at + timedelta(seconds=1),
            request=RunRequest(
                database_path=next(
                    pack.database_path
                    for pack_id, pack in prepared.packs
                    if pack_id == selected.pack_id
                ),
                question=case.question,
                answer_schema=case.answer_schema,
                model=invocation.model_configuration,
                policy=invocation.policy,
            ),
            outcome=outcome,
        )
        retained = write_terminal_record(terminal, run_directory)
        predictions.append(
            prediction.model_copy(
                update={"terminal_sha256": terminal_sha256 or retained.sha256}
            )
        )
    selected = selected.model_copy(update={"predictions": tuple(predictions)})
    retain_benchmark_cell_receipt(selected, invocation.receipt_path)
    return selected


def complete_report_evidence(
    tmp_path: Path,
    *,
    accepted_answer: object = None,
) -> tuple[PreparedBenchmark, EvidenceReader]:
    prepared = prepared_benchmark(tmp_path)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    selected_receipt = receipt(invocation)
    if accepted_answer is not None:
        prediction = selected_receipt.predictions[0].model_copy(
            update={"answer": accepted_answer}
        )
        selected_receipt = selected_receipt.model_copy(
            update={"predictions": (prediction,)}
        )
    selected_receipt = retain_receipt_with_terminals(
        prepared,
        invocation,
        selected_receipt,
    )
    exact = accepted_answer in (None, {"count": 3})
    runs = {
        selected_receipt.evaluation_run_id: BenchmarkMlflowRun(
            run_id=selected_receipt.evaluation_run_id,
            status="FINISHED",
            lifecycle_stage="active",
            metrics=successful_evaluation_metrics(exact=exact),
            tags=evaluation_tags(invocation),
        ),
        "tracking-run": BenchmarkMlflowRun(
            run_id="tracking-run",
            status="FINISHED",
            lifecycle_stage="active",
            metrics={
                "dsa.elapsed_seconds": 2.5,
                "dsa.usage.input_tokens": 100,
                "dsa.usage.output_tokens": 25,
                "dsa.usage.requests": 2,
                "dsa.usage.total_tokens": 125,
            },
            params={"dsa.model_name": prepared.study.models[0].configuration.name},
            tags={
                "dsa.component": "analysis",
                "dsa.outcome": "succeeded",
                "dsa.run_id": "run-id",
            },
        ),
    }
    return prepared, EvidenceReader(runs)


def build_report(tmp_path: Path, *, accepted_answer: object = None) -> BenchmarkReport:
    prepared, reader = complete_report_evidence(
        tmp_path,
        accepted_answer=accepted_answer,
    )
    return build_benchmark_report(
        prepared.study,
        prepared.runtime,
        reporter_revision=REPORTER_REVISION,
        pack_loader=lambda _reference: prepared.packs[0][1],
        evidence_reader=reader,
    )


def test_report_recomputes_exact_counts_and_observed_usage(tmp_path: Path) -> None:
    report = build_report(tmp_path)

    assert report.study_sha256 == report.study.sha256
    assert report.reporter_revision == REPORTER_REVISION
    assert report.overall.case_count == 1
    assert report.overall.metrics.end_to_end_exact_success.model_dump() == {
        "denominator": 1,
        "numerator": 1,
        "rate": 1.0,
    }
    assert report.overall.metrics.end_to_end_policy_success.numerator == 1
    assert report.overall.metrics.completion_rate.numerator == 1
    assert report.overall.metrics.conditional_exact_accuracy.denominator == 1
    assert report.overall.metrics.conditional_policy_accuracy.denominator == 1
    assert report.overall.metrics.agent_failure.numerator == 0
    assert report.overall.metrics.infrastructure_failure.numerator == 0
    assert report.overall.observations.elapsed_seconds.total == 2.5
    assert report.overall.observations.total_tokens.total == 125.0
    assert report.provider_cost.status == "unavailable"
    assert report.cells[0].cases[0].end_to_end_exact_success is True
    assert report.cells[0].cases[0].end_to_end_policy_success is True
    assert not hasattr(report.cells[0].cases[0], "answer")
    assert report.canonical_json.endswith("\n")


def test_report_uses_task_counts_instead_of_averaging_cell_rates(
    tmp_path: Path,
) -> None:
    report = build_report(tmp_path, accepted_answer={"count": 4})

    assert report.overall.metrics.end_to_end_exact_success.numerator == 0
    assert report.overall.metrics.completion_rate.numerator == 1
    assert report.overall.metrics.conditional_exact_accuracy.numerator == 0
    assert report.overall.metrics.conditional_policy_accuracy.numerator == 0
    assert report.overall.metrics.agent_failure.numerator == 1
    assert report.overall.metrics.infrastructure_failure.numerator == 0
    assert report.cells[0].cases[0].conditional_exact_json is False


def test_report_keeps_tolerance_policy_matches_distinct_from_exactness(
    tmp_path: Path,
) -> None:
    request = valid_request(tmp_path)
    pack = evaluation_pack(
        tmp_path,
        EvaluationPackCase(
            case_id="tiny-count",
            case_version="1",
            question=request.question,
            answer_schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"ratio": {"type": "number"}},
                "required": ["ratio"],
                "additionalProperties": False,
            },
            expected_answer={"ratio": 0.1},
            metadata=EvaluationCaseMetadata(family="counting", source_level="small"),
        ),
        scorer={
            "name": "json-numeric-tolerance",
            "version": "1",
            "relative_tolerance": 1e-9,
            "absolute_tolerance": 1e-12,
        },
    )
    prepared = prepared_benchmark(tmp_path, pack=pack)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    selected = receipt(invocation)
    selected = selected.model_copy(
        update={
            "predictions": (
                selected.predictions[0].model_copy(
                    update={"answer": {"ratio": 0.10000000001}}
                ),
            )
        }
    )
    selected = retain_receipt_with_terminals(prepared, invocation, selected)
    reader = EvidenceReader(
        {
            selected.evaluation_run_id: BenchmarkMlflowRun(
                run_id=selected.evaluation_run_id,
                status="FINISHED",
                lifecycle_stage="active",
                metrics=successful_evaluation_metrics(exact=False, policy=True),
                tags=evaluation_tags(invocation),
            ),
            "tracking-run": BenchmarkMlflowRun(
                run_id="tracking-run",
                status="FINISHED",
                lifecycle_stage="active",
                params={"dsa.model_name": invocation.model_configuration.name},
                tags={
                    "dsa.component": "analysis",
                    "dsa.outcome": "succeeded",
                    "dsa.run_id": "run-id",
                },
            ),
        }
    )

    report = build_benchmark_report(
        prepared.study,
        prepared.runtime,
        reporter_revision=REPORTER_REVISION,
        pack_loader=lambda _reference: pack,
        evidence_reader=reader,
    )

    assert report.overall.metrics.end_to_end_exact_success.numerator == 0
    assert report.overall.metrics.end_to_end_policy_success.numerator == 1
    assert report.overall.metrics.conditional_exact_accuracy.numerator == 0
    assert report.overall.metrics.conditional_policy_accuracy.numerator == 1
    assert report.cells[0].cases[0].conditional_exact_json is False
    assert report.cells[0].cases[0].conditional_policy_match is True


def test_report_rejects_incomplete_or_foreign_evidence_before_publication(
    tmp_path: Path,
) -> None:
    prepared = prepared_benchmark(tmp_path)
    reader = EvidenceReader({})

    with pytest.raises(ValueError, match="complete cell receipts"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],
            evidence_reader=reader,
        )

    assert reader.requested == []


def test_report_rejects_mlflow_metric_or_identity_mismatch(tmp_path: Path) -> None:
    prepared, reader = complete_report_evidence(tmp_path)
    reader.runs["evaluation-run"] = reader.runs["evaluation-run"].model_copy(
        update={"metrics": {"end_to_end_exact_success/mean": 0.0}}
    )

    with pytest.raises(ValueError, match="MLflow evaluation metrics"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],
            evidence_reader=reader,
        )


@pytest.mark.parametrize(
    ("run_id", "update"),
    (
        ("evaluation-run", {"status": "RUNNING"}),
        ("evaluation-run", {"lifecycle_stage": "deleted"}),
        ("tracking-run", {"status": "FAILED"}),
    ),
)
def test_report_requires_completed_active_mlflow_runs(
    tmp_path: Path,
    run_id: str,
    update: dict[str, str],
) -> None:
    prepared, reader = complete_report_evidence(tmp_path)
    reader.runs[run_id] = reader.runs[run_id].model_copy(update=update)

    with pytest.raises(ValueError, match="completed active evidence"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],
            evidence_reader=reader,
        )


def test_report_rejects_an_unverified_terminal_digest(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    selected = retain_receipt_with_terminals(
        prepared,
        invocation,
        receipt(invocation),
        terminal_sha256="0" * 64,
    )
    reader = EvidenceReader({})

    with pytest.raises(ValueError, match="terminal digest"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],
            evidence_reader=reader,
        )

    assert selected.predictions[0].terminal_sha256 == "0" * 64
    assert reader.requested == []


def test_report_rejects_a_foreign_terminal_run_identity(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    retain_receipt_with_terminals(
        prepared,
        invocation,
        receipt(invocation),
        terminal_run_id="foreign-run",
    )
    reader = EvidenceReader({})

    with pytest.raises(ValueError, match="terminal run identity"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],
            evidence_reader=reader,
        )

    assert reader.requested == []


def test_report_revalidates_mutated_typed_inputs(tmp_path: Path) -> None:
    prepared, reader = complete_report_evidence(tmp_path)
    prepared.study.models[0].configuration.settings["api_key"] = "SECRET"

    with pytest.raises(ValueError, match="credentials or endpoints"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],
            evidence_reader=reader,
        )


def test_report_excludes_answers_questions_paths_and_raw_diagnostics(
    tmp_path: Path,
) -> None:
    report = build_report(tmp_path)
    content = report.canonical_json

    assert '"count":3' not in content
    assert "How many" not in content
    assert str(tmp_path) not in content
    assert "DATABRICKS_TOKEN" not in content
    assert '"answer":null' not in content


def test_publication_is_deterministic_idempotent_and_no_overwrite(
    tmp_path: Path,
) -> None:
    report = build_report(tmp_path)
    output = tmp_path / "publication"

    first = publish_benchmark_report(report, output)
    second = publish_benchmark_report(report, output)

    assert first == second
    assert first.json_path.read_bytes() == report.canonical_json.encode()
    markdown = first.markdown_path.read_text()
    assert f"Report JSON SHA-256: `{report.sha256}`" in markdown
    assert "1/1 (1.000000)" in markdown
    assert "Provider cost: unavailable" in markdown
    assert str(tmp_path) not in markdown

    first.markdown_path.write_text("different\n")
    with pytest.raises(ValueError, match="conflict"):
        publish_benchmark_report(report, output)


def test_report_model_rejects_a_study_digest_mismatch(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    value = report.model_dump(mode="python")
    value["study_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="study digest"):
        BenchmarkReport.model_validate(value)


def test_report_model_rejects_cell_metrics_changed_after_derivation(
    tmp_path: Path,
) -> None:
    report = build_report(tmp_path)
    value = report.model_dump(mode="python")
    value["cells"][0]["metrics"]["end_to_end_exact_success"] = {
        "denominator": 1,
        "numerator": 0,
        "rate": 0.0,
    }

    with pytest.raises(ValueError, match="cell metrics"):
        BenchmarkReport.model_validate(value)


def test_report_aggregates_multiple_cells_from_case_counts(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path, cells=2)
    runs: dict[str, BenchmarkMlflowRun] = {}
    for index, cell in enumerate(prepared.cells):
        invocation = BenchmarkCellInvocation.from_prepared(prepared, cell, 1)
        base = receipt(invocation)
        exact = index == 0
        tracking_id = f"tracking-{index}"
        run_id = f"run-{index}"
        prediction = base.predictions[0].model_copy(
            update={
                "answer": {"count": 3 if exact else 4},
                "reporting": base.predictions[0].reporting.model_copy(
                    update={
                        "trace_id": f"trace-{index}",
                        "tracking_run_id": tracking_id,
                    }
                ),
                "run_id": run_id,
            }
        )
        selected = base.model_copy(
            update={
                "evaluation_run_id": f"evaluation-{index}",
                "predictions": (prediction,),
            }
        )
        selected = retain_receipt_with_terminals(prepared, invocation, selected)
        runs[selected.evaluation_run_id] = BenchmarkMlflowRun(
            run_id=selected.evaluation_run_id,
            status="FINISHED",
            lifecycle_stage="active",
            metrics=successful_evaluation_metrics(exact=exact),
            tags=evaluation_tags(invocation),
        )
        runs[tracking_id] = BenchmarkMlflowRun(
            run_id=tracking_id,
            status="FINISHED",
            lifecycle_stage="active",
            metrics={"dsa.usage.total_tokens": float(100 + index)},
            params={"dsa.model_name": invocation.model_configuration.name},
            tags={
                "dsa.component": "analysis",
                "dsa.outcome": "succeeded",
                "dsa.run_id": run_id,
            },
        )

    report = build_benchmark_report(
        prepared.study,
        prepared.runtime,
        reporter_revision=REPORTER_REVISION,
        pack_loader=lambda _reference: prepared.packs[0][1],
        evidence_reader=EvidenceReader(runs),
    )

    assert report.overall.case_count == 2
    assert report.overall.metrics.end_to_end_exact_success.numerator == 1
    assert report.overall.metrics.end_to_end_exact_success.denominator == 2
    assert [item.result.metrics.end_to_end_exact_success.numerator for item in report.models] == [
        1,
        0,
    ]
    assert report.overall.observations.total_tokens.total == 201.0
    assert report.overall.observations.total_tokens.observed_count == 2


def test_report_rejects_a_symlinked_cell_directory(tmp_path: Path) -> None:
    prepared, reader = complete_report_evidence(tmp_path)
    cell_directory = prepared.runtime.workspace_root / prepared.cells[0].cell_id
    outside = tmp_path / "outside-cell"
    cell_directory.rename(outside)
    cell_directory.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="exact cell directories"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],
            evidence_reader=reader,
        )

    assert reader.requested == []


def test_report_reverifies_pack_database_bytes(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path)
    prepared.packs[0][1].database_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="pack database"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],
            evidence_reader=EvidenceReader({}),
        )


def test_production_report_output_cannot_modify_the_benchmark_workspace(
    tmp_path: Path,
) -> None:
    prepared, _reader = complete_report_evidence(tmp_path)

    with pytest.raises(BenchmarkReportConfigurationError, match="outside"):
        create_benchmark_report(
            prepared.study,
            prepared.runtime,
            output_directory=prepared.runtime.workspace_root / "reports",
            reporter_revision=REPORTER_REVISION,
        )


def test_failed_reporting_has_no_conditional_mlflow_exception(
    tmp_path: Path,
) -> None:
    prepared = prepared_benchmark(tmp_path)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    base = receipt(invocation)
    prediction = MlflowEvaluationPrediction(
        case_id=base.predictions[0].case_id,
        run_id="failed-run",
        terminal_sha256="1" * 64,
        accepted=False,
        failure_stage="model",
        failure_code="model_http_error",
        reporting=MlflowReporting(
            status="failed",
            failure_code="mlflow_export_failed",
        ),
    )
    failed_receipt = base.model_copy(update={"predictions": (prediction,)})
    failed_receipt = retain_receipt_with_terminals(
        prepared,
        invocation,
        failed_receipt,
    )
    reader = EvidenceReader(
        {
            "evaluation-run": BenchmarkMlflowRun(
                run_id="evaluation-run",
                status="FINISHED",
                lifecycle_stage="active",
                metrics={
                    "agent_failure/mean": 0.0,
                    "end_to_end_exact_success/mean": 0.0,
                    "end_to_end_policy_success/mean": 0.0,
                    "infrastructure_failure/mean": 1.0,
                },
                tags=evaluation_tags(invocation),
            )
        }
    )

    report = build_benchmark_report(
        prepared.study,
        prepared.runtime,
        reporter_revision=REPORTER_REVISION,
        pack_loader=lambda _reference: prepared.packs[0][1],
        evidence_reader=reader,
    )

    assert report.overall.metrics.conditional_exact_accuracy.denominator == 0
    assert report.overall.metrics.conditional_exact_accuracy.rate is None
    assert report.overall.metrics.infrastructure_failure.numerator == 1
    assert reader.requested == ["evaluation-run"]
    assert json.loads(report.canonical_json)["format"] == "dsa-benchmark-report/v2"


def test_mlflow_reader_retains_only_the_bounded_safe_projection() -> None:
    raw = SimpleNamespace(
        info=SimpleNamespace(
            run_id="evaluation-run",
            status="FINISHED",
            lifecycle_stage="active",
        ),
        data=SimpleNamespace(
            metrics={
                "end_to_end_exact_success/mean": 1.0,
                "secret.metric": 42.0,
            },
            params={
                "dsa.model_name": "test",
                "private_endpoint": "https://private.example",
            },
            tags={
                "dsa.benchmark.cell_id": "cell-" + "a" * 64,
                "credential": "SECRET",
            },
        ),
    )

    class Client:
        def get_run(self, run_id: str) -> object:
            assert run_id == "evaluation-run"
            return raw

    selected = MlflowBenchmarkEvidenceReader(Client()).get_run("evaluation-run")

    assert selected.metrics == {"end_to_end_exact_success/mean": 1.0}
    assert selected.status == "FINISHED"
    assert selected.lifecycle_stage == "active"
    assert selected.params == {"dsa.model_name": "test"}
    assert selected.tags == {"dsa.benchmark.cell_id": "cell-" + "a" * 64}
    assert "SECRET" not in selected.model_dump_json()
