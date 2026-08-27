"""Read-only immutable benchmark report contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dsa.benchmark import (
    BenchmarkCellInvocation,
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
from dsa.evaluation import MlflowEvaluationPrediction
from dsa.reporting import MlflowReporting

from .test_benchmark_execution import prepared_benchmark, receipt

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
    retain_benchmark_cell_receipt(selected_receipt, invocation.receipt_path)
    exact = accepted_answer in (None, {"count": 3})
    runs = {
        selected_receipt.evaluation_run_id: BenchmarkMlflowRun(
            run_id=selected_receipt.evaluation_run_id,
            metrics={
                "agent_failure/mean": 0.0 if exact else 1.0,
                "conditional_exact_json/mean": 1.0 if exact else 0.0,
                "end_to_end_exact_success/mean": 1.0 if exact else 0.0,
                "infrastructure_failure/mean": 0.0,
            },
            tags=evaluation_tags(invocation),
        ),
        "tracking-run": BenchmarkMlflowRun(
            run_id="tracking-run",
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
    assert report.overall.metrics.completion_rate.numerator == 1
    assert report.overall.metrics.conditional_exact_accuracy.denominator == 1
    assert report.overall.metrics.agent_failure.numerator == 0
    assert report.overall.metrics.infrastructure_failure.numerator == 0
    assert report.overall.observations.elapsed_seconds.total == 2.5
    assert report.overall.observations.total_tokens.total == 125.0
    assert report.provider_cost.status == "unavailable"
    assert report.cells[0].cases[0].end_to_end_exact_success is True
    assert not hasattr(report.cells[0].cases[0], "answer")
    assert report.canonical_json.endswith("\n")


def test_report_uses_task_counts_instead_of_averaging_cell_rates(
    tmp_path: Path,
) -> None:
    report = build_report(tmp_path, accepted_answer={"count": 4})

    assert report.overall.metrics.end_to_end_exact_success.numerator == 0
    assert report.overall.metrics.completion_rate.numerator == 1
    assert report.overall.metrics.conditional_exact_accuracy.numerator == 0
    assert report.overall.metrics.agent_failure.numerator == 1
    assert report.overall.metrics.infrastructure_failure.numerator == 0
    assert report.cells[0].cases[0].conditional_exact_json is False


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
        retain_benchmark_cell_receipt(selected, invocation.receipt_path)
        runs[selected.evaluation_run_id] = BenchmarkMlflowRun(
            run_id=selected.evaluation_run_id,
            metrics={
                "agent_failure/mean": 0.0 if exact else 1.0,
                "conditional_exact_json/mean": 1.0 if exact else 0.0,
                "end_to_end_exact_success/mean": 1.0 if exact else 0.0,
                "infrastructure_failure/mean": 0.0,
            },
            tags=evaluation_tags(invocation),
        )
        runs[tracking_id] = BenchmarkMlflowRun(
            run_id=tracking_id,
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
    retain_benchmark_cell_receipt(failed_receipt, invocation.receipt_path)
    reader = EvidenceReader(
        {
            "evaluation-run": BenchmarkMlflowRun(
                run_id="evaluation-run",
                metrics={
                    "agent_failure/mean": 0.0,
                    "end_to_end_exact_success/mean": 0.0,
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
    assert json.loads(report.canonical_json)["format"] == "dsa-benchmark-report/v1"


def test_mlflow_reader_retains_only_the_bounded_safe_projection() -> None:
    raw = SimpleNamespace(
        info=SimpleNamespace(run_id="evaluation-run"),
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
    assert selected.params == {"dsa.model_name": "test"}
    assert selected.tags == {"dsa.benchmark.cell_id": "cell-" + "a" * 64}
    assert "SECRET" not in selected.model_dump_json()
