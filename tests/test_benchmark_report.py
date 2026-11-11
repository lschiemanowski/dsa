"""Read-only immutable benchmark report contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from dsa.benchmark import (
    BenchmarkCellInvocation,
    BenchmarkCellReceipt,
    PreparedBenchmark,
    retain_benchmark_cell_receipt,
)
from dsa.benchmark_report import (
    BenchmarkCaseOutcome,
    BenchmarkReport,
    BenchmarkReportConfigurationError,
    benchmark_report_markdown,
    build_benchmark_report,
    create_benchmark_report,
    publish_benchmark_report,
)
from dsa.contract import DerivationRequest, RunRequest
from dsa.cost import unavailable_provider_cost
from dsa.evaluation import EvaluationPrediction
from dsa.pack import EvaluationCaseMetadata, EvaluationPackCase, LoadedEvaluationPack
from dsa.record import (
    DatabaseRecord,
    DerivationVerification,
    Failure,
    RunFailure,
    RunSuccess,
    TerminalRecord,
    write_terminal_record,
)
from dsa.reporting import LogfireReporting

from .test_benchmark_execution import prepared_benchmark, receipt
from .test_derivation import sample_derivation
from .test_episode import valid_request
from .test_evaluation import evaluation_pack

REPORTER_REVISION = "e" * 40








def retain_receipt_with_terminals(
    prepared: PreparedBenchmark,
    invocation: BenchmarkCellInvocation,
    selected_receipt: BenchmarkCellReceipt,
    *,
    terminal_run_id: str | None = None,
    terminal_question: str | None = None,
    terminal_sha256: str | None = None,
    provider_costs: tuple[float | None, ...] = (0.001, 0.002),
    provider_response_ids: tuple[str, ...] | None = None,
    total_tokens: int = 125,
) -> BenchmarkCellReceipt:
    selected = BenchmarkCellReceipt.model_validate_json(selected_receipt.model_dump_json())
    cases = {
        case.case_id: case
        for pack_id, pack in prepared.packs
        if pack_id == selected.pack_id
        for case in pack.cases
    }
    predictions: list[EvaluationPrediction] = []
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    for prediction in selected.predictions:
        case = cases[prediction.case_id]
        run_directory = invocation.attempt_directory / "runs" / prediction.run_id
        run_directory.mkdir(parents=True)
        verification: DerivationVerification | None = None
        if prediction.accepted:
            if case.derivation is None:
                outcome = RunSuccess(answer=prediction.answer)
            else:
                derivation = sample_derivation()
                derivation_bytes = json.dumps(
                    derivation.model_dump(mode="json"),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                answer_bytes = json.dumps(
                    prediction.answer,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                notebook_bytes = b'{"cells":[],"nbformat":4,"nbformat_minor":5}\n'
                (run_directory / "derivation.ipynb").write_bytes(notebook_bytes)
                source_sha256 = next(
                    pack.manifest.database.sha256
                    for pack_id, pack in prepared.packs
                    if pack_id == selected.pack_id
                )
                verification = DerivationVerification(
                    derivation_sha256=sha256(derivation_bytes).hexdigest(),
                    result_sha256=sha256(answer_bytes).hexdigest(),
                    source_database_sha256=source_sha256,
                    runtime_identity="docker/test",
                    notebook_sha256=sha256(notebook_bytes).hexdigest(),
                    notebook_byte_length=len(notebook_bytes),
                )
                outcome = RunSuccess(
                    answer=prediction.answer,
                    derivation=derivation,
                    derivation_verification=verification,
                )
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
        response_ids = provider_response_ids or tuple(
            f"gen-{prediction.run_id}-{index}"
            for index in range(1, len(provider_costs) + 1)
        )
        if len(response_ids) != len(provider_costs):
            raise AssertionError("test provider response evidence is misaligned")
        source_sha256 = next(
            pack.manifest.database.sha256
            for pack_id, pack in prepared.packs
            if pack_id == selected.pack_id
        )
        terminal = TerminalRecord(
            schema_version="2" if case.derivation is not None else "1",
            run_id=terminal_run_id or prediction.run_id,
            started_at=started_at,
            finished_at=started_at + timedelta(seconds=2.5),
            usage={
                "input_tokens": 100,
                "output_tokens": 25,
                "requests": 2,
                "total_tokens": total_tokens,
            },
            request=RunRequest(
                database_path=next(
                    pack.database_path
                    for pack_id, pack in prepared.packs
                    if pack_id == selected.pack_id
                ),
                question=terminal_question or case.question,
                answer_schema=case.answer_schema,
                derivation=case.derivation,
                model=invocation.model_configuration,
                policy=invocation.policy,
            ),
            messages=tuple(
                {
                    "kind": "response",
                    "provider_name": "openrouter",
                    "provider_response_id": response_id,
                    "provider_details": (
                        {"cost": cost} if cost is not None else {}
                    ),
                }
                for response_id, cost in zip(
                    response_ids,
                    provider_costs,
                    strict=True,
                )
            ),
            database=(
                DatabaseRecord(
                    source_sha256=source_sha256,
                    final_sha256=source_sha256,
                )
            ),
            outcome=outcome,
        )
        retained = write_terminal_record(terminal, run_directory)
        predictions.append(
            prediction.model_copy(
                update={
                    "terminal_sha256": terminal_sha256 or retained.sha256,
                    "derivation_sha256": (
                        verification.derivation_sha256
                        if verification is not None
                        else None
                    ),
                }
            )
        )
    selected = selected.model_copy(update={"predictions": tuple(predictions)})
    retain_benchmark_cell_receipt(selected, invocation.receipt_path)
    return selected


def complete_report_evidence(
    tmp_path: Path,
    *,
    accepted_answer: object = None,
    provider_costs: tuple[float | None, ...] = (0.001, 0.002),
    pack: LoadedEvaluationPack | None = None,
    reporting: LogfireReporting | None = None,
) -> PreparedBenchmark:
    prepared = prepared_benchmark(tmp_path, pack=pack)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    selected_receipt = receipt(invocation)
    prediction = selected_receipt.predictions[0].model_copy(
        update={"reporting": reporting or LogfireReporting()}
    )
    if accepted_answer is not None:
        prediction = prediction.model_copy(update={"answer": accepted_answer})
    selected_receipt = selected_receipt.model_copy(update={"predictions": (prediction,)})
    selected_receipt = retain_receipt_with_terminals(
        prepared,
        invocation,
        selected_receipt,
        provider_costs=provider_costs,
    )
    return prepared


def build_report(
    tmp_path: Path,
    *,
    accepted_answer: object = None,
    provider_costs: tuple[float | None, ...] = (0.001, 0.002),
    reporting: LogfireReporting | None = None,
) -> BenchmarkReport:
    prepared = complete_report_evidence(
        tmp_path,
        accepted_answer=accepted_answer,
        provider_costs=provider_costs,
        reporting=reporting,
    )
    return build_benchmark_report(
        prepared.study,
        prepared.runtime,
        reporter_revision=REPORTER_REVISION,
        pack_loader=lambda _reference: prepared.packs[0][1],

    )


@pytest.mark.parametrize(
    "reporting",
    [LogfireReporting(), LogfireReporting(status="failed", failure_code="logfire_token_missing")],
)
def test_report_recomputes_exact_counts_and_observed_usage(
    tmp_path: Path, reporting: LogfireReporting
) -> None:
    report = build_report(tmp_path, reporting=reporting)

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
    assert report.overall.provider_cost.model_dump() == {
        "status": "observed",
        "observed_amount": "0.003",
        "currency": "USD",
        "source": "openrouter_usage_cost",
        "complete_case_count": 1,
        "incomplete_case_count": 0,
        "observed_generation_count": 2,
        "unavailable_generation_count": 0,
    }
    assert report.cells[0].cases[0].end_to_end_exact_success is True
    assert report.cells[0].cases[0].end_to_end_policy_success is True
    assert not hasattr(report.cells[0].cases[0], "answer")
    assert report.canonical_json.endswith("\n")


def test_report_hashes_exact_local_derivation_notebook(tmp_path: Path) -> None:
    request = valid_request(tmp_path)
    case = EvaluationPackCase(
        case_id="tiny-count",
        case_version="1",
        question=request.question,
        answer_schema=request.answer_schema,
        derivation=DerivationRequest(),
        expected_answer={"count": 3},
        metadata=EvaluationCaseMetadata(family="counting", source_level="small"),
    )
    pack = evaluation_pack(tmp_path, case)
    prepared = complete_report_evidence(tmp_path, pack=pack)

    report = build_benchmark_report(
        prepared.study,
        prepared.runtime,
        reporter_revision=REPORTER_REVISION,
        pack_loader=lambda _reference: pack,

    )

    assert report.overall.metrics.end_to_end_exact_success.numerator == 1
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    notebook = invocation.attempt_directory / "runs" / "run-id" / "derivation.ipynb"
    notebook.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="derivation notebook"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: pack,

        )


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


def test_report_marks_incomplete_provider_cost_evidence_as_partial(
    tmp_path: Path,
) -> None:
    report = build_report(tmp_path, provider_costs=(0.001, None))

    assert report.overall.provider_cost.status == "partial"
    assert report.overall.provider_cost.observed_amount == "0.001"
    assert report.overall.provider_cost.complete_case_count == 0
    assert report.overall.provider_cost.incomplete_case_count == 1
    assert report.overall.provider_cost.observed_generation_count == 1
    assert report.overall.provider_cost.unavailable_generation_count == 1
    markdown = benchmark_report_markdown(report)
    assert "partial $0.001 observed; per-case mean unavailable" in markdown
    assert "$0.001/case" not in markdown


def test_report_preserves_a_positive_tiny_cost_in_markdown(tmp_path: Path) -> None:
    report = build_report(tmp_path, provider_costs=(1e-13,))

    markdown = benchmark_report_markdown(report)
    assert "$0.0000000000001; $1e-13/case" in markdown
    assert "$0/case" not in markdown






def test_report_rejects_response_id_reuse_when_one_cost_is_unavailable(
    tmp_path: Path,
) -> None:
    prepared = prepared_benchmark(tmp_path, cells=2)
    for index, cell in enumerate(prepared.cells):
        invocation = BenchmarkCellInvocation.from_prepared(prepared, cell, 1)
        base = receipt(invocation)
        prediction = base.predictions[0].model_copy(
            update={"run_id": f"run-reused-provider-id-{index}"}
        )
        selected = base.model_copy(update={"predictions": (prediction,)})
        retain_receipt_with_terminals(
            prepared,
            invocation,
            selected,
            provider_costs=(0.001 if index == 0 else None,),
            provider_response_ids=("gen-reused",),
        )

    with pytest.raises(ValueError, match="provider response identities must be unique"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],

        )


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

    report = build_benchmark_report(
        prepared.study,
        prepared.runtime,
        reporter_revision=REPORTER_REVISION,
        pack_loader=lambda _reference: pack,

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

    with pytest.raises(ValueError, match="complete cell receipts"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],

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

    with pytest.raises(ValueError, match="terminal digest"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],

        )

    assert selected.predictions[0].terminal_sha256 == "0" * 64



def test_report_rejects_a_foreign_terminal_run_identity(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    retain_receipt_with_terminals(
        prepared,
        invocation,
        receipt(invocation),
        terminal_run_id="foreign-run",
    )

    with pytest.raises(ValueError, match="terminal run identity"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],

        )



def test_report_revalidates_mutated_typed_inputs(tmp_path: Path) -> None:
    prepared = complete_report_evidence(tmp_path)
    prepared.study.models[0].configuration.settings["api_key"] = "SECRET"

    with pytest.raises(ValueError, match="credentials or endpoints"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],

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
    assert "Provider inference cost (OpenRouter usage, USD): $0.003" in markdown
    assert first.json_path.name.endswith(".report-v4.json")
    assert first.markdown_path.name.endswith(".report-v4.md")
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


def test_report_model_rejects_cell_cost_changed_after_derivation(
    tmp_path: Path,
) -> None:
    report = build_report(tmp_path)
    value = report.model_dump(mode="python")
    value["cells"][0]["provider_cost"]["observed_amount"] = "0.004"

    with pytest.raises(ValueError, match="cell provider cost"):
        BenchmarkReport.model_validate(value)


def test_report_case_rejects_exact_success_without_a_policy_match() -> None:
    with pytest.raises(ValueError, match="exact match must satisfy the pack policy"):
        BenchmarkCaseOutcome(
            case_id="case-1",
            run_id="run-1",
            terminal_sha256="a" * 64,
            accepted=True,
            reporting=LogfireReporting(
                status="recorded",
                trace_id="a" * 32,
            ),
            end_to_end_exact_success=True,
            end_to_end_policy_success=False,
            conditional_exact_json=True,
            conditional_policy_match=False,
            agent_failure=True,
            infrastructure_failure=False,
            provider_cost=unavailable_provider_cost(),
        )


def test_report_aggregates_multiple_cells_from_case_counts(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path, cells=2)
    for index, cell in enumerate(prepared.cells):
        invocation = BenchmarkCellInvocation.from_prepared(prepared, cell, 1)
        base = receipt(invocation)
        prediction = base.predictions[0].model_copy(
            update={
                "answer": {"count": 3 if index == 0 else 4},
                "run_id": f"run-{index}",
            }
        )
        selected = base.model_copy(update={"predictions": (prediction,)})
        retain_receipt_with_terminals(
            prepared, invocation, selected, total_tokens=100 + index
        )

    report = build_benchmark_report(
        prepared.study,
        prepared.runtime,
        reporter_revision=REPORTER_REVISION,
        pack_loader=lambda _reference: prepared.packs[0][1],

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
    assert report.overall.provider_cost.observed_amount == "0.006"
    assert [item.result.provider_cost.observed_amount for item in report.models] == [
        "0.003",
        "0.003",
    ]


def test_report_rejects_a_symlinked_cell_directory(tmp_path: Path) -> None:
    prepared = complete_report_evidence(tmp_path)
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

        )



def test_report_reverifies_pack_database_bytes(tmp_path: Path) -> None:
    prepared = prepared_benchmark(tmp_path)
    prepared.packs[0][1].database_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="pack database"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],

        )


def test_production_report_output_cannot_modify_the_benchmark_workspace(
    tmp_path: Path,
) -> None:
    prepared = complete_report_evidence(tmp_path)

    with pytest.raises(BenchmarkReportConfigurationError, match="outside"):
        create_benchmark_report(
            prepared.study,
            prepared.runtime,
            output_directory=prepared.runtime.workspace_root / "reports",
            reporter_revision=REPORTER_REVISION,
        )


def test_failed_analysis_still_counts_when_telemetry_fails(
    tmp_path: Path,
) -> None:
    prepared = prepared_benchmark(tmp_path)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    base = receipt(invocation)
    prediction = EvaluationPrediction(
        case_id=base.predictions[0].case_id,
        run_id="failed-run",
        terminal_sha256="1" * 64,
        accepted=False,
        failure_stage="model",
        failure_code="model_http_error",
        reporting=LogfireReporting(
            status="failed",
            failure_code="logfire_recording_failed",
        ),
    )
    failed_receipt = base.model_copy(update={"predictions": (prediction,)})
    failed_receipt = retain_receipt_with_terminals(
        prepared,
        invocation,
        failed_receipt,
    )

    report = build_benchmark_report(
        prepared.study,
        prepared.runtime,
        reporter_revision=REPORTER_REVISION,
        pack_loader=lambda _reference: prepared.packs[0][1],

    )

    assert report.overall.metrics.conditional_exact_accuracy.denominator == 0
    assert report.overall.metrics.conditional_exact_accuracy.rate is None
    assert report.overall.metrics.infrastructure_failure.numerator == 1

    assert json.loads(report.canonical_json)["format"] == "dsa-benchmark-report/v4"



def test_report_rejects_terminal_for_another_question_even_with_matching_digest(
    tmp_path: Path,
) -> None:
    prepared = prepared_benchmark(tmp_path)
    invocation = BenchmarkCellInvocation.from_prepared(prepared, prepared.cells[0], 1)
    retain_receipt_with_terminals(
        prepared, invocation, receipt(invocation), terminal_question="A different question"
    )

    with pytest.raises(ValueError, match="terminal request does not match"):
        build_benchmark_report(
            prepared.study,
            prepared.runtime,
            reporter_revision=REPORTER_REVISION,
            pack_loader=lambda _reference: prepared.packs[0][1],
        )
