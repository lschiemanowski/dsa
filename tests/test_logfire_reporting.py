"""Check that only safe analysis metadata reaches Logfire."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from dsa import RunCompletion, RunFailure, RunSuccess, evaluation_telemetry, reporting, run_analysis

from .test_episode import clock, valid_request
from .test_evaluation import evaluation_pack, prediction


async def test_missing_token_preserves_exact_canonical_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
    request = valid_request(tmp_path)

    async def run(enabled: bool) -> RunCompletion:
        return await run_analysis(
            request,
            runs_directory=tmp_path / str(enabled),
            model=TestModel(call_tools=[], custom_output_args={"count": 3}),
            report_to_logfire=enabled,
            identity_factory=lambda: "run-example",
            clock=clock(),
        )

    disabled, failed = await asyncio.gather(run(False), run(True))
    assert isinstance(disabled.outcome, RunSuccess)
    assert disabled.reporting.status == "disabled"
    assert failed.reporting.status == "failed"
    assert failed.reporting.failure_code == "logfire_token_missing"
    assert failed.record == disabled.record.model_copy(update={"messages": failed.record.messages})
    terminal = json.loads(failed.retained_record.path.read_bytes())
    assert terminal["outcome"] == {"answer": {"count": 3}, "status": "succeeded"}
    assert "reporting" not in terminal
    assert len(list((tmp_path / "True").glob("*/terminal.json"))) == 1


async def test_logfire_traces_omit_payloads_and_raw_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("logfire")
    import logfire
    from logfire.testing import TestExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = TestExporter()
    sdk = logfire.configure(
        local=True,
        send_to_logfire=False,
        console=False,
        metrics=False,
        additional_span_processors=[SimpleSpanProcessor(exporter)],
    )
    monkeypatch.setattr(reporting, "_analysis_logfire", lambda: sdk)
    request = valid_request(tmp_path).model_copy(update={"question": "private-question-sentinel"})

    async def provider_error(
        _messages: list[ModelMessage], _info: AgentInfo
    ) -> ModelResponse:
        raise ModelHTTPError(401, "test", body={"detail": "private-error-sentinel"})

    traced, disabled = await asyncio.gather(
        run_analysis(
            request,
            runs_directory=tmp_path / "traced",
            model=FunctionModel(provider_error),
            report_to_logfire=True,
            identity_factory=lambda: "run-traced",
        ),
        run_analysis(
            request,
            runs_directory=tmp_path / "disabled",
            model=TestModel(call_tools=[], custom_output_args={"count": 3}),
            identity_factory=lambda: "run-disabled",
        ),
    )

    assert isinstance(traced.outcome, RunFailure)
    assert traced.reporting.status == "recorded"
    assert isinstance(disabled.outcome, RunSuccess)
    assert disabled.reporting.status == "disabled"
    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    assert [span["name"] for span in spans] == ["dsa.model.request", "dsa.run"]
    assert spans[-1]["attributes"]["dsa.failure_code"] == "model_http_error"
    serialized = json.dumps(spans, default=str)
    for private in (
        "private-question-sentinel", "private-error-sentinel", str(request.database_path),
        "run-disabled",
    ):
        assert private not in serialized


def test_logfire_experiment_exports_safe_scores_from_retained_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pydantic_evals")
    import logfire
    from logfire.testing import TestExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = TestExporter()
    sdk = logfire.configure(
        send_to_logfire=False,
        console=False,
        metrics=False,
        additional_span_processors=[SimpleSpanProcessor(exporter)],
    )
    def configure(*, local: bool) -> logfire.Logfire:
        assert local is False
        return sdk

    monkeypatch.setattr(evaluation_telemetry, "configure_logfire", configure)
    pack = evaluation_pack(tmp_path)
    selected = prediction(case_id=pack.cases[0].case_id, answer={"count": 87654321})
    try:
        result = evaluation_telemetry.record_experiment(
            pack,
            (selected,),
            pack_id="fixture",
            model_id="fixture-model",
            metadata={"dsa.benchmark.cell_id": "cell-fixture"},
        )
    finally:
        logfire.configure(send_to_logfire=False, console=False, metrics=False)

    assert result.status == "recorded", result
    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    case = next(span for span in spans if span["name"] == "case: {case_name}")
    attributes = case["attributes"]
    assert attributes["inputs"] == pack.cases[0].case_id
    assert attributes["output"] == {"run_id": selected.run_id}
    assert attributes.get("expected_output") in (None, "null")
    assert attributes["assertions"]["end_to_end_exact_success"]["value"] is False
    assert attributes["assertions"]["agent_failure"]["value"] is True
    serialized = json.dumps(spans, default=str)
    assert "87654321" not in serialized
    assert pack.cases[0].question not in serialized
