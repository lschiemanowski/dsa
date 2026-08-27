from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import JsonValue, ValidationError
from pydantic_ai.models.test import TestModel

from dsa import MlflowEvaluationCase, MlflowEvaluationError, MlflowReporting
from dsa.evaluation import (
    MlflowEvaluationPrediction,
    agent_failure,
    conditional_exact_json,
    end_to_end_exact_success,
    exact_json_equal,
    infrastructure_failure,
    run_mlflow_evaluation,
)

from .test_episode import valid_request


def prediction(**updates: object) -> MlflowEvaluationPrediction:
    values: dict[str, object] = {
        "case_id": "case-1",
        "run_id": "run-1",
        "terminal_sha256": "a" * 64,
        "accepted": True,
        "answer": {"count": 3},
        "reporting": {
            "status": "reported",
            "tracking_run_id": "tracking",
            "trace_id": "trace",
        },
    }
    values.update(updates)
    return MlflowEvaluationPrediction.model_validate(values)


def test_case_snapshots_expectation_and_keeps_it_out_of_inputs(tmp_path: Path) -> None:
    expected: dict[str, JsonValue] = {"count": 3}
    case = MlflowEvaluationCase(
        case_id="tiny-count",
        request=valid_request(tmp_path),
        expected_answer=expected,
    )
    expected["count"] = 99

    row = case.dataset_record()

    assert row["expectations"] == {"answer": {"count": 3}}
    inputs = row["inputs"]
    assert isinstance(inputs, dict)
    assert "expected_answer" not in inputs
    request = inputs["request"]
    assert isinstance(request, dict)
    assert "expected" not in request


def test_case_rejects_expectation_outside_answer_schema(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="must satisfy"):
        MlflowEvaluationCase(
            case_id="bad",
            request=valid_request(tmp_path),
            expected_answer={"count": "three"},
        )


def test_exact_json_and_failure_scorers_cover_the_all_case_denominator() -> None:
    expectations: dict[str, JsonValue] = {"answer": {"count": 3}}
    exact = prediction()
    wrong = prediction(answer={"count": 4})
    provider = prediction(
        accepted=False,
        answer=None,
        failure_stage="model",
        failure_code="model_http_error",
    )
    invalid = prediction(
        accepted=False,
        answer=None,
        failure_stage="answer_validation",
        failure_code="attempts_exhausted",
    )
    reporting_failed = prediction(
        reporting=MlflowReporting(
            status="failed",
            failure_code="mlflow_export_failed",
        )
    )

    assert exact_json_equal({"b": 2, "a": 1}, {"a": 1, "b": 2}) is True
    assert exact_json_equal(1, 1.0) is False
    assert end_to_end_exact_success(exact, expectations) is True
    assert conditional_exact_json(exact, expectations) is True
    assert agent_failure(exact, expectations) is False
    assert infrastructure_failure(exact) is False

    assert end_to_end_exact_success(wrong, expectations) is False
    assert conditional_exact_json(wrong, expectations) is False
    assert agent_failure(wrong, expectations) is True

    assert end_to_end_exact_success(provider, expectations) is False
    assert conditional_exact_json(provider, expectations) is None
    assert infrastructure_failure(provider) is True
    assert agent_failure(provider, expectations) is False

    assert infrastructure_failure(invalid) is False
    assert agent_failure(invalid, expectations) is True

    assert end_to_end_exact_success(reporting_failed, expectations) is False
    assert conditional_exact_json(reporting_failed, expectations) is True
    assert infrastructure_failure(reporting_failed) is True
    assert agent_failure(reporting_failed, expectations) is False


def test_native_evaluation_uses_dataset_expectations_and_normal_reporting_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "123")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

    class Dataset:
        dataset_id = "dataset-1"
        digest = "digest-1"

        def __init__(self) -> None:
            self.records: list[dict[str, object]] = []

        def merge_records(self, records: list[dict[str, object]]) -> Dataset:
            self.records.extend(records)
            return self

    class Column(list[object]):
        def tolist(self) -> list[object]:
            return list(self)

    class Frame:
        def __init__(self, output: dict[str, Any]) -> None:
            self.output = output

        def __contains__(self, key: object) -> bool:
            return key == "outputs"

        def __getitem__(self, key: str) -> Column:
            assert key == "outputs"
            return Column([self.output])

    class Result:
        def __init__(self, output: dict[str, Any]) -> None:
            self.run_id = "evaluation-run"
            self.metrics = {"end_to_end_exact_success/mean": 0.0}
            self.result_df = Frame(output)

    class Api:
        def __init__(self) -> None:
            self.dataset = Dataset()
            self.output: dict[str, Any] | None = None
            self.scorer_names: list[str] = []

        def create_dataset(self, *, name: str, experiment_id: str) -> Dataset:
            assert name == "tiny-dataset"
            assert experiment_id == "123"
            return self.dataset

        def scorer(self, function: object, *, name: str) -> object:
            self.scorer_names.append(name)
            return function

        def invalid_feedback(self, name: str, rationale: str) -> object:
            return {"name": name, "rationale": rationale, "valid": False}

        def evaluate(
            self,
            *,
            data: Dataset,
            predict_fn: Any,
            scorers: list[object],
        ) -> Result:
            assert data is self.dataset
            assert len(scorers) == 4
            inputs = data.records[0]["inputs"]
            assert isinstance(inputs, dict)
            monkeypatch.delenv("DATABRICKS_TOKEN")
            self.output = asyncio.run(predict_fn(**inputs))
            assert self.output is not None
            return Result(self.output)

    api = Api()
    case = MlflowEvaluationCase(
        case_id="tiny-count",
        request=valid_request(tmp_path),
        expected_answer={"count": 3},
    )

    result = run_mlflow_evaluation(
        [case],
        dataset_name="tiny-dataset",
        runs_directory=tmp_path / "runs",
        model_factory=lambda _case: TestModel(
            call_tools=[], custom_output_args={"count": 3}
        ),
        api=cast(Any, api),
    )

    assert result.dataset_id == "dataset-1"
    assert result.dataset_digest == "digest-1"
    assert result.evaluation_run_id == "evaluation-run"
    assert len(result.predictions) == 1
    assert result.predictions[0].answer == {"count": 3}
    assert api.scorer_names == [
        "end_to_end_exact_success",
        "conditional_exact_json",
        "agent_failure",
        "infrastructure_failure",
    ]
    assert api.dataset.records[0]["expectations"] == {"answer": {"count": 3}}
    assert api.output is not None
    assert api.output["accepted"] is True
    assert api.output["reporting"] == {
        "status": "failed",
        "tracking_run_id": None,
        "trace_id": None,
        "failure_code": "mlflow_configuration_missing",
    }


def test_native_evaluation_rejects_local_or_incomplete_tracking_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = MlflowEvaluationCase(
        case_id="tiny-count",
        request=valid_request(tmp_path),
        expected_answer={"count": 3},
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:/tmp/mlruns")

    with pytest.raises(MlflowEvaluationError) as caught:
        run_mlflow_evaluation(
            [case],
            dataset_name="tiny",
            runs_directory=tmp_path / "runs",
        )

    assert caught.value.code == "mlflow_tracking_uri_invalid"
