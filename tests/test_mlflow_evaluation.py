from __future__ import annotations

import asyncio
import os
from contextlib import AbstractContextManager, contextmanager, nullcontext
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
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

    for code in (
        "run_timeout",
        "tool_result_limit_exceeded",
        "usage_limit_exceeded",
    ):
        host_limit = prediction(
            accepted=False,
            answer=None,
            failure_stage="orchestration",
            failure_code=code,
        )
        assert infrastructure_failure(host_limit) is True
        assert agent_failure(host_limit, expectations) is False

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
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "false")

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
            assert os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] == "true"
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
    assert os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] == "false"


def test_native_mlflow_api_executes_one_analysis_per_dataset_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mlflow = pytest.importorskip("mlflow")
    pandas = pytest.importorskip("pandas")
    base_module: Any = import_module("mlflow.genai.evaluation.base")
    harness_module: Any = import_module("mlflow.genai.evaluation.harness")
    scorers_module: Any = import_module("mlflow.genai.scorers")
    entities_module: Any = import_module("mlflow.entities")

    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "123")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")
    monkeypatch.delenv("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", raising=False)

    @contextmanager
    def evaluation_run() -> Any:
        yield SimpleNamespace(
            info=SimpleNamespace(run_id="native-evaluation-run"),
            data=SimpleNamespace(tags={}),
        )

    class Client:
        def set_tag(self, *_args: object, **_kwargs: object) -> None:
            return None

    outputs: list[dict[str, Any]] = []

    def harness_run(
        *,
        predict_fn: Any,
        eval_df: Any,
        scorers: list[object],
        run_id: str,
    ) -> Any:
        assert predict_fn is not None
        assert len(scorers) == 4
        assert run_id == "native-evaluation-run"
        for inputs in eval_df["inputs"].tolist():
            outputs.append(predict_fn(inputs))
        return SimpleNamespace(
            run_id=run_id,
            metrics={"end_to_end_exact_success/mean": 0.0},
            result_df=pandas.DataFrame({"outputs": outputs}),
        )

    def evaluation_autologging(
        **_kwargs: object,
    ) -> AbstractContextManager[None]:
        return nullcontext()

    def no_log(*_args: object, **_kwargs: object) -> None:
        return None

    def no_display(*_args: object) -> None:
        return None

    monkeypatch.setattr(base_module, "_start_run_or_reuse_active_run", evaluation_run)
    monkeypatch.setattr(
        base_module,
        "configure_autologging_for_evaluation",
        evaluation_autologging,
    )
    monkeypatch.setattr(base_module, "_log_dataset_input", no_log)
    monkeypatch.setattr(base_module, "MlflowClient", Client)
    monkeypatch.setattr(base_module, "display_evaluation_output", no_display)
    monkeypatch.setattr(harness_module, "run", harness_run)

    class Dataset:
        dataset_id = "native-dataset"
        digest = "native-digest"

        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        def merge_records(self, records: list[dict[str, Any]]) -> Dataset:
            self.records.extend(records)
            return self

    class NativeApi:
        def __init__(self) -> None:
            self.dataset = Dataset()

        def create_dataset(self, *, name: str, experiment_id: str) -> Dataset:
            assert name == "native-dataset"
            assert experiment_id == "123"
            return self.dataset

        def scorer(self, function: Any, *, name: str) -> object:
            return scorers_module.scorer(function, name=name)

        def invalid_feedback(self, name: str, rationale: str) -> object:
            return entities_module.Feedback(
                name=name,
                value=None,
                rationale=rationale,
                valid=False,
            )

        def evaluate(
            self,
            *,
            data: Dataset,
            predict_fn: Any,
            scorers: list[object],
        ) -> object:
            monkeypatch.delenv("DATABRICKS_TOKEN")
            return mlflow.genai.evaluate(
                data=data.records,
                predict_fn=predict_fn,
                scorers=scorers,
            )

    model_factory_calls = 0

    def model_factory(_case: MlflowEvaluationCase) -> TestModel:
        nonlocal model_factory_calls
        model_factory_calls += 1
        return TestModel(call_tools=[], custom_output_args={"count": 3})

    first_case = MlflowEvaluationCase(
        case_id="first-native-case",
        request=valid_request(tmp_path),
        expected_answer={"count": 3},
    )
    second_case = MlflowEvaluationCase(
        case_id="second-native-case",
        request=valid_request(tmp_path),
        expected_answer={"count": 3},
    )
    result = run_mlflow_evaluation(
        [first_case, second_case],
        dataset_name="native-dataset",
        runs_directory=tmp_path / "runs",
        model_factory=model_factory,
        api=cast(Any, NativeApi()),
    )

    assert model_factory_calls == 2
    assert len(outputs) == 2
    assert len(list((tmp_path / "runs").glob("*/terminal.json"))) == 2
    assert len(result.predictions) == 2
    assert "MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION" not in os.environ


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
