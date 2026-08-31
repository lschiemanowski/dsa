from __future__ import annotations

import asyncio
import json
import os
from contextlib import AbstractContextManager, contextmanager, nullcontext
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import JsonValue, ValidationError
from pydantic_ai.models.test import TestModel

from dsa import MlflowEvaluationError, MlflowReporting
from dsa.evaluation import (
    MlflowEvaluationPrediction,
    agent_failure,
    conditional_exact_json,
    conditional_policy_match,
    end_to_end_exact_success,
    end_to_end_policy_success,
    exact_json_equal,
    infrastructure_failure,
    json_answers_equal,
    run_mlflow_evaluation,
)
from dsa.pack import (
    EvaluationCaseMetadata,
    EvaluationPackCase,
    EvaluationPackManifest,
    HuggingFacePackReference,
    LoadedEvaluationPack,
)

from .test_episode import valid_request


def case_metadata() -> EvaluationCaseMetadata:
    return EvaluationCaseMetadata(family="counting", source_level="small")


def evaluation_pack(
    tmp_path: Path,
    *cases: EvaluationPackCase,
    scorer: dict[str, JsonValue] | None = None,
) -> LoadedEvaluationPack:
    request = valid_request(tmp_path)
    selected_cases = cases or (
        EvaluationPackCase(
            case_id="tiny-count",
            case_version="1",
            question=request.question,
            answer_schema=request.answer_schema,
            expected_answer={"count": 3},
            metadata=case_metadata(),
        ),
    )
    database_bytes = request.database_path.read_bytes()
    case_bytes = b"".join(
        json.dumps(
            case.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for case in selected_cases
    )
    manifest = EvaluationPackManifest.model_validate(
        {
            "format": "dsa-evaluation-pack/v1",
            "pack_id": "tiny-pack",
            "version": "1.0.0",
            "license": "CC0-1.0",
            "database": {
                "id": "tiny-database-1.0.0",
                "path": "database/tiny.duckdb",
                "size_bytes": len(database_bytes),
                "sha256": sha256(database_bytes).hexdigest(),
            },
            "cases": {
                "path": "cases.jsonl",
                "case_count": len(selected_cases),
                "size_bytes": len(case_bytes),
                "sha256": sha256(case_bytes).hexdigest(),
            },
            "scorer": scorer or {"name": "exact-json", "version": "1"},
            "provenance": {
                "source_datasets": (
                    {
                        "name": "tiny-source",
                        "version": "1",
                        "case_count": len(selected_cases),
                        "export_sha256": "b" * 64,
                    },
                )
            },
        }
    )
    manifest_bytes = (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    return LoadedEvaluationPack(
        reference=HuggingFacePackReference(
            repo_id="example/tiny",
            revision="c" * 40,
            path="tiny/1.0.0",
            manifest_sha256=sha256(manifest_bytes).hexdigest(),
        ),
        manifest=manifest,
        database_path=request.database_path,
        cases=selected_cases,
    )


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
    request = valid_request(tmp_path)
    case = EvaluationPackCase(
        case_id="tiny-count",
        case_version="1",
        question=request.question,
        answer_schema=request.answer_schema,
        expected_answer=expected,
        metadata=case_metadata(),
    )
    expected["count"] = 99
    pack = evaluation_pack(tmp_path, case)

    row = case.dataset_record(pack.manifest)

    assert row["expectations"] == {
        "answer": '{"count":3}',
        "scorer": '{"name":"exact-json","version":"1"}',
    }
    inputs = row["inputs"]
    assert isinstance(inputs, dict)
    assert "expected_answer" not in inputs
    assert "request" not in inputs
    assert "database_path" not in inputs
    assert "model" not in inputs
    assert "policy" not in inputs
    assert inputs["database_id"] == "tiny-database-1.0.0"
    assert row["tags"] == {
        "family": "counting",
        "pack": "tiny-pack",
        "pack_version": "1.0.0",
        "source_level": "small",
    }


def test_case_rejects_expectation_outside_answer_schema(tmp_path: Path) -> None:
    request = valid_request(tmp_path)
    with pytest.raises(ValidationError, match="must satisfy"):
        EvaluationPackCase(
            case_id="bad",
            case_version="1",
            question=request.question,
            answer_schema=request.answer_schema,
            expected_answer={"count": "three"},
            metadata=case_metadata(),
        )


def test_exact_json_and_failure_scorers_cover_the_all_case_denominator() -> None:
    expectations: dict[str, JsonValue] = {
        "answer": '{"count":3}',
        "scorer": '{"name":"exact-json","version":"1"}',
    }
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


def test_numeric_tolerance_is_explicit_recursive_and_never_weakens_integers() -> None:
    scorer = {
        "name": "json-numeric-tolerance",
        "version": "1",
        "relative_tolerance": 1e-9,
        "absolute_tolerance": 1e-12,
    }
    expectations: dict[str, JsonValue] = {
        "answer": '{"count":3,"ratio":0.1,"values":[1.0,true]}',
        "scorer": json.dumps(scorer, sort_keys=True, separators=(",", ":")),
    }

    assert json_answers_equal(
        {"count": 3, "ratio": 0.10000000001, "values": [1, True]},
        expectations,
    ) is True
    tolerated = prediction(
        answer={"count": 3, "ratio": 0.10000000001, "values": [1, True]}
    )
    assert conditional_exact_json(tolerated, expectations) is False
    assert end_to_end_exact_success(tolerated, expectations) is False
    assert conditional_policy_match(tolerated, expectations) is True
    assert end_to_end_policy_success(tolerated, expectations) is True
    assert agent_failure(tolerated, expectations) is False
    assert json_answers_equal(
        {"count": 3.0, "ratio": 0.1, "values": [1, True]},
        expectations,
    ) is False
    assert json_answers_equal(
        {"count": 3, "ratio": 0.1001, "values": [1, True]},
        expectations,
    ) is False
    assert json_answers_equal(
        {"count": 3, "ratio": 0.1, "values": [1, 1]},
        expectations,
    ) is False
    assert json_answers_equal(
        {"count": 3, "ratio": 10**1000, "values": [1, True]},
        expectations,
    ) is False


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
            self.dataset.digest = "pre-merge-digest"
            self.refreshed_dataset = Dataset()
            self.refreshed_dataset.digest = "post-merge-digest"
            self.output: dict[str, Any] | None = None
            self.scorer_names: list[str] = []
            self.run_tags: dict[str, str] | None = None

        def create_dataset(self, *, name: str, experiment_id: str) -> Dataset:
            assert name == "tiny-dataset"
            assert experiment_id == "123"
            return self.dataset

        def get_dataset(self, *, name: str) -> Dataset:
            assert name == "tiny-dataset"
            self.refreshed_dataset.records = self.dataset.records
            return self.refreshed_dataset

        def scorer(self, function: object, *, name: str) -> object:
            self.scorer_names.append(name)
            return function

        def invalid_feedback(self, name: str, rationale: str) -> object:
            return {"name": name, "rationale": rationale, "valid": False}

        @contextmanager
        def evaluation_context(self, tags: dict[str, str]) -> Any:
            self.run_tags = tags
            yield object()

        def evaluate(
            self,
            *,
            data: Dataset,
            predict_fn: Any,
            scorers: list[object],
        ) -> Result:
            assert data is self.refreshed_dataset
            assert len(scorers) == 6
            assert os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] == "true"
            inputs = data.records[0]["inputs"]
            assert isinstance(inputs, dict)
            monkeypatch.delenv("DATABRICKS_TOKEN")
            self.output = asyncio.run(predict_fn(**inputs))
            assert self.output is not None
            return Result(self.output)

    api = Api()
    request = valid_request(tmp_path)
    case = EvaluationPackCase(
        case_id="tiny-count",
        case_version="1",
        question=request.question,
        answer_schema=request.answer_schema,
        expected_answer={"count": 3},
        metadata=case_metadata(),
    )
    pack = evaluation_pack(tmp_path, case)

    result = run_mlflow_evaluation(
        pack,
        dataset_name="tiny-dataset",
        runs_directory=tmp_path / "runs",
        model_configuration=request.model,
        policy=request.policy,
        model_factory=lambda _case: TestModel(
            call_tools=[], custom_output_args={"count": 3}
        ),
        api=cast(Any, api),
        run_tags={
            "dsa.benchmark.cell_id": "cell-1",
            "dsa.benchmark.study_sha256": "a" * 64,
        },
    )

    assert result.dataset_id == "dataset-1"
    assert result.dataset_digest == "post-merge-digest"
    assert result.evaluation_run_id == "evaluation-run"
    assert len(result.predictions) == 1
    assert result.predictions[0].answer == {"count": 3}
    assert api.scorer_names == [
        "end_to_end_exact_success",
        "conditional_exact_json",
        "end_to_end_policy_success",
        "conditional_policy_match",
        "agent_failure",
        "infrastructure_failure",
    ]
    assert api.run_tags == {
        "dsa.benchmark.cell_id": "cell-1",
        "dsa.benchmark.study_sha256": "a" * 64,
    }
    assert api.dataset.records[0]["expectations"] == {
        "answer": '{"count":3}',
        "scorer": '{"name":"exact-json","version":"1"}',
    }
    assert api.output is not None
    assert api.output["accepted"] is True
    assert api.output["reporting"] == {
        "status": "failed",
        "tracking_run_id": None,
        "trace_id": None,
        "failure_code": "mlflow_configuration_missing",
    }
    assert os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] == "false"


def test_evaluation_reserves_a_case_before_paid_or_persistent_work(
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
            self.records: list[dict[str, Any]] = []

        def merge_records(self, records: list[dict[str, Any]]) -> Dataset:
            self.records.extend(records)
            return self

    class Api:
        def __init__(self) -> None:
            self.dataset = Dataset()

        def create_dataset(self, **_kwargs: object) -> Dataset:
            return self.dataset

        def get_dataset(self, **_kwargs: object) -> Dataset:
            return self.dataset

        def scorer(self, function: object, *, name: str) -> object:
            return function

        def invalid_feedback(self, name: str, rationale: str) -> object:
            return {"name": name, "rationale": rationale, "valid": False}

        def evaluate(
            self,
            *,
            data: Dataset,
            predict_fn: Any,
            scorers: list[object],
        ) -> object:
            inputs = data.records[0]["inputs"]
            assert isinstance(inputs, dict)
            monkeypatch.delenv("DATABRICKS_TOKEN")

            async def invoke_duplicate() -> None:
                await predict_fn(**inputs)
                await predict_fn(**inputs)

            asyncio.run(invoke_duplicate())
            raise AssertionError("duplicate invocation should have been rejected")

    model_factory_calls = 0

    def model_factory(_case: EvaluationPackCase) -> TestModel:
        nonlocal model_factory_calls
        model_factory_calls += 1
        return TestModel(call_tools=[], custom_output_args={"count": 3})

    request = valid_request(tmp_path)
    with pytest.raises(MlflowEvaluationError) as caught:
        run_mlflow_evaluation(
            evaluation_pack(tmp_path),
            dataset_name="tiny-dataset",
            runs_directory=tmp_path / "runs",
            model_configuration=request.model,
            policy=request.policy,
            model_factory=model_factory,
            api=cast(Any, Api()),
        )

    assert caught.value.code == "mlflow_evaluation_failed"
    assert model_factory_calls == 1
    assert len(list((tmp_path / "runs").glob("*/terminal.json"))) == 1


def test_evaluation_rejects_unsafe_benchmark_tags_before_mlflow_work(
    tmp_path: Path,
) -> None:
    request = valid_request(tmp_path)

    with pytest.raises(ValueError, match="safe benchmark identities"):
        run_mlflow_evaluation(
            evaluation_pack(tmp_path),
            dataset_name="tiny-dataset",
            runs_directory=tmp_path / "runs",
            model_configuration=request.model,
            policy=request.policy,
            run_tags={"dsa.benchmark.cell_id": "https://private.example/secret"},
        )


def test_native_api_starts_the_evaluation_run_with_benchmark_tags() -> None:
    captured: list[dict[str, str]] = []

    class Mlflow:
        def start_run(self, *, tags: dict[str, str]) -> AbstractContextManager[None]:
            captured.append(tags)
            return nullcontext()

    api_type: Any = vars(import_module("dsa.evaluation"))["_MlflowEvaluationApi"]
    api = api_type.__new__(api_type)
    api.mlflow = Mlflow()

    with api.evaluation_context({"dsa.benchmark.cell_id": "cell-1"}):
        pass

    assert captured == [{"dsa.benchmark.cell_id": "cell-1"}]


def test_native_api_reuses_an_existing_dataset_and_only_creates_when_missing() -> None:
    class MissingDataset(Exception):
        error_code = "RESOURCE_DOES_NOT_EXIST"

    existing = SimpleNamespace(dataset_id="existing", digest="existing-digest")
    created = SimpleNamespace(dataset_id="created", digest="created-digest")

    class Datasets:
        def __init__(self) -> None:
            self.found = True
            self.create_conflict = False
            self.created: list[tuple[str, str]] = []

        def get_dataset(self, *, name: str) -> object:
            assert name == "catalog.schema.dataset"
            if self.found:
                return existing
            raise MissingDataset

        def create_dataset(self, *, name: str, experiment_id: str) -> object:
            self.created.append((name, experiment_id))
            if self.create_conflict:
                self.found = True
                raise RuntimeError("another worker created the dataset")
            return created

    api_type: Any = vars(import_module("dsa.evaluation"))["_MlflowEvaluationApi"]
    api = api_type.__new__(api_type)
    api.datasets = Datasets()

    assert api.create_dataset(
        name="catalog.schema.dataset", experiment_id="123"
    ) is existing
    assert api.datasets.created == []

    api.datasets.found = False
    assert api.create_dataset(
        name="catalog.schema.dataset", experiment_id="123"
    ) is created
    assert api.datasets.created == [("catalog.schema.dataset", "123")]

    api.datasets.found = False
    api.datasets.create_conflict = True
    assert api.create_dataset(
        name="catalog.schema.dataset", experiment_id="123"
    ) is existing
    assert api.datasets.created == [
        ("catalog.schema.dataset", "123"),
        ("catalog.schema.dataset", "123"),
    ]


def test_native_api_does_not_hide_dataset_lookup_failures() -> None:
    class Datasets:
        def get_dataset(self, *, name: str) -> object:
            raise RuntimeError(f"lookup failed for {name}")

    api_type: Any = vars(import_module("dsa.evaluation"))["_MlflowEvaluationApi"]
    api = api_type.__new__(api_type)
    api.datasets = Datasets()

    with pytest.raises(RuntimeError, match="lookup failed"):
        api.create_dataset(name="catalog.schema.dataset", experiment_id="123")


def test_native_api_does_not_hide_dataset_creation_failures() -> None:
    class MissingDataset(Exception):
        error_code = "RESOURCE_DOES_NOT_EXIST"

    class Datasets:
        def get_dataset(self, *, name: str) -> object:
            raise MissingDataset(name)

        def create_dataset(self, *, name: str, experiment_id: str) -> object:
            raise RuntimeError(f"creation failed for {name} in {experiment_id}")

    api_type: Any = vars(import_module("dsa.evaluation"))["_MlflowEvaluationApi"]
    api = api_type.__new__(api_type)
    api.datasets = Datasets()

    with pytest.raises(RuntimeError, match="creation failed"):
        api.create_dataset(name="catalog.schema.dataset", experiment_id="123")


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
        assert len(scorers) == 6
        assert run_id == "native-evaluation-run"
        for inputs in reversed(eval_df["inputs"].tolist()):
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

        def get_dataset(self, *, name: str) -> Dataset:
            assert name == "native-dataset"
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

    def model_factory(_case: EvaluationPackCase) -> TestModel:
        nonlocal model_factory_calls
        model_factory_calls += 1
        return TestModel(call_tools=[], custom_output_args={"count": 3})

    request = valid_request(tmp_path)
    first_case = EvaluationPackCase(
        case_id="first-native-case",
        case_version="1",
        question=request.question,
        answer_schema=request.answer_schema,
        expected_answer={"count": 3},
        metadata=case_metadata(),
    )
    second_case = EvaluationPackCase(
        case_id="second-native-case",
        case_version="1",
        question=request.question,
        answer_schema=request.answer_schema,
        expected_answer={"count": 3},
        metadata=case_metadata(),
    )
    pack = evaluation_pack(tmp_path, first_case, second_case)
    result = run_mlflow_evaluation(
        pack,
        dataset_name="native-dataset",
        runs_directory=tmp_path / "runs",
        model_configuration=request.model,
        policy=request.policy,
        model_factory=model_factory,
        api=cast(Any, NativeApi()),
    )

    assert model_factory_calls == 2
    assert len(outputs) == 2
    assert len(list((tmp_path / "runs").glob("*/terminal.json"))) == 2
    assert len(result.predictions) == 2
    assert tuple(item.case_id for item in result.predictions) == (
        "first-native-case",
        "second-native-case",
    )
    assert "MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION" not in os.environ


def test_native_evaluation_rejects_local_or_incomplete_tracking_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = valid_request(tmp_path)
    case = EvaluationPackCase(
        case_id="tiny-count",
        case_version="1",
        question=request.question,
        answer_schema=request.answer_schema,
        expected_answer={"count": 3},
        metadata=case_metadata(),
    )
    pack = evaluation_pack(tmp_path, case)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:/tmp/mlruns")

    with pytest.raises(MlflowEvaluationError) as caught:
        run_mlflow_evaluation(
            pack,
            dataset_name="tiny",
            runs_directory=tmp_path / "runs",
            model_configuration=request.model,
            policy=request.policy,
        )

    assert caught.value.code == "mlflow_tracking_uri_invalid"


def test_native_evaluation_reverifies_the_pack_database_before_remote_work(
    tmp_path: Path,
) -> None:
    """A changed cached database cannot be evaluated under its released identity."""
    request = valid_request(tmp_path)
    pack = evaluation_pack(tmp_path)
    pack.database_path.write_bytes(b"changed after pack resolution")

    with pytest.raises(MlflowEvaluationError) as caught:
        run_mlflow_evaluation(
            pack,
            dataset_name="tiny",
            runs_directory=tmp_path / "runs",
            model_configuration=request.model,
            policy=request.policy,
        )

    assert caught.value.code == "evaluation_database_mismatch"
    assert not (tmp_path / "runs").exists()
