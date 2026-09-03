from __future__ import annotations

import asyncio
import json
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from dsa import MlflowReporting, RetainedDerivationNotebook, RunSuccess, run_analysis
from dsa import reporting as reporting_module
from dsa.record import ArtifactRecord

from .test_derivation import ResultExecutor, sample_derivation
from .test_episode import clock, expected_derivation_receipt, valid_request


def test_reporting_result_states_are_closed_and_safe() -> None:
    assert MlflowReporting().model_dump(mode="json") == {
        "status": "disabled",
        "tracking_run_id": None,
        "trace_id": None,
        "failure_code": None,
    }
    assert MlflowReporting(
        status="reported",
        tracking_run_id="a" * 32,
        trace_id="tr-123",
    ).status == "reported"
    assert MlflowReporting(
        status="failed",
        tracking_run_id="a" * 32,
        failure_code="mlflow_export_failed",
    ).status == "failed"

    with pytest.raises(ValidationError, match="requires both remote identifiers"):
        MlflowReporting(status="reported", tracking_run_id="run")
    with pytest.raises(ValidationError, match="stable failure code"):
        MlflowReporting(status="failed")
    with pytest.raises(ValidationError):
        MlflowReporting(
            status="failed",
            failure_code="https://private.example/TOKEN",
        )


async def test_reporting_flag_is_strict_before_identity_or_filesystem(
    tmp_path: Path,
) -> None:
    identity_called = False
    runs_directory = tmp_path / "runs"

    def identity() -> str:
        nonlocal identity_called
        identity_called = True
        return "should-not-be-used"

    with pytest.raises(TypeError, match="report_to_mlflow must be a boolean"):
        await run_analysis(
            valid_request(tmp_path),
            runs_directory=runs_directory,
            model=TestModel(call_tools=[], custom_output_args={"count": 3}),
            identity_factory=identity,
            report_to_mlflow=cast(Any, 1),
        )

    assert identity_called is False
    assert not runs_directory.exists()


async def test_disabled_reporting_is_absent_from_canonical_terminal(
    tmp_path: Path,
) -> None:
    completion = await run_analysis(
        valid_request(tmp_path),
        runs_directory=tmp_path / "runs",
        model=TestModel(call_tools=[], custom_output_args={"count": 3}),
        identity_factory=lambda: "run-reporting-disabled",
        clock=clock(),
    )

    assert completion.reporting == MlflowReporting()
    terminal = json.loads(completion.retained_record.path.read_bytes())
    assert "reporting" not in terminal
    assert "report_to_mlflow" not in terminal["request"]
    assert completion.record.model_dump(mode="json") == terminal


async def test_missing_databricks_configuration_does_not_change_analysis_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    completion = await run_analysis(
        valid_request(tmp_path),
        runs_directory=tmp_path / "runs",
        model=TestModel(call_tools=[], custom_output_args={"count": 3}),
        identity_factory=lambda: "run-reporting-missing-config",
        clock=clock(),
        report_to_mlflow=True,
    )

    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == {"count": 3}
    assert completion.reporting == MlflowReporting(
        status="failed",
        failure_code="mlflow_configuration_missing",
    )
    assert "mlflow" not in completion.retained_record.path.read_text()


async def test_terminal_export_uses_exact_bytes_and_safe_failure_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline(function: Any, *args: object, **kwargs: object) -> Any:
        return function(*args, **kwargs)

    # This unit exercises export semantics, not thread scheduling; keep the fake inline.
    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    completion = await run_analysis(
        valid_request(tmp_path),
        runs_directory=tmp_path / "runs",
        model=TestModel(call_tools=[], custom_output_args={"count": 3}),
        identity_factory=lambda: "run-reporting-export",
        clock=clock(),
    )
    terminal_text = completion.retained_record.path.read_text()

    class Entity:
        def __init__(self, *values: object) -> None:
            self.values = values

    class Mlflow:
        def flush_trace_async_logging(self) -> None:
            return None

    class Client:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.text: list[tuple[str, str]] = []
            self.terminated: list[str] = []

        def create_run(self, *_args: object, **_kwargs: object) -> Any:
            return SimpleNamespace(info=SimpleNamespace(run_id="tracking"))

        def log_text(self, _run_id: str, text: str, path: str) -> None:
            if self.fail:
                raise RuntimeError("https://private.example/?token=SECRET")
            self.text.append((path, text))

        def log_batch(self, *_args: object, **_kwargs: object) -> None:
            return None

        def link_traces_to_run(self, *_args: object) -> None:
            return None

        def set_terminated(self, _run_id: str, status: str) -> None:
            self.terminated.append(status)

    backend_type = reporting_module.__dict__["_MlflowBackend"]
    backend = backend_type.__new__(backend_type)
    backend.mlflow = Mlflow()
    backend.metric_type = Entity
    backend.param_type = Entity
    backend.tag_type = Entity

    cost_record = completion.record.model_copy(
        update={
            "messages": (
                {
                    "kind": "response",
                    "provider_name": "openrouter",
                    "provider_response_id": "gen-cost-1",
                    "provider_details": {"cost": 0.1},
                },
                {
                    "kind": "response",
                    "provider_name": "openrouter",
                    "provider_response_id": "gen-cost-2",
                    "provider_details": {"cost": 0.2},
                },
            )
        }
    )
    metrics, _, tags = backend._run_metadata(cost_record)
    metric_values = {item.values[0]: item.values[1] for item in metrics}
    tag_values = {item.values[0]: item.values[1] for item in tags}
    assert metric_values["dsa.provider_cost_usd"] == 0.3
    assert metric_values["dsa.provider_cost_generations"] == 2
    assert tag_values["dsa.provider_cost_status"] == "observed"

    client = Client()
    selected_client = client

    def client_factory(*, tracking_uri: str) -> Client:
        assert tracking_uri == "databricks"
        return selected_client

    backend.client_type = client_factory
    result = await backend._export_completion("123", "trace", completion)

    assert result.status == "reported"
    assert client.text[0] == ("dsa/terminal.json", terminal_text)
    assert json.loads(client.text[1][1]) == {
        "schema_version": "1",
        "run_id": "run-reporting-export",
        "artifacts": [],
    }
    assert client.terminated == ["FINISHED"]

    derived_request = valid_request(tmp_path)
    derived_request = derived_request.model_validate(
        {
            **derived_request.model_dump(mode="python", round_trip=True),
            "derivation": {"format": "dsa-derivation/v1"},
        }
    )
    derivation_json = sample_derivation().model_dump(mode="json")
    receipt = expected_derivation_receipt(
        derived_request,
        derivation_json,
        {"count": 3},
    )
    derived_calls = 0

    async def respond_derived(
        _messages: list[Any],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal derived_calls
        derived_calls += 1
        if derived_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "validate_derivation",
                        derivation_json,
                        "validate-derivation",
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "final_answer",
                    {"answer": {"count": 3}, "derivation_receipt": receipt},
                    "answer",
                )
            ]
        )

    derived_completion = await run_analysis(
        derived_request,
        runs_directory=tmp_path / "derived-runs",
        model=FunctionModel(respond_derived, model_name="test"),
        python_executor=ResultExecutor({"count": 3}),
        identity_factory=lambda: "run-reporting-derived",
        clock=clock(),
    )
    assert derived_completion.retained_notebook is not None
    notebook_path = derived_completion.retained_notebook.path
    notebook_text = notebook_path.read_text()
    notebook_client = Client()
    selected_client = notebook_client
    notebook_result = await backend._export_completion(
        "123", "trace", derived_completion
    )
    assert notebook_result.status == "reported"
    assert [path for path, _text in notebook_client.text] == [
        "dsa/terminal.json",
        "dsa/derivation.ipynb",
        "dsa/artifacts.json",
    ]
    assert notebook_client.text[1][1] == notebook_text

    forged_notebook = completion.model_copy(
        update={
            "retained_notebook": RetainedDerivationNotebook(
                path=notebook_path,
                sha256=sha256(notebook_text.encode()).hexdigest(),
                byte_length=len(notebook_text.encode()),
            )
        }
    )
    forged_client = Client()
    selected_client = forged_client
    forged_result = await backend._export_completion("123", "trace", forged_notebook)
    assert forged_result.failure_code == "mlflow_export_failed"
    assert forged_client.text == []

    notebook_path.write_text("tampered\n")
    notebook_integrity_client = Client()
    selected_client = notebook_integrity_client
    notebook_integrity_failure = await backend._export_completion(
        "123", "trace", derived_completion
    )
    assert notebook_integrity_failure.failure_code == "mlflow_export_failed"
    assert notebook_integrity_client.text == []
    notebook_path.write_text(notebook_text)

    failed_client = Client(fail=True)
    selected_client = failed_client
    failed = await backend._export_completion("123", "trace", completion)
    assert failed.failure_code == "mlflow_export_failed"
    assert "SECRET" not in failed.model_dump_json()

    completion.retained_record.path.write_bytes(b"tampered\n")
    integrity_client = Client()
    selected_client = integrity_client
    integrity_failure = await backend._export_completion("123", "trace", completion)
    assert integrity_failure.failure_code == "mlflow_export_failed"
    assert integrity_client.text == []


async def test_trace_setup_failure_executes_analysis_once_without_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline(function: Any, *args: object, **kwargs: object) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "123")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")
    monkeypatch.setitem(reporting_module.__dict__, "_autolog_initialized", False)

    completion = await run_analysis(
        valid_request(tmp_path),
        runs_directory=tmp_path / "runs",
        model=TestModel(call_tools=[], custom_output_args={"count": 3}),
        identity_factory=lambda: "run-before-trace-setup",
        clock=clock(),
    )
    calls = 0

    async def operation() -> Any:
        nonlocal calls
        calls += 1
        return completion

    @contextmanager
    def broken_span(**_kwargs: object) -> Any:
        raise RuntimeError("private reporting setup detail")
        yield

    class PydanticAi:
        def autolog(self, **_kwargs: object) -> None:
            return None

    class Client:
        terminated: ClassVar[list[str]] = []

        def __init__(self, *, tracking_uri: str) -> None:
            assert tracking_uri == "databricks"

        def create_run(self, *_args: object, **_kwargs: object) -> Any:
            return SimpleNamespace(info=SimpleNamespace(run_id="tracking-setup"))

        def set_terminated(self, _run_id: str, status: str) -> None:
            self.terminated.append(status)

    backend_type = reporting_module.__dict__["_MlflowBackend"]
    backend = backend_type.__new__(backend_type)

    def tracing_context(**_kwargs: object) -> AbstractContextManager[None]:
        return nullcontext()

    def location(*, experiment_id: str) -> str:
        return experiment_id

    backend.mlflow = SimpleNamespace(
        tracing=SimpleNamespace(context=tracing_context),
        start_span=broken_span,
    )
    backend.pydantic_ai = PydanticAi()
    backend.client_type = Client
    backend.location_type = location
    backend.span_type = SimpleNamespace(AGENT="AGENT")

    result = await backend.run(
        enabled=True,
        run_id="run-trace-setup",
        request=completion.record.request,
        operation=operation,
    )

    assert calls == 1
    assert isinstance(result.outcome, RunSuccess)
    assert result.reporting == MlflowReporting(
        status="failed",
        failure_code="mlflow_trace_setup_failed",
    )
    assert Client.terminated == []


async def test_enabled_and_disabled_reporting_contexts_are_task_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline(function: Any, *args: object, **kwargs: object) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "123")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")
    monkeypatch.setitem(reporting_module.__dict__, "_autolog_initialized", False)

    disabled_completion = await run_analysis(
        valid_request(tmp_path),
        runs_directory=tmp_path / "disabled-runs",
        model=TestModel(call_tools=[], custom_output_args={"count": 3}),
        identity_factory=lambda: "run-disabled-context",
        clock=clock(),
    )
    enabled_completion = await run_analysis(
        valid_request(tmp_path),
        runs_directory=tmp_path / "enabled-runs",
        model=TestModel(call_tools=[], custom_output_args={"count": 3}),
        identity_factory=lambda: "run-enabled-context",
        clock=clock(),
    )

    enabled_state: ContextVar[bool | None] = ContextVar("enabled_state", default=None)

    @contextmanager
    def tracing_context(
        *, enabled: bool, **_kwargs: object
    ) -> Any:
        token = enabled_state.set(enabled)
        try:
            yield
        finally:
            enabled_state.reset(token)

    class Span:
        trace_id = "trace-enabled"

        def set_inputs(self, _value: object) -> None:
            return None

        def set_outputs(self, _value: object) -> None:
            return None

    @contextmanager
    def start_span(**_kwargs: object) -> Any:
        yield Span()

    class PydanticAi:
        def autolog(self, **_kwargs: object) -> None:
            return None

    class Entity:
        def __init__(self, *_values: object) -> None:
            return None

    class Client:
        def __init__(self, *, tracking_uri: str) -> None:
            assert tracking_uri == "databricks"

        def create_run(self, *_args: object, **_kwargs: object) -> Any:
            return SimpleNamespace(info=SimpleNamespace(run_id="tracking-enabled"))

        def log_text(self, *_args: object) -> None:
            return None

        def log_batch(self, *_args: object, **_kwargs: object) -> None:
            return None

        def link_traces_to_run(self, *_args: object) -> None:
            return None

        def set_terminated(self, *_args: object) -> None:
            return None

    class Mlflow:
        def __init__(self) -> None:
            self.tracing = SimpleNamespace(context=tracing_context)
            self.start_span = start_span

        def flush_trace_async_logging(self) -> None:
            return None

    def location(*, experiment_id: str) -> str:
        return experiment_id

    backend_type = reporting_module.__dict__["_MlflowBackend"]
    backend = backend_type.__new__(backend_type)
    backend.mlflow = Mlflow()
    backend.pydantic_ai = PydanticAi()
    backend.client_type = Client
    backend.location_type = location
    backend.span_type = SimpleNamespace(AGENT="AGENT")
    backend.metric_type = Entity
    backend.param_type = Entity
    backend.tag_type = Entity

    disabled_entered = asyncio.Event()
    enabled_entered = asyncio.Event()
    release = asyncio.Event()

    async def disabled_operation() -> Any:
        assert enabled_state.get() is False
        disabled_entered.set()
        await enabled_entered.wait()
        assert enabled_state.get() is False
        release.set()
        return disabled_completion

    async def enabled_operation() -> Any:
        await disabled_entered.wait()
        assert enabled_state.get() is True
        enabled_entered.set()
        await release.wait()
        assert enabled_state.get() is True
        return enabled_completion

    disabled_task = asyncio.create_task(
        backend.run(
            enabled=False,
            run_id="run-disabled-context",
            request=disabled_completion.record.request,
            operation=disabled_operation,
        )
    )
    enabled_task = asyncio.create_task(
        backend.run(
            enabled=True,
            run_id="run-enabled-context",
            request=enabled_completion.record.request,
            operation=enabled_operation,
        )
    )
    disabled_result, enabled_result = await asyncio.gather(disabled_task, enabled_task)

    assert disabled_result.reporting.status == "disabled"
    assert enabled_result.reporting.status == "reported"
    assert enabled_state.get() is None


async def test_broken_disabled_tracing_context_cannot_block_analysis(
    tmp_path: Path,
) -> None:
    completion = await run_analysis(
        valid_request(tmp_path),
        runs_directory=tmp_path / "runs",
        model=TestModel(call_tools=[], custom_output_args={"count": 3}),
        identity_factory=lambda: "run-broken-disabled-context",
        clock=clock(),
    )
    calls = 0

    async def operation() -> Any:
        nonlocal calls
        calls += 1
        return completion

    def broken_context(**_kwargs: object) -> AbstractContextManager[None]:
        raise RuntimeError("broken optional tracing context")

    backend_type = reporting_module.__dict__["_MlflowBackend"]
    backend = backend_type.__new__(backend_type)
    backend.mlflow = SimpleNamespace(
        tracing=SimpleNamespace(context=broken_context),
    )

    result = await backend.run(
        enabled=False,
        run_id="run-broken-disabled-context",
        request=completion.record.request,
        operation=operation,
    )

    assert calls == 1
    assert isinstance(result.outcome, RunSuccess)
    assert result.reporting.status == "disabled"


def test_artifact_manifest_contains_metadata_not_artifact_bytes(tmp_path: Path) -> None:
    secret = "ARTIFACT-CONTENT-MUST-NOT-BE-UPLOADED"
    artifact_path = tmp_path / "a1.json"
    artifact_path.write_text(secret)
    record = ArtifactRecord(
        handle="a1",
        relative_path="artifacts/a1.json",
        media_type="application/json",
        size_bytes=len(secret),
        sha256="a" * 64,
        producer_tool_call_id="tool-1",
    )
    base = SimpleNamespace(
        schema_version="1",
        run_id="run-manifest",
        artifacts=(record,),
    )

    manifest_text = reporting_module.__dict__["_artifact_manifest_text"]
    text = cast(str, manifest_text(base))

    assert secret not in text
    assert str(artifact_path) not in text
    assert json.loads(text)["artifacts"] == [record.model_dump(mode="json")]
