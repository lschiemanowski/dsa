from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import duckdb
import pyarrow.parquet as parquet
import pytest
from pydantic import JsonValue, ValidationError
from pydantic_ai import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from dsa import RunFailure, RunRequest, RunSuccess, run_analysis
from dsa.environment import PythonExecutionRequest, PythonExecutionResult

from .test_contract import DRAFT_2020_12, local_reference_answer_schema, request_value
from .test_derivation import ClassifiedFailureExecutor, ResultExecutor, sample_derivation


def valid_request(tmp_path: Path, **policy: int) -> RunRequest:
    database = tmp_path / "source.duckdb"
    connection = duckdb.connect(str(database))
    connection.close()
    raw = request_value(database)
    raw_model = raw["model"]
    assert isinstance(raw_model, dict)
    raw_model["name"] = "test"
    raw["policy"] = policy
    return RunRequest.model_validate(raw)


def clock() -> Callable[[], datetime]:
    current = datetime(2026, 8, 26, 8, 0, tzinfo=UTC)

    def now() -> datetime:
        nonlocal current
        result = current
        current += timedelta(seconds=1)
        return result

    return now


async def test_invalid_request_has_no_identity_model_call_or_filesystem_effect(
    tmp_path: Path,
) -> None:
    called = False

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal called
        called = True
        return ModelResponse(parts=[])

    raw = request_value(tmp_path / "source.duckdb")
    raw["question"] = "  "
    runs_directory = tmp_path / "runs"

    with pytest.raises(ValidationError, match="question must not be blank"):
        await run_analysis(
            raw,
            runs_directory=runs_directory,
            model=FunctionModel(respond),
            identity_factory=lambda: "should-not-be-used",
        )

    assert called is False
    assert not runs_directory.exists()


async def test_mutated_typed_request_is_revalidated_before_side_effects(tmp_path: Path) -> None:
    request = valid_request(tmp_path)
    request.model.settings["api_key"] = "SECRET"
    identity_called = False
    runs_directory = tmp_path / "runs"

    def identity() -> str:
        nonlocal identity_called
        identity_called = True
        return "should-not-be-used"

    with pytest.raises(ValidationError, match="credentials or endpoints"):
        await run_analysis(
            request,
            runs_directory=runs_directory,
            model=TestModel(call_tools=[], custom_output_args={"count": 3}),
            identity_factory=identity,
        )

    assert identity_called is False
    assert not runs_directory.exists()


async def test_valid_structured_answer_succeeds_and_retains_native_messages(
    tmp_path: Path,
) -> None:
    completion = await run_analysis(
        valid_request(tmp_path),
        runs_directory=tmp_path / "runs",
        model=TestModel(call_tools=[], custom_output_args={"count": 3}),
        identity_factory=lambda: "run-001",
        clock=clock(),
    )

    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == {"count": 3}
    assert completion.record.usage["requests"] == 1
    assert [message["kind"] for message in completion.record.messages] == [
        "request",
        "response",
        "request",
    ]
    retained = json.loads(completion.retained_record.path.read_bytes())
    assert retained["outcome"] == {"status": "succeeded", "answer": {"count": 3}}
    assert retained["messages"] == list(completion.record.messages)


async def test_opted_in_derivation_is_replayed_and_retained_with_a_notebook(
    tmp_path: Path,
) -> None:
    request = valid_request(tmp_path)
    request = RunRequest.model_validate(
        {
            **request.model_dump(mode="python", round_trip=True),
            "derivation": {"format": "dsa-derivation/v1"},
        }
    )
    derivation = sample_derivation()

    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=TestModel(
            call_tools=[],
            custom_output_args={
                "answer": {"count": 3},
                "derivation": derivation.model_dump(mode="json"),
            },
        ),
        python_executor=ResultExecutor({"count": 3}),
        identity_factory=lambda: "run-derived",
        clock=clock(),
    )

    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == {"count": 3}
    assert completion.outcome.derivation == derivation
    assert completion.outcome.derivation_verification is not None
    assert completion.record.schema_version == "2"
    assert completion.retained_notebook is not None
    assert completion.retained_notebook.path.is_file()
    retained = json.loads(completion.retained_record.path.read_bytes())
    assert retained["request"]["derivation"] == {"format": "dsa-derivation/v1"}
    assert retained["outcome"]["derivation"]["cells"][0]["type"] == "markdown"
    assert retained["outcome"]["derivation_verification"]["status"] == "verified"


async def test_derivation_envelope_preserves_local_answer_schema_references(
    tmp_path: Path,
) -> None:
    request = valid_request(tmp_path)
    caller_schema = local_reference_answer_schema()
    request = RunRequest.model_validate(
        {
            **request.model_dump(mode="python", round_trip=True),
            "answer_schema": caller_schema,
            "derivation": {"format": "dsa-derivation/v1"},
        }
    )

    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=TestModel(
            call_tools=[],
            custom_output_args={
                "answer": {"count": 3},
                "derivation": sample_derivation().model_dump(mode="json"),
            },
        ),
        python_executor=ResultExecutor({"count": 3}),
        identity_factory=lambda: "run-derived-local-reference",
        clock=clock(),
    )

    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == {"count": 3}
    assert completion.record.request.answer_schema == caller_schema
    assert "$id" not in completion.record.request.answer_schema


async def test_answer_only_run_keeps_legacy_terminal_and_creates_no_notebook(
    tmp_path: Path,
) -> None:
    completion = await run_analysis(
        valid_request(tmp_path),
        runs_directory=tmp_path / "runs",
        model=TestModel(call_tools=[], custom_output_args={"count": 3}),
        identity_factory=lambda: "run-answer-only",
        clock=clock(),
    )

    assert isinstance(completion.outcome, RunSuccess)
    assert completion.record.schema_version == "1"
    assert completion.outcome.derivation is None
    assert completion.retained_notebook is None
    retained = json.loads(completion.retained_record.path.read_bytes())
    assert retained["outcome"] == {"status": "succeeded", "answer": {"count": 3}}
    assert "derivation" not in retained["request"]


async def test_python_tool_is_absent_without_an_injected_executor(tmp_path: Path) -> None:
    async def respond(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        assert [tool.name for tool in info.function_tools] == [
            "inspect_database",
            "query_database",
        ]
        return ModelResponse(
            parts=[ToolCallPart("final_answer", {"count": 3}, "answer")]
        )

    completion = await run_analysis(
        valid_request(tmp_path),
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        identity_factory=lambda: "run-no-python-executor",
        clock=clock(),
    )

    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == {"count": 3}


async def test_framework_output_schema_does_not_mutate_caller_schema(tmp_path: Path) -> None:
    request = valid_request(tmp_path)
    expected_schema = deepcopy(request.answer_schema)

    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=TestModel(call_tools=[], custom_output_args={"count": 3}),
        identity_factory=lambda: "run-schema-snapshot",
        clock=clock(),
    )

    assert request.answer_schema == expected_schema
    assert completion.record.request.answer_schema == expected_schema
    assert json.loads(completion.retained_record.path.read_bytes())["request"][
        "answer_schema"
    ] == expected_schema


async def test_typed_request_is_snapshotted_before_model_execution(tmp_path: Path) -> None:
    request = valid_request(tmp_path, max_validation_attempts=1)
    expected_request = request.model_dump(mode="json")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        entered.set()
        await release.wait()
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"count": 3}, "answer")]
        )

    task = asyncio.create_task(
        run_analysis(
            request,
            runs_directory=tmp_path / "runs",
            model=FunctionModel(respond, model_name="test"),
            identity_factory=lambda: "run-request-snapshot",
            clock=clock(),
        )
    )
    await entered.wait()
    properties = request.answer_schema["properties"]
    assert isinstance(properties, dict)
    count_schema = properties["count"]
    assert isinstance(count_schema, dict)
    count_schema["type"] = "string"
    request.model.settings["temperature"] = 1
    release.set()

    completion = await task

    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == {"count": 3}
    assert completion.record.request.model_dump(mode="json") == expected_request
    assert json.loads(completion.retained_record.path.read_bytes())["request"] == expected_request


async def test_schema_validation_feedback_retries_then_accepts_exact_answer(
    tmp_path: Path,
) -> None:
    calls = 0

    async def respond(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        count = -1 if calls == 1 else 2
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"count": count},
                    f"answer-{calls}",
                )
            ]
        )

    completion = await run_analysis(
        valid_request(tmp_path, max_validation_attempts=2),
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        identity_factory=lambda: "run-retry",
        clock=clock(),
    )

    assert calls == 2
    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == {"count": 2}
    transcript = json.dumps(completion.record.messages)
    assert "answer_schema_validation_failed" in transcript
    assert "minimum" in transcript


async def test_exhausted_answer_validation_is_a_typed_terminal_failure(
    tmp_path: Path,
) -> None:
    calls = 0

    async def respond(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"count": -1}, f"bad-{calls}")]
        )

    completion = await run_analysis(
        valid_request(tmp_path, max_validation_attempts=2),
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        identity_factory=lambda: "run-invalid-answer",
        clock=clock(),
    )

    assert calls == 2
    assert isinstance(completion.outcome, RunFailure)
    assert completion.outcome.failure.stage == "answer_validation"
    assert completion.outcome.failure.code == "attempts_exhausted"
    assert completion.outcome.failure.diagnostics["attempts"] == 2
    assert "answer_schema_validation_failed" in json.dumps(completion.record.messages)


async def test_provider_failure_preserves_specific_safe_diagnostics(tmp_path: Path) -> None:
    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(503, "test", {"error": "temporarily unavailable"})

    completion = await run_analysis(
        valid_request(tmp_path),
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        identity_factory=lambda: "run-model-failure",
        clock=clock(),
    )

    assert isinstance(completion.outcome, RunFailure)
    failure = completion.outcome.failure
    assert failure.stage == "model"
    assert failure.code == "model_http_error"
    assert failure.diagnostics["status_code"] == 503
    assert failure.diagnostics["model_name"] == "test"


async def test_provider_failure_does_not_retain_raw_api_error_text(tmp_path: Path) -> None:
    private_detail = "https://private.invalid/api?token=do-not-retain"

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        raise ModelAPIError("test", f"request failed at {private_detail}")

    completion = await run_analysis(
        valid_request(tmp_path),
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        identity_factory=lambda: "run-api-failure",
        clock=clock(),
    )

    assert isinstance(completion.outcome, RunFailure)
    failure = completion.outcome.failure
    assert failure.code == "model_api_error"
    assert failure.diagnostics == {
        "model_name": "test",
        "exception_type": "ModelAPIError",
    }
    assert private_detail not in completion.retained_record.path.read_text()


async def test_provider_failure_after_invalid_answer_is_not_misclassified(
    tmp_path: Path,
) -> None:
    calls = 0

    async def respond(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {"count": -1}, "bad-answer")]
            )
        raise ModelHTTPError(429, "test", {"error": "rate limited"})

    completion = await run_analysis(
        valid_request(tmp_path, max_validation_attempts=3),
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        identity_factory=lambda: "run-retry-provider-failure",
        clock=clock(),
    )

    assert isinstance(completion.outcome, RunFailure)
    assert completion.outcome.failure.stage == "model"
    assert completion.outcome.failure.code == "model_http_error"
    assert completion.outcome.failure.diagnostics["status_code"] == 429
    assert "answer_schema_validation_failed" in json.dumps(completion.record.messages)


async def test_missing_source_is_terminal_analysis_environment_failure_without_model(
    tmp_path: Path,
) -> None:
    called = False

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal called
        called = True
        return ModelResponse(parts=[])

    raw = request_value(tmp_path / "missing.duckdb")
    raw_model = raw["model"]
    assert isinstance(raw_model, dict)
    raw_model["name"] = "test"

    completion = await run_analysis(
        RunRequest.model_validate(raw),
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        identity_factory=lambda: "run-missing-source",
        clock=clock(),
    )

    assert called is False
    assert isinstance(completion.outcome, RunFailure)
    assert completion.outcome.failure.stage == "analysis_environment"
    assert completion.outcome.failure.code == "source_database_not_file"
    assert completion.retained_record.path.is_file()


async def test_non_duckdb_source_is_terminal_environment_failure_without_model(
    tmp_path: Path,
) -> None:
    called = False

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal called
        called = True
        return ModelResponse(parts=[])

    request = valid_request(tmp_path)
    request.database_path.write_bytes(b"not a DuckDB database")

    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        identity_factory=lambda: "run-invalid-database",
        clock=clock(),
    )

    assert called is False
    assert isinstance(completion.outcome, RunFailure)
    assert completion.outcome.failure.stage == "analysis_environment"
    assert completion.outcome.failure.code == "source_database_invalid"


async def test_non_object_caller_schema_is_unwrapped_at_the_public_boundary(
    tmp_path: Path,
) -> None:
    request = valid_request(tmp_path)
    request = RunRequest.model_validate(
        {
            **request.model_dump(),
            "answer_schema": {
                "$schema": DRAFT_2020_12,
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
            },
        }
    )

    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=TestModel(call_tools=[], custom_output_args={"value": [1, 2]}),
        identity_factory=lambda: "run-array",
        clock=clock(),
    )

    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == [1, 2]


async def test_cancellation_retains_terminal_record_and_propagates(tmp_path: Path) -> None:
    entered = asyncio.Event()

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    task = asyncio.create_task(
        run_analysis(
            valid_request(tmp_path),
            runs_directory=tmp_path / "runs",
            model=FunctionModel(respond, model_name="test"),
            identity_factory=lambda: "run-cancelled",
            clock=clock(),
        )
    )
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as captured:
        await task

    error = cast(Any, captured.value)
    record = error.terminal_record
    assert isinstance(record.outcome, RunFailure)
    assert record.outcome.failure.stage == "cancelled"
    assert error.retained_record.path.is_file()


def valid_database_request(tmp_path: Path, **policy: int) -> RunRequest:
    database = tmp_path / "analysis.duckdb"
    connection = duckdb.connect(str(database))
    try:
        connection.execute(
            "create table events as "
            "select i::integer as event_id, ('event-' || i)::varchar as label "
            "from range(12) values(i)"
        )
    finally:
        connection.close()
    raw = request_value(database)
    raw_model = raw["model"]
    assert isinstance(raw_model, dict)
    raw_model["name"] = "test"
    raw["policy"] = policy
    return RunRequest.model_validate(raw)


class EpisodePythonExecutor:
    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        table = parquet.read_table(  # pyright: ignore[reportUnknownMemberType]
            request.inputs_directory / "a1.parquet"
        )
        (request.output_directory / "answer.json").write_text(
            json.dumps({"count": table.num_rows}), encoding="utf-8"
        )
        return PythonExecutionResult(stdout="", stderr="")


class MutatingEpisodeExecutor:
    def __init__(self) -> None:
        self.database_path: Path | None = None

    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        self.database_path = request.database_path
        assert request.environment == {
            "DSAGENT_DATABASE": "/database/database.duckdb",
            "DSAGENT_INPUTS": "/inputs",
            "DSAGENT_OUTPUTS": "/outputs",
        }
        connection = duckdb.connect(str(request.database_path))
        try:
            connection.execute("insert into events values (12, 'event-12')")
        finally:
            connection.close()
        return PythonExecutionResult(
            runtime_identity="docker/test|sha256:" + "1" * 64,
        )


class MutatingAndDerivingExecutor:
    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        connection = duckdb.connect(str(request.database_path))
        try:
            if not request.expected_outputs:
                connection.execute("insert into events values (12, 'event-12')")
            else:
                row = cast(
                    tuple[int],
                    connection.execute("select count(*) from events").fetchone(),
                )
                count = row[0]
                (request.output_directory / "result.json").write_text(
                    json.dumps({"count": count}),
                    encoding="utf-8",
                )
        finally:
            connection.close()
        return PythonExecutionResult(
            runtime_identity="docker/test|sha256:" + "2" * 64,
        )


class SequenceDerivationExecutor:
    def __init__(self, results: list[JsonValue]) -> None:
        self.results = results
        self.calls = 0

    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        result = self.results[self.calls]
        self.calls += 1
        (request.output_directory / "result.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return PythonExecutionResult(runtime_identity="docker/test")


class ArtifactAndDerivationExecutor:
    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        if request.expected_outputs == ("answer.json",):
            table = parquet.read_table(  # pyright: ignore[reportUnknownMemberType]
                request.inputs_directory / "a1.parquet"
            )
            result = {"count": table.num_rows}
            name = "answer.json"
        else:
            assert request.expected_outputs == ("result.json",)
            connection = duckdb.connect(str(request.database_path), read_only=True)
            try:
                row = cast(
                    tuple[int],
                    connection.execute("select count(*) from events").fetchone(),
                )
                count = row[0]
            finally:
                connection.close()
            result = {"count": count}
            name = "result.json"
        (request.output_directory / name).write_text(json.dumps(result), encoding="utf-8")
        return PythonExecutionResult(runtime_identity="docker/test")


class BlockingDerivationExecutor:
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        del request
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def test_run_uses_private_database_commits_mutation_and_cleans_workspace(
    tmp_path: Path,
) -> None:
    """Successful Python mutates only the run copy and later SQL sees that commit."""
    request = valid_database_request(tmp_path)
    source = request.database_path
    source_before = source.read_bytes()
    executor = MutatingEpisodeExecutor()
    calls = 0

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_python",
                        {"source": "# mutate", "inputs": [], "expected_outputs": []},
                        "python-mutate",
                    )
                ]
            )
        if calls == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "query_database",
                        {"sql": "select count(*) from events"},
                        "query-mutated",
                    )
                ]
            )
        return ModelResponse(
            parts=[ToolCallPart("final_answer", {"count": 13}, "answer")]
        )

    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        python_executor=executor,
        identity_factory=lambda: "run-private-database",
        clock=clock(),
    )

    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == {"count": 13}
    assert executor.database_path is not None
    assert executor.database_path != source
    assert source.read_bytes() == source_before
    assert completion.record.database is not None
    assert completion.record.database.source_sha256 == sha256(source_before).hexdigest()
    assert completion.record.database.final_sha256 != completion.record.database.source_sha256
    assert not (tmp_path / "runs" / "run-private-database" / "work").exists()
    retained = json.loads(completion.retained_record.path.read_text())
    tool_results = [
        json.loads(part["content"])
        for message in retained["messages"]
        for part in message["parts"]
        if part["part_kind"] == "tool-return"
        and part.get("tool_name") == "query_database"
    ]
    assert any(result.get("rows") == [[13]] for result in tool_results)


async def test_derivation_replays_from_pristine_source_not_exploratory_state(
    tmp_path: Path,
) -> None:
    request = valid_database_request(tmp_path)
    request = RunRequest.model_validate(
        {
            **request.model_dump(mode="python", round_trip=True),
            "derivation": {"format": "dsa-derivation/v1"},
        }
    )
    calls = 0

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_python",
                        {"source": "# exploratory mutation", "inputs": [], "expected_outputs": []},
                        "mutate",
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "final_answer",
                    {
                        "answer": {"count": 12},
                        "derivation": sample_derivation().model_dump(mode="json"),
                    },
                    "derived-answer",
                )
            ]
        )

    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        python_executor=MutatingAndDerivingExecutor(),
        identity_factory=lambda: "run-pristine-derivation",
        clock=clock(),
    )

    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == {"count": 12}
    assert completion.record.database is not None
    assert completion.record.database.final_sha256 != completion.record.database.source_sha256


async def test_derivation_mismatch_retries_then_accepts_reproduced_answer(
    tmp_path: Path,
) -> None:
    request = valid_request(tmp_path, max_validation_attempts=2)
    request = RunRequest.model_validate(
        {
            **request.model_dump(mode="python", round_trip=True),
            "derivation": {"format": "dsa-derivation/v1"},
        }
    )
    model_calls = 0

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "final_answer",
                    {
                        "answer": {"count": 3},
                        "derivation": sample_derivation().model_dump(mode="json"),
                    },
                    f"answer-{model_calls}",
                )
            ]
        )

    executor = SequenceDerivationExecutor([{"count": 4}, {"count": 3}])
    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        python_executor=executor,
        identity_factory=lambda: "run-derivation-retry",
        clock=clock(),
    )

    assert model_calls == 2
    assert executor.calls == 2
    assert isinstance(completion.outcome, RunSuccess)
    assert "derivation_result_mismatch" in json.dumps(completion.record.messages)


async def test_exhausted_derivation_mismatches_are_a_typed_agent_failure(
    tmp_path: Path,
) -> None:
    request = valid_request(tmp_path, max_validation_attempts=2)
    request = RunRequest.model_validate(
        {
            **request.model_dump(mode="python", round_trip=True),
            "derivation": {"format": "dsa-derivation/v1"},
        }
    )

    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=TestModel(
            call_tools=[],
            custom_output_args={
                "answer": {"count": 3},
                "derivation": sample_derivation().model_dump(mode="json"),
            },
        ),
        python_executor=SequenceDerivationExecutor([{"count": 4}, {"count": 4}]),
        identity_factory=lambda: "run-derivation-exhausted",
        clock=clock(),
    )

    assert isinstance(completion.outcome, RunFailure)
    assert completion.outcome.failure.stage == "derivation_validation"
    assert completion.outcome.failure.code == "attempts_exhausted"
    assert completion.outcome.failure.diagnostics == {"attempts": 2}
    assert completion.retained_notebook is None


async def test_invalid_derivation_cell_sequence_uses_bounded_retries(
    tmp_path: Path,
) -> None:
    request = valid_request(tmp_path, max_validation_attempts=2)
    request = RunRequest.model_validate(
        {
            **request.model_dump(mode="python", round_trip=True),
            "derivation": {"format": "dsa-derivation/v1"},
        }
    )
    invalid = {
        "format": "dsa-derivation/v1",
        "cells": [
            {"type": "code", "source": "value = 3"},
            {"type": "code", "source": "result = {'count': value}"},
        ],
    }
    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=TestModel(
            call_tools=[],
            custom_output_args={"answer": {"count": 3}, "derivation": invalid},
        ),
        python_executor=ResultExecutor({"count": 3}),
        identity_factory=lambda: "run-invalid-derivation-shape",
        clock=clock(),
    )

    assert isinstance(completion.outcome, RunFailure)
    assert completion.outcome.failure.stage == "derivation_validation"
    assert completion.outcome.failure.code == "attempts_exhausted"
    assert completion.outcome.failure.diagnostics == {"attempts": 2}


async def test_derivation_container_cleanup_failure_is_not_retried(
    tmp_path: Path,
) -> None:
    request = valid_request(tmp_path, max_validation_attempts=3)
    request = RunRequest.model_validate(
        {
            **request.model_dump(mode="python", round_trip=True),
            "derivation": {"format": "dsa-derivation/v1"},
        }
    )
    executor = ClassifiedFailureExecutor("python_container_cleanup")
    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=TestModel(
            call_tools=[],
            custom_output_args={
                "answer": {"count": 3},
                "derivation": sample_derivation().model_dump(mode="json"),
            },
        ),
        python_executor=executor,
        identity_factory=lambda: "run-derivation-cleanup-failure",
        clock=clock(),
    )

    assert executor.calls == 1
    assert isinstance(completion.outcome, RunFailure)
    assert completion.outcome.failure.stage == "analysis_environment"
    assert completion.outcome.failure.code == "python_container_cleanup"
    assert completion.retained_notebook is None


async def test_derivation_can_accompany_a_retained_artifact_answer(
    tmp_path: Path,
) -> None:
    request = valid_database_request(tmp_path)
    request = RunRequest.model_validate(
        {
            **request.model_dump(mode="python", round_trip=True),
            "derivation": {"format": "dsa-derivation/v1"},
        }
    )
    calls = 0

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "query_database",
                        {"sql": "select * from events order by event_id"},
                        "query-for-derived-artifact",
                    )
                ]
            )
        if calls == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_python",
                        {
                            "source": "# write answer.json from a1",
                            "inputs": ["a1"],
                            "expected_outputs": ["answer.json"],
                        },
                        "write-derived-artifact",
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "answer_from_artifact",
                    {
                        "handle": "a2",
                        "derivation": sample_derivation().model_dump(mode="json"),
                    },
                    "submit-derived-artifact",
                )
            ]
        )

    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        python_executor=ArtifactAndDerivationExecutor(),
        identity_factory=lambda: "run-derived-artifact",
        clock=clock(),
    )

    assert calls == 3
    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == {"count": 12}
    assert completion.outcome.derivation_verification is not None
    assert completion.retained_notebook is not None


async def test_cancellation_during_derivation_replay_retains_terminal_only(
    tmp_path: Path,
) -> None:
    request = valid_request(tmp_path)
    request = RunRequest.model_validate(
        {
            **request.model_dump(mode="python", round_trip=True),
            "derivation": {"format": "dsa-derivation/v1"},
        }
    )
    executor = BlockingDerivationExecutor()
    task = asyncio.create_task(
        run_analysis(
            request,
            runs_directory=tmp_path / "runs",
            model=TestModel(
                call_tools=[],
                custom_output_args={
                    "answer": {"count": 3},
                    "derivation": sample_derivation().model_dump(mode="json"),
                },
            ),
            python_executor=executor,
            identity_factory=lambda: "run-cancelled-derivation",
            clock=clock(),
        )
    )
    await executor.entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as captured:
        await task

    error = cast(Any, captured.value)
    assert error.terminal_record.outcome.failure.stage == "cancelled"
    assert error.retained_record.path.is_file()
    run_directory = tmp_path / "runs" / "run-cancelled-derivation"
    assert not (run_directory / "derivation.ipynb").exists()
    assert not (run_directory / "work").exists()


async def test_operator_can_retain_private_database_for_debugging(tmp_path: Path) -> None:
    """The operator-only override retains the final run copy without changing the request."""
    request = valid_database_request(tmp_path)

    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=TestModel(call_tools=[], custom_output_args={"count": 12}),
        identity_factory=lambda: "run-retained-work",
        clock=clock(),
        keep_workdir=True,
    )

    working = tmp_path / "runs" / "run-retained-work" / "work" / "database.duckdb"
    assert working.is_file()
    assert completion.record.database is not None
    assert completion.record.database.final_sha256 == sha256(working.read_bytes()).hexdigest()
    assert "keep_workdir" not in completion.record.request.model_dump()


async def test_source_with_wal_is_rejected_before_model_execution(tmp_path: Path) -> None:
    """A potentially changing DuckDB source cannot become an inconsistent private copy."""
    request = valid_database_request(tmp_path)
    Path(f"{request.database_path}.wal").write_bytes(b"active writer")
    called = False

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal called
        called = True
        return ModelResponse(parts=[])

    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        identity_factory=lambda: "run-source-wal",
        clock=clock(),
    )

    assert called is False
    assert isinstance(completion.outcome, RunFailure)
    assert completion.outcome.failure.code == "source_database_wal"
    assert completion.record.database is None
    assert not (tmp_path / "runs" / "run-source-wal" / "work").exists()


async def test_episode_uses_database_artifact_python_locator_and_json_final_output(
    tmp_path: Path,
) -> None:
    calls = 0

    async def respond(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        assert [tool.name for tool in info.function_tools] == [
            "inspect_database",
            "query_database",
            "run_python",
        ]
        assert [tool.name for tool in info.output_tools] == [
            "final_answer",
            "answer_from_artifact",
        ]
        if calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("inspect_database", {}, "inspect-1")]
            )
        if calls == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "query_database",
                        {"sql": "select * from events order by event_id"},
                        "query-1",
                    )
                ]
            )
        if calls == 3:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_python",
                        {
                            "source": "# consume DSAGENT_INPUTS/a1.parquet",
                            "inputs": ["a1"],
                            "expected_outputs": ["answer.json"],
                        },
                        "python-1",
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "answer_from_artifact",
                    {"handle": "a2"},
                    "answer-artifact-1",
                )
            ]
        )

    completion = await run_analysis(
        valid_database_request(tmp_path),
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        python_executor=EpisodePythonExecutor(),
        identity_factory=lambda: "run-artifact-answer",
        clock=clock(),
    )

    assert calls == 4
    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == {"count": 12}
    assert [artifact.handle for artifact in completion.record.artifacts] == ["a1", "a2"]
    assert [artifact.producer_tool_call_id for artifact in completion.record.artifacts] == [
        "query-1",
        "python-1",
    ]
    terminal = completion.retained_record.path.read_text()
    assert "event-0" in terminal
    assert "event-11" not in terminal
    retained = json.loads(terminal)
    assert retained["outcome"]["answer"] == {"count": 12}
    assert len(retained["artifacts"]) == 2


async def test_cumulative_model_visible_tool_result_budget_is_terminal(
    tmp_path: Path,
) -> None:
    calls = 0

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[ToolCallPart("inspect_database", {}, "inspect-budget")]
        )

    completion = await run_analysis(
        valid_database_request(tmp_path, max_total_tool_result_bytes=20),
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        identity_factory=lambda: "run-tool-budget",
        clock=clock(),
    )

    assert calls == 1
    assert isinstance(completion.outcome, RunFailure)
    assert completion.outcome.failure.stage == "orchestration"
    assert completion.outcome.failure.code == "tool_result_limit_exceeded"


class InvalidAnswerExecutor:
    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        (request.output_directory / "answer.json").write_text(
            json.dumps({"count": -1}), encoding="utf-8"
        )
        return PythonExecutionResult()


async def test_schema_invalid_json_artifact_retries_then_accepts_direct_answer(
    tmp_path: Path,
) -> None:
    calls = 0

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "query_database",
                        {"sql": "select * from events order by event_id"},
                        "query-invalid-final",
                    )
                ]
            )
        if calls == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_python",
                        {
                            "source": "# write an answer artifact",
                            "inputs": ["a1"],
                            "expected_outputs": ["answer.json"],
                        },
                        "python-invalid-final",
                    )
                ]
            )
        if calls == 3:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "answer_from_artifact",
                        {"handle": "a2"},
                        "invalid-artifact-answer",
                    )
                ]
            )
        return ModelResponse(
            parts=[ToolCallPart("final_answer", {"count": 12}, "direct-recovery")]
        )

    completion = await run_analysis(
        valid_database_request(tmp_path, max_validation_attempts=2),
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        python_executor=InvalidAnswerExecutor(),
        identity_factory=lambda: "run-invalid-artifact-retry",
        clock=clock(),
    )

    assert calls == 4
    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == {"count": 12}
    assert "answer_schema_validation_failed" in json.dumps(completion.record.messages)


async def test_validation_retry_feedback_obeys_per_result_byte_limit(
    tmp_path: Path,
) -> None:
    request = valid_request(
        tmp_path,
        max_tool_result_bytes=128,
        max_validation_attempts=2,
    )
    request = RunRequest.model_validate(
        {
            **request.model_dump(),
            "answer_schema": {
                "$schema": DRAFT_2020_12,
                "type": "string",
                "maxLength": 5,
            },
        }
    )
    calls = 0

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        value = "x" * 1_000 if calls == 1 else "valid"
        return ModelResponse(
            parts=[ToolCallPart("final_answer", {"value": value}, f"answer-{calls}")]
        )

    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond, model_name="test"),
        identity_factory=lambda: "run-bounded-retry",
        clock=clock(),
    )

    assert calls == 2
    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == "valid"
    for message in completion.record.messages:
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        for raw_part in parts:
            if not isinstance(raw_part, dict):
                continue
            part = cast(dict[str, JsonValue], raw_part)
            if part.get("part_kind") == "retry-prompt":
                content = part.get("content")
                assert isinstance(content, str)
                assert len(content.encode("utf-8")) <= 128
