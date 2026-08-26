from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from dsa import RunFailure, RunRequest, RunSuccess, run_analysis

from .test_contract import DRAFT_2020_12, request_value


def valid_request(tmp_path: Path, **policy: int) -> RunRequest:
    database = tmp_path / "source.duckdb"
    database.write_bytes(b"fixture")
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
