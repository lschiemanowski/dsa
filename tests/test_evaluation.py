from __future__ import annotations

import asyncio
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue, ValidationError
from pydantic_ai.models.test import TestModel

from dsa import EvaluationError, LogfireReporting, RunCompletion, RunRequest
from dsa.evaluation import (
    EvaluationPrediction,
    agent_failure,
    conditional_exact_json,
    conditional_policy_match,
    end_to_end_exact_success,
    end_to_end_policy_success,
    exact_json_equal,
    infrastructure_failure,
    json_answers_equal,
    run_evaluation,
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


def prediction(**updates: object) -> EvaluationPrediction:
    values: dict[str, object] = {
        "case_id": "case-1",
        "run_id": "run-1",
        "terminal_sha256": "a" * 64,
        "accepted": True,
        "answer": {"count": 3},
        "reporting": LogfireReporting(),
    }
    values.update(updates)
    return EvaluationPrediction.model_validate(values)






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
        reporting=LogfireReporting(
            status="failed",
            failure_code="logfire_recording_failed",
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

    assert end_to_end_exact_success(reporting_failed, expectations) is True
    assert conditional_exact_json(reporting_failed, expectations) is True
    assert infrastructure_failure(reporting_failed) is False
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


async def test_local_evaluation_runs_verified_cases_once_in_pack_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dsa import evaluation
    from dsa.runner import run_analysis

    case = evaluation_pack(tmp_path).cases[0]
    pack = evaluation_pack(
        tmp_path,
        *(case.model_copy(update={"case_id": f"count-{i}"}) for i in range(4)),
    )
    request = valid_request(tmp_path)
    entered: list[str] = []
    in_flight = peak = 0
    two_started = asyncio.Event()

    async def observe(
        selected: RunRequest | object, **kwargs: Any
    ) -> RunCompletion:
        nonlocal in_flight, peak
        assert isinstance(selected, RunRequest)
        assert selected.question == case.question
        assert set(selected.model_dump()) == {
            "database_path", "question", "answer_schema", "model", "policy"
        }
        assert kwargs["report_to_logfire"] is False
        in_flight += 1
        peak = max(peak, in_flight)
        if in_flight == 2:
            two_started.set()
        try:
            await asyncio.wait_for(two_started.wait(), timeout=5)
            return await run_analysis(selected, **kwargs)
        finally:
            in_flight -= 1

    def model_factory(item: EvaluationPackCase) -> TestModel:
        entered.append(item.case_id)
        return TestModel(call_tools=[], custom_output_args={"count": 3})

    monkeypatch.setattr(evaluation, "run_analysis", observe)
    result = await evaluation.run_evaluation(
        pack,
        runs_directory=tmp_path / "runs",
        model_configuration=request.model,
        policy=request.policy,
        case_workers=2,
        model_factory=model_factory,
    )

    expected = [item.case_id for item in pack.cases]
    assert [item.case_id for item in result.predictions] == expected
    assert sorted(entered) == sorted(expected)
    assert peak == 2
    assert all(item.accepted and item.answer == {"count": 3} for item in result.predictions)
    assert len(list((tmp_path / "runs").glob("*/terminal.json"))) == len(expected)


async def test_evaluation_rechecks_database_before_starting_any_case(tmp_path: Path) -> None:
    pack = evaluation_pack(tmp_path)
    request = valid_request(tmp_path)
    pack.database_path.write_bytes(b"tampered")

    with pytest.raises(EvaluationError, match="evaluation_database_mismatch"):
        await run_evaluation(
            pack,
            runs_directory=tmp_path / "runs",
            model_configuration=request.model,
            policy=request.policy,
        )
    assert not (tmp_path / "runs").exists()
