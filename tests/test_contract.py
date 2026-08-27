from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue, ValidationError

from dsa import ModelConfiguration, RunPolicy, RunRequest

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
ANSWER_SCHEMA = {
    "$schema": DRAFT_2020_12,
    "type": "object",
    "properties": {"count": {"type": "integer", "minimum": 0}},
    "required": ["count"],
    "additionalProperties": False,
}


def request_value(database_path: Path) -> dict[str, object]:
    return {
        "database_path": database_path,
        "question": "How many rows are in the events relation?",
        "answer_schema": deepcopy(ANSWER_SCHEMA),
        "model": {
            "name": "openrouter:example/model",
            "settings": {"temperature": 0, "seed": 7},
        },
        "policy": {},
    }


def test_request_is_compact_strict_and_json_serializable(tmp_path: Path) -> None:
    raw = request_value(tmp_path / "source.duckdb")

    request = RunRequest.model_validate(raw)
    retained = request.model_dump(mode="json")

    assert retained == {
        "database_path": str(tmp_path / "source.duckdb"),
        "question": "How many rows are in the events relation?",
        "answer_schema": ANSWER_SCHEMA,
        "model": {
            "name": "openrouter:example/model",
            "settings": {"seed": 7, "temperature": 0},
        },
        "policy": RunPolicy().model_dump(mode="json"),
    }
    assert set(retained) == {
        "database_path",
        "question",
        "answer_schema",
        "model",
        "policy",
    }

    with pytest.raises(ValidationError, match="extra_forbidden"):
        RunRequest.model_validate({**raw, "expectations": {"count": 3}})


def test_request_snapshots_caller_owned_json(tmp_path: Path) -> None:
    raw = request_value(tmp_path / "source.duckdb")
    request = RunRequest.model_validate(raw)

    schema = raw["answer_schema"]
    settings = raw["model"]
    assert isinstance(schema, dict)
    assert isinstance(settings, dict)
    schema["type"] = "array"
    model_settings = cast(dict[str, object], settings["settings"])
    assert isinstance(model_settings, dict)
    model_settings["temperature"] = 1

    assert request.answer_schema == ANSWER_SCHEMA
    assert request.model.settings == {"seed": 7, "temperature": 0}


@pytest.mark.parametrize("question", ["", "  \n\t"])
def test_question_must_be_completely_specific_nonblank_text(
    tmp_path: Path,
    question: str,
) -> None:
    raw = request_value(tmp_path / "source.duckdb")
    raw["question"] = question

    with pytest.raises(ValidationError, match="question must not be blank"):
        RunRequest.model_validate(raw)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object"},
        {"$schema": DRAFT_2020_12, "type": "not-a-json-schema-type"},
    ],
)
def test_answer_contract_must_be_valid_draft_2020_12(
    tmp_path: Path,
    schema: dict[str, object],
) -> None:
    raw = request_value(tmp_path / "source.duckdb")
    raw["answer_schema"] = schema

    with pytest.raises(ValidationError, match="Draft 2020-12"):
        RunRequest.model_validate(raw)


def test_answer_contract_allows_local_refs_but_rejects_external_refs(tmp_path: Path) -> None:
    local = request_value(tmp_path / "source.duckdb")
    local["answer_schema"] = {
        "$schema": DRAFT_2020_12,
        "$defs": {"count": {"type": "integer", "minimum": 0}},
        "type": "object",
        "properties": {"count": {"$ref": "#/$defs/count"}},
        "required": ["count"],
    }
    external = deepcopy(local)
    external_schema = external["answer_schema"]
    assert isinstance(external_schema, dict)
    external_schema["properties"] = {
        "count": {"$ref": "https://example.invalid/schema.json"}
    }

    assert RunRequest.model_validate(local).answer_schema["$defs"]
    with pytest.raises(ValidationError, match="external references"):
        RunRequest.model_validate(external)


def test_answer_contract_inspects_references_only_in_schema_locations(tmp_path: Path) -> None:
    literal = request_value(tmp_path / "source.duckdb")
    literal["answer_schema"] = {
        "$schema": DRAFT_2020_12,
        "const": {"$ref": "https://example.invalid/literal-value"},
    }
    dynamic = request_value(tmp_path / "source.duckdb")
    dynamic["answer_schema"] = {
        "$schema": DRAFT_2020_12,
        "$dynamicRef": "https://example.invalid/schema.json#node",
    }

    assert RunRequest.model_validate(literal).answer_schema["const"] == {
        "$ref": "https://example.invalid/literal-value"
    }
    with pytest.raises(ValidationError, match="external references"):
        RunRequest.model_validate(dynamic)


@pytest.mark.parametrize(
    "settings",
    [
        {"api_key": "secret"},
        {"base_url": "https://provider.invalid"},
        {"extra_body": {"access_token": "secret"}},
        {"extra_headers": {"Authorization": "Bearer secret"}},
    ],
)
def test_credentials_and_endpoints_are_not_serializable_model_settings(
    settings: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="credentials or endpoints"):
        ModelConfiguration(
            name="openrouter:example/model",
            settings=cast(dict[str, JsonValue], settings),
        )


def test_policy_is_flat_positive_and_caps_inline_previews() -> None:
    dumped = RunPolicy().model_dump()

    assert dumped
    assert all(not isinstance(value, dict) for value in dumped.values())
    assert dumped["max_python_cpus"] == 2
    assert dumped["max_python_processes"] == 64

    with pytest.raises(ValidationError):
        RunPolicy(max_model_requests=0)
    with pytest.raises(ValidationError):
        RunPolicy(max_preview_rows=6)
    with pytest.raises(ValidationError):
        RunPolicy(max_total_tool_result_bytes=0)
    with pytest.raises(ValidationError):
        RunPolicy(max_python_cpus=0)
    with pytest.raises(ValidationError):
        RunPolicy(max_python_processes=0)
