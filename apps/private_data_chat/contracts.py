"""Versioned, serializable contracts for the Private Data Chat boundary."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_MAX_ANSWER_SCHEMA_BYTES = 32 * 1024
_MAX_ANALYSIS_GUIDANCE_LENGTH = 12_000
_MAX_SCHEMA_DEPTH = 16
_MAX_SCHEMA_NODES = 512
_MAX_MOCK_CONTEXT_BYTES = 64 * 1024
_SCHEMA_KEYS = frozenset(
    {
        "$schema",
        "additionalProperties",
        "const",
        "description",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "multipleOf",
        "properties",
        "required",
        "title",
        "type",
        "uniqueItems",
    }
)
_SCHEMA_TYPES = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})


class AppContract(BaseModel):
    """Strict immutable base for data crossing an application trust boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProposalStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


class QuantitativeInterpretation(AppContract):
    """Human-readable decisions that make a quantitative question unambiguous."""

    measure: str = Field(min_length=1, max_length=1_000)
    population: str = Field(min_length=1, max_length=1_000)
    group_by: tuple[str, ...] = Field(default=(), max_length=16)
    filters: tuple[str, ...] = Field(default=(), max_length=32)
    time_window: str | None = Field(default=None, min_length=1, max_length=1_000)
    units: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("measure", "population", "time_window", "units")
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("interpretation text must not be blank")
        return value

    @field_validator("group_by", "filters", mode="before")
    @classmethod
    def snapshot_sequences(cls, value: object) -> object:
        return tuple(cast(list[object], value)) if isinstance(value, list) else value

    @field_validator("group_by", "filters")
    @classmethod
    def reject_blank_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("interpretation items must not be blank")
        return value


class ProposalPayload(AppContract):
    """The complete model-authored content that a user approves."""

    format: Literal["dsa-question-proposal/v1"] = "dsa-question-proposal/v1"
    question: str = Field(min_length=1, max_length=8_000)
    interpretation: QuantitativeInterpretation
    answer_schema: dict[str, JsonValue]
    analysis_guidance: str | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_ANALYSIS_GUIDANCE_LENGTH,
    )

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("analysis_guidance")
    @classmethod
    def reject_blank_guidance(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("analysis guidance must not be blank")
        return value

    @field_validator("answer_schema")
    @classmethod
    def validate_safe_answer_schema(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        schema = deepcopy(value)
        _validate_answer_schema(schema)
        return schema


class ClarifierTurn(AppContract):
    """One strictly parsed response from the untrusted clarification model."""

    format: Literal["dsa-clarifier-turn/v1"] = "dsa-clarifier-turn/v1"
    kind: Literal["clarification", "proposal"]
    message: str = Field(min_length=1, max_length=8_000)
    proposal: ProposalPayload | None = None

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("clarifier message must not be blank")
        return value

    @model_validator(mode="after")
    def validate_turn_shape(self) -> ClarifierTurn:
        if (self.kind == "proposal") != (self.proposal is not None):
            raise ValueError("only proposal turns may contain a proposal")
        return self


class ProposalBinding(AppContract):
    """Host-controlled identity and private-data binding excluded from model control."""

    user_id: str = Field(min_length=1, max_length=256)
    conversation_id: str = Field(min_length=1, max_length=256)
    data_source_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,127}$")


class ArtifactIdentity(AppContract):
    """Opaque retained-artifact identity safe to expose outside the trusted host."""

    artifact_id: str = Field(pattern=r"^artifact-[A-Za-z0-9_-]{16,128}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: Annotated[int, Field(gt=0)]
    media_type: str = Field(min_length=1, max_length=200)


class AnalysisRequest(AppContract):
    """Narrow request sent to the host-owned DSA execution adapter."""

    format: Literal["private-data-analysis/v1"] = "private-data-analysis/v1"
    run_id: str = Field(pattern=r"^analysis-[A-Za-z0-9_-]{16,128}$")
    proposal_id: str = Field(pattern=r"^proposal-[A-Za-z0-9_-]{16,128}$")
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_source_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,127}$")
    question: str = Field(min_length=1, max_length=8_000)
    answer_schema: dict[str, JsonValue]
    analysis_guidance: str | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_ANALYSIS_GUIDANCE_LENGTH,
    )
    request_derivation: Literal[True] = True

    @field_validator("analysis_guidance")
    @classmethod
    def reject_blank_guidance(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("analysis guidance must not be blank")
        return value

    @field_validator("answer_schema")
    @classmethod
    def validate_answer_schema(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        schema = deepcopy(value)
        _validate_answer_schema(schema)
        return schema


class AnalysisResult(AppContract):
    """Stable executor result; never contains host paths or raw exception text."""

    format: Literal["private-data-analysis-result/v1"] = "private-data-analysis-result/v1"
    run_id: str = Field(pattern=r"^analysis-[A-Za-z0-9_-]{16,128}$")
    proposal_id: str = Field(pattern=r"^proposal-[A-Za-z0-9_-]{16,128}$")
    status: Literal["succeeded", "failed"]
    answer: JsonValue | None = None
    terminal: ArtifactIdentity | None = None
    notebook: ArtifactIdentity | None = None
    failure_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,127}$")

    @model_validator(mode="after")
    def validate_outcome_shape(self) -> AnalysisResult:
        if self.status == "succeeded":
            if self.answer is None or self.terminal is None or self.failure_code is not None:
                raise ValueError(
                    "successful analysis requires an answer, terminal, and no failure code"
                )
        elif self.answer is not None or self.notebook is not None or self.failure_code is None:
            raise ValueError("failed analysis requires only a stable failure code")
        if self.terminal is not None and self.terminal.media_type != "application/json":
            raise ValueError("terminal artifact must use application/json")
        if self.notebook is not None and self.notebook.media_type != "application/x-ipynb+json":
            raise ValueError("notebook artifact must use application/x-ipynb+json")
        _ensure_finite_json(self.answer)
        return self


class ProposalRecord(AppContract):
    """Server-owned state for one immutable proposal and its at-most-once execution."""

    format: Literal["dsa-question-proposal-record/v1"] = "dsa-question-proposal-record/v1"
    proposal_id: str = Field(pattern=r"^proposal-[A-Za-z0-9_-]{16,128}$")
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: ProposalPayload
    binding: ProposalBinding
    status: ProposalStatus
    created_at: datetime
    expires_at: datetime
    approved_at: datetime | None = None
    run_id: str | None = Field(default=None, pattern=r"^analysis-[A-Za-z0-9_-]{16,128}$")
    result: AnalysisResult | None = None

    @model_validator(mode="after")
    def validate_state(self) -> ProposalRecord:
        timestamps = (self.created_at, self.expires_at, self.approved_at)
        if any(value is not None and value.utcoffset() is None for value in timestamps):
            raise ValueError("proposal timestamps must be timezone-aware")
        if self.expires_at <= self.created_at:
            raise ValueError("proposal expiry must follow creation")
        if self.approved_at is not None and not (
            self.created_at <= self.approved_at < self.expires_at
        ):
            raise ValueError("proposal approval time must precede expiry")
        expected_digest = proposal_digest(self.binding, self.payload)
        if self.proposal_sha256 != expected_digest:
            raise ValueError("proposal digest does not match its bound content")
        if self.status is ProposalStatus.PROPOSED:
            if self.approved_at is not None or self.run_id is not None or self.result is not None:
                raise ValueError("unapproved proposal contains execution state")
        elif self.status is ProposalStatus.EXPIRED:
            if self.run_id is not None or self.result is not None:
                raise ValueError("expired proposal contains execution state")
        elif self.status is ProposalStatus.APPROVED:
            if self.approved_at is None or self.run_id is not None or self.result is not None:
                raise ValueError("approved proposal has invalid execution state")
        elif self.status is ProposalStatus.RUNNING:
            if self.approved_at is None or self.run_id is None or self.result is not None:
                raise ValueError("running proposal has invalid execution state")
        else:
            if self.approved_at is None or self.run_id is None or self.result is None:
                raise ValueError("terminal proposal is missing execution state")
            if self.result.run_id != self.run_id or self.result.proposal_id != self.proposal_id:
                raise ValueError("analysis result does not match its proposal")
            expected_status = (
                ProposalStatus.SUCCEEDED
                if self.result.status == "succeeded"
                else ProposalStatus.FAILED
            )
            if self.status is not expected_status:
                raise ValueError("proposal status does not match its analysis result")
        return self


class MockRelation(AppContract):
    """One relation description containing synthetic examples only."""

    name: str = Field(min_length=1, max_length=256)
    columns: tuple[str, ...] = Field(min_length=1, max_length=100)
    sample_rows: tuple[dict[str, JsonValue], ...] = Field(default=(), max_length=5)

    @field_validator("columns", "sample_rows", mode="before")
    @classmethod
    def snapshot_sequences(cls, value: object) -> object:
        return tuple(cast(list[object], value)) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_rows(self) -> MockRelation:
        if len(set(self.columns)) != len(self.columns) or any(
            not item.strip() for item in self.columns
        ):
            raise ValueError("mock relation columns must be unique and nonblank")
        column_set = set(self.columns)
        for row in self.sample_rows:
            if set(row) != column_set:
                raise ValueError("every mock row must have exactly the declared columns")
            _ensure_finite_json(row)
        return self


class MockDatabaseContext(AppContract):
    """Bounded, explicitly synthetic context safe for the untrusted model."""

    format: Literal["dsa-mock-database/v1"] = "dsa-mock-database/v1"
    data_source_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=256)
    synthetic: Literal[True] = True
    relations: tuple[MockRelation, ...] = Field(min_length=1, max_length=20)

    @field_validator("relations", mode="before")
    @classmethod
    def snapshot_relations(cls, value: object) -> object:
        return tuple(cast(list[object], value)) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_context(self) -> MockDatabaseContext:
        names = [relation.name for relation in self.relations]
        if len(set(names)) != len(names):
            raise ValueError("mock relation names must be unique")
        if len(canonical_json_bytes(self)) > _MAX_MOCK_CONTEXT_BYTES:
            raise ValueError("mock database context exceeds its byte limit")
        return self


def canonical_json_bytes(value: BaseModel | dict[str, JsonValue]) -> bytes:
    """Return deterministic UTF-8 JSON for hashing and size enforcement."""
    raw = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        raw,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def proposal_payload_json(payload: ProposalPayload) -> dict[str, JsonValue]:
    """Project the exact proposal content shown to the user and covered by its digest."""
    raw = cast(dict[str, JsonValue], payload.model_dump(mode="json"))
    if raw.get("analysis_guidance") is None:
        raw.pop("analysis_guidance", None)
    return raw


def proposal_digest(binding: ProposalBinding, payload: ProposalPayload) -> str:
    """Bind exactly what the user approves to its host-controlled context."""
    envelope = {
        "binding": cast(JsonValue, binding.model_dump(mode="json")),
        "payload": cast(JsonValue, proposal_payload_json(payload)),
    }
    return sha256(canonical_json_bytes(envelope)).hexdigest()


def _validate_answer_schema(schema: dict[str, JsonValue]) -> None:
    if schema.get("$schema") != DRAFT_2020_12:
        raise ValueError("answer schema must declare JSON Schema Draft 2020-12")
    _ensure_finite_json(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ValueError("answer schema must be valid JSON Schema Draft 2020-12") from error
    if len(canonical_json_bytes(schema)) > _MAX_ANSWER_SCHEMA_BYTES:
        raise ValueError("answer schema exceeds its byte limit")
    node_count = [0]
    _validate_schema_node(schema, depth=0, node_count=node_count, root=True)


def _validate_schema_node(
    node: JsonValue,
    *,
    depth: int,
    node_count: list[int],
    root: bool = False,
) -> None:
    if depth > _MAX_SCHEMA_DEPTH:
        raise ValueError("answer schema exceeds its depth limit")
    node_count[0] += 1
    if node_count[0] > _MAX_SCHEMA_NODES:
        raise ValueError("answer schema exceeds its node limit")
    if not isinstance(node, dict):
        raise ValueError("answer schema must use explicit object subschemas")
    mapping = cast(dict[str, JsonValue], node)
    unsupported = sorted(set(mapping) - _SCHEMA_KEYS)
    if unsupported:
        raise ValueError(f"answer schema contains unsupported keywords: {', '.join(unsupported)}")
    schema_type = mapping.get("type")
    if not isinstance(schema_type, str) or schema_type not in _SCHEMA_TYPES:
        raise ValueError("every answer subschema must declare one supported type")
    if root and schema_type != "object":
        raise ValueError("answer schema root must be an object")
    properties = mapping.get("properties")
    required = mapping.get("required")
    if schema_type == "object":
        if not isinstance(properties, dict) or not properties:
            raise ValueError("object answer schemas require nonempty properties")
        if mapping.get("additionalProperties") is not False:
            raise ValueError("object answer schemas must forbid additional properties")
        property_names = set(properties)
        if not isinstance(required, list) or set(required) != property_names:
            raise ValueError("object answer schemas must require every declared property")
        if len(required) != len(property_names) or not all(
            isinstance(item, str) for item in required
        ):
            raise ValueError("object answer schema required keys must be unique strings")
        for child in properties.values():
            _validate_schema_node(child, depth=depth + 1, node_count=node_count)
    elif properties is not None or required is not None or "additionalProperties" in mapping:
        raise ValueError("object keywords require an object answer schema")
    items = mapping.get("items")
    if schema_type == "array":
        if items is None:
            raise ValueError("array answer schemas require items")
        _validate_schema_node(items, depth=depth + 1, node_count=node_count)
    elif items is not None:
        raise ValueError("items requires an array answer schema")


def _ensure_finite_json(value: JsonValue | None) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON values must be finite")
    if isinstance(value, dict):
        for child in cast(dict[str, JsonValue], value).values():
            _ensure_finite_json(child)
    elif isinstance(value, list):
        for child in cast(list[JsonValue], value):
            _ensure_finite_json(child)
