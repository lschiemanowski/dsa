"""Serializable public input contract for one analysis run."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Literal, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from pydantic_core import PydanticCustomError
from referencing.jsonschema import DRAFT202012

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
PositiveInt = Annotated[int, Field(gt=0)]
_MAX_DERIVATION_CELLS = 24
_MAX_DERIVATION_SOURCE_BYTES = 64 * 1024
_SAFE_MODEL_SETTING_KEYS = frozenset(
    {
        "frequency_penalty",
        "logit_bias",
        "max_tokens",
        "parallel_tool_calls",
        "presence_penalty",
        "seed",
        "service_tier",
        "stop_sequences",
        "temperature",
        "thinking",
        "timeout",
        "tool_choice",
        "top_k",
        "top_p",
    }
)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelConfiguration(ContractModel):
    """Serializable model selection without credentials or provider endpoints."""

    name: str
    settings: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise PydanticCustomError("model_name_blank", "model name must not be blank")
        return value

    @field_validator("settings")
    @classmethod
    def snapshot_safe_settings(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        unsupported = sorted(set(value) - _SAFE_MODEL_SETTING_KEYS)
        if unsupported:
            raise PydanticCustomError(
                "unsafe_model_setting",
                "model settings must not contain credentials or endpoints",
                {"keys": unsupported},
            )
        ensure_finite_json(value, "model settings")
        return {key: deepcopy(value[key]) for key in sorted(value)}


class RunPolicy(ContractModel):
    """Flat host-enforced limits for an analysis run."""

    max_run_seconds: PositiveInt = 900
    max_model_requests: PositiveInt = 20
    max_total_tokens: PositiveInt = 200_000
    max_model_output_tokens: PositiveInt = 16_384
    max_tool_calls: PositiveInt = 40
    max_validation_attempts: PositiveInt = 3
    max_tool_result_bytes: PositiveInt = 32 * 1024
    max_total_tool_result_bytes: PositiveInt = 256 * 1024
    max_preview_rows: Annotated[int, Field(ge=0, le=5)] = 5
    max_inspection_seconds: PositiveInt = 30
    max_inspection_result_bytes: PositiveInt = 64 * 1024
    max_query_seconds: PositiveInt = 120
    max_query_memory_bytes: PositiveInt = 512 * 1024 * 1024
    max_query_rows: PositiveInt = 1_000_000
    max_query_result_bytes: PositiveInt = 64 * 1024 * 1024
    max_python_seconds: PositiveInt = 300
    max_python_memory_bytes: PositiveInt = 4 * 1024 * 1024 * 1024
    max_python_cpus: PositiveInt = 2
    max_python_processes: PositiveInt = 64
    max_python_output_bytes: PositiveInt = 64 * 1024
    max_python_scratch_bytes: PositiveInt = 1024 * 1024 * 1024
    max_artifact_count: PositiveInt = 20
    max_artifact_bytes: PositiveInt = 512 * 1024 * 1024
    max_total_artifact_bytes: PositiveInt = 1024 * 1024 * 1024

    @model_validator(mode="after")
    def validate_related_limits(self) -> RunPolicy:
        if self.max_artifact_bytes > self.max_total_artifact_bytes:
            raise ValueError("max_artifact_bytes must not exceed max_total_artifact_bytes")
        return self


class DerivationRequest(ContractModel):
    """Opt into one versioned replayable human-verification derivation."""

    format: Literal["dsa-derivation/v1"] = "dsa-derivation/v1"
    allow_plots: bool = Field(default=False, exclude_if=lambda value: not value)


class DerivationPlot(ContractModel):
    """A declared static plot produced by replay, not a model-supplied image."""

    filename: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.png$")
    title: str = Field(min_length=1, max_length=200)
    caption: str | None = Field(default=None, max_length=1000)

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("plot title must not be blank")
        return value


class DerivationMarkdownCell(ContractModel):
    """One concise explanatory cell in a replayable derivation."""

    type: Literal["markdown"] = "markdown"
    source: str = Field(min_length=1, max_length=16 * 1024)

    @field_validator("source")
    @classmethod
    def source_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("derivation cell source must not be blank")
        return value


class DerivationCodeCell(ContractModel):
    """One plain-Python code cell in a replayable derivation."""

    type: Literal["code"] = "code"
    source: str = Field(min_length=1, max_length=32 * 1024)

    @field_validator("source")
    @classmethod
    def source_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("derivation cell source must not be blank")
        return value


DerivationCell = Annotated[
    DerivationMarkdownCell | DerivationCodeCell,
    Field(discriminator="type"),
]


class Derivation(ContractModel):
    """Model-authored notebook cells without outputs or execution metadata."""

    format: Literal["dsa-derivation/v1"] = "dsa-derivation/v1"
    plots: tuple[DerivationPlot, ...] = Field(
        default=(),
        max_length=3,
        exclude_if=lambda value: not value,
    )

    @field_validator("plots", mode="before")
    @classmethod
    def snapshot_plots(cls, value: object) -> object:
        return tuple(cast(list[object], value)) if isinstance(value, list) else value

    cells: tuple[DerivationCell, ...] = Field(
        min_length=2,
        max_length=_MAX_DERIVATION_CELLS,
    )

    @field_validator("cells", mode="before")
    @classmethod
    def snapshot_cells(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(cast(list[object], value))
        return value

    @model_validator(mode="after")
    def validate_human_verification_shape(self) -> Derivation:
        if len({plot.filename for plot in self.plots}) != len(self.plots):
            raise ValueError("plot filenames must be distinct")
        if not isinstance(self.cells[0], DerivationMarkdownCell):
            raise ValueError("derivation must begin with an explanatory markdown cell")
        if not isinstance(self.cells[-1], DerivationCodeCell):
            raise ValueError("derivation must end with a code cell")
        if not any(isinstance(cell, DerivationCodeCell) for cell in self.cells):
            raise ValueError("derivation must contain a code cell")
        total_bytes = sum(len(cell.source.encode("utf-8")) for cell in self.cells)
        if total_bytes > _MAX_DERIVATION_SOURCE_BYTES:
            raise ValueError("derivation source exceeds its byte limit")
        return self


class RunRequest(ContractModel):
    """Everything a caller may place in the model-visible analysis contract."""

    database_path: Path
    question: str
    answer_schema: dict[str, JsonValue]
    derivation: DerivationRequest | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    model: ModelConfiguration
    policy: RunPolicy

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise PydanticCustomError("question_blank", "question must not be blank")
        return value

    @field_validator("answer_schema")
    @classmethod
    def snapshot_valid_schema(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return snapshot_answer_schema(value)


def snapshot_answer_schema(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Validate and isolate one caller-owned Draft 2020-12 answer schema."""
    if value.get("$schema") != DRAFT_2020_12:
        raise PydanticCustomError(
            "answer_schema_draft",
            "answer schema must declare JSON Schema Draft 2020-12",
        )
    ensure_finite_json(value, "answer schema")
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError as error:
        raise PydanticCustomError(
            "answer_schema_invalid",
            "answer schema must be valid JSON Schema Draft 2020-12",
        ) from error
    external_reference = _first_external_reference(value)
    if external_reference is not None:
        raise PydanticCustomError(
            "answer_schema_external_reference",
            "answer schema may use local references but not external references",
            {"reference": external_reference},
        )
    return deepcopy(value)


def _first_external_reference(value: dict[str, JsonValue]) -> str | None:
    resources = [DRAFT202012.create_resource(value)]
    while resources:
        resource = resources.pop()
        contents = resource.contents
        if isinstance(contents, dict):
            mapping = cast(dict[str, JsonValue], contents)
            for keyword in ("$ref", "$dynamicRef"):
                reference = mapping.get(keyword)
                if isinstance(reference, str) and not reference.startswith("#"):
                    return reference
        resources.extend(resource.subresources())
    return None


def ensure_finite_json(value: JsonValue, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PydanticCustomError(
            "non_finite_json",
            "JSON values must be finite",
            {"field": label},
        )
    if isinstance(value, dict):
        mapping = cast(dict[str, JsonValue], value)
        for child in mapping.values():
            ensure_finite_json(child, label)
    elif isinstance(value, list):
        sequence = cast(list[JsonValue], value)
        for child in sequence:
            ensure_finite_json(child, label)
