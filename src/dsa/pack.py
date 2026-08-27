"""Strict versioned evaluation packs resolved through Hugging Face Hub."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from hashlib import sha256
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Protocol, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import Field, JsonValue, field_validator, model_validator

from dsa.contract import ContractModel, ensure_finite_json, snapshot_answer_schema

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveSize = Annotated[int, Field(gt=0)]
SafeName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
_SAFE_PATH_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_CASE_BYTES = 16 * 1024 * 1024


class PackDownload(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        filename: str,
    ) -> str: ...


class EvaluationPackError(RuntimeError):
    """Stable sanitized failure at the evaluation-pack boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"evaluation pack failed ({code})")


class PackFile(ContractModel):
    """One exact regular file beneath a selected pack directory."""

    path: str
    size_bytes: PositiveSize
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _safe_relative_path(value)


class PackDatabase(PackFile):
    """The source database and its portable identity."""

    id: SafeName


class PackCases(PackFile):
    """The one canonical case export in a pack."""

    case_count: Annotated[int, Field(gt=0, le=10_000)]


class PackScorer(ContractModel):
    """The deterministic scorer required by this pack format."""

    name: Literal["exact-json"]
    version: Literal["1"]


class SourceDataset(ContractModel):
    """One frozen predecessor dataset retained only as provenance."""

    name: SafeName
    version: SafeName
    case_count: Annotated[int, Field(gt=0, le=10_000)]
    export_sha256: Sha256


class PackProvenance(ContractModel):
    """Frozen source datasets from which a released pack was migrated."""

    source_datasets: tuple[SourceDataset, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def source_identities_are_unique(self) -> PackProvenance:
        identities = tuple((item.name, item.version) for item in self.source_datasets)
        if len(identities) != len(set(identities)):
            raise ValueError("source dataset identities must be unique")
        return self


class EvaluationPackManifest(ContractModel):
    """Portable content identities for one released evaluation pack."""

    format: Literal["dsa-evaluation-pack/v1"]
    pack_id: SafeName
    version: str = Field(
        pattern=(
            r"^(?:0|[1-9][0-9]*)\."
            r"(?:0|[1-9][0-9]*)\."
            r"(?:0|[1-9][0-9]*)$"
        )
    )
    license: SafeName
    database: PackDatabase
    cases: PackCases
    scorer: PackScorer
    provenance: PackProvenance

    @model_validator(mode="after")
    def validate_complete_manifest(self) -> EvaluationPackManifest:
        if self.database.path == self.cases.path:
            raise ValueError("database and cases must identify different files")
        source_count = sum(item.case_count for item in self.provenance.source_datasets)
        if source_count != self.cases.case_count:
            raise ValueError("source dataset counts must match the case export")
        return self


class EvaluationCaseMetadata(ContractModel):
    """Non-model-visible descriptive metadata retained with one case."""

    family: SafeName
    source_level: SafeName


class EvaluationPackCase(ContractModel):
    """One exact analytical question and its host-only expected answer."""

    case_id: SafeName
    case_version: SafeName
    question: str
    answer_schema: dict[str, JsonValue]
    expected_answer: JsonValue
    metadata: EvaluationCaseMetadata

    @field_validator("question")
    @classmethod
    def question_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("answer_schema")
    @classmethod
    def validate_answer_schema(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return snapshot_answer_schema(value)

    @field_validator("expected_answer")
    @classmethod
    def snapshot_expected_answer(cls, value: JsonValue) -> JsonValue:
        ensure_finite_json(value, "expected answer")
        return deepcopy(value)

    @model_validator(mode="after")
    def expected_answer_satisfies_schema(self) -> EvaluationPackCase:
        errors: list[JsonSchemaValidationError] = list(
            cast(Any, Draft202012Validator(self.answer_schema)).iter_errors(
                self.expected_answer
            )
        )
        if errors:
            raise ValueError("expected answer must satisfy the answer schema")
        return self

    def dataset_record(
        self,
        manifest: EvaluationPackManifest,
    ) -> dict[str, JsonValue]:
        """Project portable inputs, host-only expectations, and descriptive tags."""
        return {
            "inputs": {
                "case_id": self.case_id,
                "case_version": self.case_version,
                "database_id": manifest.database.id,
                "database_sha256": manifest.database.sha256,
                "question": self.question,
                "answer_schema": deepcopy(self.answer_schema),
            },
            "expectations": {"answer": deepcopy(self.expected_answer)},
            "tags": {
                "family": self.metadata.family,
                "pack": manifest.pack_id,
                "pack_version": manifest.version,
                "source_level": self.metadata.source_level,
            },
        }


class HuggingFacePackReference(ContractModel):
    """One immutable pack locator in a Hugging Face dataset repository."""

    format: Literal["dsa-huggingface-pack/v1"] = "dsa-huggingface-pack/v1"
    repo_id: str = Field(
        pattern=(
            r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/"
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
        )
    )
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str
    manifest_sha256: Sha256

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _safe_relative_path(value)


class LoadedEvaluationPack(ContractModel):
    """A completely verified manifest, database path, and ordered cases."""

    reference: HuggingFacePackReference
    manifest: EvaluationPackManifest
    database_path: Path
    cases: tuple[EvaluationPackCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def cases_match_manifest_identity(self) -> LoadedEvaluationPack:
        identities = tuple(case.case_id for case in self.cases)
        if len(identities) != len(set(identities)):
            raise ValueError("evaluation pack case identities must be unique")
        if identities != tuple(sorted(identities)):
            raise ValueError("evaluation pack cases must be ordered by identity")
        value = _canonical_case_bytes(self.cases)
        if (
            len(self.cases) != self.manifest.cases.case_count
            or len(value) != self.manifest.cases.size_bytes
            or sha256(value).hexdigest() != self.manifest.cases.sha256
        ):
            raise ValueError("evaluation pack cases do not match the manifest")
        return self


def load_huggingface_evaluation_pack(
    reference: HuggingFacePackReference | object,
    *,
    downloader: PackDownload | None = None,
) -> LoadedEvaluationPack:
    """Download and verify one exact pack revision without executing remote code."""
    try:
        selected_reference = (
            HuggingFacePackReference.model_validate_json(reference.model_dump_json())
            if isinstance(reference, HuggingFacePackReference)
            else HuggingFacePackReference.model_validate(reference)
        )
    except Exception:
        raise EvaluationPackError("pack_reference_invalid") from None
    selected_downloader = downloader or _default_download
    manifest_name = f"{selected_reference.path}/pack.json"
    manifest_path = _download(selected_downloader, selected_reference, manifest_name)
    manifest_bytes = _read_bounded_file(manifest_path, _MAX_MANIFEST_BYTES)
    if sha256(manifest_bytes).hexdigest() != selected_reference.manifest_sha256:
        raise EvaluationPackError("pack_manifest_digest_mismatch")
    try:
        manifest = EvaluationPackManifest.model_validate_json(manifest_bytes)
    except Exception:
        raise EvaluationPackError("pack_manifest_invalid") from None
    if manifest_bytes != _canonical_json(manifest.model_dump(mode="json")) + b"\n":
        raise EvaluationPackError("pack_manifest_invalid")

    database_name = f"{selected_reference.path}/{manifest.database.path}"
    database_path = _download(
        selected_downloader,
        selected_reference,
        database_name,
    )
    _verify_file(database_path, manifest.database)

    cases_name = f"{selected_reference.path}/{manifest.cases.path}"
    cases_path = _download(selected_downloader, selected_reference, cases_name)
    case_bytes = _verified_bounded_bytes(cases_path, manifest.cases, _MAX_CASE_BYTES)
    cases = _parse_cases(case_bytes, manifest.cases.case_count)
    return LoadedEvaluationPack(
        reference=selected_reference,
        manifest=manifest,
        database_path=database_path,
        cases=cases,
    )


def _default_download(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    filename: str,
) -> str:
    try:
        module = import_module("huggingface_hub")
    except ModuleNotFoundError:
        raise EvaluationPackError("pack_dependency_missing") from None
    download = cast(PackDownload, module.hf_hub_download)
    return download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        filename=filename,
    )


def _download(
    downloader: PackDownload,
    reference: HuggingFacePackReference,
    filename: str,
) -> Path:
    try:
        value = downloader(
            repo_id=reference.repo_id,
            repo_type="dataset",
            revision=reference.revision,
            filename=filename,
        )
    except EvaluationPackError:
        raise
    except Exception:
        raise EvaluationPackError("pack_download_failed") from None
    if not value:
        raise EvaluationPackError("pack_download_failed")
    return Path(value)


def _read_bounded_file(path: Path, maximum: int) -> bytes:
    try:
        with path.open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            if size <= 0 or size > maximum:
                raise EvaluationPackError("pack_file_size_invalid")
            value = source.read(maximum + 1)
    except EvaluationPackError:
        raise
    except OSError:
        raise EvaluationPackError("pack_file_unavailable") from None
    if len(value) != size:
        raise EvaluationPackError("pack_file_unavailable")
    return value


def _verify_file(path: Path, expected: PackFile) -> None:
    try:
        with path.open("rb") as source:
            if os.fstat(source.fileno()).st_size != expected.size_bytes:
                raise EvaluationPackError("pack_file_size_mismatch")
            digest = sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except EvaluationPackError:
        raise
    except OSError:
        raise EvaluationPackError("pack_file_unavailable") from None
    if digest.hexdigest() != expected.sha256:
        raise EvaluationPackError("pack_file_digest_mismatch")


def _verified_bounded_bytes(path: Path, expected: PackFile, maximum: int) -> bytes:
    if expected.size_bytes > maximum:
        raise EvaluationPackError("pack_file_size_invalid")
    try:
        with path.open("rb") as source:
            if os.fstat(source.fileno()).st_size != expected.size_bytes:
                raise EvaluationPackError("pack_file_size_mismatch")
            value = source.read(maximum + 1)
    except EvaluationPackError:
        raise
    except OSError:
        raise EvaluationPackError("pack_file_unavailable") from None
    if len(value) != expected.size_bytes:
        raise EvaluationPackError("pack_file_unavailable")
    if sha256(value).hexdigest() != expected.sha256:
        raise EvaluationPackError("pack_file_digest_mismatch")
    return value


def _parse_cases(value: bytes, expected_count: int) -> tuple[EvaluationPackCase, ...]:
    if not value.endswith(b"\n") or b"\r" in value:
        raise EvaluationPackError("pack_cases_not_canonical")
    lines = value.removesuffix(b"\n").split(b"\n")
    if not lines or any(not line for line in lines):
        raise EvaluationPackError("pack_cases_not_canonical")
    cases: list[EvaluationPackCase] = []
    try:
        for line in lines:
            cases.append(EvaluationPackCase.model_validate(json.loads(line)))
    except Exception:
        raise EvaluationPackError("pack_case_invalid") from None
    identities = tuple(case.case_id for case in cases)
    if len(identities) != len(set(identities)):
        raise EvaluationPackError("pack_case_id_duplicate")
    if len(cases) != expected_count:
        raise EvaluationPackError("pack_case_count_mismatch")
    canonical = _canonical_case_bytes(tuple(sorted(cases, key=lambda item: item.case_id)))
    if canonical != value:
        raise EvaluationPackError("pack_cases_not_canonical")
    return tuple(cases)


def _canonical_case_bytes(cases: tuple[EvaluationPackCase, ...]) -> bytes:
    return b"".join(
        _canonical_json(case.model_dump(mode="json")) + b"\n" for case in cases
    )


def _safe_relative_path(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError("path must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(_SAFE_PATH_SEGMENT.fullmatch(part) is None for part in path.parts)
    ):
        raise ValueError("path must be a safe relative POSIX path")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
