"""Canonical terminal record and atomic local retention."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import Field, JsonValue, model_validator

from dsa.contract import ContractModel, Derivation, RunRequest

FailureStage = Literal[
    "model",
    "analysis_environment",
    "answer_validation",
    "derivation_validation",
    "orchestration",
    "cancelled",
]


class _AnswerValidator(Protocol):
    def iter_errors(self, instance: JsonValue) -> Iterable[JsonSchemaValidationError]: ...


class Failure(ContractModel):
    """Safe terminal explanation of an unsuccessful run."""

    stage: FailureStage
    code: str
    message: str
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)


class DerivationVerification(ContractModel):
    """Integrity evidence from replaying a derivation in an injected executor."""

    status: Literal["verified"] = "verified"
    derivation_sha256: str = Field(pattern=r"[0-9a-f]{64}")
    result_sha256: str = Field(pattern=r"[0-9a-f]{64}")
    source_database_sha256: str = Field(pattern=r"[0-9a-f]{64}")
    runtime_identity: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        exclude_if=lambda value: value is None,
    )
    notebook_relative_path: Literal["derivation.ipynb"] = "derivation.ipynb"
    notebook_sha256: str = Field(pattern=r"[0-9a-f]{64}")
    notebook_byte_length: Annotated[int, Field(gt=0)]


class RunSuccess(ContractModel):
    """A caller-schema-valid answer."""

    status: Literal["succeeded"] = "succeeded"
    answer: JsonValue
    derivation: Derivation | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    derivation_verification: DerivationVerification | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class RunFailure(ContractModel):
    """A typed terminal failure after a run has started."""

    status: Literal["failed"] = "failed"
    failure: Failure


RunOutcome = Annotated[RunSuccess | RunFailure, Field(discriminator="status")]


class ArtifactRecord(ContractModel):
    """Integrity metadata for one run-private retained artifact."""

    handle: str = Field(pattern=r"a[1-9][0-9]*")
    relative_path: str
    media_type: str
    size_bytes: Annotated[int, Field(ge=0)]
    sha256: str = Field(pattern=r"[0-9a-f]{64}")
    producer_tool_call_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_managed_path(self) -> ArtifactRecord:
        path = Path(self.relative_path)
        if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "artifacts":
            raise ValueError("artifact path must remain inside the managed artifact directory")
        if path.name.split(".", 1)[0] != self.handle:
            raise ValueError("artifact path must be named by its handle")
        expected_suffix = {
            "application/json": ".json",
            "application/vnd.apache.parquet": ".parquet",
        }.get(self.media_type)
        if expected_suffix is None or path.suffix != expected_suffix:
            raise ValueError("artifact path must match its supported media type")
        return self


class DatabaseRecord(ContractModel):
    """Content identities for the immutable source and final private run state."""

    source_sha256: str = Field(pattern=r"[0-9a-f]{64}")
    final_sha256: str = Field(pattern=r"[0-9a-f]{64}")


class TerminalRecord(ContractModel):
    """The single canonical debugging and reproducibility record for a run."""

    schema_version: Literal["1", "2"] = "1"
    run_id: str
    started_at: datetime
    finished_at: datetime
    request: RunRequest
    messages: tuple[dict[str, JsonValue], ...] = ()
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: tuple[ArtifactRecord, ...] = ()
    database: DatabaseRecord | None = None
    outcome: RunOutcome

    @model_validator(mode="after")
    def validate_terminal_record(self) -> TerminalRecord:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.run_id) is None:
            raise ValueError("run_id must be a safe managed identity")
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("terminal timestamps must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        expected_schema_version = "2" if self.request.derivation is not None else "1"
        if self.schema_version != expected_schema_version:
            raise ValueError("terminal schema version must match the derivation contract")
        if isinstance(self.outcome, RunSuccess):
            validator = cast(
                _AnswerValidator,
                Draft202012Validator(self.request.answer_schema),
            )
            errors = list(validator.iter_errors(self.outcome.answer))
            if errors:
                raise ValueError("successful terminal answer violates the caller schema")
            derivation_requested = self.request.derivation is not None
            derivation_present = self.outcome.derivation is not None
            verification = self.outcome.derivation_verification
            if (not derivation_requested and derivation_present) or derivation_present != (
                verification is not None
            ):
                raise ValueError(
                    "successful terminal derivation must be requested and verified"
                )
            if derivation_present:
                assert self.outcome.derivation is not None
                assert verification is not None
                derivation_bytes = _canonical_json_bytes(
                    self.outcome.derivation.model_dump(mode="json")
                )
                result_bytes = _canonical_json_bytes(self.outcome.answer)
                if verification.derivation_sha256 != sha256(derivation_bytes).hexdigest():
                    raise ValueError("derivation digest contradicts terminal content")
                if verification.result_sha256 != sha256(result_bytes).hexdigest():
                    raise ValueError("derivation result digest contradicts terminal answer")
                if (
                    self.database is None
                    or verification.source_database_sha256
                    != self.database.source_sha256
                ):
                    raise ValueError(
                        "derivation source digest contradicts terminal database"
                    )
        expected_handles = [f"a{index}" for index in range(1, len(self.artifacts) + 1)]
        if [artifact.handle for artifact in self.artifacts] != expected_handles:
            raise ValueError("artifact handles must be unique and sequential")
        return self


class RetainedTerminalRecord(ContractModel):
    """Integrity reference to the exact retained terminal bytes."""

    path: Path
    sha256: str
    byte_length: Annotated[int, Field(gt=0)]


class RetainedDerivationNotebook(ContractModel):
    """Integrity reference to one deterministic notebook projection."""

    path: Path
    sha256: str = Field(pattern=r"[0-9a-f]{64}")
    byte_length: Annotated[int, Field(gt=0)]


def validate_retained_derivation_notebook(
    record: TerminalRecord,
    retained_record: RetainedTerminalRecord,
    retained_notebook: RetainedDerivationNotebook | None,
) -> None:
    """Require one exact notebook reference for one verified derivation success."""
    verification = (
        record.outcome.derivation_verification
        if isinstance(record.outcome, RunSuccess)
        else None
    )
    if verification is None:
        if retained_notebook is not None:
            raise ValueError("terminal without verified derivation must not retain a notebook")
        return
    if retained_notebook is None:
        raise ValueError("verified derivation terminal requires its retained notebook")
    expected_path = retained_record.path.parent / verification.notebook_relative_path
    if (
        retained_notebook.path != expected_path
        or retained_notebook.sha256 != verification.notebook_sha256
        or retained_notebook.byte_length != verification.notebook_byte_length
    ):
        raise ValueError("retained notebook does not match terminal derivation evidence")


def write_terminal_record(
    record: TerminalRecord,
    run_directory: Path,
) -> RetainedTerminalRecord:
    """Write canonical bytes once, atomically, and return their sole digest."""
    if not run_directory.is_dir() or run_directory.is_symlink():
        raise ValueError("run directory must be an existing managed directory")
    destination = run_directory / "terminal.json"
    if destination.exists():
        raise FileExistsError("terminal record already exists")

    content = (
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    temporary = run_directory / f".terminal.{uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        _fsync_directory(run_directory)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise

    return RetainedTerminalRecord(
        path=destination,
        sha256=sha256(content).hexdigest(),
        byte_length=len(content),
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
