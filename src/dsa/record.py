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

from dsa.contract import ContractModel, RunRequest

FailureStage = Literal[
    "model",
    "analysis_environment",
    "answer_validation",
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


class RunSuccess(ContractModel):
    """A caller-schema-valid answer."""

    status: Literal["succeeded"] = "succeeded"
    answer: JsonValue


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

    schema_version: Literal["1"] = "1"
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
        if isinstance(self.outcome, RunSuccess):
            validator = cast(
                _AnswerValidator,
                Draft202012Validator(self.request.answer_schema),
            )
            errors = list(validator.iter_errors(self.outcome.answer))
            if errors:
                raise ValueError("successful terminal answer violates the caller schema")
        expected_handles = [f"a{index}" for index in range(1, len(self.artifacts) + 1)]
        if [artifact.handle for artifact in self.artifacts] != expected_handles:
            raise ValueError("artifact handles must be unique and sequential")
        return self


class RetainedTerminalRecord(ContractModel):
    """Integrity reference to the exact retained terminal bytes."""

    path: Path
    sha256: str
    byte_length: Annotated[int, Field(gt=0)]


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


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
