"""Pinned, typed database cards for the Private Data Chat application."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable
from hashlib import sha256
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol, cast

from pydantic import Field, field_validator, model_validator

from apps.private_data_chat.contracts import AppContract

_MAX_CARD_BYTES = 64 * 1024
_SAFE_PATH_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CardDownload(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        filename: str,
    ) -> str: ...


class HuggingFaceDatabaseCardReference(AppContract):
    """One exact JSON card in a Hugging Face dataset repository."""

    format: Literal["dsa-huggingface-database-card/v1"] = (
        "dsa-huggingface-database-card/v1"
    )
    repo_id: str = Field(
        pattern=(
            r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/"
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
        )
    )
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def require_safe_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or len(value) > 1024
            or any(
                part in {"", ".", ".."} or _SAFE_PATH_SEGMENT.fullmatch(part) is None
                for part in path.parts
            )
        ):
            raise ValueError("database-card path must be a safe relative path")
        return value


class DatabaseColumn(AppContract):
    """One user-relevant column in a database relation."""

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    data_type: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1_000)

    @field_validator("data_type", "description")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("database-column text must not be blank")
        return value


class DatabaseRelation(AppContract):
    """One table or view exposed by the packaged DuckDB."""

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
    kind: Literal["table", "view"]
    row_count: Annotated[int, Field(ge=0)]
    description: str = Field(min_length=1, max_length=2_000)
    columns: tuple[DatabaseColumn, ...] = Field(min_length=1, max_length=128)

    @field_validator("columns", mode="before")
    @classmethod
    def snapshot_columns(cls, value: object) -> object:
        return tuple(cast(list[object], value)) if isinstance(value, list) else value

    @field_validator("description")
    @classmethod
    def reject_blank_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("relation description must not be blank")
        return value

    @model_validator(mode="after")
    def require_unique_columns(self) -> DatabaseRelation:
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("relation column names must be unique")
        return self


class DatabaseCoverage(AppContract):
    """A compact set of facts that orients a user to the dataset."""

    period_start: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?$")
    period_end: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?$")
    transaction_lines: Annotated[int, Field(ge=0)]
    invoices: Annotated[int, Field(ge=0)]
    identified_customers: Annotated[int, Field(ge=0)]
    product_codes: Annotated[int, Field(ge=0)]
    country_values: Annotated[int, Field(ge=0)]


class DatabaseCard(AppContract):
    """Dataset-owned content with separate user and model projections."""

    format: Literal["dsa-database-card/v1"] = "dsa-database-card/v1"
    title: str = Field(min_length=1, max_length=256)
    summary: str = Field(min_length=1, max_length=4_000)
    coverage: DatabaseCoverage
    primary_relation: str = Field(
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$"
    )
    relations: tuple[DatabaseRelation, ...] = Field(min_length=1, max_length=32)
    example_questions: tuple[str, ...] = Field(min_length=1, max_length=12)
    analysis_notes: tuple[str, ...] = Field(default=(), max_length=64)

    @field_validator("relations", "example_questions", "analysis_notes", mode="before")
    @classmethod
    def snapshot_sequences(cls, value: object) -> object:
        return tuple(cast(list[object], value)) if isinstance(value, list) else value

    @field_validator("title", "summary")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("database-card text must not be blank")
        return value

    @field_validator("example_questions", "analysis_notes")
    @classmethod
    def reject_blank_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() or len(item) > 2_000 for item in value):
            raise ValueError("database-card list items must be nonblank and bounded")
        return value

    @model_validator(mode="after")
    def validate_relations(self) -> DatabaseCard:
        names = [relation.name for relation in self.relations]
        if len(names) != len(set(names)):
            raise ValueError("database relation names must be unique")
        if self.primary_relation not in names:
            raise ValueError("primary relation must be present in relations")
        return self


class DatabaseCardError(RuntimeError):
    """Stable failure at the public database-card boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"database card failed ({code})")


def load_database_card(
    reference: HuggingFaceDatabaseCardReference | object,
    *,
    downloader: CardDownload | None = None,
) -> DatabaseCard:
    """Download, verify, parse, and snapshot one bounded JSON card."""
    try:
        selected = (
            HuggingFaceDatabaseCardReference.model_validate_json(
                reference.model_dump_json()
            )
            if isinstance(reference, HuggingFaceDatabaseCardReference)
            else HuggingFaceDatabaseCardReference.model_validate(reference)
        )
    except Exception:
        raise DatabaseCardError("database_card_reference_invalid") from None
    download = downloader or _default_download
    try:
        downloaded = download(
            repo_id=selected.repo_id,
            repo_type="dataset",
            revision=selected.revision,
            filename=selected.path,
        )
    except DatabaseCardError:
        raise
    except Exception:
        raise DatabaseCardError("database_card_download_failed") from None
    if not downloaded:
        raise DatabaseCardError("database_card_download_failed")
    content = _read_downloaded_file(Path(downloaded))
    if sha256(content).hexdigest() != selected.sha256:
        raise DatabaseCardError("database_card_digest_mismatch")
    try:
        return DatabaseCard.model_validate_json(content)
    except Exception:
        raise DatabaseCardError("database_card_invalid") from None


def render_database_overview(card: DatabaseCard) -> str:
    """Render the user-facing projection without model-only analysis notes."""
    selected = DatabaseCard.model_validate_json(card.model_dump_json())
    coverage = selected.coverage
    primary = next(
        relation for relation in selected.relations if relation.name == selected.primary_relation
    )
    relations = "\n".join(
        f"- `{relation.name}` ({relation.kind}, {relation.row_count:,} "
        f"{'row' if relation.row_count == 1 else 'rows'}): "
        f"{relation.description}"
        for relation in selected.relations
    )
    columns = "\n".join(
        f"| `{column.name}` | `{column.data_type}` | {column.description} |"
        for column in primary.columns
    )
    examples = "\n".join(f"- {question}" for question in selected.example_questions)
    return (
        f"{selected.summary}\n\n"
        "### At a glance\n\n"
        f"- **Coverage:** {coverage.period_start} through {coverage.period_end}\n"
        f"- **Transaction lines:** {coverage.transaction_lines:,}\n"
        f"- **Invoices:** {coverage.invoices:,}\n"
        f"- **Identified customers:** {coverage.identified_customers:,}\n"
        f"- **Product codes:** {coverage.product_codes:,}\n"
        f"- **Country values:** {coverage.country_values:,}\n\n"
        "### Available data\n\n"
        f"{relations}\n\n"
        f"### Main fields in `{primary.name}`\n\n"
        "| Column | Type | Contents |\n"
        "| --- | --- | --- |\n"
        f"{columns}\n\n"
        "### Example questions\n\n"
        f"{examples}"
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
        raise DatabaseCardError("database_card_dependency_missing") from None
    download = cast(Callable[..., str], module.hf_hub_download)
    return download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        filename=filename,
    )


def _read_downloaded_file(path: Path) -> bytes:
    try:
        with path.open("rb") as source:
            metadata = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size <= 0
                or metadata.st_size > _MAX_CARD_BYTES
            ):
                raise DatabaseCardError("database_card_file_size_invalid")
            content = source.read(_MAX_CARD_BYTES + 1)
    except DatabaseCardError:
        raise
    except OSError:
        raise DatabaseCardError("database_card_file_unavailable") from None
    if not content or len(content) > _MAX_CARD_BYTES:
        raise DatabaseCardError("database_card_file_size_invalid")
    return content
