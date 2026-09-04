"""Pinned public DuckDB descriptions for the Private Data Chat application."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable
from hashlib import sha256
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol, cast

from pydantic import Field, field_validator

from apps.private_data_chat.contracts import AppContract

_MAX_DESCRIPTION_BYTES = 32 * 1024
_SAFE_PATH_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class DescriptionDownload(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        filename: str,
    ) -> str: ...


class HuggingFaceDescriptionReference(AppContract):
    """One exact Markdown description in a Hugging Face dataset repository."""

    format: Literal["dsa-huggingface-database-description/v1"] = (
        "dsa-huggingface-database-description/v1"
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
            raise ValueError("description path must be a safe relative path")
        return value


class DescriptionError(RuntimeError):
    """Stable failure at the public database-description boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"database description failed ({code})")


def load_database_description(
    reference: HuggingFaceDescriptionReference | object,
    *,
    downloader: DescriptionDownload | None = None,
) -> str:
    """Download and verify one bounded UTF-8 Markdown description."""
    try:
        selected = (
            HuggingFaceDescriptionReference.model_validate_json(reference.model_dump_json())
            if isinstance(reference, HuggingFaceDescriptionReference)
            else HuggingFaceDescriptionReference.model_validate(reference)
        )
    except Exception:
        raise DescriptionError("description_reference_invalid") from None
    download = downloader or _default_download
    try:
        downloaded = download(
            repo_id=selected.repo_id,
            repo_type="dataset",
            revision=selected.revision,
            filename=selected.path,
        )
    except DescriptionError:
        raise
    except Exception:
        raise DescriptionError("description_download_failed") from None
    if not downloaded:
        raise DescriptionError("description_download_failed")
    content = _read_downloaded_file(Path(downloaded))
    if sha256(content).hexdigest() != selected.sha256:
        raise DescriptionError("description_digest_mismatch")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise DescriptionError("description_invalid") from None
    if not text.strip() or "\x00" in text:
        raise DescriptionError("description_invalid")
    return text.rstrip()


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
        raise DescriptionError("description_dependency_missing") from None
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
                or metadata.st_size > _MAX_DESCRIPTION_BYTES
            ):
                raise DescriptionError("description_file_size_invalid")
            content = source.read(_MAX_DESCRIPTION_BYTES + 1)
    except DescriptionError:
        raise
    except OSError:
        raise DescriptionError("description_file_unavailable") from None
    if not content or len(content) > _MAX_DESCRIPTION_BYTES:
        raise DescriptionError("description_file_size_invalid")
    return content
