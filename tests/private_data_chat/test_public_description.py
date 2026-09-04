"""Tests for exact public DuckDB description loading."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from apps.private_data_chat.public_description import (
    DescriptionError,
    HuggingFaceDescriptionReference,
    load_database_description,
)

REVISION = "a" * 40
CONTENT = b"## About the DuckDB\n\nUse `analysis.lines`.\n"


class Downloader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls: list[dict[str, str]] = []

    def __call__(self, **kwargs: str) -> str:
        self.calls.append(kwargs)
        return str(self.path)


def reference(content: bytes = CONTENT) -> HuggingFaceDescriptionReference:
    return HuggingFaceDescriptionReference(
        repo_id="lschiemanowski/dsa-datasets",
        revision=REVISION,
        path="online-retail-ii/1.0.0/DATABASE.md",
        sha256=sha256(content).hexdigest(),
    )


def test_loads_one_exact_description_from_a_snapshot_symlink(tmp_path: Path) -> None:
    blob = tmp_path / "blobs" / sha256(CONTENT).hexdigest()
    blob.parent.mkdir()
    blob.write_bytes(CONTENT)
    snapshot = tmp_path / "snapshot" / "DATABASE.md"
    snapshot.parent.mkdir()
    snapshot.symlink_to(blob)
    downloader = Downloader(snapshot)

    description = load_database_description(reference(), downloader=downloader)

    assert description == CONTENT.decode().rstrip()
    assert downloader.calls == [
        {
            "repo_id": "lschiemanowski/dsa-datasets",
            "repo_type": "dataset",
            "revision": REVISION,
            "filename": "online-retail-ii/1.0.0/DATABASE.md",
        }
    ]


@pytest.mark.parametrize(
    "updates",
    [
        {"revision": "main"},
        {"path": "../private.md"},
        {"repo_id": "missing-namespace"},
    ],
)
def test_reference_requires_an_exact_revision_and_safe_path(updates: dict[str, str]) -> None:
    values: dict[str, Any] = {
        "repo_id": "lschiemanowski/dsa-datasets",
        "revision": REVISION,
        "path": "online-retail-ii/1.0.0/DATABASE.md",
        "sha256": "b" * 64,
    }
    values.update(updates)
    with pytest.raises(ValidationError):
        HuggingFaceDescriptionReference.model_validate(values)


def test_revalidates_mutated_typed_reference_before_download(tmp_path: Path) -> None:
    path = tmp_path / "DATABASE.md"
    path.write_bytes(CONTENT)
    selected = reference().model_copy(update={"revision": "main"})
    downloader = Downloader(path)

    with pytest.raises(DescriptionError) as caught:
        load_database_description(selected, downloader=downloader)

    assert caught.value.code == "description_reference_invalid"
    assert downloader.calls == []


def test_rejects_digest_mismatch_without_retaining_transport_details(tmp_path: Path) -> None:
    path = tmp_path / "DATABASE.md"
    path.write_bytes(CONTENT)
    selected = reference().model_copy(update={"sha256": "0" * 64})

    with pytest.raises(DescriptionError) as caught:
        load_database_description(selected, downloader=Downloader(path))

    assert caught.value.code == "description_digest_mismatch"
    assert str(path) not in str(caught.value)


@pytest.mark.parametrize("content", [b"", b"\x00", b"\xff"])
def test_rejects_empty_binary_or_non_utf8_descriptions(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "DATABASE.md"
    path.write_bytes(content)

    with pytest.raises(DescriptionError):
        load_database_description(reference(content), downloader=Downloader(path))
