"""Descriptive tests for immutable Hugging Face evaluation packs."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dsa.pack import (
    EvaluationPackError,
    HuggingFacePackReference,
    LoadedEvaluationPack,
    load_huggingface_evaluation_pack,
)

SHA = "a" * 64
REVISION = "b" * 40


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _case(case_id: str = "case-01") -> dict[str, Any]:
    return {
        "case_id": case_id,
        "case_version": "1",
        "question": "How many rows are present?",
        "answer_schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
            "additionalProperties": False,
        },
        "expected_answer": {"count": 3},
        "metadata": {"family": "data_quality", "source_level": "small"},
    }


def _write_pack(
    root: Path,
    *,
    cases: list[dict[str, Any]] | None = None,
    canonical_cases: bool = True,
    manifest_updates: dict[str, Any] | None = None,
) -> tuple[HuggingFacePackReference, dict[str, Path]]:
    selected_cases = cases or [_case()]
    database = b"tiny duckdb bytes"
    if canonical_cases:
        case_bytes = b"".join(
            _canonical_json(case) + b"\n"
            for case in sorted(selected_cases, key=lambda value: value["case_id"])
        )
    else:
        case_bytes = json.dumps(selected_cases[0]).encode() + b"\n"
    manifest: dict[str, Any] = {
        "format": "dsa-evaluation-pack/v1",
        "pack_id": "online-retail-ii",
        "version": "1.0.0",
        "license": "CC-BY-4.0",
        "database": {
            "id": "online_retail_ii-1.0.0",
            "path": "database/online_retail_ii.duckdb",
            "size_bytes": len(database),
            "sha256": sha256(database).hexdigest(),
        },
        "cases": {
            "path": "cases.jsonl",
            "case_count": len(selected_cases),
            "size_bytes": len(case_bytes),
            "sha256": sha256(case_bytes).hexdigest(),
        },
        "scorer": {"name": "exact-json", "version": "1"},
        "provenance": {
            "source_datasets": [
                {
                    "name": "online-retail-ii-core-v4",
                    "version": "4",
                    "case_count": len(selected_cases),
                    "export_sha256": SHA,
                }
            ]
        },
    }
    if manifest_updates:
        manifest.update(manifest_updates)
    manifest_bytes = _canonical_json(manifest) + b"\n"
    files = {
        "online-retail-ii/1.0.0/pack.json": root / "pack.json",
        "online-retail-ii/1.0.0/database/online_retail_ii.duckdb": root
        / "database.duckdb",
        "online-retail-ii/1.0.0/cases.jsonl": root / "cases.jsonl",
    }
    files["online-retail-ii/1.0.0/pack.json"].write_bytes(manifest_bytes)
    files[
        "online-retail-ii/1.0.0/database/online_retail_ii.duckdb"
    ].write_bytes(database)
    files["online-retail-ii/1.0.0/cases.jsonl"].write_bytes(case_bytes)
    reference = HuggingFacePackReference(
        repo_id="lschiemanowski/dsa-datasets",
        revision=REVISION,
        path="online-retail-ii/1.0.0",
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
    )
    return reference, files


class Downloader:
    def __init__(self, files: dict[str, Path]) -> None:
        self.files = files
        self.calls: list[dict[str, object]] = []
        self.failure: Exception | None = None

    def __call__(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return str(self.files[str(kwargs["filename"])])


def test_loader_requests_only_exact_pinned_pack_files_and_verifies_content(
    tmp_path: Path,
) -> None:
    """A loaded pack is fully verified and transport remains Hub-owned."""
    reference, files = _write_pack(tmp_path)
    downloader = Downloader(files)

    pack = load_huggingface_evaluation_pack(reference, downloader=downloader)

    assert pack.manifest.pack_id == "online-retail-ii"
    assert pack.database_path == tmp_path / "database.duckdb"
    assert tuple(case.case_id for case in pack.cases) == ("case-01",)
    assert pack.cases[0].expected_answer == {"count": 3}
    assert [call["filename"] for call in downloader.calls] == [
        "online-retail-ii/1.0.0/pack.json",
        "online-retail-ii/1.0.0/database/online_retail_ii.duckdb",
        "online-retail-ii/1.0.0/cases.jsonl",
    ]
    for call in downloader.calls:
        assert call == {
            "repo_id": "lschiemanowski/dsa-datasets",
            "repo_type": "dataset",
            "revision": REVISION,
            "filename": call["filename"],
        }


def test_loader_resolves_and_reverifies_a_snapshot_database_symlink(
    tmp_path: Path,
) -> None:
    """The retained database path satisfies the runner's no-follow boundary."""
    reference, files = _write_pack(tmp_path)
    published = files[
        "online-retail-ii/1.0.0/database/online_retail_ii.duckdb"
    ]
    blob = tmp_path / "cache" / "blobs" / sha256(published.read_bytes()).hexdigest()
    blob.parent.mkdir(parents=True)
    published.replace(blob)
    published.symlink_to(blob)

    pack = load_huggingface_evaluation_pack(reference, downloader=Downloader(files))

    assert pack.database_path == blob.resolve(strict=True)
    assert pack.database_path.is_file()
    assert not pack.database_path.is_symlink()
    descriptor = os.open(
        pack.database_path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    os.close(descriptor)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"repo_id": "missing-namespace"}, "repo_id"),
        ({"revision": "main"}, "revision"),
        ({"path": "../private"}, "path"),
        ({"path": "/absolute"}, "path"),
    ],
)
def test_reference_requires_an_exact_repository_revision_and_safe_subdirectory(
    updates: dict[str, str],
    message: str,
) -> None:
    """Mutable revisions and paths outside the selected pack are inadmissible."""
    values = {
        "repo_id": "lschiemanowski/dsa-datasets",
        "revision": REVISION,
        "path": "online-retail-ii/1.0.0",
        "manifest_sha256": SHA,
    }
    values.update(updates)
    with pytest.raises(ValidationError, match=message):
        HuggingFacePackReference.model_validate(values)


def test_loader_revalidates_mutated_typed_references(tmp_path: Path) -> None:
    """A model-copy mutation cannot bypass the immutable revision boundary."""
    reference, files = _write_pack(tmp_path)
    mutated = reference.model_copy(update={"revision": "main"})
    downloader = Downloader(files)

    with pytest.raises(EvaluationPackError) as caught:
        load_huggingface_evaluation_pack(mutated, downloader=downloader)

    assert caught.value.code == "pack_reference_invalid"
    assert downloader.calls == []


def test_loaded_pack_cannot_be_rebound_to_different_case_content(tmp_path: Path) -> None:
    """Direct construction cannot claim a manifest digest for different cases."""
    reference, files = _write_pack(tmp_path)
    pack = load_huggingface_evaluation_pack(reference, downloader=Downloader(files))
    changed = pack.cases[0].model_copy(update={"expected_answer": {"count": 4}})

    with pytest.raises(ValidationError, match="do not match the manifest"):
        LoadedEvaluationPack(
            reference=pack.reference,
            manifest=pack.manifest,
            database_path=pack.database_path,
            cases=(changed,),
        )


def test_loaded_pack_cannot_bind_a_different_manifest_to_a_trusted_locator(
    tmp_path: Path,
) -> None:
    """Direct construction still binds canonical manifest bytes to the locator."""
    reference, files = _write_pack(tmp_path)
    pack = load_huggingface_evaluation_pack(reference, downloader=Downloader(files))
    changed_manifest = pack.manifest.model_copy(update={"pack_id": "different-pack"})

    with pytest.raises(ValidationError, match="manifest does not match its locator"):
        LoadedEvaluationPack(
            reference=pack.reference,
            manifest=changed_manifest,
            database_path=pack.database_path,
            cases=pack.cases,
        )


def test_loader_rejects_manifest_file_and_canonical_export_mismatches(
    tmp_path: Path,
) -> None:
    """No downloaded bytes are trusted through names or cache metadata alone."""
    reference, files = _write_pack(tmp_path)
    downloader = Downloader(files)
    bad_reference = reference.model_copy(update={"manifest_sha256": SHA})
    with pytest.raises(EvaluationPackError) as caught:
        load_huggingface_evaluation_pack(bad_reference, downloader=downloader)
    assert caught.value.code == "pack_manifest_digest_mismatch"

    reference, files = _write_pack(tmp_path)
    files["online-retail-ii/1.0.0/cases.jsonl"].write_bytes(b"tampered")
    with pytest.raises(EvaluationPackError) as caught:
        load_huggingface_evaluation_pack(reference, downloader=Downloader(files))
    assert caught.value.code == "pack_file_size_mismatch"


def test_loader_requires_canonical_order_unique_cases_and_valid_expectations(
    tmp_path: Path,
) -> None:
    """The retained case export has one deterministic identity and valid answers."""
    reference, files = _write_pack(tmp_path, canonical_cases=False)
    with pytest.raises(EvaluationPackError) as caught:
        load_huggingface_evaluation_pack(reference, downloader=Downloader(files))
    assert caught.value.code == "pack_cases_not_canonical"

    duplicate = [_case(), _case()]
    reference, files = _write_pack(tmp_path, cases=duplicate)
    with pytest.raises(EvaluationPackError) as caught:
        load_huggingface_evaluation_pack(reference, downloader=Downloader(files))
    assert caught.value.code == "pack_case_id_duplicate"

    invalid = _case()
    invalid["expected_answer"] = {"count": "three"}
    reference, files = _write_pack(tmp_path, cases=[invalid])
    with pytest.raises(EvaluationPackError) as caught:
        load_huggingface_evaluation_pack(reference, downloader=Downloader(files))
    assert caught.value.code == "pack_case_invalid"


def test_loader_rejects_unsafe_manifest_paths_before_downloading_content(
    tmp_path: Path,
) -> None:
    """A remote manifest cannot escape its selected version directory."""
    reference, files = _write_pack(
        tmp_path,
        manifest_updates={
            "database": {
                "id": "online_retail_ii-1.0.0",
                "path": "../../credential",
                "size_bytes": 1,
                "sha256": SHA,
            }
        },
    )
    downloader = Downloader(files)
    with pytest.raises(EvaluationPackError) as caught:
        load_huggingface_evaluation_pack(reference, downloader=downloader)
    assert caught.value.code == "pack_manifest_invalid"
    assert len(downloader.calls) == 1


def test_loader_sanitizes_hub_and_local_file_failures(tmp_path: Path) -> None:
    """Remote URLs, tokens, and cache paths do not cross the pack boundary."""
    reference, files = _write_pack(tmp_path)
    downloader = Downloader(files)
    downloader.failure = RuntimeError("https://signed.example/?token=SECRET")
    with pytest.raises(EvaluationPackError) as caught:
        load_huggingface_evaluation_pack(reference, downloader=downloader)
    assert caught.value.code == "pack_download_failed"
    assert "SECRET" not in str(caught.value)

    files["online-retail-ii/1.0.0/database/online_retail_ii.duckdb"] = (
        tmp_path / "missing.duckdb"
    )
    with pytest.raises(EvaluationPackError) as caught:
        load_huggingface_evaluation_pack(reference, downloader=Downloader(files))
    assert caught.value.code == "pack_file_unavailable"
    assert str(tmp_path) not in str(caught.value)
