"""Pinned identity and opt-in live verification of Online Retail II 1.0.0."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dsa.pack import HuggingFacePackReference, load_huggingface_evaluation_pack

ROOT = Path(__file__).resolve().parents[1]
LOCATOR = ROOT / "evaluation-packs/online-retail-ii-1.0.0.json"
CASE_IDS = (
    "cohort-01",
    "cohort-02",
    "cohort-04",
    "cohort-05",
    "cohort-06",
    "dq-01",
    "dq-02",
    "dq-03",
    "dq-04",
    "dq-06",
    "mp-01",
    "mp-02",
    "mp-04",
    "mp-05",
    "mp-06",
    "product-01",
    "product-02",
    "product-04",
    "product-05",
    "product-06",
)


def reference() -> HuggingFacePackReference:
    return HuggingFacePackReference.model_validate_json(LOCATOR.read_bytes())


def test_repository_pins_one_exact_public_pack_revision() -> None:
    """The codebase retains only an immutable locator, not duplicated pack rows."""
    value = reference()

    assert value.repo_id == "lschiemanowski/dsa-datasets"
    assert value.revision == "897212ab5d9ad03631abccb5cc3e93f6a4396e65"
    assert value.path == "online-retail-ii/1.0.0"
    assert value.manifest_sha256 == (
        "703a821304f96a1ca7e301dcb5391a2c858ff4c265a2283d14739ef003e3e33d"
    )
    assert not (ROOT / "evaluation-packs/cases.jsonl").exists()
    assert not (ROOT / "evaluation-packs/online_retail_ii.duckdb").exists()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("DSA_HUGGINGFACE_TEST") != "1",
    reason="set DSA_HUGGINGFACE_TEST=1 to download the exact public pack revision",
)
def test_exact_public_online_retail_pack_download() -> None:
    """The pinned public revision resolves to the released database and twenty cases."""
    pack = load_huggingface_evaluation_pack(reference())

    assert pack.manifest.pack_id == "online-retail-ii"
    assert pack.manifest.version == "1.0.0"
    assert pack.manifest.database.sha256 == (
        "7439eff27b091d2cb4622ca9320e7f6aefccdc72af1895f983d838c7a518cbaf"
    )
    assert pack.manifest.cases.sha256 == (
        "5cb9492096e7b8a7d3cbb5b033df42a51c30666eb1dcbb9aa1cf1faa17b06219"
    )
    assert tuple(case.case_id for case in pack.cases) == CASE_IDS
