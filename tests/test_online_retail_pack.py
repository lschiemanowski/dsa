"""Pinned identity and opt-in live verification of Online Retail II 1.1.0."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_ai.models.test import TestModel

from dsa import ModelConfiguration, RunPolicy, RunRequest, RunSuccess, run_analysis
from dsa.pack import HuggingFacePackReference, load_huggingface_evaluation_pack

ROOT = Path(__file__).resolve().parents[1]
LOCATOR = ROOT / "evaluation-packs/online-retail-ii-1.1.0.json"
LEGACY_CASE_IDS = (
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
    assert value.revision == "58958007cdf38eb9e563356f16ecd8011d5a3d67"
    assert value.path == "online-retail-ii/1.1.0"
    assert value.manifest_sha256 == (
        "d514cc6083264c0edabe9360d60949465ba8a97955a6378cbc088a1fffb75d30"
    )
    assert not (ROOT / "evaluation-packs/cases.jsonl").exists()
    assert not (ROOT / "evaluation-packs/online_retail_ii.duckdb").exists()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("DSA_HUGGINGFACE_TEST") != "1",
    reason="set DSA_HUGGINGFACE_TEST=1 to download the exact public pack revision",
)
async def test_exact_public_online_retail_pack_download(tmp_path: Path) -> None:
    """The exact public pack supplies a database usable by an ordinary DSA run."""
    pack = load_huggingface_evaluation_pack(reference())

    assert pack.manifest.pack_id == "online-retail-ii"
    assert pack.manifest.version == "1.1.0"
    assert pack.manifest.database.sha256 == (
        "7439eff27b091d2cb4622ca9320e7f6aefccdc72af1895f983d838c7a518cbaf"
    )
    assert pack.manifest.cases.sha256 == (
        "fb9b4e1b70bd2e5e8e5075f0344fc399e82b30c4e7c087e2db9f2ebe03be7618"
    )
    assert len(pack.cases) == 100
    assert tuple(case.case_id for case in pack.cases[:20]) == LEGACY_CASE_IDS
    assert all(case.case_id.startswith("retail-v2-") for case in pack.cases[20:])
    families = {case.metadata.family for case in pack.cases}
    assert {
        "basket-analysis",
        "customer-retention",
        "data-quality",
        "geography",
        "market-basket",
        "product-performance",
        "returns",
        "time-series",
    } <= families
    assert pack.database_path.is_file()
    assert not pack.database_path.is_symlink()

    case = pack.cases[0]
    request = RunRequest(
        database_path=pack.database_path,
        question=case.question,
        answer_schema=case.answer_schema,
        model=ModelConfiguration(name="test"),
        policy=RunPolicy(),
    )
    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=TestModel(
            call_tools=[],
            custom_output_args=cast(dict[str, Any], case.expected_answer),
        ),
        identity_factory=lambda: "pinned-pack-smoke",
    )

    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.answer == case.expected_answer
