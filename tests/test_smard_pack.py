"""Pinned identity and opt-in live verification of SMARD DE/LU 2024."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from dsa.pack import HuggingFacePackReference, load_huggingface_evaluation_pack

ROOT = Path(__file__).resolve().parents[1]
LOCATOR = ROOT / "evaluation-packs/smard-de-lu-2024-1.2.0.json"
CASE_IDS = (
    "smard-01-annual-wind-solar",
    "smard-02-peak-pv-day",
    "smard-03-peak-renewable-share-day",
    "smard-04-peak-grid-load",
    "smard-05-negative-price-prevalence",
    "smard-06-longest-negative-price-run",
    "smard-07-load-forecast-error",
    "smard-08-renewable-forecast-error",
    "smard-09-price-renewables-correlation",
    "smard-10-net-export-during-negative-prices",
    "smard-11-wind-solar-exceeds-load",
    "smard-12-largest-residual-load-increase",
    "smard-13-annual-renewable-share",
    "smard-14-peak-offshore-wind-interval",
    "smard-15-peak-onshore-wind-day",
    "smard-16-widest-daily-price-range",
    "smard-17-lowest-residual-load-interval",
    "smard-18-net-import-prevalence",
    "smard-19-largest-load-forecast-error",
    "smard-20-highest-monthly-average-price",
)
ADDED_EXPECTED_ANSWERS: dict[str, JsonValue] = {
    "smard-13-annual-renewable-share": {
        "eligible_intervals": 35132,
        "renewable_mwh": 257258671.5,
        "grid_load_mwh": 470336234.84,
        "share_percent": 54.6968,
    },
    "smard-14-peak-offshore-wind-interval": {
        "interval_start_utc": "2024-11-22T16:00:00Z",
        "interval_start_local": "2024-11-22T17:00:00",
        "wind_offshore_mwh": 1868.25,
    },
    "smard-15-peak-onshore-wind-day": {
        "date": "2024-02-06",
        "wind_onshore_mwh": 1064138.0,
        "interval_count": 96,
    },
    "smard-16-widest-daily-price-range": {
        "date": "2024-12-12",
        "minimum_price_eur_per_mwh": 107.35,
        "maximum_price_eur_per_mwh": 936.28,
        "spread_eur_per_mwh": 828.93,
        "eligible_intervals": 96,
    },
    "smard-17-lowest-residual-load-interval": {
        "interval_start_utc": "2024-05-01T11:30:00Z",
        "interval_start_local": "2024-05-01T13:30:00",
        "residual_load_mwh": -2155.0,
    },
    "smard-18-net-import-prevalence": {
        "importing_intervals": 25952,
        "eligible_intervals": 35136,
        "percent": 73.8616,
    },
    "smard-19-largest-load-forecast-error": {
        "interval_start_utc": "2024-05-10T06:30:00Z",
        "actual_grid_load_mwh": 12108.0,
        "forecast_grid_load_mwh": 14895.75,
        "error_mwh": 2787.75,
        "absolute_error_mwh": 2787.75,
    },
    "smard-20-highest-monthly-average-price": {
        "month": "2024-11",
        "average_price_eur_per_mwh": 113.91,
        "eligible_intervals": 2880,
    },
}


def reference() -> HuggingFacePackReference:
    return HuggingFacePackReference.model_validate_json(LOCATOR.read_bytes())


def test_repository_pins_one_exact_public_smard_pack_revision() -> None:
    """The codebase retains an immutable locator, not duplicated SMARD rows."""
    value = reference()

    assert value.repo_id == "lschiemanowski/dsa-datasets"
    assert value.revision == "58958007cdf38eb9e563356f16ecd8011d5a3d67"
    assert value.path == "smard-de-lu-2024/1.2.0"
    assert value.manifest_sha256 == (
        "8afc8246308c445cb6e792ca344da66194826c79c744bdb824fb688ac7badea1"
    )
    assert not (ROOT / "evaluation-packs/smard-cases.jsonl").exists()
    assert not (ROOT / "evaluation-packs/smard_de_lu_2024.duckdb").exists()
    assert not (ROOT / "evaluation-packs/smard-de-lu-2024-1.0.0.json").exists()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("DSA_HUGGINGFACE_TEST") != "1",
    reason="set DSA_HUGGINGFACE_TEST=1 to download the exact public pack revision",
)
def test_exact_public_smard_pack_download() -> None:
    """The exact public pack supplies the complete verified SMARD suite."""
    pack = load_huggingface_evaluation_pack(reference())

    assert pack.manifest.pack_id == "smard-de-lu-2024"
    assert pack.manifest.version == "1.2.0"
    assert pack.manifest.database.id == "smard_de_lu_2024-1.0.0"
    assert pack.manifest.database.sha256 == (
        "249143dd8b399a6be84d666190f16b89ebc71545783362a50efaa40dc02bc45b"
    )
    assert pack.manifest.cases.sha256 == (
        "fa452ec4c64e6da08d3d68c1f769ef71ccf88b77b2974898808cba72f284777b"
    )
    assert len(pack.cases) == 100
    assert tuple(case.case_id for case in pack.cases[:20]) == CASE_IDS
    assert all(case.case_id.startswith("smard-v2-") for case in pack.cases[20:])
    families = {case.metadata.family for case in pack.cases}
    assert {
        "calendar-integrity",
        "capture-price",
        "cross-border-trade",
        "forecast-error",
        "generation-mix",
        "load-profile",
        "price-distribution",
        "provenance",
        "system-conditions",
    } <= families
    assert {
        case.case_id: case.expected_answer
        for case in pack.cases
        if case.case_id in CASE_IDS[12:]
    } == ADDED_EXPECTED_ANSWERS
    assert pack.database_path.is_file()
    assert not pack.database_path.is_symlink()

    records = tuple(case.dataset_record(pack.manifest) for case in pack.cases)
    inputs = tuple(cast(dict[str, JsonValue], record["inputs"]) for record in records)
    assert all("expected_answer" not in value for value in inputs)
    assert all("answer" not in value for value in inputs)
