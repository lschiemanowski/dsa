"""Pinned identity and opt-in live verification of EEA air quality 2018-2024."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from dsa.pack import HuggingFacePackReference, load_huggingface_evaluation_pack

ROOT = Path(__file__).resolve().parents[1]
LOCATOR = ROOT / "evaluation-packs/eea-air-quality-six-cities-2018-2024-1.1.0.json"
CASE_IDS = tuple(
    f"eea-{number:02d}-{suffix}"
    for number, suffix in (
        (1, "valid-measurements"),
        (2, "lowest-pm25-2024"),
        (3, "highest-pm25-2018"),
        (4, "largest-no2-decrease"),
        (5, "largest-pm10-relative-decrease"),
        (6, "peak-complete-o3-day"),
        (7, "peak-no2-city-hour"),
        (8, "peak-valid-pm10-measurement"),
        (9, "most-pm10-hours-above-50"),
        (10, "most-pm25-days-above-15"),
        (11, "longest-no2-run-above-40"),
        (12, "strongest-pm25-no2-correlation"),
        (13, "madrid-peak-o3-calendar-month"),
        (14, "largest-o3-increase"),
        (15, "rank-2024-pm25-cities"),
        (16, "largest-no2-drop-2019-2020"),
        (17, "most-well-covered-so2-years"),
        (18, "highest-invalid-pollutant-rate"),
        (19, "highest-below-detection-rate"),
        (20, "simultaneous-pm10-hours-2024"),
    )
)
SELECTED_EXPECTED_ANSWERS: dict[str, JsonValue] = {
    "eea-11-longest-no2-run-above-40": {
        "city": "Madrid",
        "start_reporting_time": "2021-01-11T15:00:00",
        "end_reporting_time": "2021-01-20T23:00:00",
        "consecutive_hours": 225,
    },
    "eea-12-strongest-pm25-no2-correlation": {
        "city": "Paris",
        "paired_hours": 61368,
        "correlation": 0.571188,
    },
    "eea-19-highest-below-detection-rate": {
        "city": "Berlin",
        "pollutant": "SO2",
        "below_detection_count": 26178,
        "analyzable_count": 31795,
        "percent": 82.3337,
    },
    "eea-20-simultaneous-pm10-hours-2024": {
        "common_hours": 8782,
        "simultaneous_hours": 7,
    },
}


def reference() -> HuggingFacePackReference:
    return HuggingFacePackReference.model_validate_json(LOCATOR.read_bytes())


def test_repository_pins_one_exact_public_eea_pack_revision() -> None:
    """The repository retains an immutable locator, not duplicated EEA rows."""
    value = reference()

    assert value.repo_id == "lschiemanowski/dsa-datasets"
    assert value.path == "eea-air-quality-six-cities-2018-2024/1.1.0"
    assert value.revision == "58958007cdf38eb9e563356f16ecd8011d5a3d67"
    assert value.manifest_sha256 == (
        "d2d29215ada662b7e6dcc70188a91ea9b7e50df8f77a20ee24521b869b137317"
    )
    assert not (ROOT / "evaluation-packs/eea-air-quality-cases.jsonl").exists()
    assert not (ROOT / "evaluation-packs/eea_air_quality_six_cities_2018_2024.duckdb").exists()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("DSA_HUGGINGFACE_TEST") != "1",
    reason="set DSA_HUGGINGFACE_TEST=1 to download the exact public EEA pack revision",
)
def test_exact_public_eea_pack_download() -> None:
    """The exact public pack supplies the complete verified EEA suite."""
    pack = load_huggingface_evaluation_pack(reference())

    assert pack.manifest.pack_id == "eea-air-quality-six-cities-2018-2024"
    assert pack.manifest.version == "1.1.0"
    assert pack.manifest.database.id == ("eea_air_quality_six_cities_2018_2024-1.0.0")
    assert pack.manifest.database.sha256 == (
        "974df8df6bd3af538c1ca098f3c6c6bbe80f07e7a63ac565dc61c3ba6d7a3fd5"
    )
    assert pack.manifest.cases.sha256 == (
        "9f94d99fc98cac7a6cc9927e54b4f7887f2fac1cf052b6150743aacedcd57a34"
    )
    assert len(pack.cases) == 100
    assert tuple(case.case_id for case in pack.cases[:20]) == CASE_IDS
    assert all(case.case_id.startswith("eea-v2-") for case in pack.cases[20:])
    families = {case.metadata.family for case in pack.cases}
    assert {
        "annual-trend",
        "city-dashboard",
        "coverage",
        "cross-city-correlation",
        "diurnal-profile",
        "provenance",
        "sampling-point-variation",
        "seasonality",
        "validity-audit",
    } <= families
    assert {
        case.case_id: case.expected_answer
        for case in pack.cases
        if case.case_id in SELECTED_EXPECTED_ANSWERS
    } == SELECTED_EXPECTED_ANSWERS
    assert pack.database_path.is_file()
    assert not pack.database_path.is_symlink()

    records = tuple(case.dataset_record(pack.manifest) for case in pack.cases)
    inputs = tuple(cast(dict[str, JsonValue], record["inputs"]) for record in records)
    assert all("expected_answer" not in value for value in inputs)
    assert all("answer" not in value for value in inputs)
