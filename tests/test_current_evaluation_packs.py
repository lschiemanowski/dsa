"""Pin the revised problem exports, not merely the number of cases."""

import os
from pathlib import Path

import pytest

from dsa.pack import HuggingFacePackReference, load_huggingface_evaluation_pack

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "online-retail-ii": "dd5a9983411d533f503f3e3a44f34c63f4955c290ca6a042286bc6c27a32b9c9",
    "smard-de-lu-2024": "63f952c0551c246ea556dd6337a1de02f8672a9d37f07d133201233424854931",
    "eea-air-quality-six-cities-2018-2024": (
        "e4977bcecfe050a7166fbf909362a242f276b2a5140353a64e967e62ecc304e4"
    ),
}


@pytest.mark.parametrize("slug", CASES)
def test_current_locator_uses_versionless_path_and_immutable_revision(slug: str) -> None:
    ref = HuggingFacePackReference.model_validate_json(
        (ROOT / "examples/evaluation" / f"{slug}.json").read_bytes()
    )
    assert ref.path == slug
    assert ref.revision == "f45a22879769fc02731f3dc55d126a8d6705d4b9"


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("DSA_HUGGINGFACE_TEST") != "1", reason="opt-in HF download")
@pytest.mark.parametrize("slug", CASES)
def test_current_pack_contains_the_latest_revised_cases(slug: str) -> None:
    ref = HuggingFacePackReference.model_validate_json(
        (ROOT / "examples/evaluation" / f"{slug}.json").read_bytes()
    )
    pack = load_huggingface_evaluation_pack(ref)
    assert len(pack.cases) == 100
    assert pack.manifest.cases.sha256 == CASES[slug]
    assert pack.manifest.version == "2.0.0"
    assert not pack.database_path.is_symlink()
