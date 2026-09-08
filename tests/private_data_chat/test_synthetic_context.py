from __future__ import annotations

import json
import os
import tomllib
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from apps.private_data_chat.contracts import MockDatabaseContext
from apps.private_data_chat.database_card import (
    DatabaseCard,
    DatabaseCardError,
    HuggingFaceDatabaseCardReference,
    load_database_card,
    render_database_overview,
)
from apps.private_data_chat.settings import PrivateDataChatConfiguration, load_configuration
from apps.private_data_chat.synthetic_context import (
    HuggingFaceSyntheticContextReference,
    load_synthetic_context,
    validate_context_card,
)
from tests.private_data_chat.test_database_card import Downloader, card, card_values
from tests.private_data_chat.test_settings import environment


def context() -> MockDatabaseContext:
    return MockDatabaseContext.model_validate(
        {
            "synthetic": True,
            "data_source_id": "energy",
            "display_name": "Example energy",
            "relations": [
                {
                    "name": "analysis.lines",
                    "columns": ["invoice_id", "quantity"],
                    "sample_rows": [{"invoice_id": "FAKE-1", "quantity": 2}],
                }
            ],
        }
    )


def reference(content: bytes) -> HuggingFaceSyntheticContextReference:
    return HuggingFaceSyntheticContextReference(
        repo_id="example/datasets",
        revision="a" * 40,
        path="energy/synthetic-context.json",
        sha256=sha256(content).hexdigest(),
    )


def test_verified_snapshot_download_accepts_hf_cache_symlink(tmp_path: Path) -> None:
    content = context().model_dump_json().encode()
    blob = tmp_path / "blob"
    blob.write_bytes(content)
    snapshot = tmp_path / "snapshot"
    snapshot.symlink_to(blob)
    downloader = Downloader(snapshot)
    loaded = load_synthetic_context(reference(content), downloader=downloader)
    assert loaded == context()
    validate_context_card(loaded, card())
    assert downloader.calls[0]["revision"] == "a" * 40
    blob.write_bytes(content + b" ")
    with pytest.raises(DatabaseCardError, match="digest_mismatch"):
        load_synthetic_context(reference(content), downloader=downloader)


@pytest.mark.parametrize("marker", [False, None])
def test_explicit_synthetic_marker_required(tmp_path: Path, marker: bool | None) -> None:
    raw = context().model_dump(mode="json")
    if marker is None:
        raw.pop("synthetic")
    else:
        raw["synthetic"] = marker
    content = json.dumps(raw).encode()
    path = tmp_path / "context.json"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="explicitly"):
        load_synthetic_context(reference(content), downloader=Downloader(path))


def test_invalid_or_mutated_locator_is_revalidated(tmp_path: Path) -> None:
    ref = reference(b"unused").model_copy(update={"revision": "main"})
    downloader = Downloader(tmp_path / "unused")
    with pytest.raises(ValidationError):
        load_synthetic_context(ref, downloader=downloader)
    assert not downloader.calls


@pytest.mark.parametrize(
    "name,columns",
    [
        ("unknown.table", ["invoice_id", "quantity"]),
        ("analysis.lines", ["quantity"]),
    ],
)
def test_card_and_synthetic_shape_must_agree(name: str, columns: list[str]) -> None:
    raw = context().model_dump(mode="python")
    raw["relations"] = [{"name": name, "columns": columns}]
    with pytest.raises(ValueError, match="do not match"):
        validate_context_card(MockDatabaseContext.model_validate(raw), card())


def test_generic_coverage_is_escaped_and_has_no_retail_requirements() -> None:
    raw = card_values()
    raw.update(
        format="dsa-database-card/v2",
        data_source_id="energy",
        coverage={
            "facts": [
                {"label": "Hours", "value": 8784},
                {"label": "![bad](https://example)", "value": "[link](https://example)"},
            ],
        },
    )
    selected = DatabaseCard.model_validate(raw)
    rendered = render_database_overview(selected)
    assert "8,784" in rendered
    assert "**Invoices:**" not in rendered
    assert "![bad](https://example)" not in rendered
    assert "[link](https://example)" not in rendered
    validate_context_card(context(), selected)
    with pytest.raises(ValueError, match="IDs differ"):
        validate_context_card(context(), selected.model_copy(update={"data_source_id": "other"}))


def test_wrong_card_version_cannot_change_coverage_semantics() -> None:
    with pytest.raises(ValidationError, match="v2 cards"):
        DatabaseCard.model_validate({**card_values(), "format": "dsa-database-card/v2"})


def test_remote_configuration_requires_one_source_and_matching_pins(tmp_path: Path) -> None:
    raw = load_configuration(environment(tmp_path)).model_dump(mode="python")
    ref = reference(b"unused")
    raw["synthetic_context"] = ref.model_dump(mode="python")
    raw["database_card"] = {
        "repo_id": ref.repo_id,
        "revision": ref.revision,
        "path": "energy/database-card.json",
        "sha256": "a" * 64,
    }
    with pytest.raises(ValidationError, match="exactly one"):
        PrivateDataChatConfiguration.model_validate(raw)
    raw["mock_context_path"] = None
    assert PrivateDataChatConfiguration.model_validate(raw).synthetic_context == ref
    raw["database_card"] = None
    with pytest.raises(ValidationError, match="same repository revision"):
        PrivateDataChatConfiguration.model_validate(raw)
    raw["synthetic_context"] = None
    with pytest.raises(ValidationError, match="exactly one"):
        PrivateDataChatConfiguration.model_validate(raw)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("DSA_HUGGINGFACE_TEST") != "1", reason="opt-in HF chat context")
def test_published_chat_context_and_card_match() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "apps/private_data_chat/online-retail-ii.chat.toml.example"
    )
    raw = tomllib.loads(path.read_text())
    card_ref = HuggingFaceDatabaseCardReference.model_validate(raw["database_card"])
    context_ref = HuggingFaceSyntheticContextReference.model_validate(raw["synthetic_context"])
    assert card_ref.revision == context_ref.revision
    loaded_card = load_database_card(card_ref)
    loaded_context = load_synthetic_context(context_ref)
    validate_context_card(loaded_context, loaded_card)
    assert loaded_card.format == "dsa-database-card/v2"
    assert sum(len(relation.sample_rows) for relation in loaded_context.relations) == 2


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("DSA_HUGGINGFACE_TEST") != "1", reason="opt-in HF chat context")
@pytest.mark.parametrize(
    "dataset,card_digest,context_digest,relation_count,sample_count",
    [
        (
            "smard-de-lu-2024",
            "221aba0560d42e05cb9759ae88cfc51bad234db0fa34de28dc25ce93b19b9019",
            "583efd06ebe60398006c660a23168bd58f93ad6193c62c005126cccaa23499b7",
            2, 2,
        ),
        (
            "eea-air-quality-six-cities-2018-2024",
            "beabbb895a7794f797f7318772ea1518eba3ba82cde7de4fed144bc469f8990e",
            "83d971bca7b6fa01aa70673ed6bc9b51a1fb9e68da8bf43afd0960b770bc8bbb",
            5, 4,
        ),
    ],
)
def test_published_smard_and_eea_context(
    dataset: str, card_digest: str, context_digest: str,
    relation_count: int, sample_count: int,
) -> None:
    revision = "c3dbcd3375a678eee85843f7ad738a1ded9edec3"
    card = load_database_card(HuggingFaceDatabaseCardReference(
        repo_id="lschiemanowski/dsa-datasets", revision=revision,
        path=f"{dataset}/database-card.json", sha256=card_digest,
    ))
    synthetic = load_synthetic_context(HuggingFaceSyntheticContextReference(
        repo_id="lschiemanowski/dsa-datasets", revision=revision,
        path=f"{dataset}/synthetic-context.json", sha256=context_digest,
    ))
    validate_context_card(synthetic, card)
    assert card.data_source_id == dataset
    assert len(card.relations) == relation_count
    assert sum(len(relation.sample_rows) for relation in synthetic.relations) == sample_count
    assert "synthetic examples only" in synthetic.display_name
