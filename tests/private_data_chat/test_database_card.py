"""Tests for exact typed database-card loading and projections."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from apps.private_data_chat.contracts import canonical_json_bytes
from apps.private_data_chat.database_card import (
    DatabaseCard,
    DatabaseCardError,
    HuggingFaceDatabaseCardReference,
    load_database_card,
    render_database_overview,
)

REVISION = "a" * 40
MODEL_ONLY_NOTE = "MODEL ONLY: signed quantities must be retained."


def card_values() -> dict[str, object]:
    return {
        "format": "dsa-database-card/v1",
        "title": "Retail data",
        "summary": "The original first paragraph.",
        "coverage": {
            "period_start": "2009-12-01 07:45",
            "period_end": "2011-12-09 12:50",
            "transaction_lines": 1_044_848,
            "invoices": 53_628,
            "identified_customers": 5_942,
            "product_codes": 5_305,
            "country_values": 43,
        },
        "primary_relation": "analysis.lines",
        "relations": [
            {
                "name": "analysis.lines",
                "kind": "view",
                "row_count": 1_044_848,
                "description": "The main transaction-line view.",
                "columns": [
                    {
                        "name": "invoice_id",
                        "data_type": "VARCHAR",
                        "description": "Invoice identifier.",
                    },
                    {
                        "name": "quantity",
                        "data_type": "INTEGER",
                        "description": "Signed item quantity.",
                    },
                ],
            },
            {
                "name": "metadata.dataset",
                "kind": "table",
                "row_count": 1,
                "description": "Source and build metadata.",
                "columns": [
                    {
                        "name": "source_url",
                        "data_type": "VARCHAR",
                        "description": "Original dataset page.",
                    }
                ],
            },
        ],
        "example_questions": ["How did monthly sales change?"],
        "analysis_notes": [MODEL_ONLY_NOTE],
    }


def card() -> DatabaseCard:
    return DatabaseCard.model_validate(card_values())


class Downloader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls: list[dict[str, str]] = []

    def __call__(self, **kwargs: str) -> str:
        self.calls.append(kwargs)
        return str(self.path)


def reference(content: bytes) -> HuggingFaceDatabaseCardReference:
    return HuggingFaceDatabaseCardReference(
        repo_id="lschiemanowski/dsa-datasets",
        revision=REVISION,
        path="online-retail-ii/1.0.0/database-card.json",
        sha256=sha256(content).hexdigest(),
    )


def test_loads_one_exact_card_from_a_snapshot_symlink(tmp_path: Path) -> None:
    content = canonical_json_bytes(card())
    blob = tmp_path / "blobs" / sha256(content).hexdigest()
    blob.parent.mkdir()
    blob.write_bytes(content)
    snapshot = tmp_path / "snapshot" / "database-card.json"
    snapshot.parent.mkdir()
    snapshot.symlink_to(blob)
    downloader = Downloader(snapshot)

    loaded = load_database_card(reference(content), downloader=downloader)

    assert loaded == card()
    assert downloader.calls == [
        {
            "repo_id": "lschiemanowski/dsa-datasets",
            "repo_type": "dataset",
            "revision": REVISION,
            "filename": "online-retail-ii/1.0.0/database-card.json",
        }
    ]


def test_user_projection_describes_contents_but_omits_analysis_notes() -> None:
    overview = render_database_overview(card())

    assert overview.startswith(r"The original first paragraph\.")
    assert "1,044,848" in overview
    assert "`analysis.lines` (view, 1,044,848 rows)" in overview
    assert r"| `quantity` | INTEGER | Signed item quantity\. |" in overview
    assert r"- How did monthly sales change\?" in overview
    assert MODEL_ONLY_NOTE not in overview


@pytest.mark.parametrize(
    "updates",
    [
        {"revision": "main"},
        {"path": "../private.json"},
        {"repo_id": "missing-namespace"},
    ],
)
def test_reference_requires_an_exact_revision_and_safe_path(updates: dict[str, str]) -> None:
    values: dict[str, Any] = {
        "repo_id": "lschiemanowski/dsa-datasets",
        "revision": REVISION,
        "path": "online-retail-ii/1.0.0/database-card.json",
        "sha256": "b" * 64,
    }
    values.update(updates)
    with pytest.raises(ValidationError):
        HuggingFaceDatabaseCardReference.model_validate(values)


def test_revalidates_mutated_typed_reference_before_download(tmp_path: Path) -> None:
    content = canonical_json_bytes(card())
    path = tmp_path / "database-card.json"
    path.write_bytes(content)
    selected = reference(content).model_copy(update={"revision": "main"})
    downloader = Downloader(path)

    with pytest.raises(DatabaseCardError) as caught:
        load_database_card(selected, downloader=downloader)

    assert caught.value.code == "database_card_reference_invalid"
    assert downloader.calls == []


def test_rejects_digest_mismatch_without_retaining_transport_details(tmp_path: Path) -> None:
    content = canonical_json_bytes(card())
    path = tmp_path / "database-card.json"
    path.write_bytes(content)
    selected = reference(content).model_copy(update={"sha256": "0" * 64})

    with pytest.raises(DatabaseCardError) as caught:
        load_database_card(selected, downloader=Downloader(path))

    assert caught.value.code == "database_card_digest_mismatch"
    assert str(path) not in str(caught.value)


@pytest.mark.parametrize("content", [b"", b"\xff", b"{}", json.dumps([]).encode()])
def test_rejects_empty_binary_or_invalid_cards(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "database-card.json"
    path.write_bytes(content)

    with pytest.raises(DatabaseCardError):
        load_database_card(reference(content), downloader=Downloader(path))


def test_card_rejects_duplicate_relations_and_columns() -> None:
    values = card_values()
    relations = cast(list[dict[str, object]], values["relations"])
    values["relations"] = [relations[0], relations[0]]
    with pytest.raises(ValidationError, match="relation names must be unique"):
        DatabaseCard.model_validate(values)

    values = card_values()
    relations = cast(list[dict[str, object]], values["relations"])
    relation = dict(relations[0])
    columns = cast(list[dict[str, object]], relation["columns"])
    relation["columns"] = [columns[0], columns[0]]
    values["relations"] = [relation]
    with pytest.raises(ValidationError, match="column names must be unique"):
        DatabaseCard.model_validate(values)


def test_user_projection_revalidates_a_mutated_typed_card() -> None:
    selected = card().model_copy(update={"relations": ()})

    with pytest.raises(ValidationError, match="at least 1 item"):
        render_database_overview(selected)


def test_user_projection_neutralizes_markdown_in_every_free_text_field() -> None:
    values = card_values()
    values["summary"] = "![summary](https://attacker.example/summary)"
    values["example_questions"] = ["[Question](https://attacker.example/question)"]
    relations = cast(list[dict[str, object]], values["relations"])
    primary = dict(relations[0])
    primary["description"] = "![relation](https://attacker.example/relation)"
    columns = cast(list[dict[str, object]], primary["columns"])
    first_column = dict(columns[0])
    first_column["data_type"] = "[type](https://attacker.example/type)"
    first_column["description"] = "![column](https://attacker.example/column)"
    primary["columns"] = [first_column, *columns[1:]]
    values["relations"] = [primary, *relations[1:]]

    overview = render_database_overview(DatabaseCard.model_validate(values))

    assert "![" not in overview
    assert "](https://" not in overview
    assert "\\!\\[summary\\]\\(https\\:\\/\\/attacker\\.example\\/summary\\)" in overview
    assert "\\[Question\\]\\(https\\:\\/\\/attacker\\.example\\/question\\)" in overview
