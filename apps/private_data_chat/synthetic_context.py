"""Explicitly authored synthetic context, loaded from a content-pinned repository."""

from __future__ import annotations

import json
from typing import Literal, cast

from apps.private_data_chat.contracts import MockDatabaseContext
from apps.private_data_chat.database_card import (
    CardDownload,
    DatabaseCard,
    HuggingFaceContextReference,
    download_context_bytes,
)


class HuggingFaceSyntheticContextReference(HuggingFaceContextReference):
    format: Literal["dsa-huggingface-synthetic-context/v1"] = "dsa-huggingface-synthetic-context/v1"


def load_synthetic_context(
    reference: HuggingFaceSyntheticContextReference,
    *,
    downloader: CardDownload | None = None,
) -> MockDatabaseContext:
    """Never sample a private database or substitute a local fallback on failure."""
    content = download_context_bytes(reference, downloader=downloader)
    return parse_synthetic_context(content)


def parse_synthetic_context(content: bytes) -> MockDatabaseContext:
    """Require the explicit synthetic marker even though the internal model defaults it."""
    raw = json.loads(content)
    if not isinstance(raw, dict) or cast(dict[str, object], raw).get("synthetic") is not True:
        raise ValueError("published context must explicitly declare synthetic=true")
    return MockDatabaseContext.model_validate_json(content)


def validate_context_card(context: MockDatabaseContext, card: DatabaseCard) -> None:
    """Require matching identities and exact columns for each synthetic relation."""
    selected = MockDatabaseContext.model_validate_json(context.model_dump_json())
    selected_card = DatabaseCard.model_validate_json(card.model_dump_json())
    if (
        selected_card.data_source_id is not None
        and selected_card.data_source_id != selected.data_source_id
    ):
        raise ValueError("synthetic context and card data source IDs differ")
    relations = {relation.name: relation for relation in selected_card.relations}
    for relation in selected.relations:
        actual = relations.get(relation.name)
        if actual is None or set(relation.columns) != {column.name for column in actual.columns}:
            raise ValueError("synthetic context relation/columns do not match the card")
