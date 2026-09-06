from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue, ValidationError

from apps.private_data_chat.contracts import (
    AnalysisResult,
    ArtifactIdentity,
    ClarifierTurn,
    MockDatabaseContext,
    MockRelation,
    ProposalBinding,
    ProposalPayload,
    QuantitativeInterpretation,
    proposal_digest,
    proposal_payload_json,
)


def answer_schema() -> dict[str, JsonValue]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "total": {"type": "number", "description": "Total net sales in GBP"},
            "months": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["total", "months"],
        "additionalProperties": False,
    }


def proposal_payload() -> ProposalPayload:
    return ProposalPayload(
        question="What was monthly net sales in GBP during 2011?",
        interpretation=QuantitativeInterpretation(
            measure="Sum quantity times unit price, excluding cancellations",
            population="Completed invoice lines",
            group_by=("calendar month",),
            filters=("invoice date is in 2011",),
            time_window="2011-01-01 inclusive through 2012-01-01 exclusive",
            units="GBP",
        ),
        answer_schema=answer_schema(),
    )


def artifact(name: str = "terminal") -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_id=f"artifact-{name}-0123456789abcdef",
        sha256="a" * 64,
        byte_length=123,
        media_type=(
            "application/x-ipynb+json" if name == "notebook" else "application/json"
        ),
    )


def test_proposal_copies_and_accepts_the_bounded_schema() -> None:
    source = answer_schema()
    payload = ProposalPayload(
        question="Calculate net sales.",
        interpretation=QuantitativeInterpretation(measure="Net sales", population="All sales"),
        answer_schema=source,
    )

    properties = source["properties"]
    assert isinstance(properties, dict)
    properties["secret"] = {"type": "string"}

    retained_properties = payload.answer_schema["properties"]
    assert isinstance(retained_properties, dict)
    assert "secret" not in retained_properties


def test_analysis_guidance_is_optional_bounded_and_bound_by_the_proposal_digest() -> None:
    without_guidance = proposal_payload()
    assert without_guidance.analysis_guidance is None
    assert "analysis_guidance" not in proposal_payload_json(without_guidance)

    guidance = "1. Filter to 2011.\n2. Aggregate line value by calendar month."
    with_guidance = ProposalPayload.model_validate(
        {
            **without_guidance.model_dump(mode="python"),
            "analysis_guidance": guidance,
        }
    )
    binding = ProposalBinding(
        user_id="user-1",
        conversation_id="chat-1",
        data_source_id="retail",
    )
    assert proposal_payload_json(with_guidance)["analysis_guidance"] == guidance
    assert proposal_digest(binding, without_guidance) != proposal_digest(
        binding,
        with_guidance,
    )

    for invalid in ("   ", "x" * 12_001):
        with pytest.raises(ValidationError):
            ProposalPayload.model_validate(
                {
                    **without_guidance.model_dump(mode="python"),
                    "analysis_guidance": invalid,
                }
            )


@pytest.mark.parametrize(
    "mutation",
    [
        {"$ref": "file:///private/database"},
        {"oneOf": [{"type": "number"}, {"type": "string"}]},
        {"patternProperties": {".*": {"type": "string"}}},
    ],
)
def test_proposal_rejects_schema_features_outside_the_audited_subset(
    mutation: dict[str, object],
) -> None:
    schema = answer_schema()
    schema.update(cast(dict[str, JsonValue], mutation))
    with pytest.raises(ValidationError, match="unsupported keywords"):
        ProposalPayload(
            question="Calculate net sales.",
            interpretation=QuantitativeInterpretation(measure="Net sales", population="All sales"),
            answer_schema=schema,
        )


def test_proposal_cannot_supply_privileged_execution_fields() -> None:
    raw = proposal_payload().model_dump(mode="python")
    raw["database_path"] = "/private/data.duckdb"
    raw["model"] = {"name": "attacker-choice"}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProposalPayload.model_validate(raw)


def test_mock_context_is_explicitly_synthetic_and_row_shaped() -> None:
    context = MockDatabaseContext(
        data_source_id="retail",
        display_name="Retail mock",
        relations=(
            MockRelation(
                name="invoice_lines",
                columns=("invoice_id", "quantity"),
                sample_rows=({"invoice_id": "FAKE-1", "quantity": 2},),
            ),
        ),
    )
    assert context.synthetic is True

    with pytest.raises(ValidationError, match="exactly the declared columns"):
        MockRelation(
            name="invoice_lines",
            columns=("invoice_id", "quantity"),
            sample_rows=({"invoice_id": "FAKE-1"},),
        )


def test_checked_in_mock_context_is_valid_and_explicitly_synthetic() -> None:
    path = Path(__file__).parents[2] / "apps/private_data_chat/mock-database.example.json"
    context = MockDatabaseContext.model_validate_json(path.read_bytes())
    assert context.synthetic is True
    assert context.data_source_id == "online_retail_ii"
    assert tuple(relation.name for relation in context.relations) == (
        "analysis.transaction_lines_nonoverlapping",
    )
    assert context.relations[0].columns == (
        "line_id",
        "source_sheet",
        "source_row",
        "invoice_id",
        "stock_code",
        "description",
        "quantity",
        "invoice_ts",
        "unit_price_gbp",
        "customer_id",
        "country",
    )
    assert all(
        "FAKE" in str(value)
        for relation in context.relations
        for row in relation.sample_rows
        for key, value in row.items()
        if key in {"invoice_id", "stock_code", "customer_id"}
    )


def test_clarifier_turn_requires_proposal_only_for_proposal_kind() -> None:
    clarification = ClarifierTurn(
        kind="clarification",
        message="Which calendar year should I use?",
    )
    assert clarification.proposal is None

    with pytest.raises(ValidationError, match="only proposal turns"):
        ClarifierTurn(
            kind="proposal",
            message="Ready.",
        )


def test_analysis_result_enforces_success_and_failure_evidence_shapes() -> None:
    success = AnalysisResult(
        run_id="analysis-0123456789abcdef",
        proposal_id="proposal-0123456789abcdef",
        status="succeeded",
        answer={"total": 1.25},
        terminal=artifact(),
        notebook=artifact("notebook"),
    )
    assert success.notebook is not None

    with pytest.raises(ValidationError, match="failed analysis"):
        AnalysisResult(
            run_id=success.run_id,
            proposal_id=success.proposal_id,
            status="failed",
            answer={"total": 1.25},
            terminal=artifact(),
            failure_code="analysis_failed",
        )

    with pytest.raises(ValidationError, match="successful analysis"):
        AnalysisResult(
            run_id=success.run_id,
            proposal_id=success.proposal_id,
            status="succeeded",
            answer={"total": 1.25},
        )


def test_typed_proposal_nested_mutation_is_observable_to_boundary_revalidation() -> None:
    payload = proposal_payload()
    mutated = deepcopy(payload.answer_schema)
    mutated["$ref"] = "file:///private/database"
    object.__setattr__(payload, "answer_schema", mutated)

    with pytest.raises(ValidationError, match="unsupported keywords"):
        ProposalPayload.model_validate(payload.model_dump(mode="python", round_trip=True))
