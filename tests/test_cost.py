"""Tests for provider-native monetary evidence extraction."""

from __future__ import annotations

from pydantic import JsonValue

from dsa.cost import (
    combine_provider_costs,
    provider_cost_from_messages,
    safe_openrouter_response_ids,
)


def response(identity: str, cost: JsonValue = None) -> dict[str, JsonValue]:
    details: dict[str, JsonValue] = {} if cost is None else {"cost": cost}
    return {
        "kind": "response",
        "parts": [],
        "provider_name": "openrouter",
        "provider_response_id": identity,
        "provider_details": details,
    }


def test_openrouter_cost_is_summed_as_exact_decimal_evidence() -> None:
    cost = provider_cost_from_messages(
        (response("gen-one", 0.1), response("gen-two", 0.2))
    )

    assert cost.model_dump(mode="json") == {
        "status": "observed",
        "observed_amount": "0.3",
        "currency": "USD",
        "source": "openrouter_usage_cost",
        "complete_case_count": 1,
        "incomplete_case_count": 0,
        "observed_generation_count": 2,
        "unavailable_generation_count": 0,
    }


def test_missing_foreign_or_duplicate_cost_evidence_is_partial() -> None:
    foreign = response("foreign", 4)
    foreign["provider_name"] = "other"
    cost = provider_cost_from_messages(
        (
            response("gen-one", 0.125),
            response("gen-missing"),
            response("gen-one", 0.25),
            foreign,
        )
    )

    assert cost.status == "partial"
    assert cost.observed_amount == "0.125"
    assert cost.complete_case_count == 0
    assert cost.incomplete_case_count == 1
    assert cost.observed_generation_count == 1
    assert cost.unavailable_generation_count == 3


def test_no_response_or_no_observed_cost_is_unavailable_not_zero() -> None:
    empty = provider_cost_from_messages(())
    missing = provider_cost_from_messages((response("gen-missing"),))

    assert empty.status == "unavailable"
    assert empty.observed_amount is None
    assert empty.incomplete_case_count == 1
    assert empty.unavailable_generation_count == 0
    assert missing.status == "unavailable"
    assert missing.unavailable_generation_count == 1


def test_safe_response_identity_is_retained_when_cost_is_unavailable() -> None:
    messages = (response("gen-missing"), response("gen-invalid", "not-a-number"))

    assert safe_openrouter_response_ids(messages) == (
        "gen-missing",
        "gen-invalid",
    )


def test_combination_preserves_amount_and_case_and_generation_coverage() -> None:
    complete = provider_cost_from_messages((response("gen-one", 0.1),))
    partial = provider_cost_from_messages(
        (response("gen-two", 0.2), response("gen-three"))
    )

    combined = combine_provider_costs((complete, partial))

    assert combined.status == "partial"
    assert combined.observed_amount == "0.3"
    assert combined.complete_case_count == 1
    assert combined.incomplete_case_count == 1
    assert combined.observed_generation_count == 2
    assert combined.unavailable_generation_count == 1
