"""Provider-native monetary evidence derived from retained model responses."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, field_validator, model_validator

from dsa.contract import ContractModel

NonNegativeInt = Annotated[int, Field(ge=0)]
CostStatus = Literal["observed", "partial", "unavailable"]
_SAFE_RESPONSE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_DECIMAL_AMOUNT = re.compile(r"(?:0|[1-9][0-9]{0,17})(?:\.[0-9]{1,18})?")
_MAX_PROVIDER_COST_USD = Decimal("1000000000")


class ProviderCostSummary(ContractModel):
    """Exact observed spend and explicit case/generation coverage."""

    status: CostStatus
    observed_amount: str | None = None
    currency: Literal["USD"] | None = None
    source: Literal["openrouter_usage_cost"] | None = None
    complete_case_count: NonNegativeInt
    incomplete_case_count: NonNegativeInt
    observed_generation_count: NonNegativeInt
    unavailable_generation_count: NonNegativeInt

    @field_validator("observed_amount")
    @classmethod
    def canonical_decimal_amount(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            amount = Decimal(value)
        except InvalidOperation:
            raise ValueError("provider cost must be a canonical decimal amount") from None
        if (
            _DECIMAL_AMOUNT.fullmatch(value) is None
            or amount < 0
            or amount > _MAX_PROVIDER_COST_USD
            or value != _decimal_text(amount)
        ):
            raise ValueError("provider cost must be a canonical decimal amount")
        return value

    @model_validator(mode="after")
    def state_matches_coverage(self) -> ProviderCostSummary:
        has_amount = self.observed_amount is not None
        case_count = self.complete_case_count + self.incomplete_case_count
        if has_amount != (self.currency is not None and self.source is not None):
            raise ValueError("observed provider cost requires currency and source")
        if has_amount != (self.observed_generation_count > 0):
            raise ValueError("provider cost amount does not match generation coverage")
        if self.complete_case_count and not has_amount:
            raise ValueError("complete provider cost cases require observed generations")
        if has_amount and not case_count:
            raise ValueError("observed provider cost requires case coverage")
        if self.unavailable_generation_count and not self.incomplete_case_count:
            raise ValueError("unavailable generations require incomplete case coverage")
        expected: CostStatus
        if not has_amount:
            expected = "unavailable"
        elif self.incomplete_case_count or self.unavailable_generation_count:
            expected = "partial"
        else:
            expected = "observed"
        if self.status != expected:
            raise ValueError("provider cost status does not match its coverage")
        return self

    @property
    def amount_decimal(self) -> Decimal | None:
        return Decimal(self.observed_amount) if self.observed_amount is not None else None


def provider_cost_from_messages(
    messages: Iterable[Mapping[str, JsonValue]],
) -> ProviderCostSummary:
    """Derive one case's billed OpenRouter spend without estimating prices."""
    response_count = 0
    observed: list[Decimal] = []
    unavailable_count = 0
    response_ids: set[str] = set()
    for message in messages:
        if message.get("kind") != "response":
            continue
        response_count += 1
        response_id = message.get("provider_response_id")
        provider_name = message.get("provider_name")
        raw_details = message.get("provider_details")
        details: Mapping[str, JsonValue] = (
            cast(dict[str, JsonValue], raw_details)
            if isinstance(raw_details, dict)
            else cast(dict[str, JsonValue], {})
        )
        amount = _provider_amount(details.get("cost"))
        if (
            provider_name != "openrouter"
            or not isinstance(response_id, str)
            or _SAFE_RESPONSE_ID.fullmatch(response_id) is None
            or response_id in response_ids
            or amount is None
        ):
            unavailable_count += 1
            continue
        response_ids.add(response_id)
        observed.append(amount)

    complete = response_count > 0 and unavailable_count == 0
    return _cost_summary(
        amount=sum(observed, Decimal(0)) if observed else None,
        complete_case_count=int(complete),
        incomplete_case_count=int(not complete),
        observed_generation_count=len(observed),
        unavailable_generation_count=unavailable_count,
    )


def safe_openrouter_response_ids(
    messages: Iterable[Mapping[str, JsonValue]],
) -> tuple[str, ...]:
    """Return safe OpenRouter response identities independently of cost validity."""
    result: list[str] = []
    for message in messages:
        response_id = message.get("provider_response_id")
        if (
            message.get("kind") == "response"
            and message.get("provider_name") == "openrouter"
            and isinstance(response_id, str)
            and _SAFE_RESPONSE_ID.fullmatch(response_id) is not None
        ):
            result.append(response_id)
    return tuple(result)


def combine_provider_costs(
    values: Iterable[ProviderCostSummary],
) -> ProviderCostSummary:
    """Combine exact observed monetary evidence without float coercion."""
    selected = tuple(values)
    amounts = [
        amount
        for item in selected
        if (amount := item.amount_decimal) is not None
    ]
    return _cost_summary(
        amount=sum(amounts, Decimal(0)) if amounts else None,
        complete_case_count=sum(item.complete_case_count for item in selected),
        incomplete_case_count=sum(item.incomplete_case_count for item in selected),
        observed_generation_count=sum(
            item.observed_generation_count for item in selected
        ),
        unavailable_generation_count=sum(
            item.unavailable_generation_count for item in selected
        ),
    )


def unavailable_provider_cost(*, case_count: int = 1) -> ProviderCostSummary:
    """Construct explicit unavailable cost evidence for cases without responses."""
    return _cost_summary(
        amount=None,
        complete_case_count=0,
        incomplete_case_count=case_count,
        observed_generation_count=0,
        unavailable_generation_count=0,
    )


def _cost_summary(
    *,
    amount: Decimal | None,
    complete_case_count: int,
    incomplete_case_count: int,
    observed_generation_count: int,
    unavailable_generation_count: int,
) -> ProviderCostSummary:
    if amount is None:
        status: CostStatus = "unavailable"
    elif incomplete_case_count or unavailable_generation_count:
        status = "partial"
    else:
        status = "observed"
    return ProviderCostSummary(
        status=status,
        observed_amount=_decimal_text(amount) if amount is not None else None,
        currency="USD" if amount is not None else None,
        source="openrouter_usage_cost" if amount is not None else None,
        complete_case_count=complete_case_count,
        incomplete_case_count=incomplete_case_count,
        observed_generation_count=observed_generation_count,
        unavailable_generation_count=unavailable_generation_count,
    )


def _provider_amount(value: JsonValue | None) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    amount = Decimal(str(value))
    text = _decimal_text(amount)
    if (
        amount < 0
        or amount > _MAX_PROVIDER_COST_USD
        or _DECIMAL_AMOUNT.fullmatch(text) is None
    ):
        return None
    return amount


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"
