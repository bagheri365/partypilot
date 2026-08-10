from __future__ import annotations

from datetime import date, time
from decimal import Decimal

import pytest
from pydantic import ValidationError

from partypilot.domain import AgeRange, PartyRequest


def test_party_request_accepts_supported_fields() -> None:
    request = PartyRequest(
        location="Brooklyn, NY",
        event_date=date(2026, 9, 12),
        event_time=time(14, 30),
        guest_count=18,
        child_age_range=AgeRange(minimum=6, maximum=9),
        total_budget=Decimal("1250.00"),
        theme_preferences=("space", "hands-on activities"),
        allergies=("peanuts",),
        dietary_restrictions=("vegetarian",),
        accessibility_needs=("step-free access",),
        other_constraints=("indoor backup required",),
    )

    assert request.location == "Brooklyn, NY"
    assert request.event_time == time(14, 30)
    assert request.guest_count == 18
    assert request.total_budget == Decimal("1250.00")
    assert request.child_age_range == AgeRange(minimum=6, maximum=9)


def test_party_request_allows_single_child_age_and_optional_time() -> None:
    request = PartyRequest(
        location="Queens, NY",
        event_date=date(2026, 10, 1),
        guest_count=10,
        child_age=7,
        total_budget=Decimal("0"),
    )

    assert request.event_time is None
    assert request.child_age == 7
    assert request.total_budget == Decimal("0")


@pytest.mark.parametrize("guest_count", [0, -1])
def test_party_request_rejects_invalid_guest_count(guest_count: int) -> None:
    with pytest.raises(ValidationError):
        PartyRequest(
            location="Manhattan, NY",
            event_date=date(2026, 8, 20),
            guest_count=guest_count,
            total_budget=Decimal("500"),
        )


def test_party_request_rejects_negative_budget() -> None:
    with pytest.raises(ValidationError):
        PartyRequest(
            location="Manhattan, NY",
            event_date=date(2026, 8, 20),
            guest_count=12,
            total_budget=Decimal("-0.01"),
        )


def test_age_range_rejects_reversed_bounds() -> None:
    with pytest.raises(ValidationError, match="maximum age cannot be less than minimum age"):
        AgeRange(minimum=10, maximum=4)


def test_party_request_rejects_single_age_and_range_together() -> None:
    with pytest.raises(ValidationError, match="either child_age or child_age_range"):
        PartyRequest(
            location="Bronx, NY",
            event_date=date(2026, 11, 8),
            guest_count=8,
            child_age=8,
            child_age_range=AgeRange(minimum=7, maximum=9),
            total_budget=Decimal("750"),
        )


def test_party_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PartyRequest.model_validate(
            {
                "location": "Staten Island, NY",
                "event_date": date(2026, 12, 5),
                "guest_count": 6,
                "total_budget": Decimal("400"),
                "unsupported_option": True,
            }
        )
