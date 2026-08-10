"""Tests for structured resource domain models."""

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from partypilot.domain import (
    AccessibilityAttribute,
    Activity,
    AgeRange,
    Caterer,
    ResourceCategory,
    TimeWindow,
    Venue,
)


def _availability() -> tuple[TimeWindow, ...]:
    return (
        TimeWindow(
            start=datetime(2026, 8, 15, 9),
            end=datetime(2026, 8, 15, 18),
        ),
    )


def test_venue_contains_structured_common_fields() -> None:
    venue = Venue(
        resource_id="venue-1",
        name="Community Hall",
        location="Brooklyn, NY",
        price=Decimal("650.00"),
        capacity=80,
        availability=_availability(),
        accessibility_attributes=frozenset(
            {
                AccessibilityAttribute.WHEELCHAIR_ACCESSIBLE,
                AccessibilityAttribute.ACCESSIBLE_RESTROOM,
            }
        ),
    )

    assert venue.category is ResourceCategory.VENUE
    assert venue.capacity == 80
    assert venue.price == Decimal("650.00")
    assert venue.availability[0].contains(datetime(2026, 8, 15, 12))


def test_caterer_capacity_is_optional() -> None:
    caterer = Caterer(
        resource_id="caterer-1",
        name="Neighborhood Kitchen",
        location="Brooklyn, NY",
        price=Decimal("25.50"),
    )

    assert caterer.category is ResourceCategory.CATERER
    assert caterer.capacity is None


def test_activity_supports_age_restrictions() -> None:
    activity = Activity(
        resource_id="activity-1",
        name="Junior Climbing",
        location="Brooklyn, NY",
        price=Decimal("300"),
        capacity=20,
        age_restrictions=AgeRange(minimum=8, maximum=14),
    )

    assert activity.category is ResourceCategory.ACTIVITY
    assert activity.age_restrictions == AgeRange(minimum=8, maximum=14)


@pytest.mark.parametrize(
    ("resource_type", "kwargs"),
    [
        (Venue, {"capacity": 10}),
        (Caterer, {}),
        (Activity, {"capacity": 10}),
    ],
)
def test_resources_reject_negative_prices(resource_type: type, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        resource_type(
            resource_id="resource-1",
            name="Example",
            location="Brooklyn, NY",
            price=Decimal("-0.01"),
            **kwargs,
        )


def test_capacity_must_be_positive_when_present() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Venue(
            resource_id="venue-1",
            name="Example",
            location="Brooklyn, NY",
            price=Decimal("0"),
            capacity=0,
        )


def test_resource_identity_fields_cannot_be_blank() -> None:
    with pytest.raises(ValidationError, match="value cannot be blank"):
        Caterer(
            resource_id=" ",
            name="Example",
            location="Brooklyn, NY",
            price=Decimal("10"),
        )


def test_concrete_resource_rejects_wrong_category() -> None:
    with pytest.raises(ValidationError, match="venue category must be 'venue'"):
        Venue(
            resource_id="venue-1",
            name="Example",
            location="Brooklyn, NY",
            price=Decimal("100"),
            capacity=10,
            category=ResourceCategory.ACTIVITY,
        )


def test_resource_models_are_immutable_and_forbid_unknown_fields() -> None:
    venue = Venue(
        resource_id="venue-1",
        name="Example",
        location="Brooklyn, NY",
        price=Decimal("100"),
        capacity=10,
    )

    with pytest.raises(ValidationError):
        venue.name = "Changed"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Venue.model_validate(
            {
                "resource_id": "venue-2",
                "name": "Example",
                "location": "Brooklyn, NY",
                "price": Decimal("100"),
                "capacity": 10,
                "policy_text": "long unstructured policy",
            }
        )
