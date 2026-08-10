from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import assert_type

import pytest
from pydantic import ValidationError

from partypilot.domain.resources import Resource, ResourceCategory, Venue
from partypilot.domain.temporal import TimeWindow
from partypilot.ports import ResourceSearchCriteria, ResourceStore


class FakeResourceStore:
    def __init__(self, resources: tuple[Resource, ...]) -> None:
        self.resources = resources
        self.last_criteria: ResourceSearchCriteria | None = None

    def search(self, criteria: ResourceSearchCriteria) -> tuple[Resource, ...]:
        self.last_criteria = criteria
        return self.resources


def test_search_criteria_supports_structured_filters() -> None:
    availability = TimeWindow(
        start=datetime(2026, 9, 1, 10, 0),
        end=datetime(2026, 9, 1, 14, 0),
    )

    criteria = ResourceSearchCriteria(
        location="  Brooklyn  ",
        minimum_capacity=20,
        maximum_price=Decimal("750.00"),
        availability=availability,
        category=ResourceCategory.VENUE,
    )

    assert criteria.location == "Brooklyn"
    assert criteria.minimum_capacity == 20
    assert criteria.maximum_price == Decimal("750.00")
    assert criteria.availability == availability
    assert criteria.category is ResourceCategory.VENUE


def test_search_criteria_allows_partial_filters() -> None:
    criteria = ResourceSearchCriteria(category=ResourceCategory.ACTIVITY)

    assert criteria.location is None
    assert criteria.minimum_capacity is None
    assert criteria.maximum_price is None
    assert criteria.availability is None


def test_search_criteria_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        ResourceSearchCriteria(location="   ")
    with pytest.raises(ValidationError):
        ResourceSearchCriteria(minimum_capacity=0)
    with pytest.raises(ValidationError):
        ResourceSearchCriteria(maximum_price=Decimal("-0.01"))


def test_resource_store_is_structurally_typed() -> None:
    venue = Venue(
        resource_id="venue-1",
        name="Garden Room",
        location="Brooklyn",
        price=Decimal("500"),
        capacity=30,
    )
    store: ResourceStore = FakeResourceStore((venue,))
    criteria = ResourceSearchCriteria(category=ResourceCategory.VENUE)

    results = store.search(criteria)

    assert_type(results, tuple[Resource, ...])
    assert results == (venue,)
    assert isinstance(store, FakeResourceStore)
    assert store.last_criteria == criteria
