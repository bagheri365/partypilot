"""Deterministic in-memory resource-store adapter and fixture data."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from partypilot.domain.party_request import AgeRange
from partypilot.domain.resources import (
    AccessibilityAttribute,
    Activity,
    Caterer,
    Resource,
    Venue,
)
from partypilot.domain.temporal import TimeWindow
from partypilot.ports.resource_store import ResourceSearchCriteria


def _window(start_hour: int, end_hour: int) -> TimeWindow:
    return TimeWindow(
        start=datetime(2026, 9, 20, start_hour),
        end=datetime(2026, 9, 20, end_hour),
    )


DEFAULT_RESOURCES: tuple[Resource, ...] = (
    Venue(
        resource_id="venue-brooklyn-loft",
        name="Brooklyn Party Loft",
        location="Brooklyn, NY",
        price=Decimal("700.00"),
        capacity=50,
        availability=(_window(10, 20),),
        accessibility_attributes=frozenset(
            {
                AccessibilityAttribute.WHEELCHAIR_ACCESSIBLE,
                AccessibilityAttribute.ACCESSIBLE_RESTROOM,
            }
        ),
    ),
    Venue(
        resource_id="venue-small-studio",
        name="Small Studio",
        location="Brooklyn, NY",
        price=Decimal("300.00"),
        capacity=8,
        availability=(_window(12, 16),),
    ),
    Caterer(
        resource_id="caterer-family-table",
        name="Family Table Catering",
        location="Brooklyn, NY",
        price=Decimal("425.00"),
        capacity=60,
        availability=(_window(9, 19),),
    ),
    Caterer(
        resource_id="caterer-premium",
        name="Premium Banquets",
        location="Brooklyn, NY",
        price=Decimal("1800.00"),
        capacity=120,
        availability=(_window(9, 21),),
    ),
    Activity(
        resource_id="activity-craft-party",
        name="Craft Party",
        location="Brooklyn, NY",
        price=Decimal("250.00"),
        capacity=30,
        availability=(_window(11, 18),),
        age_restrictions=AgeRange(minimum=5, maximum=12),
    ),
    Activity(
        resource_id="activity-teen-climbing",
        name="Teen Climbing",
        location="Queens, NY",
        price=Decimal("350.00"),
        capacity=20,
        availability=(_window(14, 18),),
        age_restrictions=AgeRange(minimum=13, maximum=17),
    ),
)


class InMemoryResourceStore:
    """ResourceStore implementation backed by an immutable resource tuple."""

    def __init__(self, resources: tuple[Resource, ...] = DEFAULT_RESOURCES) -> None:
        self._resources = resources

    def search(self, criteria: ResourceSearchCriteria) -> tuple[Resource, ...]:
        """Return resources matching every supplied structured criterion."""
        return tuple(resource for resource in self._resources if self._matches(resource, criteria))

    @staticmethod
    def _matches(resource: Resource, criteria: ResourceSearchCriteria) -> bool:
        if (
            criteria.location is not None
            and resource.location.casefold() != criteria.location.casefold()
        ):
            return False
        if criteria.minimum_capacity is not None and (
            resource.capacity is None or resource.capacity < criteria.minimum_capacity
        ):
            return False
        if criteria.maximum_price is not None and resource.price > criteria.maximum_price:
            return False
        if criteria.category is not None and resource.category is not criteria.category:
            return False
        return not (
            criteria.availability is not None
            and not any(
                window.contains_window(criteria.availability) for window in resource.availability
            )
        )
