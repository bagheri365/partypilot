"""Tests for the deterministic in-memory resource adapter."""

from datetime import datetime
from decimal import Decimal
from typing import assert_type

from partypilot.adapters import DEFAULT_RESOURCES, InMemoryResourceStore
from partypilot.domain import Resource, ResourceCategory, TimeWindow
from partypilot.ports import ResourceSearchCriteria, ResourceStore


def test_adapter_satisfies_resource_store_protocol() -> None:
    store: ResourceStore = InMemoryResourceStore()
    results = store.search(ResourceSearchCriteria())

    assert_type(results, tuple[Resource, ...])
    assert results == DEFAULT_RESOURCES


def test_default_fixtures_cover_all_resource_categories() -> None:
    categories = {resource.category for resource in DEFAULT_RESOURCES}

    assert categories == {
        ResourceCategory.VENUE,
        ResourceCategory.CATERER,
        ResourceCategory.ACTIVITY,
    }


def test_default_fixtures_include_feasible_and_infeasible_capacity_examples() -> None:
    store = InMemoryResourceStore()

    venues = store.search(
        ResourceSearchCriteria(
            location="Brooklyn, NY",
            category=ResourceCategory.VENUE,
        )
    )

    assert {venue.resource_id for venue in venues} == {
        "venue-brooklyn-loft",
        "venue-small-studio",
    }
    assert any(venue.capacity is not None and venue.capacity >= 25 for venue in venues)
    assert any(venue.capacity is not None and venue.capacity < 25 for venue in venues)


def test_search_filters_by_location_case_insensitively() -> None:
    store = InMemoryResourceStore()

    results = store.search(ResourceSearchCriteria(location="brooklyn, ny"))

    assert results
    assert all(resource.location == "Brooklyn, NY" for resource in results)


def test_search_filters_by_minimum_capacity() -> None:
    store = InMemoryResourceStore()

    results = store.search(ResourceSearchCriteria(minimum_capacity=50))

    assert results
    assert all(resource.capacity is not None and resource.capacity >= 50 for resource in results)


def test_search_filters_by_maximum_price() -> None:
    store = InMemoryResourceStore()

    results = store.search(ResourceSearchCriteria(maximum_price=Decimal("300.00")))

    assert {resource.resource_id for resource in results} == {
        "venue-small-studio",
        "activity-craft-party",
    }


def test_search_filters_by_category() -> None:
    store = InMemoryResourceStore()

    results = store.search(ResourceSearchCriteria(category=ResourceCategory.CATERER))

    assert {resource.resource_id for resource in results} == {
        "caterer-family-table",
        "caterer-premium",
    }


def test_search_requires_requested_window_to_be_fully_available() -> None:
    store = InMemoryResourceStore()
    requested = TimeWindow(
        start=datetime(2026, 9, 20, 17),
        end=datetime(2026, 9, 20, 19),
    )

    results = store.search(
        ResourceSearchCriteria(
            category=ResourceCategory.VENUE,
            availability=requested,
        )
    )

    assert {resource.resource_id for resource in results} == {"venue-brooklyn-loft"}


def test_search_combines_all_filters_deterministically() -> None:
    store = InMemoryResourceStore()
    requested = TimeWindow(
        start=datetime(2026, 9, 20, 12),
        end=datetime(2026, 9, 20, 16),
    )

    results = store.search(
        ResourceSearchCriteria(
            location="Brooklyn, NY",
            minimum_capacity=20,
            maximum_price=Decimal("500.00"),
            availability=requested,
            category=ResourceCategory.ACTIVITY,
        )
    )

    assert [resource.resource_id for resource in results] == ["activity-craft-party"]


def test_search_can_return_no_matches() -> None:
    store = InMemoryResourceStore()

    results = store.search(
        ResourceSearchCriteria(
            location="Manhattan, NY",
            category=ResourceCategory.VENUE,
        )
    )

    assert results == ()
