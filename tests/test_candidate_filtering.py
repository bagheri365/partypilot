"""Tests for deterministic candidate filtering."""

from datetime import datetime
from decimal import Decimal

from partypilot.adapters import DEFAULT_RESOURCES
from partypilot.application import CandidateRequirements, RejectionCode, filter_candidates
from partypilot.domain import AccessibilityAttribute, AgeRange, TimeWindow


def _reasons_for(
    resource_id: str, requirements: CandidateRequirements
) -> tuple[RejectionCode, ...]:
    result = filter_candidates(DEFAULT_RESOURCES, requirements)
    rejection = next(item for item in result.rejected if item.resource.resource_id == resource_id)
    return rejection.reasons


def test_location_filter_rejects_mismatched_resource() -> None:
    reasons = _reasons_for(
        "activity-teen-climbing",
        CandidateRequirements(location="Brooklyn, NY"),
    )

    assert RejectionCode.LOCATION_MISMATCH in reasons


def test_capacity_filter_rejects_resource_that_is_too_small() -> None:
    reasons = _reasons_for(
        "venue-small-studio",
        CandidateRequirements(guest_count=20),
    )

    assert RejectionCode.INSUFFICIENT_CAPACITY in reasons


def test_budget_filter_rejects_resource_over_budget() -> None:
    reasons = _reasons_for(
        "caterer-premium",
        CandidateRequirements(maximum_price=Decimal("500.00")),
    )

    assert RejectionCode.OVER_BUDGET in reasons


def test_age_filter_accepts_single_age_inside_resource_range() -> None:
    result = filter_candidates(
        DEFAULT_RESOURCES,
        CandidateRequirements(child_age=8),
    )

    assert "activity-craft-party" in {resource.resource_id for resource in result.eligible}


def test_age_filter_rejects_single_age_outside_resource_range() -> None:
    reasons = _reasons_for(
        "activity-craft-party",
        CandidateRequirements(child_age=13),
    )

    assert RejectionCode.AGE_RESTRICTION in reasons


def test_age_filter_requires_requested_range_to_fit_resource_range() -> None:
    reasons = _reasons_for(
        "activity-craft-party",
        CandidateRequirements(child_age_range=AgeRange(minimum=4, maximum=8)),
    )

    assert RejectionCode.AGE_RESTRICTION in reasons


def test_availability_filter_rejects_resource_without_full_window() -> None:
    requested = TimeWindow(
        start=datetime(2026, 9, 20, 17),
        end=datetime(2026, 9, 20, 19),
    )

    reasons = _reasons_for(
        "venue-small-studio",
        CandidateRequirements(availability=requested),
    )

    assert RejectionCode.UNAVAILABLE in reasons


def test_accessibility_filter_rejects_missing_required_attribute() -> None:
    reasons = _reasons_for(
        "venue-small-studio",
        CandidateRequirements(
            accessibility=frozenset({AccessibilityAttribute.WHEELCHAIR_ACCESSIBLE})
        ),
    )

    assert RejectionCode.ACCESSIBILITY_MISMATCH in reasons


def test_candidate_can_collect_multiple_rejection_reasons() -> None:
    requirements = CandidateRequirements(
        location="Queens, NY",
        guest_count=20,
        maximum_price=Decimal("200.00"),
        accessibility=frozenset({AccessibilityAttribute.WHEELCHAIR_ACCESSIBLE}),
    )

    reasons = _reasons_for("venue-small-studio", requirements)

    assert reasons == (
        RejectionCode.LOCATION_MISMATCH,
        RejectionCode.INSUFFICIENT_CAPACITY,
        RejectionCode.OVER_BUDGET,
        RejectionCode.ACCESSIBILITY_MISMATCH,
    )


def test_filter_returns_eligible_and_rejected_candidates() -> None:
    result = filter_candidates(
        DEFAULT_RESOURCES,
        CandidateRequirements(
            location="Brooklyn, NY",
            guest_count=20,
            maximum_price=Decimal("800.00"),
        ),
    )

    eligible_ids = {resource.resource_id for resource in result.eligible}
    rejected_ids = {item.resource.resource_id for item in result.rejected}

    assert eligible_ids == {
        "venue-brooklyn-loft",
        "caterer-family-table",
        "activity-craft-party",
    }
    assert eligible_ids.isdisjoint(rejected_ids)
    assert len(eligible_ids) + len(rejected_ids) == len(DEFAULT_RESOURCES)


def test_filter_is_deterministic_and_preserves_input_order() -> None:
    requirements = CandidateRequirements(location="Brooklyn, NY")

    first = filter_candidates(DEFAULT_RESOURCES, requirements)
    second = filter_candidates(DEFAULT_RESOURCES, requirements)

    assert first == second
    assert [resource.resource_id for resource in first.eligible] == [
        resource.resource_id
        for resource in DEFAULT_RESOURCES
        if resource.location == "Brooklyn, NY"
    ]
