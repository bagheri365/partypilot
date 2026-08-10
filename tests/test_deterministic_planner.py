"""Tests for the deterministic baseline planner."""

from datetime import date, time
from decimal import Decimal

from partypilot.adapters import InMemoryResourceStore
from partypilot.application import DeterministicPlanner, PlannerConfig, PreferenceWeights
from partypilot.domain import PartyRequest


def request(**overrides: object) -> PartyRequest:
    data: dict[str, object] = {
        "location": "Brooklyn, NY",
        "event_date": date(2026, 9, 20),
        "event_time": time(14, 0),
        "guest_count": 10,
        "child_age": 8,
        "total_budget": Decimal("1500.00"),
    }
    data.update(overrides)
    return PartyRequest.model_validate(data)


def test_planner_returns_feasible_combination() -> None:
    result = DeterministicPlanner(InMemoryResourceStore()).plan(request())

    assert result.feasible is True
    assert result.candidates[0].resource_ids == (
        "venue-brooklyn-loft",
        "caterer-family-table",
        "activity-craft-party",
    )
    assert result.candidates[0].total_cost == Decimal("1375.00")


def test_planner_returns_no_candidates_when_budget_is_too_low() -> None:
    result = DeterministicPlanner(InMemoryResourceStore()).plan(
        request(total_budget=Decimal("1000.00"))
    )

    assert result.feasible is False
    assert result.candidates == ()


def test_planner_rejects_wrong_event_date() -> None:
    result = DeterministicPlanner(InMemoryResourceStore()).plan(
        request(event_date=date(2026, 9, 21))
    )

    assert result.feasible is False


def test_planner_filters_age_restricted_activity() -> None:
    result = DeterministicPlanner(InMemoryResourceStore()).plan(request(child_age=13))

    assert result.feasible is False


def test_planner_surfaces_unresolved_safety_constraints() -> None:
    result = DeterministicPlanner(InMemoryResourceStore()).plan(request(allergies=("peanuts",)))

    assert result.feasible is False
    assert result.unresolved_request_constraints == ("allergies",)


def test_planner_surfaces_unknown_accessibility_need() -> None:
    result = DeterministicPlanner(InMemoryResourceStore()).plan(
        request(accessibility_needs=("sensory guide",))
    )

    assert result.unresolved_request_constraints == ("accessibility:sensory guide",)


def test_planner_ranking_is_lowest_cost_first_and_deterministic() -> None:
    planner = DeterministicPlanner(
        InMemoryResourceStore(),
        PlannerConfig(preference_weights=PreferenceWeights(cost=Decimal("2"))),
    )

    first = planner.plan(request(total_budget=Decimal("3000.00")))
    second = planner.plan(request(total_budget=Decimal("3000.00")))

    assert first == second
    assert [candidate.total_cost for candidate in first.candidates] == sorted(
        candidate.total_cost for candidate in first.candidates
    )


def test_planner_can_plan_without_explicit_event_time_using_event_date() -> None:
    result = DeterministicPlanner(InMemoryResourceStore()).plan(request(event_time=None))

    assert result.feasible is True
