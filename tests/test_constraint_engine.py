from datetime import datetime
from decimal import Decimal

import pytest

from partypilot.application.budget_validation import CostComponent
from partypilot.application.candidate_filtering import CandidateRequirements
from partypilot.application.constraint_engine import (
    ConstraintEngineInput,
    ConstraintEngineViolationCode,
    validate_constraints,
)
from partypilot.domain.constraints import Constraint, ConstraintOperator, ConstraintType
from partypilot.domain.resources import Venue
from partypilot.domain.temporal import TimeWindow


def hard(identifier: str, key: str, value: object = True) -> Constraint:
    return Constraint(
        identifier=identifier,
        key=key,
        operator=ConstraintOperator.EQ,
        value=value,  # type: ignore[arg-type]
        constraint_type=ConstraintType.HARD,
        description=f"Hard {key}",
    )


def venue(*, capacity: int = 20, price: str = "100") -> Venue:
    window = TimeWindow(
        start=datetime(2026, 8, 10, 10),
        end=datetime(2026, 8, 10, 18),
    )
    return Venue(
        resource_id="venue-1",
        name="Hall",
        location="Boston",
        price=Decimal(price),
        capacity=capacity,
        availability=(window,),
    )


def test_feasible_when_all_applicable_hard_constraints_pass() -> None:
    event = TimeWindow(start=datetime(2026, 8, 10, 12), end=datetime(2026, 8, 10, 15))
    result = validate_constraints(
        ConstraintEngineInput(
            hard_constraints=(
                hard("c-location", "location", "Boston"),
                hard("c-budget", "budget", 200),
            ),
            selected_resources=(venue(),),
            candidate_requirements=CandidateRequirements(location="Boston", guest_count=10),
            budget=Decimal("200"),
            cost_components=(
                CostComponent(component_id="venue", description="Venue", amount=Decimal("100")),
            ),
            event_window=event,
        )
    )

    assert result.feasible is True
    assert result.satisfied_constraint_ids == ("c-location", "c-budget")
    assert result.violations == ()
    assert result.unresolved_constraint_ids == ()


def test_resource_failure_remains_explicit() -> None:
    result = validate_constraints(
        ConstraintEngineInput(
            hard_constraints=(hard("c-capacity", "capacity", 30),),
            selected_resources=(venue(capacity=20),),
            candidate_requirements=CandidateRequirements(guest_count=30),
        )
    )

    assert result.feasible is False
    assert result.violations[0].code is ConstraintEngineViolationCode.RESOURCE_CONSTRAINT
    assert result.violations[0].constraint_id == "c-capacity"


def test_over_budget_plan_is_not_feasible() -> None:
    result = validate_constraints(
        ConstraintEngineInput(
            hard_constraints=(hard("c-budget", "budget", 50),),
            budget=Decimal("50"),
            cost_components=(
                CostComponent(component_id="venue", description="Venue", amount=Decimal("75")),
            ),
        )
    )

    assert result.feasible is False
    assert result.violations[0].code is ConstraintEngineViolationCode.BUDGET


def test_missing_validator_input_is_unresolved_not_ignored() -> None:
    result = validate_constraints(
        ConstraintEngineInput(hard_constraints=(hard("c-budget", "budget", 50),))
    )

    assert result.feasible is False
    assert result.unresolved_constraint_ids == ("c-budget",)


def test_unsupported_hard_constraint_is_explicit_and_infeasible() -> None:
    result = validate_constraints(
        ConstraintEngineInput(hard_constraints=(hard("c-noise", "noise_limit", 80),))
    )

    assert result.feasible is False
    assert result.unresolved_constraint_ids == ("c-noise",)
    assert result.violations[0].code is ConstraintEngineViolationCode.UNSUPPORTED_HARD_CONSTRAINT


def test_non_hard_constraints_are_rejected() -> None:
    soft = Constraint(
        identifier="soft-1",
        key="theme",
        operator=ConstraintOperator.EQ,
        value="space",
        constraint_type=ConstraintType.SOFT,
        description="Prefer space theme",
    )
    with pytest.raises(ValueError, match="HARD constraints only"):
        validate_constraints(ConstraintEngineInput(hard_constraints=(soft,)))
