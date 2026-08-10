from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from partypilot.domain import (
    Constraint,
    ConstraintOperator,
    ConstraintProvenance,
    ConstraintType,
)


def test_hard_constraint_supports_required_fields() -> None:
    constraint = Constraint(
        identifier="guest-capacity",
        key="capacity",
        operator=ConstraintOperator.GTE,
        value=25,
        constraint_type=ConstraintType.HARD,
        description="Venue capacity must cover all guests.",
    )

    assert constraint.identifier == "guest-capacity"
    assert constraint.key == "capacity"
    assert constraint.value == 25
    assert constraint.provenance is None


def test_soft_constraint_can_represent_preference() -> None:
    constraint = Constraint(
        identifier="theme-preference",
        key="theme",
        operator=ConstraintOperator.IN,
        value=("space", "science"),
        constraint_type=ConstraintType.SOFT,
        description="Prefer a space or science theme.",
    )

    assert constraint.constraint_type is ConstraintType.SOFT
    assert constraint.value == ("space", "science")


def test_derived_constraint_requires_and_retains_provenance() -> None:
    provenance = ConstraintProvenance(
        source_constraint_ids=("child-age", "activity-safety"),
        derivation_explanation="Age limit follows deterministically from the safety rule.",
    )
    constraint = Constraint(
        identifier="minimum-activity-age",
        key="minimum_age",
        operator=ConstraintOperator.LTE,
        value=8,
        constraint_type=ConstraintType.DERIVED,
        description="Activity minimum age must not exceed the child's age.",
        provenance=provenance,
    )

    assert constraint.provenance == provenance
    assert constraint.provenance.source_constraint_ids == ("child-age", "activity-safety")


def test_derived_constraint_without_provenance_is_rejected() -> None:
    with pytest.raises(ValidationError, match="derived constraints must retain provenance"):
        Constraint(
            identifier="derived-budget-cap",
            key="budget",
            operator=ConstraintOperator.LTE,
            value=Decimal("500.00"),
            constraint_type=ConstraintType.DERIVED,
            description="Derived budget cap.",
        )


@pytest.mark.parametrize("field", ["identifier", "key", "description"])
def test_required_text_fields_cannot_be_empty(field: str) -> None:
    data = {
        "identifier": "capacity",
        "key": "capacity",
        "operator": ConstraintOperator.GTE,
        "value": 10,
        "constraint_type": ConstraintType.HARD,
        "description": "Capacity requirement.",
    }
    data[field] = ""

    with pytest.raises(ValidationError):
        Constraint.model_validate(data)


def test_provenance_requires_at_least_one_source_constraint() -> None:
    with pytest.raises(ValidationError):
        ConstraintProvenance(
            source_constraint_ids=(),
            derivation_explanation="Deterministic derivation.",
        )


def test_constraint_supports_temporal_and_decimal_values() -> None:
    date_constraint = Constraint(
        identifier="event-date",
        key="available_date",
        operator=ConstraintOperator.EQ,
        value=date(2026, 9, 12),
        constraint_type=ConstraintType.HARD,
        description="Resource must be available on the event date.",
    )
    budget_constraint = Constraint(
        identifier="budget-limit",
        key="price",
        operator=ConstraintOperator.LTE,
        value=Decimal("1250.00"),
        constraint_type=ConstraintType.HARD,
        description="Total price must remain within budget.",
    )

    assert date_constraint.value == date(2026, 9, 12)
    assert budget_constraint.value == Decimal("1250.00")


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Constraint.model_validate(
            {
                "identifier": "capacity",
                "key": "capacity",
                "operator": ConstraintOperator.GTE,
                "value": 10,
                "constraint_type": ConstraintType.HARD,
                "description": "Capacity requirement.",
                "unexpected": True,
            }
        )


def test_models_are_immutable() -> None:
    constraint = Constraint(
        identifier="capacity",
        key="capacity",
        operator=ConstraintOperator.GTE,
        value=10,
        constraint_type=ConstraintType.HARD,
        description="Capacity requirement.",
    )

    with pytest.raises(ValidationError):
        constraint.key = "other"
