from decimal import Decimal

import pytest
from pydantic import ValidationError

from partypilot.application.budget_validation import (
    BudgetViolationCode,
    CostComponent,
    calculate_total_cost,
    validate_budget,
)


def _components() -> tuple[CostComponent, ...]:
    return (
        CostComponent(component_id="venue", description="Venue", amount=Decimal("120.10")),
        CostComponent(component_id="food", description="Food", amount=Decimal("79.90")),
    )


def test_calculate_total_cost_uses_exact_decimal_arithmetic() -> None:
    components = (
        CostComponent(component_id="a", description="A", amount=Decimal("0.10")),
        CostComponent(component_id="b", description="B", amount=Decimal("0.20")),
    )

    assert calculate_total_cost(components) == Decimal("0.30")


def test_under_budget_is_valid() -> None:
    result = validate_budget(Decimal("250.00"), _components())

    assert result.total_cost == Decimal("200.00")
    assert result.within_budget is True
    assert result.violation is None
    assert result.components == _components()


def test_exact_budget_boundary_is_valid() -> None:
    result = validate_budget(Decimal("200.00"), _components())

    assert result.total_cost == Decimal("200.00")
    assert result.within_budget is True
    assert result.violation is None


def test_over_budget_returns_structured_violation() -> None:
    result = validate_budget(Decimal("175.50"), _components())

    assert result.within_budget is False
    assert result.violation is not None
    assert result.violation.code is BudgetViolationCode.OVER_BUDGET
    assert result.violation.budget == Decimal("175.50")
    assert result.violation.total_cost == Decimal("200.00")
    assert result.violation.amount_over == Decimal("24.50")


def test_empty_components_total_zero() -> None:
    result = validate_budget(Decimal("0"), ())

    assert result.total_cost == Decimal("0")
    assert result.within_budget is True


def test_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="budget cannot be negative"):
        validate_budget(Decimal("-0.01"), ())


def test_negative_component_amount_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CostComponent(component_id="bad", description="Bad", amount=Decimal("-1"))


@pytest.mark.parametrize("field", ["component_id", "description"])
def test_blank_component_text_is_rejected(field: str) -> None:
    values: dict[str, object] = {
        "component_id": "id",
        "description": "description",
        "amount": Decimal("1"),
    }
    values[field] = "   "

    with pytest.raises(ValidationError):
        CostComponent.model_validate(values)
