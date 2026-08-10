"""Deterministic budget calculation and validation."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BudgetViolationCode(StrEnum):
    """Structured budget validation failure codes."""

    OVER_BUDGET = "over_budget"


class CostComponent(BaseModel):
    """A named monetary component contributing to a plan's total cost."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: str
    description: str
    amount: Decimal = Field(ge=0)

    @field_validator("component_id", "description")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be blank")
        return normalized


class BudgetViolation(BaseModel):
    """Structured details explaining an over-budget plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: BudgetViolationCode
    budget: Decimal = Field(ge=0)
    total_cost: Decimal = Field(ge=0)
    amount_over: Decimal = Field(gt=0)


class BudgetValidationResult(BaseModel):
    """Deterministic budget validation result with component-level costs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    budget: Decimal = Field(ge=0)
    components: tuple[CostComponent, ...]
    total_cost: Decimal = Field(ge=0)
    within_budget: bool
    violation: BudgetViolation | None = None


def calculate_total_cost(components: tuple[CostComponent, ...]) -> Decimal:
    """Calculate a precise total using Decimal arithmetic only."""
    return sum((component.amount for component in components), start=Decimal("0"))


def validate_budget(
    budget: Decimal,
    components: tuple[CostComponent, ...],
) -> BudgetValidationResult:
    """Validate total component cost against a non-negative budget."""
    if budget < 0:
        raise ValueError("budget cannot be negative")

    total_cost = calculate_total_cost(components)
    if total_cost <= budget:
        return BudgetValidationResult(
            budget=budget,
            components=components,
            total_cost=total_cost,
            within_budget=True,
        )

    amount_over = total_cost - budget
    violation = BudgetViolation(
        code=BudgetViolationCode.OVER_BUDGET,
        budget=budget,
        total_cost=total_cost,
        amount_over=amount_over,
    )
    return BudgetValidationResult(
        budget=budget,
        components=components,
        total_cost=total_cost,
        within_budget=False,
        violation=violation,
    )
