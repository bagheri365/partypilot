from __future__ import annotations

from datetime import date, time
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmptyString = Annotated[str, Field(min_length=1)]
PositiveGuestCount = Annotated[int, Field(gt=0)]
NonNegativeAge = Annotated[int, Field(ge=0)]
NonNegativeBudget = Annotated[Decimal, Field(ge=Decimal("0"))]


class AgeRange(BaseModel):
    """Inclusive child age range."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum: NonNegativeAge
    maximum: NonNegativeAge

    @model_validator(mode="after")
    def validate_order(self) -> AgeRange:
        if self.maximum < self.minimum:
            raise ValueError("maximum age cannot be less than minimum age")
        return self


class PartyRequest(BaseModel):
    """Validated user requirements for an event-planning request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    location: NonEmptyString
    event_date: date
    event_time: time | None = None
    guest_count: PositiveGuestCount
    child_age: NonNegativeAge | None = None
    child_age_range: AgeRange | None = None
    total_budget: NonNegativeBudget
    theme_preferences: tuple[NonEmptyString, ...] = ()
    allergies: tuple[NonEmptyString, ...] = ()
    dietary_restrictions: tuple[NonEmptyString, ...] = ()
    accessibility_needs: tuple[NonEmptyString, ...] = ()
    other_constraints: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_child_age_representation(self) -> PartyRequest:
        if self.child_age is not None and self.child_age_range is not None:
            raise ValueError("provide either child_age or child_age_range, not both")
        return self
