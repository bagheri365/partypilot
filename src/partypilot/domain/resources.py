"""Structured PartyPilot resource domain models."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from partypilot.domain.party_request import AgeRange
from partypilot.domain.temporal import TimeWindow


class ResourceCategory(StrEnum):
    """High-level structured resource categories."""

    VENUE = "venue"
    CATERER = "caterer"
    ACTIVITY = "activity"


class AccessibilityAttribute(StrEnum):
    """Common structured accessibility attributes."""

    WHEELCHAIR_ACCESSIBLE = "wheelchair_accessible"
    ACCESSIBLE_RESTROOM = "accessible_restroom"
    STEP_FREE_ACCESS = "step_free_access"
    HEARING_ASSISTANCE = "hearing_assistance"
    QUIET_SPACE = "quiet_space"


class Resource(BaseModel):
    """Common structured fields shared by PartyPilot resources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resource_id: str
    name: str
    location: str
    price: Decimal = Field(ge=0)
    capacity: int | None = Field(default=None, gt=0)
    availability: tuple[TimeWindow, ...] = ()
    age_restrictions: AgeRange | None = None
    accessibility_attributes: frozenset[AccessibilityAttribute] = frozenset()
    category: ResourceCategory

    @field_validator("resource_id", "name", "location")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be blank")
        return normalized


class Venue(Resource):
    """A structured venue resource."""

    category: ResourceCategory = ResourceCategory.VENUE
    capacity: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_category(self) -> Venue:
        if self.category is not ResourceCategory.VENUE:
            raise ValueError("venue category must be 'venue'")
        return self


class Caterer(Resource):
    """A structured caterer resource."""

    category: ResourceCategory = ResourceCategory.CATERER

    @model_validator(mode="after")
    def validate_category(self) -> Caterer:
        if self.category is not ResourceCategory.CATERER:
            raise ValueError("caterer category must be 'caterer'")
        return self


class Activity(Resource):
    """A structured activity resource."""

    category: ResourceCategory = ResourceCategory.ACTIVITY
    capacity: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_category(self) -> Activity:
        if self.category is not ResourceCategory.ACTIVITY:
            raise ValueError("activity category must be 'activity'")
        return self
