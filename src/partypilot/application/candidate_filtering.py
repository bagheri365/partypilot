"""Deterministic candidate filtering for PartyPilot resources."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from partypilot.domain.party_request import AgeRange
from partypilot.domain.resources import AccessibilityAttribute, Resource
from partypilot.domain.temporal import TimeWindow


class RejectionCode(StrEnum):
    """Structured reasons a resource can fail a hard candidate constraint."""

    LOCATION_MISMATCH = "location_mismatch"
    INSUFFICIENT_CAPACITY = "insufficient_capacity"
    OVER_BUDGET = "over_budget"
    AGE_RESTRICTION = "age_restriction"
    UNAVAILABLE = "unavailable"
    ACCESSIBILITY_MISMATCH = "accessibility_mismatch"


class CandidateRequirements(BaseModel):
    """Hard structured requirements used to filter candidate resources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    location: str | None = None
    guest_count: int | None = Field(default=None, gt=0)
    maximum_price: Decimal | None = Field(default=None, ge=0)
    child_age: int | None = Field(default=None, ge=0)
    child_age_range: AgeRange | None = None
    availability: TimeWindow | None = None
    accessibility: frozenset[AccessibilityAttribute] = frozenset()

    @field_validator("location")
    @classmethod
    def validate_location(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("location cannot be blank")
        return normalized


class CandidateRejection(BaseModel):
    """A rejected candidate with all deterministic hard-constraint failures."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resource: Resource
    reasons: tuple[RejectionCode, ...]


class CandidateFilterResult(BaseModel):
    """Eligible and rejected resources from deterministic hard filtering."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    eligible: tuple[Resource, ...]
    rejected: tuple[CandidateRejection, ...]


def filter_candidates(
    resources: tuple[Resource, ...],
    requirements: CandidateRequirements,
) -> CandidateFilterResult:
    """Filter resources against every supplied hard structured requirement."""
    eligible: list[Resource] = []
    rejected: list[CandidateRejection] = []

    for resource in resources:
        reasons = _rejection_reasons(resource, requirements)
        if reasons:
            rejected.append(CandidateRejection(resource=resource, reasons=reasons))
        else:
            eligible.append(resource)

    return CandidateFilterResult(eligible=tuple(eligible), rejected=tuple(rejected))


def _rejection_reasons(
    resource: Resource,
    requirements: CandidateRequirements,
) -> tuple[RejectionCode, ...]:
    reasons: list[RejectionCode] = []

    if (
        requirements.location is not None
        and resource.location.casefold() != requirements.location.casefold()
    ):
        reasons.append(RejectionCode.LOCATION_MISMATCH)

    if requirements.guest_count is not None and (
        resource.capacity is None or resource.capacity < requirements.guest_count
    ):
        reasons.append(RejectionCode.INSUFFICIENT_CAPACITY)

    if requirements.maximum_price is not None and resource.price > requirements.maximum_price:
        reasons.append(RejectionCode.OVER_BUDGET)

    if not _age_is_compatible(resource, requirements):
        reasons.append(RejectionCode.AGE_RESTRICTION)

    if requirements.availability is not None and not any(
        window.contains_window(requirements.availability) for window in resource.availability
    ):
        reasons.append(RejectionCode.UNAVAILABLE)

    if not requirements.accessibility.issubset(resource.accessibility_attributes):
        reasons.append(RejectionCode.ACCESSIBILITY_MISMATCH)

    return tuple(reasons)


def _age_is_compatible(resource: Resource, requirements: CandidateRequirements) -> bool:
    restriction = resource.age_restrictions
    if restriction is None:
        return True

    if requirements.child_age is not None:
        return restriction.minimum <= requirements.child_age <= restriction.maximum

    if requirements.child_age_range is not None:
        requested = requirements.child_age_range
        return restriction.minimum <= requested.minimum and requested.maximum <= restriction.maximum

    return True
