"""Typed resource-store port for structured PartyPilot resource lookup."""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from partypilot.domain.resources import Resource, ResourceCategory
from partypilot.domain.temporal import TimeWindow


class ResourceSearchCriteria(BaseModel):
    """Deterministic structured filters supported by a resource store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    location: str | None = None
    minimum_capacity: int | None = Field(default=None, gt=0)
    maximum_price: Decimal | None = Field(default=None, ge=0)
    availability: TimeWindow | None = None
    category: ResourceCategory | None = None

    @field_validator("location")
    @classmethod
    def validate_location(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("location cannot be blank")
        return normalized


class ResourceStore(Protocol):
    """Port for deterministic structured resource search.

    Implementations may use in-memory data, a database, or another persistence
    mechanism, but callers depend only on this contract.
    """

    def search(self, criteria: ResourceSearchCriteria) -> tuple[Resource, ...]:
        """Return resources matching all supplied structured criteria."""
        ...
