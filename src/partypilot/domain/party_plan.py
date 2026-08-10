"""Typed plans produced by PartyPilot planners."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from partypilot.domain.resources import Resource


class PartyPlan(BaseModel):
    """A single proposed party plan.

    For the single-pass LLM baseline, resource objects are provider claims rather
    than grounded catalog records. They are therefore validated structurally but
    remain unverified evidence-wise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    resources: tuple[Resource, ...]
    claimed_total_cost: Decimal = Field(ge=0)
    assumptions: tuple[str, ...] = ()
