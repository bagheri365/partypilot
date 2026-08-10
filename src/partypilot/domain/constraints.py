from __future__ import annotations

from datetime import date, time
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmptyString = Annotated[str, Field(min_length=1)]
type ConstraintScalar = str | int | Decimal | bool | date | time
type ConstraintValue = ConstraintScalar | tuple[ConstraintScalar, ...]


class ConstraintType(StrEnum):
    """How a constraint entered the planning problem."""

    HARD = "HARD"
    SOFT = "SOFT"
    DERIVED = "DERIVED"


class ConstraintOperator(StrEnum):
    """Deterministic operators supported by PartyPilot constraints."""

    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


class ConstraintProvenance(BaseModel):
    """Minimal provenance retained for derived constraints.

    Rich evidence provenance is introduced by a later milestone. This type records only
    the derivation lineage required by the constraint model itself.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_constraint_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    derivation_explanation: NonEmptyString


class Constraint(BaseModel):
    """A typed planning constraint with stable identity and optional provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identifier: NonEmptyString
    key: NonEmptyString
    operator: ConstraintOperator
    value: ConstraintValue
    constraint_type: ConstraintType
    description: NonEmptyString
    provenance: ConstraintProvenance | None = None

    @model_validator(mode="after")
    def validate_provenance(self) -> Constraint:
        if self.constraint_type is ConstraintType.DERIVED and self.provenance is None:
            raise ValueError("derived constraints must retain provenance")
        return self
