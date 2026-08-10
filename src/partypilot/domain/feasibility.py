from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from partypilot.domain.constraints import Constraint, ConstraintType
from partypilot.domain.evidence import EvidenceReference

NonEmptyString = Annotated[str, Field(min_length=1)]


class FeasibilityOutcome(StrEnum):
    """Terminal outcome of PartyPilot plan feasibility evaluation."""

    FEASIBLE = "FEASIBLE"
    NO_FEASIBLE_PLAN = "NO_FEASIBLE_PLAN"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"


class ValidationResult(BaseModel):
    """Structured result of deterministic and evidence-aware validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    satisfied_hard_constraints: tuple[Constraint, ...] = ()
    violated_hard_constraints: tuple[Constraint, ...] = ()
    unresolved_constraints: tuple[Constraint, ...] = ()
    warnings: tuple[NonEmptyString, ...] = ()
    evidence_references: tuple[EvidenceReference, ...] = ()
    reasons: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_hard_constraint_collections(self) -> ValidationResult:
        for constraint in (
            *self.satisfied_hard_constraints,
            *self.violated_hard_constraints,
        ):
            if constraint.constraint_type is not ConstraintType.HARD:
                raise ValueError(
                    "satisfied_hard_constraints and violated_hard_constraints "
                    "may contain only HARD constraints"
                )

        satisfied_ids = {item.identifier for item in self.satisfied_hard_constraints}
        violated_ids = {item.identifier for item in self.violated_hard_constraints}
        unresolved_ids = {item.identifier for item in self.unresolved_constraints}

        if satisfied_ids & violated_ids:
            raise ValueError("a constraint cannot be both satisfied and violated")
        if (satisfied_ids | violated_ids) & unresolved_ids:
            raise ValueError("a resolved constraint cannot also be unresolved")
        return self


class FeasibilityResult(BaseModel):
    """Terminal feasibility decision paired with the validation evidence behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: FeasibilityOutcome
    plan_feasible: bool | None
    validation: ValidationResult

    @model_validator(mode="after")
    def validate_terminal_state(self) -> FeasibilityResult:
        if self.outcome is FeasibilityOutcome.FEASIBLE:
            if self.plan_feasible is not True:
                raise ValueError("FEASIBLE outcome requires plan_feasible=True")
            if self.validation.violated_hard_constraints:
                raise ValueError("FEASIBLE outcome cannot contain violated hard constraints")
            if self.validation.unresolved_constraints:
                raise ValueError("FEASIBLE outcome cannot contain unresolved constraints")

        if self.outcome is FeasibilityOutcome.NO_FEASIBLE_PLAN and self.plan_feasible is not False:
            raise ValueError("NO_FEASIBLE_PLAN outcome requires plan_feasible=False")

        if (
            self.outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
            and self.plan_feasible is not None
        ):
            raise ValueError("HUMAN_REVIEW_REQUIRED requires plan_feasible=None")

        return self
