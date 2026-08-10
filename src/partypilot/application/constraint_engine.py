"""Deterministic hard-constraint engine for PartyPilot plans."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application.budget_validation import CostComponent, validate_budget
from partypilot.application.candidate_filtering import (
    CandidateRequirements,
    RejectionCode,
    filter_candidates,
)
from partypilot.domain.constraints import Constraint, ConstraintType
from partypilot.domain.dependencies import TaskDependency
from partypilot.domain.resources import Resource
from partypilot.domain.temporal import ScheduledInterval, TimeWindow
from partypilot.domain.temporal_validation import (
    TemporalViolation,
    TemporalViolationCode,
    validate_temporal_schedule,
)


class ConstraintEngineViolationCode(StrEnum):
    """Machine-readable hard-constraint engine failure categories."""

    RESOURCE_CONSTRAINT = "resource_constraint"
    BUDGET = "budget"
    TEMPORAL = "temporal"
    UNSUPPORTED_HARD_CONSTRAINT = "unsupported_hard_constraint"


class ConstraintEngineViolation(BaseModel):
    """A typed hard-constraint violation produced by the engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ConstraintEngineViolationCode
    constraint_id: str
    message: str
    resource_id: str | None = None
    rejection_code: RejectionCode | None = None
    temporal_violation: TemporalViolation | None = None


class ConstraintEngineInput(BaseModel):
    """All deterministic inputs needed to validate a candidate plan."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    hard_constraints: tuple[Constraint, ...]
    selected_resources: tuple[Resource, ...] = ()
    candidate_requirements: CandidateRequirements = Field(default_factory=CandidateRequirements)
    budget: Decimal | None = Field(default=None, ge=0)
    cost_components: tuple[CostComponent, ...] = ()
    tasks: tuple[TaskDependency, ...] = ()
    schedule: Mapping[str, ScheduledInterval] = Field(default_factory=dict)
    event_window: TimeWindow | None = None


class ConstraintEngineResult(BaseModel):
    """Explicit deterministic outcome for all supplied hard constraints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feasible: bool
    satisfied_constraint_ids: tuple[str, ...]
    violations: tuple[ConstraintEngineViolation, ...]
    unresolved_constraint_ids: tuple[str, ...]


_RESOURCE_KEYS: dict[str, RejectionCode] = {
    "location": RejectionCode.LOCATION_MISMATCH,
    "capacity": RejectionCode.INSUFFICIENT_CAPACITY,
    "guest_count": RejectionCode.INSUFFICIENT_CAPACITY,
    "age_restrictions": RejectionCode.AGE_RESTRICTION,
    "child_age": RejectionCode.AGE_RESTRICTION,
    "availability": RejectionCode.UNAVAILABLE,
    "accessibility": RejectionCode.ACCESSIBILITY_MISMATCH,
}
_BUDGET_KEYS = frozenset({"budget", "total_budget", "maximum_price"})
_TEMPORAL_KEYS: dict[str, frozenset[TemporalViolationCode]] = {
    "temporal": frozenset(TemporalViolationCode),
    "time_window": frozenset({TemporalViolationCode.OUTSIDE_PERMITTED_WINDOW}),
    "dependency_order": frozenset({TemporalViolationCode.DEPENDENCY_ORDER}),
    "resource_conflicts": frozenset({TemporalViolationCode.EXCLUSIVE_RESOURCE_OVERLAP}),
    "setup_time": frozenset({TemporalViolationCode.SETUP_TOO_LATE}),
    "event_end": frozenset({TemporalViolationCode.BEYOND_EVENT_END}),
}


def validate_constraints(engine_input: ConstraintEngineInput) -> ConstraintEngineResult:
    """Validate every supplied hard constraint without silently ignoring any."""
    non_hard = [
        constraint.identifier
        for constraint in engine_input.hard_constraints
        if constraint.constraint_type is not ConstraintType.HARD
    ]
    if non_hard:
        raise ValueError("constraint engine accepts HARD constraints only")

    resource_result = filter_candidates(
        engine_input.selected_resources,
        engine_input.candidate_requirements,
    )
    rejection_by_resource = {
        rejection.resource.resource_id: rejection.reasons for rejection in resource_result.rejected
    }

    budget_result = (
        validate_budget(engine_input.budget, engine_input.cost_components)
        if engine_input.budget is not None
        else None
    )
    temporal_result = (
        validate_temporal_schedule(
            engine_input.tasks,
            engine_input.schedule,
            engine_input.event_window,
        )
        if engine_input.event_window is not None
        else None
    )

    satisfied: list[str] = []
    violations: list[ConstraintEngineViolation] = []
    unresolved: list[str] = []

    for constraint in engine_input.hard_constraints:
        key = constraint.key.casefold()
        if key in _RESOURCE_KEYS:
            expected_rejection = _RESOURCE_KEYS[key]
            matches = [
                resource_id
                for resource_id, reasons in rejection_by_resource.items()
                if expected_rejection in reasons
            ]
            if matches:
                violations.extend(
                    ConstraintEngineViolation(
                        code=ConstraintEngineViolationCode.RESOURCE_CONSTRAINT,
                        constraint_id=constraint.identifier,
                        resource_id=resource_id,
                        rejection_code=expected_rejection,
                        message=(
                            f"Resource {resource_id!r} violates hard constraint "
                            f"{constraint.identifier!r}."
                        ),
                    )
                    for resource_id in matches
                )
            else:
                satisfied.append(constraint.identifier)
            continue

        if key in _BUDGET_KEYS:
            if budget_result is None:
                unresolved.append(constraint.identifier)
            elif budget_result.within_budget:
                satisfied.append(constraint.identifier)
            else:
                violations.append(
                    ConstraintEngineViolation(
                        code=ConstraintEngineViolationCode.BUDGET,
                        constraint_id=constraint.identifier,
                        message=(
                            f"Plan total {budget_result.total_cost} exceeds budget "
                            f"{budget_result.budget}."
                        ),
                    )
                )
            continue

        if key in _TEMPORAL_KEYS:
            if temporal_result is None:
                unresolved.append(constraint.identifier)
                continue
            relevant_codes = _TEMPORAL_KEYS[key]
            relevant = [v for v in temporal_result.violations if v.code in relevant_codes]
            if relevant:
                violations.extend(
                    ConstraintEngineViolation(
                        code=ConstraintEngineViolationCode.TEMPORAL,
                        constraint_id=constraint.identifier,
                        message=violation.message,
                        resource_id=violation.resource_id,
                        temporal_violation=violation,
                    )
                    for violation in relevant
                )
            else:
                satisfied.append(constraint.identifier)
            continue

        unresolved.append(constraint.identifier)
        violations.append(
            ConstraintEngineViolation(
                code=ConstraintEngineViolationCode.UNSUPPORTED_HARD_CONSTRAINT,
                constraint_id=constraint.identifier,
                message=f"Hard constraint key {constraint.key!r} is not supported by the engine.",
            )
        )

    feasible = not violations and not unresolved
    return ConstraintEngineResult(
        feasible=feasible,
        satisfied_constraint_ids=tuple(satisfied),
        violations=tuple(violations),
        unresolved_constraint_ids=tuple(unresolved),
    )
