from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from partypilot.domain.evidence import EvidenceReference
from partypilot.domain.feasibility import FeasibilityOutcome

NonEmptyString = Annotated[str, Field(min_length=1)]


class SpecialistDomain(StrEnum):
    """Supported specialist domains for minimal multi-agent coordination."""

    VENUE = "venue"
    CATERING_SAFETY = "catering_safety"
    ACCESSIBILITY = "accessibility"
    SCHEDULING_OPERATIONS = "scheduling_operations"
    BUDGET = "budget"


class ArbitrationOutcome(StrEnum):
    """Typed outcomes produced by a specialist or the global coordinator."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    REPLAN_REQUIRED = "REPLAN_REQUIRED"


class SpecialistDecision(BaseModel):
    """Structured, independently auditable specialist recommendation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: NonEmptyString
    domain: SpecialistDomain
    recommendation: NonEmptyString
    status: ArbitrationOutcome
    hard_constraints_considered: tuple[NonEmptyString, ...] = ()
    evidence_references: tuple[EvidenceReference, ...] = ()
    assumptions: tuple[NonEmptyString, ...] = ()
    unresolved_uncertainties: tuple[NonEmptyString, ...] = ()
    local_score: float | None = Field(default=None, ge=0)
    local_rank: int | None = Field(default=None, ge=1)
    recommended_resource_ids: tuple[NonEmptyString, ...] = ()
    reasons_for_rejection: tuple[NonEmptyString, ...] = ()
    dependency_decision_ids: tuple[NonEmptyString, ...] = ()
    notes: tuple[NonEmptyString, ...] = ()

    @field_validator(
        "hard_constraints_considered",
        "assumptions",
        "unresolved_uncertainties",
        "recommended_resource_ids",
        "reasons_for_rejection",
        "dependency_decision_ids",
        "notes",
    )
    @classmethod
    def validate_unique_text_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("values cannot be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("values must be unique")
        return normalized


class ArbitrationTrace(BaseModel):
    """Typed explanation for the coordinator's global decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: ArbitrationOutcome
    feasibility_outcome: FeasibilityOutcome
    selected_resource_ids: tuple[NonEmptyString, ...] = ()
    accepted_specialist_ids: tuple[NonEmptyString, ...] = ()
    rejected_specialist_ids: tuple[NonEmptyString, ...] = ()
    overridden_specialist_ids: tuple[NonEmptyString, ...] = ()
    controlling_evidence_ids: tuple[NonEmptyString, ...] = ()
    dependency_conflicts: tuple[NonEmptyString, ...] = ()
    unresolved_uncertainties: tuple[NonEmptyString, ...] = ()
    reasons: tuple[NonEmptyString, ...] = ()
    global_score: float | None = Field(default=None, ge=0)
    coordination_steps: tuple[NonEmptyString, ...] = ()

    @field_validator(
        "selected_resource_ids",
        "accepted_specialist_ids",
        "rejected_specialist_ids",
        "overridden_specialist_ids",
        "controlling_evidence_ids",
        "dependency_conflicts",
        "unresolved_uncertainties",
        "reasons",
        "coordination_steps",
    )
    @classmethod
    def validate_unique_text_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("values cannot be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("values must be unique")
        return normalized


class CoordinatedPlanResult(BaseModel):
    """Terminal result for either a baseline or coordinated planning path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: NonEmptyString
    feasibility_outcome: FeasibilityOutcome
    selected_resource_ids: tuple[NonEmptyString, ...] = ()
    total_cost: float | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    hard_constraint_validity: bool
    cross_domain_compatibility: bool
    evidence_grounded_arbitration: bool
    global_optimum: bool | None = None
    human_review_calibrated: bool | None = None
    disagreement_resolved_correctly: bool = False
    disagreement_resolved_incorrectly: bool = False
    specialist_call_count: int = Field(ge=0)
    coordination_overhead_count: int = Field(ge=0)
    arbitration: ArbitrationTrace | None = None
    specialist_decisions: tuple[SpecialistDecision, ...] = ()
    notes: tuple[NonEmptyString, ...] = ()
    failure_stage: NonEmptyString | None = None

    @field_validator("selected_resource_ids", "notes")
    @classmethod
    def validate_unique_text_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("values cannot be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("values must be unique")
        return normalized


class ArchitectureComparisonMetrics(BaseModel):
    """Aggregate metrics for comparing baseline and coordinated decision paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_count: int = Field(ge=0)
    evidence_relevant_scenario_count: int = Field(ge=0)
    global_optimum_scenario_count: int = Field(ge=0)
    human_review_scenario_count: int = Field(ge=0)
    final_decision_accuracy: float = Field(ge=0, le=1)
    hard_constraint_validity: float = Field(ge=0, le=1)
    cross_domain_compatibility_accuracy: float = Field(ge=0, le=1)
    evidence_grounded_arbitration_accuracy: float = Field(ge=0, le=1)
    global_optimum_accuracy: float = Field(ge=0, le=1)
    human_review_calibration: float = Field(ge=0, le=1)
    disagreement_resolved_correctly_count: int = Field(ge=0)
    disagreement_resolved_incorrectly_count: int = Field(ge=0)
    specialist_call_count: int = Field(ge=0)
    coordination_overhead_count: int = Field(ge=0)
    mean_latency_ms: float = Field(ge=0)


class ArchitectureComparisonResult(BaseModel):
    """Comparison between a baseline path and minimal specialist coordination."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline: CoordinatedPlanResult
    coordinated: CoordinatedPlanResult
    metrics: ArchitectureComparisonMetrics
