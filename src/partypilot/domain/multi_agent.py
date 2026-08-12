"""Typed multi-agent runtime models for PartyPilot v0.5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from partypilot.domain.coordination import (
    ArbitrationOutcome,
    CoordinatedPlanResult,
    SpecialistDecision,
    SpecialistDomain,
)
from partypilot.domain.evidence import EvidenceState
from partypilot.domain.evidence_corpus import EvidenceDocument
from partypilot.domain.planning_state import (
    PlanningDecision,
    PlanningDependency,
    PlanningState,
    PlanningStateSummary,
)
from partypilot.domain.resources import Resource

NonEmptyString = Annotated[str, Field(min_length=1)]


@dataclass(frozen=True, slots=True)
class SpecialistIdentity:
    """Canonical typed identity for one specialist role."""

    domain: SpecialistDomain
    specialist_id: str
    specialist_name: str


SPECIALIST_IDENTITIES: tuple[SpecialistIdentity, ...] = (
    SpecialistIdentity(
        domain=SpecialistDomain.VENUE,
        specialist_id="venue",
        specialist_name="VenueAgent",
    ),
    SpecialistIdentity(
        domain=SpecialistDomain.CATERING_SAFETY,
        specialist_id="catering",
        specialist_name="CateringSafetyAgent",
    ),
    SpecialistIdentity(
        domain=SpecialistDomain.ACCESSIBILITY,
        specialist_id="accessibility",
        specialist_name="AccessibilityAgent",
    ),
    SpecialistIdentity(
        domain=SpecialistDomain.SCHEDULING_OPERATIONS,
        specialist_id="scheduling",
        specialist_name="SchedulingAgent",
    ),
    SpecialistIdentity(
        domain=SpecialistDomain.BUDGET,
        specialist_id="budget",
        specialist_name="BudgetAgent",
    ),
)
SPECIALIST_IDENTITY_BY_DOMAIN = {identity.domain: identity for identity in SPECIALIST_IDENTITIES}


def specialist_identity_for_domain(domain: SpecialistDomain) -> SpecialistIdentity:
    """Return the canonical identity for a specialist domain."""

    return SPECIALIST_IDENTITY_BY_DOMAIN[domain]


def canonical_specialist_id(domain: SpecialistDomain) -> str:
    """Return the canonical specialist_id for a domain."""

    return specialist_identity_for_domain(domain).specialist_id


def canonical_specialist_name(domain: SpecialistDomain) -> str:
    """Return the canonical prompt-facing specialist name for a domain."""

    return specialist_identity_for_domain(domain).specialist_name


class SpecialistFailureKind(StrEnum):
    """Typed failure categories for specialist-agent execution."""

    PROVIDER_CONNECTION_ERROR = "provider_connection_error"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_RESPONSE_ERROR = "provider_response_error"
    STRUCTURED_OUTPUT_VALIDATION_ERROR = "structured_output_validation_error"
    SPECIALIST_DOMAIN_VALIDATION_ERROR = "specialist_domain_validation_error"
    SPECIALIST_EXECUTION_ERROR = "specialist_execution_error"
    COORDINATOR_ERROR = "coordinator_error"


class SpecialistDecisionEvidenceReference(BaseModel):
    """Minimal evidence citation returned by a specialist agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: NonEmptyString
    state: EvidenceState


class SpecialistDecisionPayload(BaseModel):
    """Strict structured output expected from a specialist LLM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: NonEmptyString
    domain: SpecialistDomain
    recommendation: NonEmptyString
    status: ArbitrationOutcome
    hard_constraints_considered: tuple[NonEmptyString, ...] = ()
    evidence_references: tuple[SpecialistDecisionEvidenceReference, ...] = ()
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


class SpecialistDecisionEnvelope(BaseModel):
    """Outer wrapper for the structured specialist response."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    decision: SpecialistDecisionPayload


class SpecialistAgentInput(BaseModel):
    """Provider-neutral input to a specialist agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: NonEmptyString
    specialist_id: NonEmptyString
    specialist_name: NonEmptyString
    domain: SpecialistDomain
    planning_state: PlanningState
    candidate_resources: tuple[Resource, ...]
    requires_resource_recommendations: bool = False
    allowed_evidence_document_ids: tuple[NonEmptyString, ...] = ()
    scoped_evidence_documents: tuple[EvidenceDocument, ...] = ()
    structured_facts: tuple[NonEmptyString, ...] = ()
    relevant_dependencies: tuple[PlanningDependency, ...] = ()
    prior_accepted_decisions: tuple[PlanningDecision, ...] = ()
    explicit_instructions: tuple[NonEmptyString, ...] = ()
    candidate_total_cost: Decimal | None = Field(default=None, ge=Decimal("0"))

    @property
    def planning_state_summary(self) -> PlanningStateSummary:
        return PlanningStateSummary.from_state(self.planning_state)


class SpecialistExecutionTrace(BaseModel):
    """Execution trace for one specialist invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: NonEmptyString
    specialist_id: NonEmptyString
    specialist_name: NonEmptyString
    domain: SpecialistDomain
    model_name: NonEmptyString | None = None
    started_at: datetime
    completed_at: datetime
    latency_ms: float = Field(ge=0)
    input_scope_summary: tuple[NonEmptyString, ...] = ()
    evidence_document_ids: tuple[NonEmptyString, ...] = ()
    validation_succeeded: bool
    recommendation_status: ArbitrationOutcome | None = None
    retry_count: int = Field(ge=0)
    failure_kind: SpecialistFailureKind | None = None
    failure_error_type: NonEmptyString | None = None
    failure_reason: NonEmptyString | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=Decimal("0"))

    @field_validator("input_scope_summary", "evidence_document_ids")
    @classmethod
    def validate_unique_text_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("values cannot be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("values must be unique")
        return normalized


class SpecialistExecutionOutcome(BaseModel):
    """Structured outcome from a specialist agent run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: SpecialistDecision | None = None
    trace: SpecialistExecutionTrace
    failure_kind: SpecialistFailureKind | None = None
    failure_error_type: NonEmptyString | None = None
    failure_reason: NonEmptyString | None = None
    raw_text: str | None = None
    raw_structured_output: object | None = None


class CandidateEvaluationResult(BaseModel):
    """Per-candidate result for the live multi-agent runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_resource_ids: tuple[NonEmptyString, ...]
    specialist_outcomes: tuple[SpecialistExecutionOutcome, ...]
    selected_resource_ids: tuple[NonEmptyString, ...]
    arbitration_outcome: ArbitrationOutcome
    coordinated_result: CoordinatedPlanResult
    total_cost: float | None = None
    latency_ms: float = Field(ge=0)

    @field_validator("candidate_resource_ids", "selected_resource_ids")
    @classmethod
    def validate_unique_text_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("values cannot be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("values must be unique")
        return normalized


class MultiAgentPlanningRuntimeResult(BaseModel):
    """Terminal result from the live multi-agent runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: NonEmptyString
    planning_state: PlanningStateSummary
    candidate_results: tuple[CandidateEvaluationResult, ...]
    final_result: CoordinatedPlanResult
    execution_traces: tuple[SpecialistExecutionTrace, ...]
    wall_clock_latency_ms: float = Field(ge=0)
    notes: tuple[NonEmptyString, ...] = ()


class MultiAgentSmokeRow(BaseModel):
    """Concise smoke-test row for one specialist invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_name: NonEmptyString
    status: NonEmptyString
    recommendation_count: int = Field(ge=0)
    evidence_ids: tuple[NonEmptyString, ...] = ()
    latency_ms: float = Field(ge=0)
    validation_succeeded: bool
    retry_count: int = Field(ge=0)
