"""Typed state and dependency models for PartyPilot v0.3 replanning research."""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from partypilot.domain.constraints import Constraint, ConstraintType
from partypilot.domain.party_request import PartyRequest
from partypilot.domain.resources import Resource

NonEmptyString = Annotated[str, Field(min_length=1)]
RevisionNumber = Annotated[int, Field(ge=0)]


class PlanningDependencyKind(StrEnum):
    """Typed dependency relationships for stateful replanning."""

    GUEST_COUNT_TO_VENUE_CAPACITY = "guest_count_to_venue_capacity"
    GUEST_COUNT_TO_CATERING_QUANTITY = "guest_count_to_catering_quantity"
    GUEST_COUNT_TO_CATERING_COST = "guest_count_to_catering_cost"
    GUEST_COUNT_TO_SEATING = "guest_count_to_seating"
    GUEST_COUNT_TO_PARKING = "guest_count_to_parking"
    VENUE_TO_APPROVED_CATERERS = "venue_to_approved_caterers"
    VENUE_TO_ACTIVITY_SPACE = "venue_to_activity_space"
    ACCESSIBILITY_TO_VENUE = "accessibility_to_venue"
    ACCESSIBILITY_TO_PATH = "accessibility_to_path"
    ACCESSIBILITY_TO_ROOM = "accessibility_to_room"
    ACCESSIBILITY_TO_RESTROOM = "accessibility_to_restroom"
    DIETARY_TO_CATERING_EVIDENCE = "dietary_to_catering_evidence"
    SCHEDULE_TO_VENDOR_AVAILABILITY = "schedule_to_vendor_availability"
    SCHEDULE_TO_SETUP_WINDOW = "schedule_to_setup_window"
    BUDGET_TO_RESOURCE_SELECTION = "budget_to_resource_selection"
    BUDGET_TO_TOTAL_COST = "budget_to_total_cost"
    FEES_TO_TOTAL_COST = "fees_to_total_cost"
    NEW_EVIDENCE_TO_POLICY_VALIDITY = "new_evidence_to_policy_validity"


class PlanningDecisionCategory(StrEnum):
    """High-level categories for planning decisions."""

    RESOURCE_SELECTION = "resource_selection"
    BUDGET = "budget"
    ACCESSIBILITY = "accessibility"
    DIETARY = "dietary"
    SCHEDULE = "schedule"
    PREFERENCE = "preference"
    REVIEW = "review"
    OTHER = "other"


class PlanningDecisionStatus(StrEnum):
    """Lifecycle state for a deterministic planning decision."""

    ACTIVE = "active"
    PRESERVED = "preserved"
    INVALIDATED = "invalidated"


class PlanningDependency(BaseModel):
    """A typed relationship that may be invalidated by a state update."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dependency_id: NonEmptyString
    kind: PlanningDependencyKind
    source: NonEmptyString
    target: NonEmptyString
    description: NonEmptyString
    notes: tuple[NonEmptyString, ...] = ()

    @field_validator("dependency_id", "source", "target", "description")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be blank")
        return normalized


class PlanningDecision(BaseModel):
    """A deterministic planning conclusion tracked across revisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: NonEmptyString
    category: PlanningDecisionCategory
    summary: NonEmptyString
    status: PlanningDecisionStatus = PlanningDecisionStatus.ACTIVE
    dependency_ids: tuple[NonEmptyString, ...] = Field(default_factory=tuple)
    prerequisite_decision_ids: tuple[NonEmptyString, ...] = Field(default_factory=tuple)
    resource_ids: tuple[NonEmptyString, ...] = Field(default_factory=tuple)
    evidence_ids: tuple[NonEmptyString, ...] = Field(default_factory=tuple)
    assumptions: tuple[NonEmptyString, ...] = Field(default_factory=tuple)
    notes: tuple[NonEmptyString, ...] = Field(default_factory=tuple)

    @field_validator("decision_id", "summary")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be blank")
        return normalized

    @field_validator(
        "dependency_ids",
        "prerequisite_decision_ids",
        "resource_ids",
        "evidence_ids",
        "assumptions",
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


class PlanningUpdateKind(StrEnum):
    """Supported deterministic state changes for replanning."""

    GUEST_COUNT_CHANGED = "guest_count_changed"
    BUDGET_CHANGED = "budget_changed"
    DATE_TIME_CHANGED = "date_time_changed"
    NEW_ALLERGY_ADDED = "new_allergy_added"
    ACCESSIBILITY_REQUIREMENT_ADDED = "accessibility_requirement_added"
    VENDOR_UNAVAILABLE = "vendor_unavailable"
    NEW_EVIDENCE_DISCOVERED = "new_evidence_discovered"
    FEE_RULE_CHANGED = "fee_rule_changed"
    NO_OP = "no_op"


class PlanningUpdate(BaseModel):
    """A deterministic change event applied to a planning state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    update_id: NonEmptyString
    kind: PlanningUpdateKind
    description: NonEmptyString
    guest_count: int | None = Field(default=None, gt=0)
    total_budget: Decimal | None = Field(default=None, ge=Decimal("0"))
    event_date: date | None = None
    event_time: time | None = None
    added_allergies: tuple[NonEmptyString, ...] = ()
    added_dietary_restrictions: tuple[NonEmptyString, ...] = ()
    added_accessibility_needs: tuple[NonEmptyString, ...] = ()
    unavailable_resource_ids: tuple[NonEmptyString, ...] = ()
    evidence_document_ids: tuple[NonEmptyString, ...] = ()
    changed_rule_ids: tuple[NonEmptyString, ...] = ()
    notes: tuple[NonEmptyString, ...] = ()

    @field_validator("update_id", "description")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be blank")
        return normalized

    @field_validator(
        "added_allergies",
        "added_dietary_restrictions",
        "added_accessibility_needs",
        "unavailable_resource_ids",
        "evidence_document_ids",
        "changed_rule_ids",
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

    @model_validator(mode="after")
    def validate_kind_specific_payload(self) -> PlanningUpdate:
        required: dict[PlanningUpdateKind, bool] = {
            PlanningUpdateKind.GUEST_COUNT_CHANGED: self.guest_count is not None,
            PlanningUpdateKind.BUDGET_CHANGED: self.total_budget is not None,
            PlanningUpdateKind.DATE_TIME_CHANGED: self.event_date is not None
            or self.event_time is not None,
            PlanningUpdateKind.NEW_ALLERGY_ADDED: bool(self.added_allergies),
            PlanningUpdateKind.ACCESSIBILITY_REQUIREMENT_ADDED: bool(
                self.added_accessibility_needs
            ),
            PlanningUpdateKind.VENDOR_UNAVAILABLE: bool(self.unavailable_resource_ids),
            PlanningUpdateKind.NEW_EVIDENCE_DISCOVERED: bool(self.evidence_document_ids),
            PlanningUpdateKind.FEE_RULE_CHANGED: bool(self.changed_rule_ids),
            PlanningUpdateKind.NO_OP: not any(
                (
                    self.guest_count is not None,
                    self.total_budget is not None,
                    self.event_date is not None,
                    self.event_time is not None,
                    self.added_allergies,
                    self.added_dietary_restrictions,
                    self.added_accessibility_needs,
                    self.unavailable_resource_ids,
                    self.evidence_document_ids,
                    self.changed_rule_ids,
                )
            ),
        }
        if not required[self.kind]:
            raise ValueError(f"{self.kind.value} update is missing required payload")
        return self


class PlanningStateTransition(BaseModel):
    """Immutable audit trail for a single planning revision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_revision_number: RevisionNumber
    to_revision_number: RevisionNumber
    updates: tuple[PlanningUpdate, ...] = Field(min_length=1)
    affected_dependency_kinds: tuple[PlanningDependencyKind, ...] = ()
    affected_dependency_ids: tuple[NonEmptyString, ...] = ()
    invalidated_decision_ids: tuple[NonEmptyString, ...] = ()
    preserved_decision_ids: tuple[NonEmptyString, ...] = ()
    recompute_steps: tuple[NonEmptyString, ...] = ()
    cycle_detected: bool = False
    cycle_decision_ids: tuple[NonEmptyString, ...] = ()
    cycle_error: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_revision_order(self) -> PlanningStateTransition:
        if self.to_revision_number <= self.from_revision_number:
            raise ValueError("to_revision_number must be greater than from_revision_number")
        return self


class PlanningState(BaseModel):
    """Provider-neutral planning state for v0.3 decomposition and replanning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    revision_number: RevisionNumber
    request: PartyRequest
    selected_resources: tuple[Resource, ...] = ()
    evidence_backed_constraints: tuple[Constraint, ...] = ()
    derived_constraints: tuple[Constraint, ...] = ()
    unresolved_uncertainties: tuple[NonEmptyString, ...] = ()
    decisions: tuple[PlanningDecision, ...] = ()
    assumptions: tuple[NonEmptyString, ...] = ()
    dependency_relationships: tuple[PlanningDependency, ...] = ()
    invalidated_decision_ids: tuple[NonEmptyString, ...] = ()
    transition_log: tuple[PlanningStateTransition, ...] = ()
    notes: tuple[NonEmptyString, ...] = ()

    @field_validator("unresolved_uncertainties", "assumptions", "invalidated_decision_ids", "notes")
    @classmethod
    def validate_unique_text_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("values cannot be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("values must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_state(self) -> PlanningState:
        decision_ids = [decision.decision_id for decision in self.decisions]
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("decision IDs must be unique")
        decision_id_set = set(decision_ids)
        for decision in self.decisions:
            if not set(decision.prerequisite_decision_ids).issubset(decision_id_set):
                raise ValueError("prerequisite decisions must exist in the state")

        dependency_ids = [dependency.dependency_id for dependency in self.dependency_relationships]
        if len(set(dependency_ids)) != len(dependency_ids):
            raise ValueError("dependency IDs must be unique")

        invalidated = set(self.invalidated_decision_ids)
        if not invalidated.issubset(set(decision_ids)):
            raise ValueError("invalidated decisions must exist in the state")

        if self.evidence_backed_constraints and any(
            constraint.constraint_type is ConstraintType.DERIVED
            for constraint in self.evidence_backed_constraints
        ):
            raise ValueError("evidence-backed constraints must not be derived")

        if self.transition_log:
            last_transition = self.transition_log[-1]
            if last_transition.to_revision_number != self.revision_number:
                raise ValueError("last transition must end at the current revision")
        return self

    @property
    def active_decisions(self) -> tuple[PlanningDecision, ...]:
        return tuple(
            decision
            for decision in self.decisions
            if decision.status is PlanningDecisionStatus.ACTIVE
        )

    @property
    def preserved_decisions(self) -> tuple[PlanningDecision, ...]:
        return tuple(
            decision
            for decision in self.decisions
            if decision.status is PlanningDecisionStatus.PRESERVED
        )

    @property
    def invalidated_decisions(self) -> tuple[PlanningDecision, ...]:
        return tuple(
            decision
            for decision in self.decisions
            if decision.status is PlanningDecisionStatus.INVALIDATED
        )


class PlanningStateSummary(BaseModel):
    """Compact serializable summary for comparison reports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    revision_number: RevisionNumber
    selected_resource_ids: tuple[NonEmptyString, ...] = ()
    invalidated_decision_ids: tuple[NonEmptyString, ...] = ()
    preserved_decision_ids: tuple[NonEmptyString, ...] = ()
    unresolved_uncertainties: tuple[NonEmptyString, ...] = ()
    notes: tuple[NonEmptyString, ...] = ()

    @classmethod
    def from_state(cls, state: PlanningState) -> PlanningStateSummary:
        return cls(
            revision_number=state.revision_number,
            selected_resource_ids=tuple(
                resource.resource_id for resource in state.selected_resources
            ),
            invalidated_decision_ids=state.invalidated_decision_ids,
            preserved_decision_ids=tuple(
                decision.decision_id for decision in state.preserved_decisions
            ),
            unresolved_uncertainties=state.unresolved_uncertainties,
            notes=state.notes,
        )


class ReplanningComparisonMetrics(BaseModel):
    """Objective metrics for comparing targeted and full replanning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correctness: float = Field(ge=0, le=1)
    final_state_correctness: float = Field(ge=0, le=1)
    preserved_decision_accuracy: float = Field(ge=0, le=1)
    invalidation_accuracy: float = Field(ge=0, le=1)
    missed_recomputation_count: int = Field(ge=0)
    full_replan_decision_count: int = Field(ge=0)
    targeted_replan_decision_count: int = Field(ge=0)
    unnecessary_recomputed_decision_count: int = Field(ge=0)
    recomputation_reduction_ratio: float = Field(ge=0, le=1)
    full_replan_latency_ms: float = Field(ge=0)
    targeted_replan_latency_ms: float = Field(ge=0)


class ReplanningComparisonReport(BaseModel):
    """Deterministic comparison between full and targeted replanning strategies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_state: PlanningStateSummary
    updates: tuple[PlanningUpdate, ...]
    full_replan: PlanningStateTransition
    targeted_replan: PlanningStateTransition
    metrics: ReplanningComparisonMetrics
    notes: tuple[NonEmptyString, ...] = ()
