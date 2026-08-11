from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from partypilot.domain.constraints import Constraint
from partypilot.domain.evidence_corpus import (
    EvidenceDocument,
    EvidenceDocumentStatus,
    EvidenceDocumentType,
)
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest
from partypilot.domain.resources import Resource

NonEmptyString = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class DatasetSplit(StrEnum):
    """Supported benchmark dataset partitions."""

    DEVELOPMENT = "development"
    FROZEN_TEST = "frozen_test"
    ADVERSARIAL = "adversarial"


class ScenarioCategory(StrEnum):
    """High-level category used to stratify evaluation scenarios."""

    FEASIBLE = "feasible"
    BUDGET = "budget"
    CAPACITY = "capacity"
    AVAILABILITY = "availability"
    AGE_RESTRICTION = "age_restriction"
    ACCESSIBILITY = "accessibility"
    TEMPORAL = "temporal"
    RESOURCE_CONFLICT = "resource_conflict"
    MULTIPLE_CHOICES = "multiple_choices"
    IMPOSSIBLE_COMBINATION = "impossible_combination"
    SAFETY_EVIDENCE = "safety_evidence"
    OTHER = "other"


class ComplexityMetadata(BaseModel):
    """Simple deterministic metadata describing scenario complexity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hard_constraint_count: NonNegativeInt = 0
    derived_constraint_count: NonNegativeInt = 0
    expected_resource_count: NonNegativeInt = 0
    notes: tuple[NonEmptyString, ...] = ()


class RetrievalGroundTruthLabel(BaseModel):
    """Human-authored retrieval ground truth for evidence-dependent scenarios."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_document_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    resource_id: NonEmptyString
    expected_version: NonEmptyString
    expected_status: EvidenceDocumentStatus
    policy_type: EvidenceDocumentType


class EvaluationScenario(BaseModel):
    """Ground-truth schema for a single PartyPilot benchmark scenario."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: NonEmptyString
    request: PartyRequest
    expected_feasibility: FeasibilityOutcome
    expected_hard_constraints: tuple[Constraint, ...] = ()
    expected_derived_constraints: tuple[Constraint, ...] = ()
    expected_resource_ids: tuple[NonEmptyString, ...] = ()
    relevant_evidence_ids: tuple[NonEmptyString, ...] = ()
    retrieval_ground_truth: tuple[RetrievalGroundTruthLabel, ...] = ()
    scenario_category: ScenarioCategory
    complexity: ComplexityMetadata
    dataset_split: DatasetSplit
    labeling_notes: tuple[NonEmptyString, ...] = ()

    @field_validator("expected_hard_constraints")
    @classmethod
    def validate_hard_constraints(
        cls, constraints: tuple[Constraint, ...]
    ) -> tuple[Constraint, ...]:
        invalid = [item.identifier for item in constraints if item.constraint_type.value != "HARD"]
        if invalid:
            raise ValueError("expected_hard_constraints may contain only HARD constraints")
        return constraints

    @field_validator("expected_derived_constraints")
    @classmethod
    def validate_derived_constraints(
        cls, constraints: tuple[Constraint, ...]
    ) -> tuple[Constraint, ...]:
        invalid = [
            item.identifier for item in constraints if item.constraint_type.value != "DERIVED"
        ]
        if invalid:
            raise ValueError("expected_derived_constraints may contain only DERIVED constraints")
        return constraints


class CapabilityBoundaryScenarioMetadata(BaseModel):
    """Typed research metadata for capability-boundary benchmark scenarios."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    capability_tags: tuple[NonEmptyString, ...] = ()
    requires_evidence: bool
    requires_semantic_interpretation: bool
    requires_state_replanning: bool
    cross_domain_dependency_count: NonNegativeInt = 0
    adversarial_flag: bool = False
    complexity_trap_flag: bool = False
    milestone_introduced: NonEmptyString
    notes: tuple[NonEmptyString, ...] = ()


class CapabilityBoundaryScenario(BaseModel):
    """Research benchmark scenario paired with forward-looking capability metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario: EvaluationScenario
    metadata: CapabilityBoundaryScenarioMetadata
    evidence_documents: tuple[EvidenceDocument, ...] = ()
    structured_resources: tuple[Resource, ...] = ()


CapabilityBoundaryScenario.model_rebuild()
