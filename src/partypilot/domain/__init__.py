"""Framework-independent PartyPilot domain models."""

from partypilot.domain.constraints import (
    Constraint,
    ConstraintOperator,
    ConstraintProvenance,
    ConstraintType,
)
from partypilot.domain.coordination import (
    ArbitrationOutcome,
    ArbitrationTrace,
    ArchitectureComparisonMetrics,
    ArchitectureComparisonResult,
    CoordinatedPlanResult,
    SpecialistDecision,
    SpecialistDomain,
)
from partypilot.domain.dependencies import (
    ResourceRequirement,
    ResourceRequirementMode,
    TaskDependency,
)
from partypilot.domain.evaluation import (
    CapabilityBoundaryScenario,
    CapabilityBoundaryScenarioMetadata,
    ComplexityMetadata,
    DatasetSplit,
    EvaluationScenario,
    RetrievalGroundTruthLabel,
    ScenarioCategory,
)
from partypilot.domain.evidence import (
    DerivationMethod,
    EvidenceReference,
    EvidenceState,
    Provenance,
)
from partypilot.domain.evidence_corpus import (
    EvidenceDocument,
    EvidenceDocumentMetadata,
    EvidenceDocumentStatus,
    EvidenceDocumentType,
)
from partypilot.domain.experiment import ExperimentConfig, ExperimentResultMetadata
from partypilot.domain.feasibility import (
    FeasibilityOutcome,
    FeasibilityResult,
    ValidationResult,
)
from partypilot.domain.party_plan import PartyPlan
from partypilot.domain.party_request import AgeRange, PartyRequest
from partypilot.domain.planning_state import (
    PlanningDecision,
    PlanningDecisionCategory,
    PlanningDecisionStatus,
    PlanningDependency,
    PlanningDependencyKind,
    PlanningState,
    PlanningStateSummary,
    PlanningStateTransition,
    PlanningUpdate,
    PlanningUpdateKind,
    ReplanningComparisonMetrics,
    ReplanningComparisonReport,
)
from partypilot.domain.resources import (
    AccessibilityAttribute,
    Activity,
    Caterer,
    Resource,
    ResourceCategory,
    Venue,
)
from partypilot.domain.temporal import Duration, ScheduledInterval, TemporalRequirements, TimeWindow
from partypilot.domain.temporal_validation import (
    ScheduledTask,
    TemporalValidationResult,
    TemporalViolation,
    TemporalViolationCode,
    validate_temporal_schedule,
)

__all__ = [
    "AccessibilityAttribute",
    "Activity",
    "AgeRange",
    "ArbitrationOutcome",
    "ArbitrationTrace",
    "ArchitectureComparisonMetrics",
    "ArchitectureComparisonResult",
    "CapabilityBoundaryScenario",
    "CapabilityBoundaryScenarioMetadata",
    "Caterer",
    "ComplexityMetadata",
    "Constraint",
    "ConstraintOperator",
    "ConstraintProvenance",
    "ConstraintType",
    "CoordinatedPlanResult",
    "DatasetSplit",
    "DerivationMethod",
    "Duration",
    "EvaluationScenario",
    "EvidenceDocument",
    "EvidenceDocumentMetadata",
    "EvidenceDocumentStatus",
    "EvidenceDocumentType",
    "EvidenceReference",
    "EvidenceState",
    "ExperimentConfig",
    "ExperimentResultMetadata",
    "FeasibilityOutcome",
    "FeasibilityResult",
    "PartyPlan",
    "PartyRequest",
    "PlanningDecision",
    "PlanningDecisionCategory",
    "PlanningDecisionStatus",
    "PlanningDependency",
    "PlanningDependencyKind",
    "PlanningState",
    "PlanningStateSummary",
    "PlanningStateTransition",
    "PlanningUpdate",
    "PlanningUpdateKind",
    "Provenance",
    "ReplanningComparisonMetrics",
    "ReplanningComparisonReport",
    "Resource",
    "ResourceCategory",
    "ResourceRequirement",
    "ResourceRequirementMode",
    "RetrievalGroundTruthLabel",
    "ScenarioCategory",
    "ScheduledInterval",
    "ScheduledTask",
    "SpecialistDecision",
    "SpecialistDomain",
    "TaskDependency",
    "TemporalRequirements",
    "TemporalValidationResult",
    "TemporalViolation",
    "TemporalViolationCode",
    "TimeWindow",
    "ValidationResult",
    "Venue",
    "validate_temporal_schedule",
]
