"""Framework-independent PartyPilot domain models."""

from partypilot.domain.constraints import (
    Constraint,
    ConstraintOperator,
    ConstraintProvenance,
    ConstraintType,
)
from partypilot.domain.dependencies import (
    ResourceRequirement,
    ResourceRequirementMode,
    TaskDependency,
)
from partypilot.domain.evaluation import (
    ComplexityMetadata,
    DatasetSplit,
    EvaluationScenario,
    ScenarioCategory,
)
from partypilot.domain.evidence import (
    DerivationMethod,
    EvidenceReference,
    EvidenceState,
    Provenance,
)
from partypilot.domain.experiment import ExperimentConfig, ExperimentResultMetadata
from partypilot.domain.feasibility import (
    FeasibilityOutcome,
    FeasibilityResult,
    ValidationResult,
)
from partypilot.domain.party_plan import PartyPlan
from partypilot.domain.party_request import AgeRange, PartyRequest
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
    "Caterer",
    "ComplexityMetadata",
    "Constraint",
    "ConstraintOperator",
    "ConstraintProvenance",
    "ConstraintType",
    "DatasetSplit",
    "DerivationMethod",
    "Duration",
    "EvaluationScenario",
    "EvidenceReference",
    "EvidenceState",
    "ExperimentConfig",
    "ExperimentResultMetadata",
    "FeasibilityOutcome",
    "FeasibilityResult",
    "PartyPlan",
    "PartyRequest",
    "Provenance",
    "Resource",
    "ResourceCategory",
    "ResourceRequirement",
    "ResourceRequirementMode",
    "ScenarioCategory",
    "ScheduledInterval",
    "ScheduledTask",
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
