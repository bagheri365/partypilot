from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from partypilot.domain.constraints import (
    Constraint,
    ConstraintOperator,
    ConstraintProvenance,
    ConstraintType,
)
from partypilot.domain.evaluation import (
    ComplexityMetadata,
    DatasetSplit,
    EvaluationScenario,
    ScenarioCategory,
)
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest


def _request() -> PartyRequest:
    return PartyRequest(
        location="Boston",
        event_date=date(2026, 10, 10),
        guest_count=20,
        total_budget=Decimal("1000.00"),
    )


def _hard_constraint() -> Constraint:
    return Constraint(
        identifier="hard-budget",
        key="budget",
        operator=ConstraintOperator.LTE,
        value=Decimal("1000.00"),
        constraint_type=ConstraintType.HARD,
        description="Total cost must not exceed budget",
    )


def _derived_constraint() -> Constraint:
    return Constraint(
        identifier="derived-age",
        key="minimum_age",
        operator=ConstraintOperator.GTE,
        value=5,
        constraint_type=ConstraintType.DERIVED,
        description="Derived minimum participant age",
        provenance=ConstraintProvenance(
            source_constraint_ids=("source-age",),
            derivation_explanation="Normalized from the supplied child age range",
        ),
    )


def test_evaluation_scenario_captures_expected_ground_truth() -> None:
    scenario = EvaluationScenario(
        scenario_id="scenario-001",
        request=_request(),
        expected_feasibility=FeasibilityOutcome.FEASIBLE,
        expected_hard_constraints=(_hard_constraint(),),
        expected_derived_constraints=(_derived_constraint(),),
        expected_resource_ids=("venue-1", "caterer-1", "activity-1"),
        relevant_evidence_ids=("evidence-1",),
        scenario_category=ScenarioCategory.FEASIBLE,
        complexity=ComplexityMetadata(
            hard_constraint_count=1,
            derived_constraint_count=1,
            expected_resource_count=3,
            notes=("Simple fixture-backed scenario",),
        ),
        dataset_split=DatasetSplit.DEVELOPMENT,
        labeling_notes=("Expected resources are deterministic fixture IDs",),
    )

    assert scenario.scenario_id == "scenario-001"
    assert scenario.expected_feasibility is FeasibilityOutcome.FEASIBLE
    assert scenario.dataset_split is DatasetSplit.DEVELOPMENT
    assert scenario.complexity.expected_resource_count == 3


def test_dataset_split_values_match_benchmark_contract() -> None:
    assert {item.value for item in DatasetSplit} == {
        "development",
        "frozen_test",
        "adversarial",
    }


def test_hard_constraint_collection_rejects_non_hard_constraint() -> None:
    soft = Constraint(
        identifier="soft-theme",
        key="theme",
        operator=ConstraintOperator.EQ,
        value="space",
        constraint_type=ConstraintType.SOFT,
        description="Prefer a space theme",
    )

    with pytest.raises(ValidationError, match="only HARD constraints"):
        EvaluationScenario(
            scenario_id="scenario-002",
            request=_request(),
            expected_feasibility=FeasibilityOutcome.FEASIBLE,
            expected_hard_constraints=(soft,),
            scenario_category=ScenarioCategory.FEASIBLE,
            complexity=ComplexityMetadata(),
            dataset_split=DatasetSplit.FROZEN_TEST,
        )


def test_derived_constraint_collection_rejects_non_derived_constraint() -> None:
    with pytest.raises(ValidationError, match="only DERIVED constraints"):
        EvaluationScenario(
            scenario_id="scenario-003",
            request=_request(),
            expected_feasibility=FeasibilityOutcome.NO_FEASIBLE_PLAN,
            expected_derived_constraints=(_hard_constraint(),),
            scenario_category=ScenarioCategory.BUDGET,
            complexity=ComplexityMetadata(),
            dataset_split=DatasetSplit.ADVERSARIAL,
        )


def test_complexity_counts_cannot_be_negative() -> None:
    with pytest.raises(ValidationError):
        ComplexityMetadata(hard_constraint_count=-1)


def test_scenario_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        EvaluationScenario.model_validate(
            {
                "scenario_id": "scenario-004",
                "request": _request(),
                "expected_feasibility": FeasibilityOutcome.FEASIBLE,
                "scenario_category": ScenarioCategory.OTHER,
                "complexity": ComplexityMetadata(),
                "dataset_split": DatasetSplit.DEVELOPMENT,
                "unexpected": True,
            }
        )
