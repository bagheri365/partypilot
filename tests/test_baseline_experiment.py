from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from partypilot.application.baseline_experiment import (
    BaselineExperimentResult,
    run_baseline_experiment,
    save_baseline_experiment_reports,
)
from partypilot.application.baseline_metrics import BaselineFailureLabel
from partypilot.application.deterministic_planner import PlannerResult
from partypilot.application.single_pass_llm_planner import SinglePassLLMPlanner
from partypilot.domain.evaluation import (
    ComplexityMetadata,
    DatasetSplit,
    EvaluationScenario,
    ScenarioCategory,
)
from partypilot.domain.experiment import ExperimentConfig, ExperimentResultMetadata
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest
from partypilot.ports.llm_provider import FakeLLMProvider, GenerationResponse, UsageMetadata


class FakeDeterministicPlanner:
    def __init__(self, result: PlannerResult) -> None:
        self.result = result

    def plan(self, request: PartyRequest) -> PlannerResult:
        return self.result


def _scenario() -> EvaluationScenario:
    return EvaluationScenario(
        scenario_id="scenario-1",
        request=PartyRequest(
            location="Boston",
            event_date=date(2026, 9, 1),
            guest_count=10,
            total_budget=Decimal("500"),
        ),
        expected_feasibility=FeasibilityOutcome.FEASIBLE,
        scenario_category=ScenarioCategory.FEASIBLE,
        complexity=ComplexityMetadata(),
        dataset_split=DatasetSplit.DEVELOPMENT,
    )


def _single_pass_planner() -> SinglePassLLMPlanner:
    response = GenerationResponse(
        text="",
        structured_output={
            "resources": [
                {
                    "resource_id": "venue-1",
                    "name": "Venue One",
                    "location": "Boston",
                    "price": "100",
                    "capacity": 20,
                    "availability": [],
                    "age_restrictions": None,
                    "accessibility_attributes": [],
                    "category": "venue",
                }
            ],
            "claimed_total_cost": "100",
            "assumptions": [],
        },
        usage=UsageMetadata(input_tokens=7, output_tokens=3, total_tokens=10),
    )
    return SinglePassLLMPlanner(FakeLLMProvider([response]))


def test_run_baseline_experiment_serializes_metadata_and_scenario_results(
    tmp_path: Path,
) -> None:
    metadata = ExperimentResultMetadata(
        config=ExperimentConfig(
            experiment_id="baseline-v0.1-development-20260810T120000Z",
            code_commit_sha="abc123",
            dataset_version="v0.1",
            architecture_variant="deterministic_plus_single_pass_llm",
            model_provider="ollama",
            model_name="fake-model",
        )
    )
    result = run_baseline_experiment(
        (_scenario(),),
        FakeDeterministicPlanner(PlannerResult(candidates=())),
        _single_pass_planner(),
        metadata=metadata,
        dataset_split=DatasetSplit.DEVELOPMENT,
    )

    assert isinstance(result, BaselineExperimentResult)
    json_path, markdown_path = save_baseline_experiment_reports(result, tmp_path)
    payload = json_path.read_text(encoding="utf-8")
    assert '"experiment_id": "baseline-v0.1-development-20260810T120000Z"' in payload
    assert '"single_pass_scenarios"' in payload
    assert '"scenario_id": "scenario-1"' in payload
    assert '"usage_total_tokens": 10' in payload

    assert result.comparison.single_pass_llm.total_input_tokens == 7
    assert result.comparison.single_pass_llm.total_output_tokens == 3
    assert result.comparison.single_pass_llm.total_tokens == 10
    assert result.single_pass_scenarios[0].usage_total_tokens == 10
    assert result.single_pass_scenarios[0].failure_labels == (
        BaselineFailureLabel.HALLUCINATED_RESOURCE,
    )
    assert result.single_pass_scenarios[0].unsupported_claim is None

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# PartyPilot v0.1 Baseline Experiment" in markdown
    assert "# Baseline Comparison" in markdown
    assert "Single-pass LLM baseline" in markdown


def test_run_baseline_experiment_tracks_latency_and_labels() -> None:
    metadata = ExperimentResultMetadata(
        config=ExperimentConfig(
            experiment_id="baseline-v0.1-development-20260810T120000Z",
            code_commit_sha="abc123",
            dataset_version="v0.1",
            architecture_variant="deterministic_plus_single_pass_llm",
            model_provider="ollama",
            model_name="fake-model",
        )
    )
    first = EvaluationScenario(
        scenario_id="scenario-1",
        request=PartyRequest(
            location="Boston",
            event_date=date(2026, 9, 1),
            guest_count=10,
            total_budget=Decimal("500"),
        ),
        expected_feasibility=FeasibilityOutcome.FEASIBLE,
        scenario_category=ScenarioCategory.FEASIBLE,
        complexity=ComplexityMetadata(),
        dataset_split=DatasetSplit.DEVELOPMENT,
    )
    second = EvaluationScenario(
        scenario_id="scenario-2",
        request=PartyRequest(
            location="Boston",
            event_date=date(2026, 9, 1),
            guest_count=10,
            total_budget=Decimal("500"),
        ),
        expected_feasibility=FeasibilityOutcome.FEASIBLE,
        scenario_category=ScenarioCategory.FEASIBLE,
        complexity=ComplexityMetadata(),
        dataset_split=DatasetSplit.DEVELOPMENT,
    )
    responses = [
        GenerationResponse(
            text="{}",
            structured_output={"wrong": "shape"},
        ),
        GenerationResponse(
            text="",
            structured_output={
                "resources": [
                    {
                        "resource_id": "venue-1",
                        "name": "Venue One",
                        "location": "Boston",
                        "price": "100",
                        "capacity": 20,
                        "availability": [],
                        "age_restrictions": None,
                        "accessibility_attributes": [],
                        "category": "venue",
                    }
                ],
                "claimed_total_cost": "100",
                "assumptions": [],
            },
        ),
    ]
    planner = SinglePassLLMPlanner(FakeLLMProvider(responses))
    clock_values = iter((1.0, 1.01, 1.02, 1.07, 2.0, 2.03, 2.04, 2.10))

    result = run_baseline_experiment(
        (first, second),
        FakeDeterministicPlanner(PlannerResult(candidates=())),
        planner,
        metadata=metadata,
        dataset_split=DatasetSplit.DEVELOPMENT,
        clock=lambda: next(clock_values),
    )

    assert result.comparison.single_pass_llm.feasibility_accuracy == 0.5
    assert result.comparison.single_pass_llm.structured_output_validity == 0.5
    assert result.comparison.single_pass_llm.hard_constraint_validity == 0.5
    assert result.comparison.single_pass_llm.mean_latency_ms == pytest.approx(55.0)
    assert result.comparison.single_pass_llm.median_latency_ms == pytest.approx(55.0)
    assert result.single_pass_scenarios[0].failure_labels == (
        BaselineFailureLabel.SCHEMA_INVALID,
        BaselineFailureLabel.FEASIBILITY_MISCLASSIFICATION,
    )
    assert result.single_pass_scenarios[1].failure_labels == (
        BaselineFailureLabel.HALLUCINATED_RESOURCE,
    )


def test_run_baseline_experiment_scores_feasibility_only_for_valid_plans() -> None:
    metadata = ExperimentResultMetadata(
        config=ExperimentConfig(
            experiment_id="baseline-v0.1-development-20260810T120000Z",
            code_commit_sha="abc123",
            dataset_version="v0.1",
            architecture_variant="deterministic_plus_single_pass_llm",
            model_provider="ollama",
            model_name="fake-model",
        )
    )
    scenario = _scenario()
    responses = [
        GenerationResponse(
            text="",
            structured_output={
                "resources": [
                    {
                        "resource_id": "venue-1",
                        "name": "Venue One",
                        "location": "Boston",
                        "price": "100",
                        "capacity": 20,
                        "availability": [],
                        "age_restrictions": None,
                        "accessibility_attributes": [],
                        "category": "venue",
                    }
                ],
                "claimed_total_cost": "100",
                "assumptions": [],
            },
        ),
        GenerationResponse(
            text="",
            structured_output={
                "resources": [
                    {
                        "resource_id": "venue-2",
                        "name": "Venue Two",
                        "location": "Cambridge",
                        "price": "100",
                        "capacity": 20,
                        "availability": [],
                        "age_restrictions": None,
                        "accessibility_attributes": [],
                        "category": "venue",
                    }
                ],
                "claimed_total_cost": "100",
                "assumptions": [],
            },
        ),
        GenerationResponse(text="{}", structured_output=None),
    ]
    planner = SinglePassLLMPlanner(FakeLLMProvider(responses))
    result = run_baseline_experiment(
        (scenario, scenario, scenario),
        FakeDeterministicPlanner(PlannerResult(candidates=())),
        planner,
        metadata=metadata,
        dataset_split=DatasetSplit.DEVELOPMENT,
    )

    first, second, third = result.single_pass_scenarios
    assert first.feasibility_correct is True
    assert second.feasibility_correct is False
    assert third.feasibility_correct is False
    assert result.comparison.single_pass_llm.feasibility_accuracy == pytest.approx(1 / 3)
    assert result.comparison.single_pass_llm.structured_output_validity == pytest.approx(2 / 3)
