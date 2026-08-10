from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from partypilot.application.deterministic_planner import PlannerResult
from partypilot.application.evaluation_runner import (
    DeterministicEvaluationRunner,
    ScenarioEvaluation,
    calculate_metrics,
    render_markdown_summary,
    save_evaluation_reports,
)
from partypilot.domain.evaluation import (
    ComplexityMetadata,
    DatasetSplit,
    EvaluationScenario,
    ScenarioCategory,
)
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest


class FakePlanner:
    def __init__(self, result: PlannerResult) -> None:
        self.result = result

    def plan(self, request: PartyRequest) -> PlannerResult:
        return self.result


def scenario(
    expected: FeasibilityOutcome = FeasibilityOutcome.NO_FEASIBLE_PLAN,
) -> EvaluationScenario:
    return EvaluationScenario(
        scenario_id="scenario-1",
        request=PartyRequest(
            location="Brooklyn, NY",
            event_date=date(2026, 9, 20),
            guest_count=12,
            total_budget=Decimal("1000"),
        ),
        expected_feasibility=expected,
        scenario_category=ScenarioCategory.BUDGET,
        complexity=ComplexityMetadata(hard_constraint_count=1),
        dataset_split=DatasetSplit.DEVELOPMENT,
    )


def test_runner_maps_empty_result_to_no_feasible_plan_and_records_latency() -> None:
    times = iter((10.0, 10.025))
    runner = DeterministicEvaluationRunner(
        FakePlanner(PlannerResult(candidates=())),
        clock=lambda: next(times),
    )

    result = runner.run((scenario(),))

    assert result.metrics.feasibility_accuracy == 1.0
    assert result.metrics.no_feasible_plan_accuracy == 1.0
    assert result.metrics.hard_constraint_validity == 1.0
    assert result.scenarios[0].latency_ms == 25.000000000000355


def test_runner_maps_unresolved_result_to_human_review() -> None:
    times = iter((1.0, 1.0))
    runner = DeterministicEvaluationRunner(
        FakePlanner(PlannerResult(candidates=(), unresolved_request_constraints=("allergies",))),
        clock=lambda: next(times),
    )

    result = runner.run((scenario(FeasibilityOutcome.HUMAN_REVIEW_REQUIRED),))

    assert result.scenarios[0].predicted_outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    assert result.metrics.feasibility_accuracy == 1.0


def test_calculate_metrics_handles_empty_results() -> None:
    metrics = calculate_metrics(())

    assert metrics.scenario_count == 0
    assert metrics.feasibility_accuracy == 0.0
    assert metrics.hard_constraint_validity == 0.0
    assert metrics.no_feasible_plan_accuracy is None
    assert metrics.mean_latency_ms == 0.0


def test_calculate_metrics_uses_no_plan_scenarios_as_denominator() -> None:
    results = (
        ScenarioEvaluation(
            scenario_id="a",
            expected_outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN,
            predicted_outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN,
            feasibility_correct=True,
            hard_constraints_valid=True,
            latency_ms=10,
        ),
        ScenarioEvaluation(
            scenario_id="b",
            expected_outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN,
            predicted_outcome=FeasibilityOutcome.FEASIBLE,
            feasibility_correct=False,
            hard_constraints_valid=False,
            latency_ms=20,
        ),
    )

    metrics = calculate_metrics(results)

    assert metrics.feasibility_accuracy == 0.5
    assert metrics.hard_constraint_validity == 0.5
    assert metrics.no_feasible_plan_accuracy == 0.5
    assert metrics.mean_latency_ms == 15.0


def test_reports_save_json_and_markdown(tmp_path: Path) -> None:
    result = DeterministicEvaluationRunner(
        FakePlanner(PlannerResult(candidates=())),
        clock=lambda: 1.0,
    ).run((scenario(),))

    json_path, markdown_path = save_evaluation_reports(result, tmp_path)

    assert '"feasibility_accuracy": 1.0' in json_path.read_text(encoding="utf-8")
    summary = markdown_path.read_text(encoding="utf-8")
    assert "# Deterministic Baseline Evaluation" in summary
    assert "Feasibility accuracy: 1.000" in summary
    assert render_markdown_summary(result) == summary
