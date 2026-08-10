from datetime import date
from decimal import Decimal
from pathlib import Path

from partypilot.application.baseline_comparison import (
    BaselineComparisonRunner,
    render_baseline_comparison_markdown,
    save_baseline_comparison_reports,
)
from partypilot.application.deterministic_planner import PlannerResult
from partypilot.application.single_pass_llm_planner import (
    LLMPlanFailureCategory,
    SinglePassLLMResult,
)
from partypilot.domain.evaluation import (
    ComplexityMetadata,
    DatasetSplit,
    EvaluationScenario,
    ScenarioCategory,
)
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest
from partypilot.ports.llm_provider import UsageMetadata


class DeterministicFake:
    def __init__(self, result: PlannerResult) -> None:
        self.result = result

    def plan(self, request: PartyRequest) -> PlannerResult:
        return self.result


class LLMFake:
    def __init__(self, result: SinglePassLLMResult) -> None:
        self.result = result

    def plan(self, request: PartyRequest) -> SinglePassLLMResult:
        return self.result


def scenario(
    expected: FeasibilityOutcome = FeasibilityOutcome.NO_FEASIBLE_PLAN,
) -> EvaluationScenario:
    return EvaluationScenario(
        scenario_id="s1",
        request=PartyRequest(
            location="Boston",
            event_date=date(2026, 9, 1),
            guest_count=10,
            total_budget=Decimal("500"),
        ),
        expected_feasibility=expected,
        scenario_category=ScenarioCategory.BUDGET,
        complexity=ComplexityMetadata(),
        dataset_split=DatasetSplit.DEVELOPMENT,
    )


def test_comparison_calculates_objective_metrics_and_usage() -> None:
    times = iter((1.0, 1.01, 2.0, 2.02))
    runner = BaselineComparisonRunner(
        DeterministicFake(PlannerResult(candidates=())),
        LLMFake(
            SinglePassLLMResult(
                plan=None,
                validation=None,
                failure_categories=(LLMPlanFailureCategory.SCHEMA_INVALID,),
                usage=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        ),
        clock=lambda: next(times),
    )
    result = runner.run((scenario(),))
    assert result.deterministic.feasibility_accuracy == 1.0
    assert result.deterministic.structured_output_validity == 1.0
    assert result.single_pass_llm.feasibility_accuracy == 0.0
    assert result.single_pass_llm.structured_output_validity == 0.0
    assert result.single_pass_llm.hard_constraint_validity == 0.0
    assert result.single_pass_llm.median_latency_ms == result.single_pass_llm.mean_latency_ms
    assert result.single_pass_llm.total_tokens == 15
    assert result.single_pass_llm.mean_latency_ms > result.deterministic.mean_latency_ms


def test_unsupported_claims_are_measured_separately() -> None:
    runner = BaselineComparisonRunner(
        DeterministicFake(PlannerResult(candidates=())),
        LLMFake(
            SinglePassLLMResult(
                plan=None,
                validation=None,
                failure_categories=(LLMPlanFailureCategory.SCHEMA_INVALID,),
            )
        ),
        clock=lambda: 1.0,
    )
    result = runner.run((scenario(),))
    assert result.single_pass_llm.unsupported_claim_rate is None


def test_empty_comparison_has_no_fake_token_or_claim_metrics() -> None:
    result = BaselineComparisonRunner(
        DeterministicFake(PlannerResult(candidates=())),
        LLMFake(SinglePassLLMResult(plan=None, validation=None, failure_categories=())),
    ).run(())
    assert result.deterministic.scenario_count == 0
    assert result.single_pass_llm.unsupported_claim_rate is None
    assert result.single_pass_llm.total_tokens is None


def test_comparison_reports_are_machine_readable_and_markdown(tmp_path: Path) -> None:
    result = BaselineComparisonRunner(
        DeterministicFake(PlannerResult(candidates=())),
        LLMFake(SinglePassLLMResult(plan=None, validation=None, failure_categories=())),
        clock=lambda: 1.0,
    ).run((scenario(),))
    json_path, md_path = save_baseline_comparison_reports(result, tmp_path)
    assert '"single_pass_llm"' in json_path.read_text(encoding="utf-8")
    markdown = md_path.read_text(encoding="utf-8")
    assert "# Baseline Comparison" in markdown
    assert "subjective plan quality is not included" in markdown
    assert render_baseline_comparison_markdown(result) == markdown
