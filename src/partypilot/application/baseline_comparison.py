"""Objective comparison of PartyPilot baseline planners."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application.deterministic_planner import PlannerResult
from partypilot.application.evaluation_runner import (
    ScenarioEvaluation,
    calculate_metrics,
)
from partypilot.application.single_pass_llm_planner import (
    SinglePassLLMResult,
)
from partypilot.domain.evaluation import EvaluationScenario
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest

Clock = Callable[[], float]


class DeterministicBaseline(Protocol):
    def plan(self, request: PartyRequest) -> PlannerResult: ...


class SinglePassBaseline(Protocol):
    def plan(self, request: PartyRequest) -> SinglePassLLMResult: ...


class BaselineObjectiveMetrics(BaseModel):
    """Objective metrics shared by baseline variants."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_count: int = Field(ge=0)
    feasibility_accuracy: float = Field(ge=0, le=1)
    hard_constraint_validity: float = Field(ge=0, le=1)
    structured_output_validity: float = Field(ge=0, le=1)
    unsupported_claim_rate: float | None = Field(default=None, ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)
    median_latency_ms: float = Field(ge=0)
    total_input_tokens: int | None = Field(default=None, ge=0)
    total_output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class BaselineComparisonResult(BaseModel):
    """Machine-readable comparison; subjective quality is intentionally absent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deterministic: BaselineObjectiveMetrics
    single_pass_llm: BaselineObjectiveMetrics


class BaselineComparisonRunner:
    """Compare deterministic and single-pass LLM baselines on objective metrics."""

    def __init__(
        self,
        deterministic: DeterministicBaseline,
        single_pass_llm: SinglePassBaseline,
        *,
        clock: Clock = perf_counter,
    ) -> None:
        self._deterministic = deterministic
        self._single_pass_llm = single_pass_llm
        self._clock = clock

    def run(self, scenarios: Sequence[EvaluationScenario]) -> BaselineComparisonResult:
        deterministic_results: list[ScenarioEvaluation] = []
        deterministic_latencies: list[float] = []
        llm_feasibility_correct = 0
        llm_hard_valid = 0
        llm_structured_valid = 0
        llm_latencies: list[float] = []
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        saw_input_tokens = False
        saw_output_tokens = False
        saw_total_tokens = False

        for scenario in scenarios:
            started = self._clock()
            deterministic_result = self._deterministic.plan(scenario.request)
            deterministic_latency = max(0.0, (self._clock() - started) * 1000.0)
            deterministic_latencies.append(deterministic_latency)
            predicted = _deterministic_outcome(deterministic_result)
            deterministic_results.append(
                ScenarioEvaluation(
                    scenario_id=scenario.scenario_id,
                    expected_outcome=scenario.expected_feasibility,
                    predicted_outcome=predicted,
                    feasibility_correct=predicted is scenario.expected_feasibility,
                    # Deterministic PlannerResult candidates are emitted only after the
                    # constraint engine accepts them; empty results contain no invalid plan.
                    hard_constraints_valid=True,
                    latency_ms=deterministic_latency,
                    candidate_resource_ids=tuple(
                        candidate.resource_ids for candidate in deterministic_result.candidates
                    ),
                )
            )

            started = self._clock()
            llm_result = self._single_pass_llm.plan(scenario.request)
            llm_latency = max(0.0, (self._clock() - started) * 1000.0)
            llm_latencies.append(llm_latency)
            llm_predicted = _llm_outcome(llm_result)
            structured_valid = llm_result.plan is not None
            llm_structured_valid += int(structured_valid)
            llm_feasibility_correct += int(
                structured_valid and llm_predicted is scenario.expected_feasibility
            )
            llm_hard_valid += int(
                structured_valid
                and llm_result.validation is not None
                and llm_result.validation.feasible
            )
            if llm_result.usage is not None:
                if llm_result.usage.input_tokens is not None:
                    saw_input_tokens = True
                    input_tokens += llm_result.usage.input_tokens
                if llm_result.usage.output_tokens is not None:
                    saw_output_tokens = True
                    output_tokens += llm_result.usage.output_tokens
                if llm_result.usage.total_tokens is not None:
                    saw_total_tokens = True
                    total_tokens += llm_result.usage.total_tokens

        deterministic_metrics = calculate_metrics(deterministic_results)
        count = len(scenarios)
        return BaselineComparisonResult(
            deterministic=BaselineObjectiveMetrics(
                scenario_count=count,
                feasibility_accuracy=deterministic_metrics.feasibility_accuracy,
                hard_constraint_validity=deterministic_metrics.hard_constraint_validity,
                structured_output_validity=1.0 if count else 0.0,
                unsupported_claim_rate=None,
                mean_latency_ms=sum(deterministic_latencies) / count if count else 0.0,
                median_latency_ms=median(deterministic_latencies) if count else 0.0,
            ),
            single_pass_llm=BaselineObjectiveMetrics(
                scenario_count=count,
                feasibility_accuracy=llm_feasibility_correct / count if count else 0.0,
                hard_constraint_validity=llm_hard_valid / count if count else 0.0,
                structured_output_validity=llm_structured_valid / count if count else 0.0,
                unsupported_claim_rate=None,
                mean_latency_ms=(sum(llm_latencies) / count if count else 0.0),
                median_latency_ms=median(llm_latencies) if count else 0.0,
                total_input_tokens=input_tokens if saw_input_tokens else None,
                total_output_tokens=output_tokens if saw_output_tokens else None,
                total_tokens=total_tokens if saw_total_tokens else None,
            ),
        )


def _deterministic_outcome(result: PlannerResult) -> FeasibilityOutcome:
    if result.feasible:
        return FeasibilityOutcome.FEASIBLE
    if result.unresolved_request_constraints:
        return FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    return FeasibilityOutcome.NO_FEASIBLE_PLAN


def _llm_outcome(result: SinglePassLLMResult) -> FeasibilityOutcome:
    if result.validation is not None and result.validation.feasible and result.plan is not None:
        return FeasibilityOutcome.FEASIBLE
    if result.validation is not None and result.validation.unresolved_constraint_ids:
        return FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    return FeasibilityOutcome.NO_FEASIBLE_PLAN


def save_baseline_comparison_reports(
    result: BaselineComparisonResult,
    output_directory: Path,
    *,
    stem: str = "baseline_comparison",
) -> tuple[Path, Path]:
    """Save machine-readable JSON and a Markdown objective-metrics comparison."""
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / f"{stem}.json"
    markdown_path = output_directory / f"{stem}.md"
    json_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_baseline_comparison_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def render_baseline_comparison_markdown(result: BaselineComparisonResult) -> str:
    """Render objective metrics only; subjective plan quality is intentionally separate."""

    def line(name: str, metrics: BaselineObjectiveMetrics) -> str:
        unsupported = (
            "n/a"
            if metrics.unsupported_claim_rate is None
            else f"{metrics.unsupported_claim_rate:.3f}"
        )
        tokens = "n/a" if metrics.total_tokens is None else str(metrics.total_tokens)
        return (
            f"| {name} | {metrics.feasibility_accuracy:.3f} | "
            f"{metrics.hard_constraint_validity:.3f} | "
            f"{metrics.structured_output_validity:.3f} | {unsupported} | "
            f"{metrics.mean_latency_ms:.3f} | {metrics.median_latency_ms:.3f} | {tokens} |"
        )

    return (
        "# Baseline Comparison\n\n"
        "Objective metrics only; subjective plan quality is not included in this report.\n\n"
        "| Variant | Feasibility accuracy | Hard-constraint validity | "
        "Structured-output validity | Unsupported-claim rate | "
        "Mean latency (ms) | Median latency (ms) | Total tokens |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        f"{line('Deterministic baseline', result.deterministic)}\n"
        f"{line('Single-pass LLM baseline', result.single_pass_llm)}\n"
    )
