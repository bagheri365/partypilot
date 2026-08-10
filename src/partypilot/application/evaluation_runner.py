"""Deterministic evaluation runner for PartyPilot baselines."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application.budget_validation import CostComponent, validate_budget
from partypilot.application.candidate_filtering import CandidateRequirements, filter_candidates
from partypilot.application.deterministic_planner import PlanCandidate, PlannerResult
from partypilot.domain.evaluation import EvaluationScenario
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest
from partypilot.domain.resources import AccessibilityAttribute
from partypilot.domain.temporal import Duration, TimeWindow

Clock = Callable[[], float]


class ScenarioPlanner(Protocol):
    """Minimal planner contract required by the evaluation runner."""

    def plan(self, request: PartyRequest) -> PlannerResult:
        """Return a planner result for one validated request."""
        ...


class ScenarioEvaluation(BaseModel):
    """Machine-readable result for one benchmark scenario."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    expected_outcome: FeasibilityOutcome
    predicted_outcome: FeasibilityOutcome
    feasibility_correct: bool
    hard_constraints_valid: bool
    latency_ms: float = Field(ge=0)
    candidate_resource_ids: tuple[tuple[str, ...], ...] = ()


class EvaluationMetrics(BaseModel):
    """Aggregate objective metrics for a deterministic evaluation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_count: int = Field(ge=0)
    feasibility_accuracy: float = Field(ge=0, le=1)
    hard_constraint_validity: float = Field(ge=0, le=1)
    no_feasible_plan_accuracy: float | None = Field(default=None, ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)


class EvaluationRunResult(BaseModel):
    """Complete deterministic benchmark output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metrics: EvaluationMetrics
    scenarios: tuple[ScenarioEvaluation, ...]


class DeterministicEvaluationRunner:
    """Evaluate a deterministic planner against labeled benchmark scenarios."""

    def __init__(
        self,
        planner: ScenarioPlanner,
        *,
        clock: Clock = perf_counter,
        event_duration: Duration | None = None,
    ) -> None:
        self._planner = planner
        self._clock = clock
        self._event_duration = event_duration or Duration.hours(2)

    def run(self, scenarios: Sequence[EvaluationScenario]) -> EvaluationRunResult:
        """Run all scenarios and calculate objective aggregate metrics."""
        results: list[ScenarioEvaluation] = []
        for scenario in scenarios:
            started = self._clock()
            planner_result = self._planner.plan(scenario.request)
            elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)
            predicted = _predicted_outcome(planner_result)
            hard_valid = all(
                _candidate_satisfies_request(candidate, scenario, self._event_duration)
                for candidate in planner_result.candidates
            )
            results.append(
                ScenarioEvaluation(
                    scenario_id=scenario.scenario_id,
                    expected_outcome=scenario.expected_feasibility,
                    predicted_outcome=predicted,
                    feasibility_correct=predicted is scenario.expected_feasibility,
                    hard_constraints_valid=hard_valid,
                    latency_ms=elapsed_ms,
                    candidate_resource_ids=tuple(
                        candidate.resource_ids for candidate in planner_result.candidates
                    ),
                )
            )

        return EvaluationRunResult(
            metrics=calculate_metrics(results),
            scenarios=tuple(results),
        )


def calculate_metrics(results: Sequence[ScenarioEvaluation]) -> EvaluationMetrics:
    """Calculate deterministic aggregate metrics from scenario-level results."""
    if not results:
        return EvaluationMetrics(
            scenario_count=0,
            feasibility_accuracy=0.0,
            hard_constraint_validity=0.0,
            no_feasible_plan_accuracy=None,
            mean_latency_ms=0.0,
        )

    count = len(results)
    no_plan_expected = [
        result
        for result in results
        if result.expected_outcome is FeasibilityOutcome.NO_FEASIBLE_PLAN
    ]
    no_plan_accuracy = None
    if no_plan_expected:
        no_plan_accuracy = sum(
            result.predicted_outcome is FeasibilityOutcome.NO_FEASIBLE_PLAN
            for result in no_plan_expected
        ) / len(no_plan_expected)

    return EvaluationMetrics(
        scenario_count=count,
        feasibility_accuracy=sum(result.feasibility_correct for result in results) / count,
        hard_constraint_validity=sum(result.hard_constraints_valid for result in results) / count,
        no_feasible_plan_accuracy=no_plan_accuracy,
        mean_latency_ms=sum(result.latency_ms for result in results) / count,
    )


def save_evaluation_reports(
    result: EvaluationRunResult,
    output_directory: Path,
    *,
    stem: str = "deterministic_baseline",
) -> tuple[Path, Path]:
    """Save machine-readable JSON and a concise Markdown summary."""
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / f"{stem}.json"
    markdown_path = output_directory / f"{stem}.md"
    json_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_summary(result), encoding="utf-8")
    return json_path, markdown_path


def render_markdown_summary(result: EvaluationRunResult) -> str:
    """Render a concise, reproducible Markdown summary of measured metrics."""
    metrics = result.metrics
    no_plan = (
        "n/a"
        if metrics.no_feasible_plan_accuracy is None
        else f"{metrics.no_feasible_plan_accuracy:.3f}"
    )
    return (
        "# Deterministic Baseline Evaluation\n\n"
        f"- Scenarios: {metrics.scenario_count}\n"
        f"- Feasibility accuracy: {metrics.feasibility_accuracy:.3f}\n"
        f"- Hard-constraint validity: {metrics.hard_constraint_validity:.3f}\n"
        f"- No-feasible-plan accuracy: {no_plan}\n"
        f"- Mean latency: {metrics.mean_latency_ms:.3f} ms\n"
    )


def _predicted_outcome(result: PlannerResult) -> FeasibilityOutcome:
    if result.feasible:
        return FeasibilityOutcome.FEASIBLE
    if result.unresolved_request_constraints:
        return FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    return FeasibilityOutcome.NO_FEASIBLE_PLAN


def _candidate_satisfies_request(
    candidate: PlanCandidate,
    scenario: EvaluationScenario,
    event_duration: Duration,
) -> bool:
    request = scenario.request
    event_window: TimeWindow | None = None
    if request.event_time is not None:
        start = datetime.combine(request.event_date, request.event_time)
        event_window = TimeWindow(start=start, end=start + event_duration.value)

    accessibility: set[AccessibilityAttribute] = set()
    for need in request.accessibility_needs:
        normalized = need.strip().casefold().replace(" ", "_")
        try:
            accessibility.add(AccessibilityAttribute(normalized))
        except ValueError:
            return False

    requirements = CandidateRequirements(
        location=request.location,
        guest_count=request.guest_count,
        child_age=request.child_age,
        child_age_range=request.child_age_range,
        availability=event_window,
        accessibility=frozenset(accessibility),
    )
    if filter_candidates(candidate.resources, requirements).rejected:
        return False

    # If no explicit time is present, resources must still be available on the event date.
    if event_window is None and any(
        not any(
            window.start.date() <= request.event_date <= window.end.date()
            for window in resource.availability
        )
        for resource in candidate.resources
    ):
        return False

    components = tuple(
        CostComponent(
            component_id=resource.resource_id,
            description=resource.name,
            amount=resource.price,
        )
        for resource in candidate.resources
    )
    return validate_budget(request.total_budget, components).within_budget
