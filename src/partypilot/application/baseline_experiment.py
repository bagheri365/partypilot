"""Canonical v0.1 baseline experiment composition and artifact writing."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application.baseline_comparison import (
    BaselineComparisonResult,
    BaselineObjectiveMetrics,
    render_baseline_comparison_markdown,
)
from partypilot.application.baseline_metrics import (
    BaselineFailureLabel,
    classify_single_pass_failure_labels,
)
from partypilot.application.deterministic_planner import PlannerResult
from partypilot.application.evaluation_runner import (
    EvaluationRunResult,
    ScenarioEvaluation,
    calculate_metrics,
)
from partypilot.application.single_pass_llm_planner import (
    LLMPlanFailureCategory,
    SinglePassLLMPlanner,
    SinglePassLLMResult,
)
from partypilot.domain.evaluation import DatasetSplit, EvaluationScenario
from partypilot.domain.experiment import ExperimentResultMetadata
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest

Clock = Callable[[], float]


class SinglePassScenarioResult(BaseModel):
    """Per-scenario result for the single-pass LLM baseline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    expected_outcome: FeasibilityOutcome
    predicted_outcome: FeasibilityOutcome
    feasibility_correct: bool
    hard_constraints_valid: bool
    structured_output_valid: bool
    unsupported_claim: bool | None = None
    latency_ms: float = Field(ge=0)
    failure_labels: tuple[BaselineFailureLabel, ...]
    failure_categories: tuple[LLMPlanFailureCategory, ...]
    errors: tuple[str, ...]
    usage_input_tokens: int | None = Field(default=None, ge=0)
    usage_output_tokens: int | None = Field(default=None, ge=0)
    usage_total_tokens: int | None = Field(default=None, ge=0)


class BaselineExperimentResult(BaseModel):
    """Full v0.1 experiment artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: ExperimentResultMetadata
    dataset_split: DatasetSplit
    comparison: BaselineComparisonResult
    deterministic: EvaluationRunResult
    single_pass_scenarios: tuple[SinglePassScenarioResult, ...]


class DeterministicBaseline(Protocol):
    def plan(self, request: PartyRequest) -> PlannerResult: ...


def run_baseline_experiment(
    scenarios: Sequence[EvaluationScenario],
    deterministic: DeterministicBaseline,
    single_pass_llm: SinglePassLLMPlanner,
    *,
    metadata: ExperimentResultMetadata,
    dataset_split: DatasetSplit,
    clock: Clock = perf_counter,
) -> BaselineExperimentResult:
    """Run the canonical v0.1 baseline comparison once and retain per-scenario output."""
    deterministic_results: list[ScenarioEvaluation] = []
    single_pass_scenarios: list[SinglePassScenarioResult] = []
    deterministic_latencies: list[float] = []
    llm_latencies: list[float] = []
    llm_feasibility_correct = 0
    llm_hard_valid = 0
    llm_structured_valid = 0
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    saw_input_tokens = False
    saw_output_tokens = False
    saw_total_tokens = False

    for scenario in scenarios:
        started = clock()
        deterministic_result = deterministic.plan(scenario.request)
        deterministic_latency = max(0.0, (clock() - started) * 1000.0)
        deterministic_latencies.append(deterministic_latency)
        deterministic_predicted = _deterministic_outcome(deterministic_result)
        deterministic_evaluation = ScenarioEvaluation(
            scenario_id=scenario.scenario_id,
            expected_outcome=scenario.expected_feasibility,
            predicted_outcome=deterministic_predicted,
            feasibility_correct=deterministic_predicted is scenario.expected_feasibility,
            hard_constraints_valid=True,
            latency_ms=deterministic_latency,
            candidate_resource_ids=tuple(
                candidate.resource_ids for candidate in deterministic_result.candidates
            ),
        )
        deterministic_results.append(deterministic_evaluation)

        started = clock()
        llm_result = single_pass_llm.plan(scenario.request)
        llm_latency = max(0.0, (clock() - started) * 1000.0)
        llm_latencies.append(llm_latency)
        llm_predicted = _llm_outcome(llm_result)
        structured_valid = llm_result.plan is not None
        llm_structured_valid += int(structured_valid)
        llm_feasibility_correct += int(
            structured_valid and llm_predicted is scenario.expected_feasibility
        )
        hard_valid = (
            structured_valid
            and llm_result.validation is not None
            and llm_result.validation.feasible
        )
        llm_hard_valid += int(hard_valid)
        failure_labels = classify_single_pass_failure_labels(
            llm_result,
            scenario.expected_feasibility,
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

        single_pass_scenarios.append(
            SinglePassScenarioResult(
                scenario_id=scenario.scenario_id,
                expected_outcome=scenario.expected_feasibility,
                predicted_outcome=llm_predicted,
                feasibility_correct=(
                    structured_valid and llm_predicted is scenario.expected_feasibility
                ),
                hard_constraints_valid=hard_valid,
                structured_output_valid=structured_valid,
                unsupported_claim=None,
                latency_ms=llm_latency,
                failure_labels=failure_labels,
                failure_categories=llm_result.failure_categories,
                errors=llm_result.errors,
                usage_input_tokens=(
                    llm_result.usage.input_tokens if llm_result.usage is not None else None
                ),
                usage_output_tokens=(
                    llm_result.usage.output_tokens if llm_result.usage is not None else None
                ),
                usage_total_tokens=(
                    llm_result.usage.total_tokens if llm_result.usage is not None else None
                ),
            )
        )

    deterministic_metrics = calculate_metrics(deterministic_results)
    count = len(scenarios)
    comparison = BaselineComparisonResult(
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
            mean_latency_ms=sum(llm_latencies) / count if count else 0.0,
            median_latency_ms=median(llm_latencies) if count else 0.0,
            total_input_tokens=input_tokens if saw_input_tokens else None,
            total_output_tokens=output_tokens if saw_output_tokens else None,
            total_tokens=total_tokens if saw_total_tokens else None,
        ),
    )
    return BaselineExperimentResult(
        metadata=metadata,
        dataset_split=dataset_split,
        comparison=comparison,
        deterministic=EvaluationRunResult(
            metrics=deterministic_metrics,
            scenarios=tuple(deterministic_results),
        ),
        single_pass_scenarios=tuple(single_pass_scenarios),
    )


def save_baseline_experiment_reports(
    result: BaselineExperimentResult,
    output_directory: Path,
    *,
    stem: str = "baseline_experiment",
) -> tuple[Path, Path]:
    """Save machine-readable and human-readable experiment artifacts."""
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / f"{stem}.json"
    markdown_path = output_directory / f"{stem}.md"
    json_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_baseline_experiment_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def render_baseline_experiment_markdown(result: BaselineExperimentResult) -> str:
    """Render a concise reproducible summary for the canonical baseline experiment."""
    metadata = result.metadata.config
    commit_sha = metadata.code_commit_sha or "unavailable"
    working_tree_dirty = (
        str(metadata.working_tree_dirty) if metadata.working_tree_dirty is not None else "unknown"
    )
    git_metadata_error = metadata.git_metadata_error or "none"
    prompt_version = metadata.prompt_version or "unavailable"
    return (
        "# PartyPilot v0.1 Baseline Experiment\n\n"
        f"- Experiment ID: {metadata.experiment_id}\n"
        f"- Dataset split: {result.dataset_split.value}\n"
        f"- Timestamp: {metadata.timestamp.isoformat()}\n"
        f"- Commit SHA: {commit_sha}\n"
        f"- Working tree dirty: {working_tree_dirty}\n"
        f"- Git metadata error: {git_metadata_error}\n"
        f"- Model provider: {metadata.model_provider or 'n/a'}\n"
        f"- Model name: {metadata.model_name or 'n/a'}\n"
        f"- Prompt version: {prompt_version}\n\n"
        f"{render_baseline_comparison_markdown(result.comparison)}"
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
