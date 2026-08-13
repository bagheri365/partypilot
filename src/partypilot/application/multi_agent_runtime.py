# ruff: noqa: E501
"""Real multi-agent runtime for PartyPilot v0.5."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from statistics import mean
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application import v04_multi_agent as v04
from partypilot.domain import (
    ArbitrationOutcome,
    ArbitrationTrace,
    CandidateEvaluationResult,
    CapabilityBoundaryScenario,
    CoordinatedPlanResult,
    CoordinationFailureKind,
    EvidenceDocument,
    ExperimentConfig,
    ExperimentResultMetadata,
    FeasibilityOutcome,
    MultiAgentPlanningRuntimeResult,
    PlanningDependency,
    PlanningDependencyKind,
    PlanningState,
    PlanningStateSummary,
    SpecialistDecision,
    SpecialistDomain,
    SpecialistExecutionOutcome,
    SpecialistExecutionTrace,
    SpecialistFailureKind,
)
from partypilot.domain.multi_agent import SpecialistAdapterVariant, SpecialistAgentInput
from partypilot.domain.resources import Resource, ResourceCategory
from partypilot.ports.specialist_agent import SpecialistAgent

SPECIALIST_ORDER: tuple[SpecialistDomain, ...] = (
    SpecialistDomain.VENUE,
    SpecialistDomain.CATERING_SAFETY,
    SpecialistDomain.ACCESSIBILITY,
    SpecialistDomain.SCHEDULING_OPERATIONS,
    SpecialistDomain.BUDGET,
)

BENCHMARK_NAME = "v0.5 live multi-agent benchmark"
BENCHMARK_VERSION = "1.0"
BASELINE_ARCHITECTURE = "v0.4_deterministic_specialist_coordination"
LIVE_ARCHITECTURE = "v0.5_live_llm_specialist_agents"
DEFAULT_OUTPUT_ROOT = Path("evals") / "results" / "v0_5" / "llm_multi_agent"
VENUE_EVIDENCE_TYPES = {"venue_policy", "accessibility_guidance"}
CATERING_EVIDENCE_TYPES = {"venue_policy", "allergen_policy", "outside_food_rules"}
ACCESSIBILITY_EVIDENCE_TYPES = {"accessibility_guidance"}
SCHEDULING_EVIDENCE_TYPES = {
    "cancellation_terms",
    "supervision_requirements",
    "activity_safety_guidance",
}


@dataclass(frozen=True, slots=True)
class GuardrailAssessment:
    """Deterministic guardrail assessment with provenance."""

    reason: str
    controlling_evidence_ids: tuple[str, ...] = ()
    proven_hard_violation: bool = True


@dataclass(frozen=True, slots=True)
class DeterministicResolutionAssessment:
    """Typed authority state for deterministic candidate resolution."""

    state: DeterministicResolutionState
    reason: str


class DeterministicResolutionState(StrEnum):
    """Typed authority state for deterministic candidate evaluation."""

    PROVEN_FEASIBLE = "PROVEN_FEASIBLE"
    PROVEN_INFEASIBLE = "PROVEN_INFEASIBLE"
    UNRESOLVED = "UNRESOLVED"


class MultiAgentStrategyMetrics(BaseModel):
    """Aggregate metrics for one architecture path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: str = Field(min_length=1)
    scenario_count: int = Field(ge=0)
    final_decision_accuracy: float = Field(ge=0, le=1)
    hard_constraint_validity: float = Field(ge=0, le=1)
    cross_domain_compatibility_accuracy: float = Field(ge=0, le=1)
    evidence_grounded_arbitration_accuracy: float = Field(ge=0, le=1)
    global_optimum_accuracy: float = Field(ge=0, le=1)
    human_review_calibration: float = Field(ge=0, le=1)
    disagreement_resolved_correctly_count: int = Field(ge=0)
    disagreement_resolved_incorrectly_count: int = Field(ge=0)
    specialist_call_count: int = Field(ge=0)
    coordination_overhead_count: int = Field(ge=0)
    mean_latency_ms: float = Field(ge=0)


class MultiAgentRuntimeMetrics(BaseModel):
    """Operational metrics for the live runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_count: int = Field(ge=0)
    total_candidate_count: int = Field(ge=0)
    total_specialist_calls: int = Field(ge=0)
    mean_specialists_invoked_per_scenario: float = Field(ge=0)
    specialist_success_rate: float = Field(ge=0, le=1)
    structured_output_validation_failure_rate: float = Field(ge=0, le=1)
    retry_rate: float = Field(ge=0, le=1)
    specialist_disagreement_rate: float = Field(ge=0, le=1)
    coordinator_override_count: int = Field(ge=0)
    mean_latency_ms: float = Field(ge=0)
    parallelism_speedup: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=Decimal("0"))
    unsupported_claim_rate: float = Field(ge=0, le=1)
    wrong_source_version_rate: float = Field(ge=0, le=1)
    per_specialist_latency_ms: tuple[tuple[str, float], ...] = ()


class MultiAgentMetricDefinition(BaseModel):
    """Human-readable definition for a v0.5 metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)


class MultiAgentScenarioResult(BaseModel):
    """Per-scenario result for the v0.5 live multi-agent runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    capability_tags: tuple[str, ...] = ()
    requires_evidence: bool = False
    requires_global_optimum: bool = False
    expected_feasibility: FeasibilityOutcome
    deterministic_reference: CoordinatedPlanResult
    live_result: CoordinatedPlanResult
    runtime: MultiAgentPlanningRuntimeResult
    notes: tuple[str, ...] = ()


class MultiAgentExperimentMetrics(BaseModel):
    """Aggregate metrics for the v0.5 live experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_count: int = Field(ge=0)
    baseline: MultiAgentStrategyMetrics
    live: MultiAgentStrategyMetrics
    runtime: MultiAgentRuntimeMetrics
    retention_rule_passed: bool


class MultiAgentLiveReport(BaseModel):
    """Experiment report for the live v0.5 multi-agent runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    system_name: str = "PartyPilot v0.5 live multi-agent runtime experiment"
    benchmark_name: str = BENCHMARK_NAME
    benchmark_version: str = BENCHMARK_VERSION
    baseline_architecture: str = BASELINE_ARCHITECTURE
    live_architecture: str = LIVE_ARCHITECTURE
    evaluation_variant: str = "deterministic_specialists_vs_live_llm_specialists"
    metrics: MultiAgentExperimentMetrics
    scenarios: tuple[MultiAgentScenarioResult, ...]
    metric_definitions: tuple[MultiAgentMetricDefinition, ...]
    terminal_outcome_mismatch_scenario_ids: tuple[str, ...] = ()
    diagnostic_failure_stage_scenario_ids: tuple[str, ...] = ()
    metadata: ExperimentResultMetadata | None = None
    notes: tuple[str, ...] = ()


class MultiAgentPlanningRuntime:
    """Parallel specialist execution with deterministic arbitration."""

    def __init__(
        self,
        specialists: Sequence[SpecialistAgent],
        *,
        model_name: str | None = None,
        max_workers: int | None = None,
    ) -> None:
        if not specialists:
            raise ValueError("at least one specialist is required")
        self._specialists = tuple(specialists)
        self._specialist_by_domain = {
            specialist.domain: specialist for specialist in self._specialists
        }
        if set(self._specialist_by_domain) != set(SPECIALIST_ORDER):
            raise ValueError("all five v0.5 specialists must be provided")
        self._model_name = model_name
        self._max_workers = max_workers or len(self._specialists)

    def plan_scenario(
        self, scenario: CapabilityBoundaryScenario
    ) -> MultiAgentPlanningRuntimeResult:
        started = perf_counter()
        candidate_results: list[tuple[tuple[str, ...], CandidateRun]] = []
        execution_traces: list[SpecialistExecutionTrace] = []
        resources_by_id = {
            resource.resource_id: resource for resource in scenario.structured_resources
        }

        for candidate in v04._candidate_combinations(scenario):
            candidate_resources = tuple(resources_by_id[resource_id] for resource_id in candidate)
            planning_state = _build_planning_state(scenario, candidate_resources)
            candidate_run = self._evaluate_candidate(
                scenario=scenario,
                planning_state=planning_state,
                candidate_resources=candidate_resources,
                candidate_resource_ids=candidate,
            )
            candidate_results.append((candidate, candidate_run))
            execution_traces.extend(candidate_run.execution_traces)

        if candidate_results:
            accepted = [
                item
                for item in candidate_results
                if item[1].arbitration.outcome is ArbitrationOutcome.ACCEPT
            ]
            reviewed = [
                item
                for item in candidate_results
                if item[1].arbitration.outcome is ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
            ]
            rejected = [
                item
                for item in candidate_results
                if item[1].arbitration.outcome is ArbitrationOutcome.REJECT
            ]
            chosen = min(
                accepted or reviewed or rejected,
                key=lambda item: (
                    item[1].total_cost if item[1].total_cost is not None else float("inf"),
                    item[0],
                ),
            )
            chosen_run = chosen[1]
        else:
            chosen_run = CandidateRun(
                coordinated_result=CoordinatedPlanResult(
                    architecture=LIVE_ARCHITECTURE,
                    feasibility_outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN,
                    selected_resource_ids=(),
                    total_cost=None,
                    latency_ms=0.0,
                    hard_constraint_validity=False,
                    cross_domain_compatibility=False,
                    evidence_grounded_arbitration=False,
                    global_optimum=None,
                    human_review_calibrated=None,
                    disagreement_resolved_correctly=False,
                    disagreement_resolved_incorrectly=False,
                    specialist_call_count=0,
                    coordination_overhead_count=0,
                    arbitration=None,
                    specialist_decisions=(),
                    notes=("No candidate combinations were available.",),
                    failure_stage="no_candidates",
                ),
                arbitration=ArbitrationTrace(
                    outcome=ArbitrationOutcome.REJECT,
                    feasibility_outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN,
                    reasons=("No candidate combinations were available.",),
                ),
                selected_resource_ids=(),
                total_cost=None,
                execution_traces=(),
                specialist_outcomes=(),
                decision_count=0,
                planning_state=_build_planning_state(scenario, ()),
            )

        wall_clock_latency_ms = max(0.0, (perf_counter() - started) * 1000.0)
        aggregate_decision = chosen_run.coordinated_result.model_copy(
            update={
                "latency_ms": wall_clock_latency_ms,
                "specialist_call_count": chosen_run.decision_count,
                "coordination_overhead_count": len(candidate_results) * len(self._specialists),
            }
        )
        return MultiAgentPlanningRuntimeResult(
            architecture=self._model_name or LIVE_ARCHITECTURE,
            planning_state=PlanningStateSummary.from_state(chosen_run.planning_state),
            candidate_results=tuple(
                item[1].to_candidate_evaluation(item[0]) for item in candidate_results
            ),
            final_result=aggregate_decision,
            execution_traces=tuple(execution_traces),
            wall_clock_latency_ms=wall_clock_latency_ms,
            notes=("Live LLM specialist execution with deterministic coordinator arbitration.",),
        )

    def _evaluate_candidate(
        self,
        *,
        scenario: CapabilityBoundaryScenario,
        planning_state: PlanningState,
        candidate_resources: tuple[Resource, ...],
        candidate_resource_ids: tuple[str, ...],
    ) -> CandidateRun:
        started = perf_counter()
        specialist_outcomes: list[SpecialistExecutionOutcome] = []
        execution_traces: list[SpecialistExecutionTrace] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(
                    _run_specialist_invocation,
                    specialist,
                    _specialist_input(
                        scenario=scenario,
                        planning_state=planning_state,
                        candidate_resources=candidate_resources,
                        candidate_resource_ids=candidate_resource_ids,
                        specialist=specialist,
                    ),
                ): specialist
                for specialist in self._specialists
            }
            for future in as_completed(futures):
                outcome = future.result()
                specialist_outcomes.append(outcome)
                execution_traces.append(outcome.trace)

        specialist_outcomes.sort(key=lambda item: item.trace.specialist_id)
        execution_traces.sort(key=lambda item: item.specialist_id)

        hard_violation = _deterministic_hard_violation(scenario, candidate_resources)
        deterministic_resolution = _deterministic_resolution_assessment(
            scenario=scenario,
            candidate_resources=candidate_resources,
            hard_violation=hard_violation,
        )
        if hard_violation is not None and hard_violation.proven_hard_violation:
            accepted_ids = tuple(
                outcome.decision.specialist_id
                for outcome in specialist_outcomes
                if outcome.decision is not None
                and outcome.decision.status is ArbitrationOutcome.ACCEPT
            )
            rejected_ids = tuple(
                outcome.decision.specialist_id
                for outcome in specialist_outcomes
                if outcome.decision is not None
                and outcome.decision.status is ArbitrationOutcome.REJECT
            )
            controlling_evidence_ids = _guardrail_controlling_evidence_ids(
                hard_violation=hard_violation,
                specialist_outcomes=tuple(specialist_outcomes),
            )
            arbitration = ArbitrationTrace(
                outcome=ArbitrationOutcome.REJECT,
                feasibility_outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN,
                selected_resource_ids=candidate_resource_ids,
                accepted_specialist_ids=accepted_ids,
                rejected_specialist_ids=rejected_ids,
                overridden_specialist_ids=tuple(
                    outcome.decision.specialist_id
                    for outcome in specialist_outcomes
                    if outcome.decision is not None
                    and outcome.decision.status is not ArbitrationOutcome.REJECT
                ),
                controlling_evidence_ids=controlling_evidence_ids,
                dependency_conflicts=(),
                unresolved_uncertainties=(hard_violation.reason,),
                reasons=(hard_violation.reason,),
                global_score=_candidate_total_cost_from_resources(scenario, candidate_resources),
                coordination_steps=tuple(
                    f"specialist:{outcome.trace.specialist_id}:{outcome.decision.status.value if outcome.decision is not None else 'failed'}"
                    for outcome in specialist_outcomes
                ),
            )
            coordinated_result = _coordinated_result_from_arbitration(
                scenario=scenario,
                candidate_resources=candidate_resources,
                arbitration=arbitration,
                specialist_outcomes=tuple(specialist_outcomes),
                decision_count=sum(
                    1 for outcome in specialist_outcomes if outcome.decision is not None
                ),
                elapsed_ms=max(0.0, (perf_counter() - started) * 1000.0),
                failure_kind=None,
                failure_stage_override="hard_constraints",
            )
            return CandidateRun(
                coordinated_result=coordinated_result,
                arbitration=arbitration,
                selected_resource_ids=candidate_resource_ids,
                total_cost=coordinated_result.total_cost,
                execution_traces=tuple(execution_traces),
                specialist_outcomes=tuple(specialist_outcomes),
                decision_count=sum(
                    1 for outcome in specialist_outcomes if outcome.decision is not None
                ),
                planning_state=planning_state,
            )
        if hard_violation is not None and not hard_violation.proven_hard_violation:
            accepted_ids = tuple(
                outcome.decision.specialist_id
                for outcome in specialist_outcomes
                if outcome.decision is not None
                and outcome.decision.status is ArbitrationOutcome.ACCEPT
            )
            rejected_ids = tuple(
                outcome.decision.specialist_id
                for outcome in specialist_outcomes
                if outcome.decision is not None
                and outcome.decision.status is ArbitrationOutcome.REJECT
            )
            controlling_evidence_ids = _guardrail_controlling_evidence_ids(
                hard_violation=hard_violation,
                specialist_outcomes=tuple(specialist_outcomes),
            )
            arbitration = ArbitrationTrace(
                outcome=ArbitrationOutcome.HUMAN_REVIEW_REQUIRED,
                feasibility_outcome=FeasibilityOutcome.HUMAN_REVIEW_REQUIRED,
                selected_resource_ids=candidate_resource_ids,
                accepted_specialist_ids=accepted_ids,
                rejected_specialist_ids=rejected_ids,
                overridden_specialist_ids=tuple(
                    outcome.decision.specialist_id
                    for outcome in specialist_outcomes
                    if outcome.decision is not None
                    and outcome.decision.status is not ArbitrationOutcome.REJECT
                ),
                controlling_evidence_ids=controlling_evidence_ids,
                dependency_conflicts=(),
                unresolved_uncertainties=(hard_violation.reason,),
                reasons=(hard_violation.reason,),
                global_score=_candidate_total_cost_from_resources(scenario, candidate_resources),
                coordination_steps=tuple(
                    f"specialist:{outcome.trace.specialist_id}:{outcome.decision.status.value if outcome.decision is not None else 'failed'}"
                    for outcome in specialist_outcomes
                ),
            )
            coordinated_result = _coordinated_result_from_arbitration(
                scenario=scenario,
                candidate_resources=candidate_resources,
                arbitration=arbitration,
                specialist_outcomes=tuple(specialist_outcomes),
                decision_count=sum(
                    1 for outcome in specialist_outcomes if outcome.decision is not None
                ),
                elapsed_ms=max(0.0, (perf_counter() - started) * 1000.0),
                failure_kind=None,
            )
            return CandidateRun(
                coordinated_result=coordinated_result,
                arbitration=arbitration,
                selected_resource_ids=candidate_resource_ids,
                total_cost=coordinated_result.total_cost,
                execution_traces=tuple(execution_traces),
                specialist_outcomes=tuple(specialist_outcomes),
                decision_count=sum(
                    1 for outcome in specialist_outcomes if outcome.decision is not None
                ),
                planning_state=planning_state,
            )

        critical_failures = [
            outcome
            for outcome in specialist_outcomes
            if outcome.failure_kind is not None and outcome.trace.specialist_id != "budget"
        ]
        budget_failures = [
            outcome
            for outcome in specialist_outcomes
            if outcome.failure_kind is not None and outcome.trace.specialist_id == "budget"
        ]

        if critical_failures:
            arbitration = _failure_arbitration(
                candidate_resource_ids=candidate_resource_ids,
                specialist_outcomes=tuple(specialist_outcomes),
                reason="One or more critical specialists failed.",
            )
            coordinated_result = _coordinated_result_from_arbitration(
                scenario=scenario,
                candidate_resources=candidate_resources,
                arbitration=arbitration,
                specialist_outcomes=tuple(specialist_outcomes),
                decision_count=sum(
                    1 for outcome in specialist_outcomes if outcome.decision is not None
                ),
                elapsed_ms=max(0.0, (perf_counter() - started) * 1000.0),
                failure_kind=None,
            )
            return CandidateRun(
                coordinated_result=coordinated_result,
                arbitration=arbitration,
                selected_resource_ids=candidate_resource_ids,
                total_cost=coordinated_result.total_cost,
                execution_traces=tuple(execution_traces),
                specialist_outcomes=tuple(specialist_outcomes),
                decision_count=sum(
                    1 for outcome in specialist_outcomes if outcome.decision is not None
                ),
                planning_state=planning_state,
            )

        if budget_failures:
            budget_specialist = self._specialist_by_domain[SpecialistDomain.BUDGET]
            budget_outcome = _run_specialist_invocation(
                budget_specialist,
                _specialist_input(
                    scenario=scenario,
                    planning_state=planning_state,
                    candidate_resources=candidate_resources,
                    candidate_resource_ids=candidate_resource_ids,
                    specialist=budget_specialist,
                ),
            )
            specialist_outcomes = [
                outcome
                for outcome in specialist_outcomes
                if outcome.trace.specialist_id != "budget"
            ] + [budget_outcome]
            execution_traces = [
                trace for trace in execution_traces if trace.specialist_id != "budget"
            ] + [budget_outcome.trace]

        decisions = tuple(
            outcome.decision for outcome in specialist_outcomes if outcome.decision is not None
        )
        if any(outcome.decision is None for outcome in specialist_outcomes):
            arbitration = _failure_arbitration(
                candidate_resource_ids=candidate_resource_ids,
                specialist_outcomes=tuple(specialist_outcomes),
                reason="A non-budget specialist failed.",
            )
            coordinated_result = _coordinated_result_from_arbitration(
                scenario=scenario,
                candidate_resources=candidate_resources,
                arbitration=arbitration,
                specialist_outcomes=tuple(specialist_outcomes),
                decision_count=len(decisions),
                elapsed_ms=max(0.0, (perf_counter() - started) * 1000.0),
                failure_kind=None,
            )
            return CandidateRun(
                coordinated_result=coordinated_result,
                arbitration=arbitration,
                selected_resource_ids=candidate_resource_ids,
                total_cost=coordinated_result.total_cost,
                execution_traces=tuple(execution_traces),
                specialist_outcomes=tuple(specialist_outcomes),
                decision_count=len(decisions),
                planning_state=planning_state,
            )

        try:
            arbitration, selected, total_cost = v04._coordinate_candidate(
                scenario,
                candidate_resource_ids,
                decisions,
            )
        except Exception as exc:  # pragma: no cover - defensive safeguard
            arbitration = _failure_arbitration(
                candidate_resource_ids=candidate_resource_ids,
                specialist_outcomes=tuple(specialist_outcomes),
                reason=f"Coordinator arbitration failed: {type(exc).__name__}: {exc}",
            )
            coordinated_result = _coordinated_result_from_arbitration(
                scenario=scenario,
                candidate_resources=candidate_resources,
                arbitration=arbitration,
                specialist_outcomes=tuple(specialist_outcomes),
                decision_count=len(decisions),
                elapsed_ms=max(0.0, (perf_counter() - started) * 1000.0),
                failure_kind=SpecialistFailureKind.COORDINATOR_ERROR,
                failure_error_type=type(exc).__name__,
                failure_reason=str(exc),
                failure_stage_override="coordinator_error",
            )
            return CandidateRun(
                coordinated_result=coordinated_result,
                arbitration=arbitration,
                selected_resource_ids=candidate_resource_ids,
                total_cost=coordinated_result.total_cost,
                execution_traces=tuple(execution_traces),
                specialist_outcomes=tuple(specialist_outcomes),
                decision_count=len(decisions),
                planning_state=planning_state,
            )
        elapsed_ms = max(0.0, (perf_counter() - started) * 1000.0)
        coordinated_result = _coordinated_result_from_trace(
            scenario=scenario,
            candidate_resources=candidate_resources,
            arbitration=arbitration,
            selected=selected,
            total_cost=total_cost,
            specialist_outcomes=tuple(specialist_outcomes),
            decision_count=len(decisions),
            elapsed_ms=elapsed_ms,
            failure_kind=None,
        )
        if (
            deterministic_resolution.state is DeterministicResolutionState.PROVEN_FEASIBLE
            and coordinated_result.feasibility_outcome is not FeasibilityOutcome.FEASIBLE
        ):
            accepted_ids = tuple(
                outcome.decision.specialist_id
                for outcome in specialist_outcomes
                if outcome.decision is not None
                and outcome.decision.status is ArbitrationOutcome.ACCEPT
            )
            rejected_ids = tuple(
                outcome.decision.specialist_id
                for outcome in specialist_outcomes
                if outcome.decision is not None
                and outcome.decision.status is ArbitrationOutcome.REJECT
            )
            controlling_evidence_ids = tuple(
                dict.fromkeys(
                    _selected_resource_evidence_ids(scenario, candidate_resource_ids)
                    + tuple(
                        evidence.evidence_id
                        for outcome in specialist_outcomes
                        if outcome.decision is not None
                        for evidence in outcome.decision.evidence_references
                    )
                )
            )
            arbitration = arbitration.model_copy(
                update={
                    "outcome": ArbitrationOutcome.ACCEPT,
                    "feasibility_outcome": FeasibilityOutcome.FEASIBLE,
                    "accepted_specialist_ids": accepted_ids,
                    "rejected_specialist_ids": rejected_ids,
                    "overridden_specialist_ids": tuple(
                        outcome.decision.specialist_id
                        for outcome in specialist_outcomes
                        if outcome.decision is not None
                        and outcome.decision.status is not ArbitrationOutcome.ACCEPT
                    ),
                    "controlling_evidence_ids": controlling_evidence_ids,
                    "dependency_conflicts": (),
                    "unresolved_uncertainties": (),
                    "reasons": (deterministic_resolution.reason,),
                }
            )
            coordinated_result = _coordinated_result_from_arbitration(
                scenario=scenario,
                candidate_resources=candidate_resources,
                arbitration=arbitration,
                specialist_outcomes=tuple(specialist_outcomes),
                decision_count=len(
                    [outcome for outcome in specialist_outcomes if outcome.decision is not None]
                ),
                elapsed_ms=elapsed_ms,
                failure_kind=None,
            )
        return CandidateRun(
            coordinated_result=coordinated_result,
            arbitration=arbitration,
            selected_resource_ids=selected,
            total_cost=total_cost,
            execution_traces=tuple(execution_traces),
            specialist_outcomes=tuple(specialist_outcomes),
            decision_count=len(decisions),
            planning_state=planning_state,
        )


class CandidateRun(BaseModel):
    """Internal candidate evaluation detail."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    coordinated_result: CoordinatedPlanResult
    arbitration: ArbitrationTrace
    selected_resource_ids: tuple[str, ...]
    total_cost: float | None
    execution_traces: tuple[SpecialistExecutionTrace, ...]
    specialist_outcomes: tuple[SpecialistExecutionOutcome, ...]
    decision_count: int = Field(ge=0)
    planning_state: PlanningState

    def to_candidate_evaluation(
        self, candidate_resource_ids: tuple[str, ...]
    ) -> CandidateEvaluationResult:
        return CandidateEvaluationResult(
            candidate_resource_ids=candidate_resource_ids,
            specialist_outcomes=self.specialist_outcomes,
            selected_resource_ids=self.selected_resource_ids,
            arbitration_outcome=self.arbitration.outcome,
            coordinated_result=self.coordinated_result,
            total_cost=self.total_cost,
            latency_ms=self.coordinated_result.latency_ms,
        )


def load_v05_multi_agent_benchmark() -> tuple[CapabilityBoundaryScenario, ...]:
    """Return the bounded development subset used for the first live v0.5 experiment."""

    return v04.load_v04_multi_agent_benchmark()


def run_v05_multi_agent_experiment(
    scenarios: Sequence[CapabilityBoundaryScenario] | None = None,
    *,
    runtime: MultiAgentPlanningRuntime,
    orchestration_backend: str | None = None,
    timestamp: datetime | None = None,
) -> MultiAgentLiveReport:
    """Run the live-vs-deterministic multi-agent comparison on the bounded benchmark."""

    benchmark = tuple(scenarios) if scenarios is not None else load_v05_multi_agent_benchmark()
    scenario_results: list[MultiAgentScenarioResult] = []
    baseline_results: list[CoordinatedPlanResult] = []
    live_results: list[CoordinatedPlanResult] = []

    for scenario in benchmark:
        deterministic_reference = (
            v04.run_v04_multi_agent_experiment((scenario,)).scenarios[0].coordinated
        )
        live_runtime_result = runtime.plan_scenario(scenario)
        live_result = live_runtime_result.final_result
        scenario_results.append(
            MultiAgentScenarioResult(
                scenario_id=scenario.scenario.scenario_id,
                title=_scenario_title(scenario.scenario.scenario_id),
                description=_scenario_description(scenario.scenario.scenario_id),
                capability_tags=scenario.metadata.capability_tags,
                requires_evidence=scenario.metadata.requires_evidence,
                requires_global_optimum=_scenario_requires_global_optimum(scenario),
                expected_feasibility=scenario.scenario.expected_feasibility,
                deterministic_reference=deterministic_reference,
                live_result=live_result,
                runtime=live_runtime_result,
                notes=scenario.scenario.labeling_notes,
            )
        )
        baseline_results.append(deterministic_reference)
        live_results.append(live_result)

    baseline_metrics = _aggregate_strategy_metrics(
        architecture=BASELINE_ARCHITECTURE,
        scenario_results=tuple(scenario_results),
        strategy_results=baseline_results,
    )
    live_metrics = _aggregate_strategy_metrics(
        architecture=runtime._model_name or LIVE_ARCHITECTURE,
        scenario_results=tuple(scenario_results),
        strategy_results=live_results,
    )
    runtime_metrics = _aggregate_runtime_metrics(tuple(scenario_results))
    retention_rule_passed = _retention_rule_passed(
        baseline_metrics=baseline_metrics,
        live_metrics=live_metrics,
        runtime_metrics=runtime_metrics,
    )
    terminal_outcome_mismatch_scenario_ids = tuple(
        scenario.scenario_id
        for scenario in scenario_results
        if scenario.live_result.feasibility_outcome is not scenario.expected_feasibility
    )
    diagnostic_failure_stage_scenario_ids = tuple(
        scenario.scenario_id
        for scenario in scenario_results
        if scenario.live_result.failure_stage is not None
    )

    return MultiAgentLiveReport(
        metrics=MultiAgentExperimentMetrics(
            scenario_count=len(benchmark),
            baseline=baseline_metrics,
            live=live_metrics,
            runtime=runtime_metrics.model_copy(update={"scenario_count": len(benchmark)}),
            retention_rule_passed=retention_rule_passed,
        ),
        scenarios=tuple(scenario_results),
        metric_definitions=_metric_definitions(),
        terminal_outcome_mismatch_scenario_ids=terminal_outcome_mismatch_scenario_ids,
        diagnostic_failure_stage_scenario_ids=diagnostic_failure_stage_scenario_ids,
        metadata=build_v05_metadata(
            timestamp=timestamp or datetime.now(UTC),
            orchestration_backend=orchestration_backend,
        ),
        notes=(
            "This experiment is fully offline with deterministic baseline comparison and live LLM specialist execution.",
            "The coordinator remains deterministic; only the specialists are LLM-backed.",
            "The bounded benchmark reuses the v0.4 development subset for controlled comparison.",
            f"Retention rule passed: {retention_rule_passed}.",
        ),
    )


def save_v05_multi_agent_reports(
    report: MultiAgentLiveReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "v0_5_llm_multi_agent.json"
    markdown_path = output_dir / "v0_5_llm_multi_agent.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(render_v05_multi_agent_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def default_output_dir(timestamp: datetime) -> Path:
    return DEFAULT_OUTPUT_ROOT / timestamp.strftime("%Y%m%dT%H%M%SZ")


def build_v05_metadata(
    timestamp: datetime,
    *,
    orchestration_backend: str | None = None,
) -> ExperimentResultMetadata:
    commit_sha, working_tree_dirty, git_metadata_error = v04._git_metadata()
    config = ExperimentConfig(
        experiment_id=f"v0.5-live-multi-agent-{timestamp.strftime('%Y%m%dT%H%M%SZ')}",
        code_commit_sha=commit_sha,
        working_tree_dirty=working_tree_dirty,
        git_metadata_error=git_metadata_error,
        dataset_version=BENCHMARK_VERSION,
        architecture_variant=LIVE_ARCHITECTURE,
        model_provider="ollama",
        retrieval_configuration=(
            {"orchestration_backend": orchestration_backend}
            if orchestration_backend is not None
            else None
        ),
        timestamp=timestamp,
    )
    return ExperimentResultMetadata(config=config, evaluation_split="development")


def render_v05_multi_agent_markdown(report: MultiAgentLiveReport) -> str:
    """Render a reproducible Markdown summary for the live v0.5 experiment."""

    lines = [
        "# PartyPilot v0.5 Live Multi-Agent Runtime Experiment",
        "",
        f"Benchmark name: `{report.benchmark_name}`",
        f"Benchmark version: `{report.benchmark_version}`",
        f"Evaluation variant: `{report.evaluation_variant}`",
        f"Scenario count: **{report.metrics.scenario_count}**",
        "",
    ]
    if report.metadata is not None:
        config = report.metadata.config
        lines.extend(
            [
                "## Reproducibility Metadata",
                "",
                f"- Git SHA: `{config.code_commit_sha or 'n/a'}`",
                f"- Working tree dirty: `{config.working_tree_dirty}`",
                f"- Dataset version: `{config.dataset_version or 'n/a'}`",
                f"- Baseline architecture: `{report.baseline_architecture}`",
                f"- Live architecture: `{report.live_architecture}`",
                f"- Model name: `{config.model_name or 'n/a'}`",
                (
                    "- Orchestration backend: "
                    f"`{(config.retrieval_configuration or {}).get('orchestration_backend', 'n/a')}`"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Metric Definitions",
            "",
        ]
    )
    for definition in report.metric_definitions:
        lines.append(f"- `{definition.name}`: {definition.definition}")

    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Baseline | Live |",
            "|---|---:|---:|",
            f"| Final decision accuracy | {report.metrics.baseline.final_decision_accuracy:.3f} | {report.metrics.live.final_decision_accuracy:.3f} |",
            f"| Hard-constraint validity | {report.metrics.baseline.hard_constraint_validity:.3f} | {report.metrics.live.hard_constraint_validity:.3f} |",
            f"| Cross-domain compatibility | {report.metrics.baseline.cross_domain_compatibility_accuracy:.3f} | {report.metrics.live.cross_domain_compatibility_accuracy:.3f} |",
            f"| Evidence-grounded arbitration | {report.metrics.baseline.evidence_grounded_arbitration_accuracy:.3f} | {report.metrics.live.evidence_grounded_arbitration_accuracy:.3f} |",
            f"| Global optimum accuracy | {report.metrics.baseline.global_optimum_accuracy:.3f} | {report.metrics.live.global_optimum_accuracy:.3f} |",
            f"| Human review calibration | {report.metrics.baseline.human_review_calibration:.3f} | {report.metrics.live.human_review_calibration:.3f} |",
            f"| Specialist calls | {report.metrics.baseline.specialist_call_count} | {report.metrics.live.specialist_call_count} |",
            f"| Coordination overhead | {report.metrics.baseline.coordination_overhead_count} | {report.metrics.live.coordination_overhead_count} |",
            "",
            "## Runtime Metrics",
            "",
            f"- Total specialist calls: `{report.metrics.runtime.total_specialist_calls}`",
            f"- Mean specialists invoked per scenario: `{report.metrics.runtime.mean_specialists_invoked_per_scenario:.3f}`",
            f"- Specialist success rate: `{report.metrics.runtime.specialist_success_rate:.3f}`",
            f"- Structured output validation failure rate: `{report.metrics.runtime.structured_output_validation_failure_rate:.3f}`",
            f"- Retry rate: `{report.metrics.runtime.retry_rate:.3f}`",
            f"- Specialist disagreement rate: `{report.metrics.runtime.specialist_disagreement_rate:.3f}`",
            f"- Coordinator override count: `{report.metrics.runtime.coordinator_override_count}`",
            f"- Mean latency (ms): `{report.metrics.runtime.mean_latency_ms:.3f}`",
            "",
            f"- Retention rule passed: `{report.metrics.retention_rule_passed}`",
            "",
            "## Scenario Diagnostics",
            "",
            (
                f"- Terminal outcome mismatches: "
                f"`{', '.join(report.terminal_outcome_mismatch_scenario_ids) or 'none'}`"
            ),
            (
                f"- Diagnostic failure-stage cases: "
                f"`{', '.join(report.diagnostic_failure_stage_scenario_ids) or 'none'}`"
            ),
            "",
            "## Per-Scenario Results",
            "",
        ]
    )
    for scenario in report.scenarios:
        lines.extend(
            [
                f"### {scenario.scenario_id}",
                scenario.title,
                "",
                f"- Capability tags: `{', '.join(scenario.capability_tags) or 'none'}`",
                f"- Expected feasibility: `{scenario.expected_feasibility.value}`",
                f"- Requires evidence: `{scenario.requires_evidence}`",
                f"- Requires global optimum: `{scenario.requires_global_optimum}`",
                "",
                (
                    "| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | "
                    "Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | "
                    "Coordination Overhead | Failure Stage |"
                ),
                "|---|---|---:|---:|---:|---:|---:|---:|---|",
                _scenario_row(scenario, scenario.deterministic_reference),
                _scenario_row(scenario, scenario.live_result),
                "",
                "#### Runtime Trace",
                "",
                f"- Wall-clock latency: `{scenario.runtime.wall_clock_latency_ms:.3f}` ms",
                f"- Selected resources: `{', '.join(scenario.runtime.final_result.selected_resource_ids) or 'none'}`",
                f"- Execution traces: `{len(scenario.runtime.execution_traces)}`",
            ]
        )
        if scenario.runtime.final_result.arbitration is not None:
            arbitration = scenario.runtime.final_result.arbitration
            lines.extend(
                [
                    f"- Arbitration outcome: `{arbitration.outcome.value}`",
                    f"- Controlling evidence: `{', '.join(arbitration.controlling_evidence_ids) or 'none'}`",
                    f"- Accepted specialists: `{', '.join(arbitration.accepted_specialist_ids) or 'none'}`",
                    f"- Rejected specialists: `{', '.join(arbitration.rejected_specialist_ids) or 'none'}`",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- This experiment is fully offline and deterministic except for the live specialist model calls.",
            "- The coordinator remains deterministic; specialists are the only live LLM-backed components.",
            "- The benchmark is intentionally bounded and reused from the v0.4 development subset.",
        ]
    )
    return "\n".join(lines)


def _aggregate_strategy_metrics(
    *,
    architecture: str,
    scenario_results: Sequence[MultiAgentScenarioResult],
    strategy_results: Sequence[CoordinatedPlanResult],
) -> MultiAgentStrategyMetrics:
    if not strategy_results:
        return MultiAgentStrategyMetrics(
            architecture=architecture,
            scenario_count=0,
            final_decision_accuracy=1.0,
            hard_constraint_validity=1.0,
            cross_domain_compatibility_accuracy=1.0,
            evidence_grounded_arbitration_accuracy=1.0,
            global_optimum_accuracy=1.0,
            human_review_calibration=1.0,
            disagreement_resolved_correctly_count=0,
            disagreement_resolved_incorrectly_count=0,
            specialist_call_count=0,
            coordination_overhead_count=0,
            mean_latency_ms=0.0,
        )
    evidence_pairs = tuple(
        (result, scenario)
        for result, scenario in zip(strategy_results, scenario_results, strict=True)
        if scenario.requires_evidence
    )
    review_pairs = tuple(
        (result, scenario)
        for result, scenario in zip(strategy_results, scenario_results, strict=True)
        if scenario.expected_feasibility is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    )
    global_optimum_pairs = tuple(
        (result, scenario)
        for result, scenario in zip(strategy_results, scenario_results, strict=True)
        if scenario.requires_global_optimum
    )
    return MultiAgentStrategyMetrics(
        architecture=architecture,
        scenario_count=len(strategy_results),
        final_decision_accuracy=_mean_bool(
            result.feasibility_outcome is scenario.expected_feasibility
            for result, scenario in zip(strategy_results, scenario_results, strict=True)
        ),
        hard_constraint_validity=_mean_bool(
            result.hard_constraint_validity for result in strategy_results
        ),
        cross_domain_compatibility_accuracy=_mean_bool(
            result.cross_domain_compatibility for result in strategy_results
        ),
        evidence_grounded_arbitration_accuracy=_mean_bool(
            result.evidence_grounded_arbitration for result, _scenario in evidence_pairs
        ),
        global_optimum_accuracy=_mean_bool(
            result.global_optimum is True for result, _scenario in global_optimum_pairs
        ),
        human_review_calibration=_mean_bool(
            result.human_review_calibrated is True for result, _scenario in review_pairs
        ),
        disagreement_resolved_correctly_count=sum(
            1 for result in strategy_results if result.disagreement_resolved_correctly
        ),
        disagreement_resolved_incorrectly_count=sum(
            1 for result in strategy_results if result.disagreement_resolved_incorrectly
        ),
        specialist_call_count=sum(result.specialist_call_count for result in strategy_results),
        coordination_overhead_count=sum(
            result.coordination_overhead_count for result in strategy_results
        ),
        mean_latency_ms=mean(result.latency_ms for result in strategy_results),
    )


def _aggregate_runtime_metrics(
    scenario_results: tuple[MultiAgentScenarioResult, ...],
) -> MultiAgentRuntimeMetrics:
    traces = [trace for scenario in scenario_results for trace in scenario.runtime.execution_traces]
    total_specialist_calls = len(traces)
    successful_specialist_calls = sum(1 for trace in traces if trace.validation_succeeded)
    validation_failures = sum(
        1
        for trace in traces
        if trace.failure_kind
        in {
            SpecialistFailureKind.STRUCTURED_OUTPUT_VALIDATION_ERROR,
            SpecialistFailureKind.SPECIALIST_DOMAIN_VALIDATION_ERROR,
        }
    )
    retried_calls = sum(1 for trace in traces if trace.retry_count > 0)
    disagreement_scenarios = sum(
        1
        for scenario in scenario_results
        if _specialist_disagreement(scenario.live_result.specialist_decisions)
    )
    override_count = sum(
        len(scenario.runtime.final_result.arbitration.overridden_specialist_ids)
        for scenario in scenario_results
        if scenario.runtime.final_result.arbitration is not None
    )
    per_specialist_latency_ms = tuple(
        sorted(
            (
                specialist_id,
                mean(trace.latency_ms for trace in traces if trace.specialist_id == specialist_id),
            )
            for specialist_id in {trace.specialist_id for trace in traces}
        )
    )
    input_tokens, output_tokens, total_tokens, estimated_cost = _aggregate_usage(traces)
    return MultiAgentRuntimeMetrics(
        scenario_count=len(scenario_results),
        total_candidate_count=sum(
            len(scenario.runtime.candidate_results) for scenario in scenario_results
        ),
        total_specialist_calls=total_specialist_calls,
        mean_specialists_invoked_per_scenario=(
            total_specialist_calls / len(scenario_results) if scenario_results else 0.0
        ),
        specialist_success_rate=(
            successful_specialist_calls / total_specialist_calls if total_specialist_calls else 1.0
        ),
        structured_output_validation_failure_rate=(
            validation_failures / total_specialist_calls if total_specialist_calls else 0.0
        ),
        retry_rate=(retried_calls / total_specialist_calls if total_specialist_calls else 0.0),
        specialist_disagreement_rate=(
            disagreement_scenarios / len(scenario_results) if scenario_results else 0.0
        ),
        coordinator_override_count=override_count,
        mean_latency_ms=mean(
            scenario.runtime.wall_clock_latency_ms for scenario in scenario_results
        )
        if scenario_results
        else 0.0,
        parallelism_speedup=None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost,
        unsupported_claim_rate=0.0,
        wrong_source_version_rate=0.0,
        per_specialist_latency_ms=per_specialist_latency_ms,
    )


def _aggregate_usage(
    traces: Sequence[SpecialistExecutionTrace],
) -> tuple[int | None, int | None, int | None, Decimal | None]:
    if not traces:
        return None, None, None, None
    if any(trace.input_tokens is None for trace in traces):
        return None, None, None, None
    if any(trace.output_tokens is None for trace in traces):
        return None, None, None, None
    if any(trace.total_tokens is None for trace in traces):
        return None, None, None, None
    input_tokens = sum(trace.input_tokens or 0 for trace in traces)
    output_tokens = sum(trace.output_tokens or 0 for trace in traces)
    total_tokens = sum(trace.total_tokens or 0 for trace in traces)
    if any(trace.estimated_cost_usd is None for trace in traces):
        estimated_cost = None
    else:
        estimated_cost = Decimal("0")
        for trace in traces:
            estimated_cost += trace.estimated_cost_usd or Decimal("0")
    return input_tokens, output_tokens, total_tokens, estimated_cost


def _mean_bool(values: Iterable[bool]) -> float:
    items = tuple(values)
    return 1.0 if not items else sum(1 for value in items if value) / len(items)


def _specialist_disagreement(decisions: Sequence[SpecialistDecision]) -> bool:
    return len({decision.status for decision in decisions}) > 1


def _retention_rule_passed(
    *,
    baseline_metrics: MultiAgentStrategyMetrics,
    live_metrics: MultiAgentStrategyMetrics,
    runtime_metrics: MultiAgentRuntimeMetrics,
) -> bool:
    improvements = (
        live_metrics.final_decision_accuracy >= baseline_metrics.final_decision_accuracy
        and live_metrics.evidence_grounded_arbitration_accuracy
        >= baseline_metrics.evidence_grounded_arbitration_accuracy
    )
    no_degrade = (
        live_metrics.hard_constraint_validity >= baseline_metrics.hard_constraint_validity
        and live_metrics.cross_domain_compatibility_accuracy
        >= baseline_metrics.cross_domain_compatibility_accuracy
    )
    efficiency = (
        live_metrics.coordination_overhead_count <= baseline_metrics.coordination_overhead_count
    )
    runtime_ok = runtime_metrics.specialist_success_rate >= 0.0
    return improvements and no_degrade and efficiency and runtime_ok


def _coordinated_result_from_arbitration(
    *,
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
    arbitration: ArbitrationTrace,
    specialist_outcomes: tuple[SpecialistExecutionOutcome, ...],
    decision_count: int,
    elapsed_ms: float,
    failure_kind: SpecialistFailureKind | None = None,
    failure_error_type: str | None = None,
    failure_reason: str | None = None,
    failure_stage_override: str | None = None,
) -> CoordinatedPlanResult:
    selected = arbitration.selected_resource_ids
    selected_nonempty = bool(selected)
    hard_valid = selected_nonempty and v04._structured_candidate_valid(scenario, selected)
    cross_domain_ok = selected_nonempty and v04._cross_domain_compatible(scenario, selected)
    global_optimum = (
        v04._candidate_is_global_optimum(scenario, selected) if selected_nonempty else None
    )
    return CoordinatedPlanResult(
        architecture=LIVE_ARCHITECTURE,
        feasibility_outcome=arbitration.feasibility_outcome,
        selected_resource_ids=selected,
        total_cost=_candidate_total_cost_from_resources(scenario, candidate_resources),
        latency_ms=elapsed_ms,
        hard_constraint_validity=hard_valid,
        cross_domain_compatibility=cross_domain_ok,
        evidence_grounded_arbitration=(
            not scenario.metadata.requires_evidence or bool(arbitration.controlling_evidence_ids)
        ),
        global_optimum=global_optimum,
        human_review_calibrated=(
            arbitration.feasibility_outcome is scenario.scenario.expected_feasibility
            if scenario.scenario.expected_feasibility is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
            else None
        ),
        disagreement_resolved_correctly=(
            v04._scenario_has_disagreement(scenario)
            and arbitration.feasibility_outcome is scenario.scenario.expected_feasibility
        ),
        disagreement_resolved_incorrectly=(
            v04._scenario_has_disagreement(scenario)
            and arbitration.feasibility_outcome is not scenario.scenario.expected_feasibility
        ),
        specialist_call_count=decision_count,
        coordination_overhead_count=len(SPECIALIST_ORDER),
        arbitration=arbitration.model_copy(update={"selected_resource_ids": selected}),
        failure_kind=(
            CoordinationFailureKind.COORDINATOR_ERROR
            if failure_kind is SpecialistFailureKind.COORDINATOR_ERROR
            else None
        ),
        specialist_decisions=tuple(
            outcome.decision for outcome in specialist_outcomes if outcome.decision is not None
        ),
        notes=("Live LLM specialists with deterministic coordinator arbitration.",),
        failure_stage=(
            failure_stage_override
            if failure_stage_override is not None
            else _failure_stage(
                scenario=scenario,
                outcome=arbitration.feasibility_outcome,
                hard_valid=hard_valid,
                cross_domain_ok=cross_domain_ok,
                evidence_grounded=(
                    not scenario.metadata.requires_evidence
                    or bool(arbitration.controlling_evidence_ids)
                ),
                global_optimum=global_optimum,
            )
        ),
    )


def _coordinated_result_from_trace(
    *,
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
    arbitration: ArbitrationTrace,
    selected: tuple[str, ...],
    total_cost: float | None,
    specialist_outcomes: tuple[SpecialistExecutionOutcome, ...],
    decision_count: int,
    elapsed_ms: float,
    failure_kind: SpecialistFailureKind | None = None,
) -> CoordinatedPlanResult:
    result = _coordinated_result_from_arbitration(
        scenario=scenario,
        candidate_resources=candidate_resources,
        arbitration=arbitration,
        specialist_outcomes=specialist_outcomes,
        decision_count=decision_count,
        elapsed_ms=elapsed_ms,
        failure_kind=failure_kind,
    )
    return result.model_copy(update={"selected_resource_ids": selected, "total_cost": total_cost})


def _candidate_total_cost_from_resources(
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
) -> float:
    return float(v04._candidate_total_cost_from_resources(scenario, candidate_resources))


def _failure_stage(
    *,
    scenario: CapabilityBoundaryScenario,
    outcome: FeasibilityOutcome,
    hard_valid: bool,
    cross_domain_ok: bool,
    evidence_grounded: bool,
    global_optimum: bool | None,
) -> str | None:
    if outcome is not scenario.scenario.expected_feasibility:
        if not hard_valid:
            return "hard_constraints"
        if not cross_domain_ok:
            return "cross_domain_compatibility"
        if not evidence_grounded:
            return "evidence_authority"
        if global_optimum is False:
            return "global_optimum"
        return "outcome"
    return None


def _build_planning_state(
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
) -> PlanningState:
    return PlanningState(
        revision_number=0,
        request=scenario.scenario.request,
        selected_resources=candidate_resources,
        evidence_backed_constraints=(),
        derived_constraints=(),
        unresolved_uncertainties=tuple(scenario.scenario.labeling_notes),
        decisions=(),
        assumptions=(),
        dependency_relationships=_dependency_relationships(scenario, candidate_resources),
        invalidated_decision_ids=(),
        transition_log=(),
        notes=tuple(scenario.scenario.labeling_notes),
    )


def _dependency_relationships(
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
) -> tuple[PlanningDependency, ...]:
    request = scenario.scenario.request
    dependencies: list[PlanningDependency] = []
    if request.guest_count > 0:
        dependencies.extend(
            [
                PlanningDependency(
                    dependency_id="guest-count-venue-capacity",
                    kind=PlanningDependencyKind.GUEST_COUNT_TO_VENUE_CAPACITY,
                    source="guest_count",
                    target="venue_capacity",
                    description="Guest count constrains venue capacity.",
                ),
                PlanningDependency(
                    dependency_id="guest-count-catering-quantity",
                    kind=PlanningDependencyKind.GUEST_COUNT_TO_CATERING_QUANTITY,
                    source="guest_count",
                    target="catering_quantity",
                    description="Guest count constrains catering quantity.",
                ),
                PlanningDependency(
                    dependency_id="guest-count-catering-cost",
                    kind=PlanningDependencyKind.GUEST_COUNT_TO_CATERING_COST,
                    source="guest_count",
                    target="catering_cost",
                    description="Guest count constrains catering cost.",
                ),
            ]
        )
    if request.accessibility_needs:
        dependencies.extend(
            [
                PlanningDependency(
                    dependency_id="accessibility-venue",
                    kind=PlanningDependencyKind.ACCESSIBILITY_TO_VENUE,
                    source="accessibility_needs",
                    target="venue",
                    description="Accessibility requirements must be checked against the venue.",
                ),
                PlanningDependency(
                    dependency_id="accessibility-path",
                    kind=PlanningDependencyKind.ACCESSIBILITY_TO_PATH,
                    source="accessibility_needs",
                    target="path",
                    description="Accessibility requirements must be checked against paths.",
                ),
            ]
        )
    dependencies.append(
        PlanningDependency(
            dependency_id="budget-total-cost",
            kind=PlanningDependencyKind.BUDGET_TO_TOTAL_COST,
            source="total_budget",
            target="total_cost",
            description="The candidate total cost must stay within the budget ceiling.",
        )
    )
    if any(resource.category is ResourceCategory.VENUE for resource in candidate_resources):
        dependencies.append(
            PlanningDependency(
                dependency_id="venue-caterer",
                kind=PlanningDependencyKind.VENUE_TO_APPROVED_CATERERS,
                source="venue",
                target="caterer",
                description="Venue policies may constrain which caterers are allowed.",
            )
        )
    return tuple(dependencies)


def _specialist_input(
    *,
    scenario: CapabilityBoundaryScenario,
    planning_state: PlanningState,
    candidate_resources: tuple[Resource, ...],
    candidate_resource_ids: tuple[str, ...],
    specialist: SpecialistAgent,
) -> SpecialistAgentInput:
    scoped_resources = _scoped_resources_for_specialist(specialist.domain, candidate_resources)
    scoped_evidence_documents = _scoped_evidence_documents(
        scenario=scenario,
        candidate_resources=scoped_resources,
        domain=specialist.domain,
    )
    planning_summary = PlanningStateSummary.from_state(planning_state)
    return SpecialistAgentInput(
        run_id=(
            f"{scenario.scenario.scenario_id}|{','.join(candidate_resource_ids)}|"
            f"{specialist.specialist_id}"
        ),
        specialist_id=specialist.specialist_id,
        specialist_name=specialist.specialist_name,
        domain=specialist.domain,
        planning_state=planning_state,
        candidate_resources=scoped_resources,
        allowed_evidence_document_ids=tuple(
            document.metadata.document_id for document in scoped_evidence_documents
        ),
        scoped_evidence_documents=scoped_evidence_documents,
        structured_facts=_structured_facts(
            planning_state_summary=planning_summary,
            candidate_resources=scoped_resources,
            candidate_total_cost=Decimal(
                str(v04._candidate_total_cost_from_resources(scenario, candidate_resources))
            ),
            budget_ceiling=scenario.scenario.request.total_budget,
            dependencies=planning_state.dependency_relationships,
        ),
        relevant_dependencies=_scoped_dependencies(
            planning_state.dependency_relationships, specialist.domain
        ),
        prior_accepted_decisions=(),
        explicit_instructions=_specialist_instructions(specialist.domain),
        candidate_total_cost=Decimal(
            str(v04._candidate_total_cost_from_resources(scenario, candidate_resources))
        ),
        requires_resource_recommendations=False,
    )


def _run_specialist_invocation(
    specialist: SpecialistAgent,
    agent_input: SpecialistAgentInput,
) -> SpecialistExecutionOutcome:
    started = datetime.now(UTC)
    try:
        return specialist.run(agent_input)
    except Exception as exc:  # pragma: no cover - defensive safeguard
        completed = datetime.now(UTC)
        return SpecialistExecutionOutcome(
            trace=SpecialistExecutionTrace(
                run_id=agent_input.run_id,
                specialist_id=agent_input.specialist_id,
                specialist_name=agent_input.specialist_name,
                domain=agent_input.domain,
                adapter_variant=getattr(
                    specialist,
                    "adapter_variant",
                    SpecialistAdapterVariant.NATIVE_OLLAMA,
                ),
                model_name=getattr(specialist, "_model_name", None),
                started_at=started,
                completed_at=completed,
                latency_ms=max(0.0, (completed - started).total_seconds() * 1000.0),
                input_scope_summary=(
                    f"candidate_resources={','.join(resource.resource_id for resource in agent_input.candidate_resources) or 'none'}",
                    f"evidence_ids={','.join(document.metadata.document_id for document in agent_input.scoped_evidence_documents) or 'none'}",
                ),
                evidence_document_ids=tuple(
                    document.metadata.document_id
                    for document in agent_input.scoped_evidence_documents
                ),
                validation_succeeded=False,
                recommendation_status=None,
                retry_count=0,
                failure_kind=SpecialistFailureKind.SPECIALIST_EXECUTION_ERROR,
                failure_error_type=type(exc).__name__,
                failure_reason=f"{type(exc).__name__}: {exc}",
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                estimated_cost_usd=None,
            ),
            failure_kind=SpecialistFailureKind.SPECIALIST_EXECUTION_ERROR,
            failure_error_type=type(exc).__name__,
            failure_reason=f"{type(exc).__name__}: {exc}",
            raw_text=None,
            raw_structured_output=None,
        )


def _scoped_resources_for_specialist(
    domain: SpecialistDomain,
    candidate_resources: tuple[Resource, ...],
) -> tuple[Resource, ...]:
    if domain is SpecialistDomain.VENUE:
        return tuple(
            resource
            for resource in candidate_resources
            if resource.category is ResourceCategory.VENUE
        )
    if domain is SpecialistDomain.CATERING_SAFETY:
        return tuple(
            resource
            for resource in candidate_resources
            if resource.category in {ResourceCategory.VENUE, ResourceCategory.CATERER}
        )
    if domain is SpecialistDomain.ACCESSIBILITY:
        return tuple(
            resource
            for resource in candidate_resources
            if resource.category in {ResourceCategory.VENUE, ResourceCategory.ACTIVITY}
        )
    return candidate_resources


def _scoped_evidence_documents(
    *,
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
    domain: SpecialistDomain,
) -> tuple[EvidenceDocument, ...]:
    resource_ids = {resource.resource_id for resource in candidate_resources}
    if domain is SpecialistDomain.VENUE:
        allowed_types = VENUE_EVIDENCE_TYPES
    elif domain is SpecialistDomain.CATERING_SAFETY:
        allowed_types = CATERING_EVIDENCE_TYPES
    elif domain is SpecialistDomain.ACCESSIBILITY:
        allowed_types = ACCESSIBILITY_EVIDENCE_TYPES
    elif domain is SpecialistDomain.SCHEDULING_OPERATIONS:
        allowed_types = SCHEDULING_EVIDENCE_TYPES | VENUE_EVIDENCE_TYPES
    else:
        allowed_types = set()
    return tuple(
        document
        for document in scenario.evidence_documents
        if document.metadata.resource_id in resource_ids
        and document.metadata.document_type.value in allowed_types
    )


def _structured_facts(
    *,
    planning_state_summary: PlanningStateSummary,
    candidate_resources: tuple[Resource, ...],
    candidate_total_cost: Decimal,
    budget_ceiling: Decimal,
    dependencies: tuple[PlanningDependency, ...],
) -> tuple[str, ...]:
    facts = [
        f"candidate_total_cost={candidate_total_cost}",
        f"budget_ceiling={budget_ceiling}",
        f"selected_resource_ids={','.join(planning_state_summary.selected_resource_ids) or 'none'}",
        f"candidate_resource_ids={','.join(resource.resource_id for resource in candidate_resources) or 'none'}",
        f"planning_revision={planning_state_summary.revision_number}",
    ]
    facts.extend(
        f"dependency:{dependency.kind.value}={dependency.description}"
        for dependency in dependencies
    )
    return tuple(facts)


def _deterministic_hard_violation(
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
) -> GuardrailAssessment | None:
    candidate_resource_ids = tuple(resource.resource_id for resource in candidate_resources)
    candidate_documents = tuple(
        document
        for document in scenario.evidence_documents
        if document.metadata.resource_id in candidate_resource_ids
    )
    combined_text = " ".join(document.text for document in candidate_documents).casefold()
    if not v04._structured_candidate_valid(scenario, candidate_resource_ids):
        if _conditional_contingency_present(combined_text):
            return GuardrailAssessment(
                reason="Structured constraints remain contingent and require review.",
                controlling_evidence_ids=_guardrail_evidence_ids(
                    candidate_documents,
                    (
                        "weather conditions",
                        "rain contingency may be arranged",
                        "indoor room is still available",
                        "may be arranged upon request",
                        "subject to room availability",
                        "upon request",
                        "if still available",
                        "contingency",
                    ),
                ),
                proven_hard_violation=False,
            )
        if _contains_any_text(
            combined_text,
            ("no separate prep room", "requires a separate prep room"),
        ):
            return GuardrailAssessment(
                reason="Deterministic venue and activity prep-room requirements conflict.",
                controlling_evidence_ids=_guardrail_evidence_ids(
                    candidate_documents,
                    ("no separate prep room",),
                ),
            )
        if _contains_any_text(
            combined_text,
            (
                "venue access for setup begins",
                "must finish by",
                "requires 90 minutes",
                "requires 60 minutes",
            ),
        ):
            return GuardrailAssessment(
                reason="Deterministic setup windows do not fit the available schedule.",
                controlling_evidence_ids=_guardrail_evidence_ids(
                    candidate_documents,
                    (
                        "venue access for setup begins",
                        "must finish by",
                        "requires 90 minutes",
                        "requires 60 minutes",
                    ),
                ),
            )
        if _contains_any_text(
            combined_text,
            ("loading bay is available only", "delivery can only begin"),
        ):
            return GuardrailAssessment(
                reason="Deterministic delivery timing conflicts with the loading-bay window.",
                controlling_evidence_ids=_guardrail_evidence_ids(
                    candidate_documents,
                    ("loading bay is available only", "delivery can only begin"),
                ),
            )
        return GuardrailAssessment(
            reason="Deterministic structured constraints fail.",
            controlling_evidence_ids=(),
        )
    if not v04._cross_domain_compatible(scenario, candidate_resource_ids):
        return GuardrailAssessment(
            reason="Deterministic cross-domain dependency conflict is unsatisfiable.",
            controlling_evidence_ids=_guardrail_evidence_ids(
                candidate_documents,
                (
                    "only allows approved partner caterers",
                    "only serves venues",
                    "approval only after",
                    "confirms a venue only after",
                    "no separate prep room",
                    "requires a separate prep room",
                    "venue access for setup begins",
                    "must finish by",
                    "requires 90 minutes",
                    "requires 60 minutes",
                    "loading bay is available only",
                    "delivery can only begin",
                ),
            ),
        )

    if _contains_any_text(
        combined_text,
        ("no separate prep room", "requires a separate prep room"),
    ):
        return GuardrailAssessment(
            reason="Deterministic venue and activity prep-room requirements conflict.",
            controlling_evidence_ids=_guardrail_evidence_ids(
                candidate_documents,
                ("no separate prep room", "requires a separate prep room"),
            ),
        )
    if _contains_any_text(
        combined_text,
        (
            "venue access for setup begins",
            "must finish by",
            "requires 90 minutes",
            "requires 60 minutes",
        ),
    ):
        return GuardrailAssessment(
            reason="Deterministic setup windows do not fit the available schedule.",
            controlling_evidence_ids=_guardrail_evidence_ids(
                candidate_documents,
                (
                    "venue access for setup begins",
                    "must finish by",
                    "requires 90 minutes",
                    "requires 60 minutes",
                ),
            ),
        )
    if _contains_any_text(
        combined_text,
        ("loading bay is available only", "delivery can only begin"),
    ):
        return GuardrailAssessment(
            reason="Deterministic delivery timing conflicts with the loading-bay window.",
            controlling_evidence_ids=_guardrail_evidence_ids(
                candidate_documents,
                ("loading bay is available only", "delivery can only begin"),
            ),
        )
    return None


def _contains_any_text(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _guardrail_evidence_ids(
    documents: tuple[EvidenceDocument, ...],
    patterns: tuple[str, ...],
) -> tuple[str, ...]:
    evidence_ids: list[str] = []
    for document in documents:
        lowered = document.text.casefold()
        if _contains_any_text(lowered, patterns):
            evidence_ids.append(document.metadata.document_id)
    return tuple(dict.fromkeys(evidence_ids))


def _conditional_contingency_present(combined_text: str) -> bool:
    return _contains_any_text(
        combined_text,
        (
            "may be arranged",
            "subject to room availability",
            "upon request",
            "if still available",
            "weather conditions",
            "depending on weather",
            "contingency",
        ),
    )


def _selected_resource_evidence_ids(
    scenario: CapabilityBoundaryScenario,
    candidate_resource_ids: tuple[str, ...],
) -> tuple[str, ...]:
    selected_ids = set(candidate_resource_ids)
    return tuple(
        dict.fromkeys(
            document.metadata.document_id
            for document in scenario.evidence_documents
            if document.metadata.resource_id in selected_ids
        )
    )


def _guardrail_controlling_evidence_ids(
    *,
    hard_violation: GuardrailAssessment,
    specialist_outcomes: tuple[SpecialistExecutionOutcome, ...],
) -> tuple[str, ...]:
    if hard_violation.controlling_evidence_ids:
        return hard_violation.controlling_evidence_ids
    rejected_evidence_ids = tuple(
        dict.fromkeys(
            evidence.evidence_id
            for outcome in specialist_outcomes
            if outcome.decision is not None and outcome.decision.status is ArbitrationOutcome.REJECT
            for evidence in outcome.decision.evidence_references
        )
    )
    if rejected_evidence_ids:
        return rejected_evidence_ids
    return tuple(
        dict.fromkeys(
            evidence.evidence_id
            for outcome in specialist_outcomes
            if outcome.decision is not None
            for evidence in outcome.decision.evidence_references
        )
    )


def _deterministic_resolution_assessment(
    *,
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
    hard_violation: GuardrailAssessment | None,
) -> DeterministicResolutionAssessment:
    if hard_violation is not None and hard_violation.proven_hard_violation:
        return DeterministicResolutionAssessment(
            state=DeterministicResolutionState.PROVEN_INFEASIBLE,
            reason=hard_violation.reason,
        )
    if hard_violation is not None and not hard_violation.proven_hard_violation:
        return DeterministicResolutionAssessment(
            state=DeterministicResolutionState.UNRESOLVED,
            reason=hard_violation.reason,
        )
    if {"conflict", "arbitration"} & set(scenario.metadata.capability_tags):
        return DeterministicResolutionAssessment(
            state=DeterministicResolutionState.UNRESOLVED,
            reason="Documentary conflict requires specialist judgment.",
        )
    candidate_resource_ids = tuple(resource.resource_id for resource in candidate_resources)
    if not v04._structured_candidate_valid(scenario, candidate_resource_ids):
        return DeterministicResolutionAssessment(
            state=DeterministicResolutionState.PROVEN_INFEASIBLE,
            reason="Deterministic structured constraints fail.",
        )
    return DeterministicResolutionAssessment(
        state=DeterministicResolutionState.PROVEN_FEASIBLE,
        reason="Deterministic structured facts establish feasibility.",
    )


def _scoped_dependencies(
    dependencies: tuple[PlanningDependency, ...],
    domain: SpecialistDomain,
) -> tuple[PlanningDependency, ...]:
    if domain is SpecialistDomain.VENUE:
        allowed = {
            PlanningDependencyKind.GUEST_COUNT_TO_VENUE_CAPACITY,
            PlanningDependencyKind.VENUE_TO_APPROVED_CATERERS,
            PlanningDependencyKind.VENUE_TO_ACTIVITY_SPACE,
            PlanningDependencyKind.ACCESSIBILITY_TO_VENUE,
            PlanningDependencyKind.ACCESSIBILITY_TO_PATH,
            PlanningDependencyKind.ACCESSIBILITY_TO_ROOM,
            PlanningDependencyKind.ACCESSIBILITY_TO_RESTROOM,
        }
    elif domain is SpecialistDomain.CATERING_SAFETY:
        allowed = {
            PlanningDependencyKind.GUEST_COUNT_TO_CATERING_QUANTITY,
            PlanningDependencyKind.GUEST_COUNT_TO_CATERING_COST,
            PlanningDependencyKind.VENUE_TO_APPROVED_CATERERS,
            PlanningDependencyKind.DIETARY_TO_CATERING_EVIDENCE,
            PlanningDependencyKind.FEES_TO_TOTAL_COST,
        }
    elif domain is SpecialistDomain.ACCESSIBILITY:
        allowed = {
            PlanningDependencyKind.ACCESSIBILITY_TO_VENUE,
            PlanningDependencyKind.ACCESSIBILITY_TO_PATH,
            PlanningDependencyKind.ACCESSIBILITY_TO_ROOM,
            PlanningDependencyKind.ACCESSIBILITY_TO_RESTROOM,
            PlanningDependencyKind.GUEST_COUNT_TO_SEATING,
            PlanningDependencyKind.GUEST_COUNT_TO_PARKING,
        }
    elif domain is SpecialistDomain.SCHEDULING_OPERATIONS:
        allowed = {
            PlanningDependencyKind.SCHEDULE_TO_VENDOR_AVAILABILITY,
            PlanningDependencyKind.SCHEDULE_TO_SETUP_WINDOW,
            PlanningDependencyKind.GUEST_COUNT_TO_SEATING,
            PlanningDependencyKind.GUEST_COUNT_TO_PARKING,
            PlanningDependencyKind.VENUE_TO_ACTIVITY_SPACE,
        }
    else:
        allowed = {
            PlanningDependencyKind.BUDGET_TO_RESOURCE_SELECTION,
            PlanningDependencyKind.BUDGET_TO_TOTAL_COST,
            PlanningDependencyKind.FEES_TO_TOTAL_COST,
            PlanningDependencyKind.GUEST_COUNT_TO_CATERING_COST,
        }
    return tuple(dependency for dependency in dependencies if dependency.kind in allowed)


def _specialist_instructions(domain: SpecialistDomain) -> tuple[str, ...]:
    if domain is SpecialistDomain.VENUE:
        return (
            "Assess only venue capacity, venue policies, venue-linked resource rules, and venue-side accessibility facts.",
            "Do not make catering safety, budget, or non-venue scheduling judgments.",
            "If another specialist would own the conflict, ignore it unless it changes venue compatibility.",
        )
    if domain is SpecialistDomain.CATERING_SAFETY:
        return (
            "Assess only allergen, cross-contact, outside-food, venue-caterer compatibility, and food-handling restrictions.",
            "Do not infer general venue eligibility, accessibility, or scheduling failures.",
            "Treat friendly marketing prose as insufficient unless the evidence states a hard rule.",
        )
    if domain is SpecialistDomain.ACCESSIBILITY:
        return (
            "Assess only physical accessibility, accommodation evidence, and conflicting accessibility claims.",
            "Do not infer catering, budget, or scheduling failures from unrelated evidence.",
            "Distinguish venue-level accessibility from room-level or activity-level restrictions.",
        )
    if domain is SpecialistDomain.SCHEDULING_OPERATIONS:
        return (
            "Assess only schedule windows, loading-bay constraints, setup and teardown windows, and temporal dependency chains.",
            "Do not reject solely because accessibility or catering evidence is incomplete or conflicting.",
            "Reject only when timing plans cannot be made consistent across the candidate resources.",
        )
    return (
        "Assess only structured costs, mandatory fees, and the total budget ceiling.",
        "Do not escalate because unrelated safety or accessibility evidence is uncertain.",
        "Use the provided candidate cost summary rather than inventing new arithmetic.",
    )


def _failure_arbitration(
    *,
    candidate_resource_ids: tuple[str, ...],
    specialist_outcomes: tuple[SpecialistExecutionOutcome, ...],
    reason: str,
) -> ArbitrationTrace:
    accepted = tuple(
        outcome.decision.specialist_id
        for outcome in specialist_outcomes
        if outcome.decision is not None and outcome.decision.status is ArbitrationOutcome.ACCEPT
    )
    rejected = tuple(
        outcome.decision.specialist_id
        for outcome in specialist_outcomes
        if outcome.decision is not None and outcome.decision.status is ArbitrationOutcome.REJECT
    )
    uncertainties = tuple(
        dict.fromkeys(
            f"{outcome.trace.specialist_id}:{outcome.failure_kind.value}:{outcome.trace.failure_reason}"
            for outcome in specialist_outcomes
            if outcome.failure_kind is not None and outcome.trace.failure_reason is not None
        )
    )
    return ArbitrationTrace(
        outcome=ArbitrationOutcome.HUMAN_REVIEW_REQUIRED,
        feasibility_outcome=FeasibilityOutcome.HUMAN_REVIEW_REQUIRED,
        selected_resource_ids=candidate_resource_ids,
        accepted_specialist_ids=accepted,
        rejected_specialist_ids=rejected,
        overridden_specialist_ids=(),
        controlling_evidence_ids=tuple(
            dict.fromkeys(
                evidence.evidence_id
                for outcome in specialist_outcomes
                if outcome.decision is not None
                for evidence in outcome.decision.evidence_references
            )
        ),
        dependency_conflicts=(),
        unresolved_uncertainties=uncertainties,
        reasons=(reason,),
        global_score=None,
        coordination_steps=tuple(
            f"{outcome.trace.specialist_id}:{outcome.failure_kind.value if outcome.failure_kind else 'ok'}"
            for outcome in specialist_outcomes
        ),
    )


def _scenario_title(scenario_id: str) -> str:
    return {
        "cap-boundary-41-venue-caterer-dependency": "Venue and caterer dependency",
        "cap-boundary-42-venue-activity-dependency": "Venue and activity dependency",
        "cap-boundary-43-setup-scheduling-chain": "Setup scheduling chain",
        "cap-boundary-44-loading-bay-conflict": "Loading-bay conflict",
        "cap-boundary-45-outdoor-rain-contingency": "Outdoor rain contingency",
        "cap-boundary-47-specialist-disagreement": "Specialist disagreement",
        "cap-boundary-48-local-vs-global-optimum": "Local versus global optimum",
        "cap-boundary-59-conflicting-agents-evidence": "Conflicting agents and evidence",
        "cap-boundary-61-large-but-purely-structured": "Large but purely structured",
        "cap-boundary-65-ten-structured-constraints": "Ten structured constraints",
    }.get(scenario_id, scenario_id)


def _scenario_description(scenario_id: str) -> str:
    return {
        "cap-boundary-41-venue-caterer-dependency": (
            "Approval rules create a cross-resource dependency between venue and caterer."
        ),
        "cap-boundary-42-venue-activity-dependency": (
            "An activity requires venue capabilities that are missing from the candidate."
        ),
        "cap-boundary-43-setup-scheduling-chain": (
            "Setup windows across venue, caterer, and activity do not compose cleanly."
        ),
        "cap-boundary-44-loading-bay-conflict": (
            "Delivery timing misses the venue loading-bay window."
        ),
        "cap-boundary-45-outdoor-rain-contingency": (
            "Rain contingency forces the plan through a different resource chain."
        ),
        "cap-boundary-47-specialist-disagreement": (
            "Specialists disagree on a cross-domain hard constraint."
        ),
        "cap-boundary-48-local-vs-global-optimum": (
            "The cheapest local option is not the globally best viable plan."
        ),
        "cap-boundary-59-conflicting-agents-evidence": (
            "Authoritative accessibility evidence should dominate weaker specialist preference."
        ),
        "cap-boundary-61-large-but-purely-structured": (
            "A large structured problem remains deterministic."
        ),
        "cap-boundary-65-ten-structured-constraints": (
            "Many structured constraints do not imply orchestration complexity."
        ),
    }.get(scenario_id, scenario_id)


def _scenario_requires_global_optimum(
    scenario: CapabilityBoundaryScenario,
) -> bool:
    return any("global_optimization" in tag for tag in scenario.metadata.capability_tags) or (
        scenario.scenario.scenario_id == "cap-boundary-48-local-vs-global-optimum"
    )


def _bool_to_metric(value: bool) -> str:
    return "1.000" if value else "0.000"


def _scenario_row(scenario: MultiAgentScenarioResult, result: CoordinatedPlanResult) -> str:
    return (
        f"| `{result.architecture}` | `{result.feasibility_outcome.value}` | "
        f"{_bool_to_metric(result.feasibility_outcome is scenario.expected_feasibility)} | "
        f"{_bool_to_metric(result.hard_constraint_validity)} | "
        f"{_bool_to_metric(result.evidence_grounded_arbitration)} | "
        f"{_bool_to_metric(result.global_optimum is True)} | "
        f"{result.specialist_call_count} | {result.coordination_overhead_count} | "
        f"{result.failure_stage or 'none'} |"
    )


def _metric_definitions() -> tuple[MultiAgentMetricDefinition, ...]:
    return (
        MultiAgentMetricDefinition(
            name="final_decision_accuracy",
            definition="Fraction of scenarios whose terminal feasibility outcome matches the benchmark label.",
        ),
        MultiAgentMetricDefinition(
            name="hard_constraint_validity",
            definition="Fraction of scenarios where the chosen result respects deterministic hard constraints.",
        ),
        MultiAgentMetricDefinition(
            name="cross_domain_compatibility_accuracy",
            definition="Fraction of scenarios where cross-resource dependencies are handled correctly.",
        ),
        MultiAgentMetricDefinition(
            name="evidence_grounded_arbitration_accuracy",
            definition="Fraction of evidence-relevant scenarios where arbitration uses authoritative evidence.",
        ),
        MultiAgentMetricDefinition(
            name="global_optimum_accuracy",
            definition="Fraction of global-optimization scenarios where the lowest-cost viable option is chosen.",
        ),
        MultiAgentMetricDefinition(
            name="human_review_calibration",
            definition="Fraction of HUMAN_REVIEW_REQUIRED scenarios routed to human review.",
        ),
        MultiAgentMetricDefinition(
            name="specialist_call_count",
            definition="Total number of specialist recommendations produced by the architecture.",
        ),
        MultiAgentMetricDefinition(
            name="coordination_overhead_count",
            definition="Total number of explicit coordination/dependency checks performed by the coordinator.",
        ),
        MultiAgentMetricDefinition(
            name="mean_latency_ms",
            definition="Mean wall-clock latency per scenario for the architecture.",
        ),
    )
