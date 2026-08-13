"""LangGraph-backed orchestration for PartyPilot v0.7a."""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from time import perf_counter
from typing import Annotated, Any, TypedDict, cast
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from partypilot.application import multi_agent_runtime as runtime_module
from partypilot.application import v04_multi_agent as v04
from partypilot.application.multi_agent_runtime import CandidateRun, MultiAgentPlanningRuntime
from partypilot.application.review_workflow import (
    HumanReviewAction,
    HumanReviewRequest,
    HumanReviewResponse,
)
from partypilot.application.state_invalidation import StateInvalidationResult, apply_updates
from partypilot.domain import (
    ArbitrationOutcome,
    ArbitrationTrace,
    CapabilityBoundaryScenario,
    FeasibilityOutcome,
    PlanningDecisionCategory,
    PlanningDependencyKind,
    PlanningState,
    PlanningUpdate,
    PlanningUpdateKind,
    Resource,
    SpecialistDomain,
    SpecialistExecutionOutcome,
    SpecialistExecutionTrace,
    SpecialistFailureKind,
)

SPECIALIST_NODE_NAMES = ("venue", "catering", "accessibility", "scheduling", "budget")
SPECIALIST_DOMAIN_BY_NODE_NAME = {
    "venue": SpecialistDomain.VENUE,
    "catering": SpecialistDomain.CATERING_SAFETY,
    "accessibility": SpecialistDomain.ACCESSIBILITY,
    "scheduling": SpecialistDomain.SCHEDULING_OPERATIONS,
    "budget": SpecialistDomain.BUDGET,
}


class GraphTraceEventKind(StrEnum):
    NODE_ENTERED = "entered"
    NODE_COMPLETED = "completed"
    ROUTED = "routed"
    REPLAN_PLANNED = "replan_planned"
    SPECIALIST_RERUN_STARTED = "specialist_rerun_started"
    SPECIALIST_RERUN_COMPLETED = "specialist_rerun_completed"
    LOOP_BOUND_EXHAUSTED = "loop_bound_exhausted"
    HUMAN_REVIEW_REQUESTED = "human_review_requested"
    GRAPH_SUSPENDED = "graph_suspended"
    GRAPH_RESUMED = "graph_resumed"
    REVIEW_ACTION = "review_action"
    STALE_REVIEW_REJECTED = "stale_review_rejected"
    POST_REVIEW_ROUTE = "post_review_route"


class GraphTraceEvent(BaseModel):
    """PartyPilot-owned execution trace for a LangGraph node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_name: str = Field(min_length=1)
    event_kind: GraphTraceEventKind
    started_at: datetime
    completed_at: datetime | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    routing_decision: str | None = None
    failure_kind: str | None = None
    outcome: str | None = None
    execution_id: str | None = None
    review_revision: int | None = Field(default=None, ge=0)
    review_action: HumanReviewAction | None = None
    details: tuple[str, ...] = ()


class CandidateGraphExecutionStatus(StrEnum):
    COMPLETED = "completed"
    SUSPENDED_FOR_HUMAN_REVIEW = "suspended_for_human_review"


class CandidateGraphExecutionResult(BaseModel):
    """PartyPilot-owned wrapper for a resumable candidate execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: str = Field(min_length=1)
    status: CandidateGraphExecutionStatus
    scenario_id: str = Field(min_length=1)
    candidate_resource_ids: tuple[str, ...] = ()
    planning_revision: int = Field(ge=0)
    wall_clock_latency_ms: float = Field(ge=0)
    review_request: HumanReviewRequest | None = None
    review_response: HumanReviewResponse | None = None
    candidate_run: CandidateRun | None = None
    graph_trace: tuple[GraphTraceEvent, ...] = ()
    notes: tuple[str, ...] = ()


class LangGraphCandidateState(TypedDict, total=False):
    """Provider-neutral orchestration state for a single candidate evaluation."""

    execution_id: str
    scenario: CapabilityBoundaryScenario
    planning_state: PlanningState
    candidate_resources: tuple[Resource, ...]
    candidate_resource_ids: tuple[str, ...]
    hard_violation: runtime_module.GuardrailAssessment | None
    deterministic_resolution: runtime_module.DeterministicResolutionAssessment | None
    candidate_run: CandidateRun
    specialist_outcomes_by_domain: Annotated[
        dict[str, SpecialistExecutionOutcome], _merge_domain_outcomes
    ]
    execution_traces_by_domain: Annotated[dict[str, SpecialistExecutionTrace], _merge_domain_traces]
    graph_trace: Annotated[list[GraphTraceEvent], operator.add]
    replan_iteration: int
    max_replan_iterations: int
    replan_reason: str | None
    targeted_specialist_domains: tuple[str, ...]
    planning_invalidation: StateInvalidationResult | None


class HumanReviewSuspendedError(RuntimeError):
    """Raised when the batch path encounters a human-review suspension."""

    def __init__(self, suspended_result: CandidateGraphExecutionResult) -> None:
        super().__init__(
            f"candidate execution {suspended_result.execution_id} suspended for human review"
        )
        self.suspended_result = suspended_result


def _merge_domain_outcomes(
    existing: dict[str, SpecialistExecutionOutcome],
    new: dict[str, SpecialistExecutionOutcome],
) -> dict[str, SpecialistExecutionOutcome]:
    return {**existing, **new}


def _merge_domain_traces(
    existing: dict[str, SpecialistExecutionTrace],
    new: dict[str, SpecialistExecutionTrace],
) -> dict[str, SpecialistExecutionTrace]:
    return {**existing, **new}


class _ReviewSessionRecord(TypedDict):
    scenario: CapabilityBoundaryScenario
    scenario_id: str
    candidate_resource_ids: tuple[str, ...]
    planning_revision: int
    started_at_perf: float
    review_request: HumanReviewRequest | None


class LangGraphMultiAgentPlanningRuntime(MultiAgentPlanningRuntime):
    """LangGraph-backed orchestration wrapper for the live multi-agent runtime."""

    def __init__(
        self,
        specialists: Sequence[Any],
        *,
        model_name: str | None = None,
        max_workers: int | None = None,
        max_replan_iterations: int = 1,
        checkpointer: Any | None = None,
    ) -> None:
        super().__init__(specialists, model_name=model_name, max_workers=max_workers)
        self._max_replan_iterations = max(1, max_replan_iterations)
        self._checkpointer = checkpointer or InMemorySaver()
        self._review_sessions: dict[str, _ReviewSessionRecord] = {}
        self._graph = self._build_graph()
        self._compiled_graph = self._graph.compile(checkpointer=self._checkpointer)
        self.graph_trace_log: tuple[tuple[GraphTraceEvent, ...], ...] = ()
        self.last_graph_trace: tuple[GraphTraceEvent, ...] = ()

    def graph_mermaid(self) -> str:
        """Return a Mermaid rendering of the graph topology for developers."""

        return self._compiled_graph.get_graph().draw_mermaid()

    def _resolve_execution_id(self, execution_id: str | None) -> str:
        return execution_id or uuid4().hex

    def _candidate_resources_for_scenario(
        self,
        *,
        scenario: CapabilityBoundaryScenario,
        candidate_resource_ids: tuple[str, ...],
    ) -> tuple[Resource, ...]:
        resources_by_id = {
            resource.resource_id: resource for resource in scenario.structured_resources
        }
        try:
            return tuple(resources_by_id[resource_id] for resource_id in candidate_resource_ids)
        except KeyError as exc:  # pragma: no cover - defensive
            missing = exc.args[0]
            raise ValueError(f"unknown candidate resource_id: {missing}") from exc

    def _invoke_candidate_graph(
        self,
        *,
        scenario: CapabilityBoundaryScenario,
        planning_state: PlanningState,
        candidate_resources: tuple[Resource, ...],
        candidate_resource_ids: tuple[str, ...],
        execution_id: str,
        resume_response: HumanReviewResponse | None = None,
    ) -> dict[str, Any]:
        state: LangGraphCandidateState = {
            "execution_id": execution_id,
            "scenario": scenario,
            "planning_state": planning_state,
            "candidate_resources": candidate_resources,
            "candidate_resource_ids": candidate_resource_ids,
            "replan_iteration": 0,
            "max_replan_iterations": self._max_replan_iterations,
            "targeted_specialist_domains": (),
            "specialist_outcomes_by_domain": {},
            "execution_traces_by_domain": {},
        }
        config: Any = {
            "configurable": {"thread_id": execution_id},
            "max_concurrency": self._max_workers,
        }
        compiled_graph: Any = self._compiled_graph
        if resume_response is not None:
            resume_command: Any = Command(resume=resume_response.model_dump(mode="json"))
            return cast(
                dict[str, Any],
                compiled_graph.invoke(resume_command, config=config),
            )
        return cast(dict[str, Any], compiled_graph.invoke(state, config=config))

    def _candidate_graph_result_from_state(
        self,
        *,
        scenario: CapabilityBoundaryScenario,
        candidate_resource_ids: tuple[str, ...],
        execution_id: str,
        started_at: float,
        final_state: dict[str, Any],
        resume_response: HumanReviewResponse | None,
        raise_on_interrupt: bool,
    ) -> CandidateGraphExecutionResult:
        wall_clock_latency_ms = max(0.0, (perf_counter() - started_at) * 1000.0)
        graph_trace: list[GraphTraceEvent] = list(final_state.get("graph_trace", ()))
        if "__interrupt__" in final_state:
            interrupts = final_state["__interrupt__"]
            interrupt_payload = interrupts[0].value if interrupts else {}
            review_request = HumanReviewRequest.model_validate(interrupt_payload)
            selected_resources = ",".join(review_request.selected_resource_ids) or "none"
            targeted_domains = ",".join(review_request.targeted_domains) or "none"
            suspended_trace = (
                self._graph_trace_event(
                    node_name="human_review",
                    event_kind=GraphTraceEventKind.HUMAN_REVIEW_REQUESTED,
                    started_at=datetime.now(UTC),
                    completed_at=datetime.now(UTC),
                    execution_id=execution_id,
                    review_revision=review_request.planning_revision,
                    details=(
                        f"scenario_id={review_request.scenario_id}",
                        f"reason={review_request.review_reason}",
                        f"selected_resources={selected_resources}",
                        f"targeted_domains={targeted_domains}",
                    ),
                ),
                self._graph_trace_event(
                    node_name="human_review",
                    event_kind=GraphTraceEventKind.GRAPH_SUSPENDED,
                    started_at=datetime.now(UTC),
                    completed_at=datetime.now(UTC),
                    execution_id=execution_id,
                    review_revision=review_request.planning_revision,
                    details=("checkpointed_state=stored",),
                ),
            )
            graph_trace = [*graph_trace, *suspended_trace]
            suspended_result = CandidateGraphExecutionResult(
                execution_id=execution_id,
                status=CandidateGraphExecutionStatus.SUSPENDED_FOR_HUMAN_REVIEW,
                scenario_id=scenario.scenario.scenario_id,
                candidate_resource_ids=candidate_resource_ids,
                planning_revision=review_request.planning_revision,
                wall_clock_latency_ms=wall_clock_latency_ms,
                review_request=review_request,
                review_response=resume_response,
                candidate_run=None,
                graph_trace=tuple(graph_trace),
            )
            self._review_sessions[execution_id] = {
                "scenario": scenario,
                "scenario_id": scenario.scenario.scenario_id,
                "candidate_resource_ids": candidate_resource_ids,
                "planning_revision": review_request.planning_revision,
                "started_at_perf": started_at,
                "review_request": review_request,
            }
            self.last_graph_trace = tuple(graph_trace)
            self.graph_trace_log = (*self.graph_trace_log, self.last_graph_trace)
            if raise_on_interrupt:
                raise HumanReviewSuspendedError(suspended_result)
            return suspended_result

        session = self._review_sessions.get(execution_id)
        stored_review_request = session["review_request"] if session is not None else None
        candidate_run = cast(CandidateRun, final_state["candidate_run"])
        completed_candidate_run = candidate_run.model_copy(
            update={
                "coordinated_result": candidate_run.coordinated_result.model_copy(
                    update={
                        "latency_ms": wall_clock_latency_ms,
                    }
                )
            }
        )
        self.last_graph_trace = tuple(graph_trace)
        self.graph_trace_log = (*self.graph_trace_log, self.last_graph_trace)
        self._review_sessions.pop(execution_id, None)
        return CandidateGraphExecutionResult(
            execution_id=execution_id,
            status=CandidateGraphExecutionStatus.COMPLETED,
            scenario_id=scenario.scenario.scenario_id,
            candidate_resource_ids=candidate_resource_ids,
            planning_revision=cast(PlanningState, final_state["planning_state"]).revision_number,
            wall_clock_latency_ms=wall_clock_latency_ms,
            review_request=stored_review_request,
            review_response=resume_response,
            candidate_run=completed_candidate_run,
            graph_trace=tuple(graph_trace),
        )

    def run_reviewable_candidate(
        self,
        *,
        scenario: CapabilityBoundaryScenario,
        candidate_resource_ids: tuple[str, ...],
        execution_id: str | None = None,
    ) -> CandidateGraphExecutionResult:
        """Run one candidate evaluation and expose suspension as a typed result."""

        execution_id = self._resolve_execution_id(execution_id)
        if execution_id in self._review_sessions:
            raise ValueError(f"execution_id {execution_id!r} is already active")
        candidate_resources = self._candidate_resources_for_scenario(
            scenario=scenario,
            candidate_resource_ids=candidate_resource_ids,
        )
        planning_state = runtime_module._build_planning_state(scenario, candidate_resources)
        started = perf_counter()
        self._review_sessions[execution_id] = {
            "scenario": scenario,
            "scenario_id": scenario.scenario.scenario_id,
            "candidate_resource_ids": candidate_resource_ids,
            "planning_revision": planning_state.revision_number,
            "started_at_perf": started,
            "review_request": None,
        }
        final_state = self._invoke_candidate_graph(
            scenario=scenario,
            planning_state=planning_state,
            candidate_resources=candidate_resources,
            candidate_resource_ids=candidate_resource_ids,
            execution_id=execution_id,
        )
        return self._candidate_graph_result_from_state(
            scenario=scenario,
            candidate_resource_ids=candidate_resource_ids,
            execution_id=execution_id,
            started_at=started,
            final_state=final_state,
            resume_response=None,
            raise_on_interrupt=False,
        )

    def resume_reviewable_candidate(
        self,
        *,
        execution_id: str,
        review_response: HumanReviewResponse,
    ) -> CandidateGraphExecutionResult:
        """Resume a suspended candidate evaluation from the stored checkpoint."""

        if execution_id not in self._review_sessions:
            raise ValueError(f"execution_id {execution_id!r} is not suspended")
        session = self._review_sessions[execution_id]
        if review_response.execution_id != execution_id:
            raise ValueError("review response execution_id does not match the suspended execution")
        started = session["started_at_perf"]
        final_state = self._invoke_candidate_graph(
            scenario=session["scenario"],
            planning_state=runtime_module._build_planning_state(
                session["scenario"],
                self._candidate_resources_for_scenario(
                    scenario=session["scenario"],
                    candidate_resource_ids=session["candidate_resource_ids"],
                ),
            ),
            candidate_resources=self._candidate_resources_for_scenario(
                scenario=session["scenario"],
                candidate_resource_ids=session["candidate_resource_ids"],
            ),
            candidate_resource_ids=session["candidate_resource_ids"],
            execution_id=execution_id,
            resume_response=review_response,
        )
        return self._candidate_graph_result_from_state(
            scenario=session["scenario"],
            candidate_resource_ids=session["candidate_resource_ids"],
            execution_id=execution_id,
            started_at=started,
            final_state=final_state,
            resume_response=review_response,
            raise_on_interrupt=False,
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
        execution_id = uuid4().hex
        final_state = self._invoke_candidate_graph(
            scenario=scenario,
            planning_state=planning_state,
            candidate_resources=candidate_resources,
            candidate_resource_ids=candidate_resource_ids,
            execution_id=execution_id,
        )
        if "__interrupt__" in final_state:
            suspended = self._candidate_graph_result_from_state(
                scenario=scenario,
                candidate_resource_ids=candidate_resource_ids,
                execution_id=execution_id,
                started_at=started,
                final_state=final_state,
                resume_response=None,
                raise_on_interrupt=False,
            )
            candidate_run = cast(CandidateRun | None, final_state.get("candidate_run"))
            if candidate_run is None:
                raise HumanReviewSuspendedError(suspended)
            finalize_started = datetime.now(UTC)
            finalize_trace = (
                self._graph_trace_event(
                    node_name="finalize",
                    event_kind=GraphTraceEventKind.NODE_ENTERED,
                    started_at=finalize_started,
                    completed_at=None,
                ),
                self._graph_trace_event(
                    node_name="finalize",
                    event_kind=GraphTraceEventKind.NODE_COMPLETED,
                    started_at=finalize_started,
                    completed_at=datetime.now(UTC),
                    latency_ms=max(0.0, (perf_counter() - started) * 1000.0),
                    routing_decision="end",
                    outcome=candidate_run.coordinated_result.feasibility_outcome.value,
                    details=(
                        f"specialist_calls={candidate_run.decision_count}",
                        f"failure_stage={candidate_run.coordinated_result.failure_stage or 'none'}",
                    ),
                ),
            )
            self.last_graph_trace = (*suspended.graph_trace, *finalize_trace)
            self.graph_trace_log = (*self.graph_trace_log, self.last_graph_trace)
            return candidate_run.model_copy(
                update={
                    "coordinated_result": candidate_run.coordinated_result.model_copy(
                        update={
                            "latency_ms": max(0.0, (perf_counter() - started) * 1000.0),
                        }
                    )
                }
            )
        candidate_run = cast(CandidateRun, final_state["candidate_run"])
        self.last_graph_trace = tuple(final_state.get("graph_trace", ()))
        self.graph_trace_log = (*self.graph_trace_log, self.last_graph_trace)
        # Keep the candidate-run latency aligned with the surrounding runtime's wall clock.
        return candidate_run.model_copy(
            update={
                "coordinated_result": candidate_run.coordinated_result.model_copy(
                    update={
                        "latency_ms": max(0.0, (perf_counter() - started) * 1000.0),
                    }
                )
            }
        )

    def _build_candidate_run(
        self,
        *,
        scenario: CapabilityBoundaryScenario,
        planning_state: PlanningState,
        candidate_resources: tuple[Resource, ...],
        candidate_resource_ids: tuple[str, ...],
        specialist_outcomes: tuple[SpecialistExecutionOutcome, ...],
    ) -> CandidateRun:
        started = perf_counter()
        ordered_outcomes = self._ordered_specialist_outcomes(specialist_outcomes)

        hard_violation = runtime_module._deterministic_hard_violation(scenario, candidate_resources)
        deterministic_resolution = runtime_module._deterministic_resolution_assessment(
            scenario=scenario,
            candidate_resources=candidate_resources,
            hard_violation=hard_violation,
        )
        if hard_violation is not None and hard_violation.proven_hard_violation:
            return self._build_terminal_candidate_run(
                scenario=scenario,
                planning_state=planning_state,
                candidate_resources=candidate_resources,
                candidate_resource_ids=candidate_resource_ids,
                hard_violation=hard_violation,
            )

        accepted_ids = tuple(
            outcome.decision.specialist_id
            for outcome in ordered_outcomes
            if outcome.decision is not None and outcome.decision.status is ArbitrationOutcome.ACCEPT
        )
        rejected_ids = tuple(
            outcome.decision.specialist_id
            for outcome in ordered_outcomes
            if outcome.decision is not None and outcome.decision.status is ArbitrationOutcome.REJECT
        )

        if hard_violation is not None and not hard_violation.proven_hard_violation:
            arbitration = ArbitrationTrace(
                outcome=ArbitrationOutcome.HUMAN_REVIEW_REQUIRED,
                feasibility_outcome=FeasibilityOutcome.HUMAN_REVIEW_REQUIRED,
                selected_resource_ids=candidate_resource_ids,
                accepted_specialist_ids=accepted_ids,
                rejected_specialist_ids=rejected_ids,
                overridden_specialist_ids=tuple(
                    outcome.decision.specialist_id
                    for outcome in ordered_outcomes
                    if outcome.decision is not None
                    and outcome.decision.status is not ArbitrationOutcome.REJECT
                ),
                controlling_evidence_ids=runtime_module._guardrail_controlling_evidence_ids(
                    hard_violation=hard_violation,
                    specialist_outcomes=ordered_outcomes,
                ),
                dependency_conflicts=(),
                unresolved_uncertainties=(hard_violation.reason,),
                reasons=(hard_violation.reason,),
                global_score=runtime_module._candidate_total_cost_from_resources(
                    scenario, candidate_resources
                ),
                coordination_steps=tuple(
                    f"specialist:{outcome.trace.specialist_id}:"
                    f"{outcome.decision.status.value if outcome.decision is not None else 'failed'}"
                    for outcome in ordered_outcomes
                ),
            )
            coordinated_result = runtime_module._coordinated_result_from_arbitration(
                scenario=scenario,
                candidate_resources=candidate_resources,
                arbitration=arbitration,
                specialist_outcomes=ordered_outcomes,
                decision_count=sum(
                    1 for outcome in ordered_outcomes if outcome.decision is not None
                ),
                elapsed_ms=max(0.0, (perf_counter() - started) * 1000.0),
                failure_kind=None,
            )
            return CandidateRun(
                coordinated_result=coordinated_result,
                arbitration=arbitration,
                selected_resource_ids=candidate_resource_ids,
                total_cost=coordinated_result.total_cost,
                execution_traces=tuple(outcome.trace for outcome in ordered_outcomes),
                specialist_outcomes=ordered_outcomes,
                decision_count=sum(
                    1 for outcome in ordered_outcomes if outcome.decision is not None
                ),
                planning_state=planning_state,
            )

        if any(outcome.failure_kind is not None for outcome in ordered_outcomes):
            critical_failures = [
                outcome
                for outcome in ordered_outcomes
                if outcome.failure_kind is not None and outcome.trace.specialist_id != "budget"
            ]
            budget_failures = [
                outcome
                for outcome in ordered_outcomes
                if outcome.failure_kind is not None and outcome.trace.specialist_id == "budget"
            ]

            if critical_failures:
                arbitration = runtime_module._failure_arbitration(
                    candidate_resource_ids=candidate_resource_ids,
                    specialist_outcomes=ordered_outcomes,
                    reason="One or more critical specialists failed.",
                )
                coordinated_result = runtime_module._coordinated_result_from_arbitration(
                    scenario=scenario,
                    candidate_resources=candidate_resources,
                    arbitration=arbitration,
                    specialist_outcomes=ordered_outcomes,
                    decision_count=sum(
                        1 for outcome in ordered_outcomes if outcome.decision is not None
                    ),
                    elapsed_ms=max(0.0, (perf_counter() - started) * 1000.0),
                    failure_kind=None,
                )
                return CandidateRun(
                    coordinated_result=coordinated_result,
                    arbitration=arbitration,
                    selected_resource_ids=candidate_resource_ids,
                    total_cost=coordinated_result.total_cost,
                    execution_traces=tuple(outcome.trace for outcome in ordered_outcomes),
                    specialist_outcomes=ordered_outcomes,
                    decision_count=sum(
                        1 for outcome in ordered_outcomes if outcome.decision is not None
                    ),
                    planning_state=planning_state,
                )

            if budget_failures:
                budget_specialist = self._specialist_by_domain[SpecialistDomain.BUDGET]
                budget_outcome = runtime_module._run_specialist_invocation(
                    budget_specialist,
                    runtime_module._specialist_input(
                        scenario=scenario,
                        planning_state=planning_state,
                        candidate_resources=candidate_resources,
                        candidate_resource_ids=candidate_resource_ids,
                        specialist=budget_specialist,
                    ),
                )
                filtered_outcomes = tuple(
                    outcome
                    for outcome in ordered_outcomes
                    if outcome.trace.specialist_id != "budget"
                )
                ordered_outcomes = (*filtered_outcomes, budget_outcome)

        decisions = tuple(
            outcome.decision for outcome in ordered_outcomes if outcome.decision is not None
        )
        if any(outcome.decision is None for outcome in ordered_outcomes):
            arbitration = runtime_module._failure_arbitration(
                candidate_resource_ids=candidate_resource_ids,
                specialist_outcomes=ordered_outcomes,
                reason="A non-budget specialist failed.",
            )
            coordinated_result = runtime_module._coordinated_result_from_arbitration(
                scenario=scenario,
                candidate_resources=candidate_resources,
                arbitration=arbitration,
                specialist_outcomes=ordered_outcomes,
                decision_count=len(decisions),
                elapsed_ms=max(0.0, (perf_counter() - started) * 1000.0),
                failure_kind=None,
            )
            return CandidateRun(
                coordinated_result=coordinated_result,
                arbitration=arbitration,
                selected_resource_ids=candidate_resource_ids,
                total_cost=coordinated_result.total_cost,
                execution_traces=tuple(outcome.trace for outcome in ordered_outcomes),
                specialist_outcomes=ordered_outcomes,
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
            arbitration = runtime_module._failure_arbitration(
                candidate_resource_ids=candidate_resource_ids,
                specialist_outcomes=ordered_outcomes,
                reason=f"Coordinator arbitration failed: {type(exc).__name__}: {exc}",
            )
            coordinated_result = runtime_module._coordinated_result_from_arbitration(
                scenario=scenario,
                candidate_resources=candidate_resources,
                arbitration=arbitration,
                specialist_outcomes=ordered_outcomes,
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
                execution_traces=tuple(outcome.trace for outcome in ordered_outcomes),
                specialist_outcomes=ordered_outcomes,
                decision_count=len(decisions),
                planning_state=planning_state,
            )

        elapsed_ms = max(0.0, (perf_counter() - started) * 1000.0)
        coordinated_result = runtime_module._coordinated_result_from_trace(
            scenario=scenario,
            candidate_resources=candidate_resources,
            arbitration=arbitration,
            selected=selected,
            total_cost=total_cost,
            specialist_outcomes=ordered_outcomes,
            decision_count=len(decisions),
            elapsed_ms=elapsed_ms,
            failure_kind=None,
        )
        if (
            any(
                outcome.decision is not None
                and outcome.decision.status == ArbitrationOutcome.REPLAN_REQUIRED
                for outcome in ordered_outcomes
            )
            and coordinated_result.feasibility_outcome == FeasibilityOutcome.FEASIBLE
        ):
            arbitration = arbitration.model_copy(
                update={
                    "outcome": ArbitrationOutcome.REPLAN_REQUIRED,
                    "feasibility_outcome": FeasibilityOutcome.HUMAN_REVIEW_REQUIRED,
                    "reasons": (
                        *arbitration.reasons,
                        "A dependency issue requires replanning.",
                    ),
                }
            )
            coordinated_result = runtime_module._coordinated_result_from_arbitration(
                scenario=scenario,
                candidate_resources=candidate_resources,
                arbitration=arbitration,
                specialist_outcomes=ordered_outcomes,
                decision_count=len(
                    [outcome for outcome in ordered_outcomes if outcome.decision is not None]
                ),
                elapsed_ms=elapsed_ms,
                failure_kind=None,
            )
        elif (
            deterministic_resolution.state
            is runtime_module.DeterministicResolutionState.PROVEN_FEASIBLE
            and coordinated_result.feasibility_outcome != FeasibilityOutcome.FEASIBLE
        ):
            accepted_ids = tuple(
                outcome.decision.specialist_id
                for outcome in ordered_outcomes
                if outcome.decision is not None
                and outcome.decision.status is ArbitrationOutcome.ACCEPT
            )
            rejected_ids = tuple(
                outcome.decision.specialist_id
                for outcome in ordered_outcomes
                if outcome.decision is not None
                and outcome.decision.status is ArbitrationOutcome.REJECT
            )
            controlling_evidence_ids = tuple(
                dict.fromkeys(
                    runtime_module._selected_resource_evidence_ids(scenario, candidate_resource_ids)
                    + tuple(
                        evidence.evidence_id
                        for outcome in ordered_outcomes
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
                        for outcome in ordered_outcomes
                        if outcome.decision is not None
                        and outcome.decision.status is not ArbitrationOutcome.ACCEPT
                    ),
                    "controlling_evidence_ids": controlling_evidence_ids,
                    "dependency_conflicts": (),
                    "unresolved_uncertainties": (),
                    "reasons": (deterministic_resolution.reason,),
                }
            )
            coordinated_result = runtime_module._coordinated_result_from_arbitration(
                scenario=scenario,
                candidate_resources=candidate_resources,
                arbitration=arbitration,
                specialist_outcomes=ordered_outcomes,
                decision_count=len(
                    [outcome for outcome in ordered_outcomes if outcome.decision is not None]
                ),
                elapsed_ms=elapsed_ms,
                failure_kind=None,
            )
        return CandidateRun(
            coordinated_result=coordinated_result,
            arbitration=arbitration,
            selected_resource_ids=selected,
            total_cost=total_cost,
            execution_traces=tuple(outcome.trace for outcome in ordered_outcomes),
            specialist_outcomes=ordered_outcomes,
            decision_count=len(decisions),
            planning_state=planning_state,
        )

    def _build_graph(self) -> StateGraph[LangGraphCandidateState]:
        graph: StateGraph[LangGraphCandidateState] = StateGraph(LangGraphCandidateState)
        graph.add_node("preflight", cast(Any, self._preflight_node))
        for node_name in SPECIALIST_NODE_NAMES:
            graph.add_node(node_name, cast(Any, self._specialist_node(node_name)))
        graph.add_node("coordinator", cast(Any, self._coordinator_node))
        graph.add_node("replan", cast(Any, self._replan_node))
        graph.add_node("human_review", cast(Any, self._human_review_node))
        graph.add_node("finalize", cast(Any, self._finalize_node))
        graph.add_edge(START, "preflight")
        graph.add_edge(list(SPECIALIST_NODE_NAMES), "coordinator")
        graph.add_edge("human_review", "finalize")
        graph.add_edge("finalize", END)
        return graph

    def _preflight_node(self, state: LangGraphCandidateState) -> Command[Any]:
        started = datetime.now(UTC)
        scenario = state["scenario"]
        candidate_resources = state["candidate_resources"]
        hard_violation = runtime_module._deterministic_hard_violation(scenario, candidate_resources)
        deterministic_resolution = runtime_module._deterministic_resolution_assessment(
            scenario=scenario,
            candidate_resources=candidate_resources,
            hard_violation=hard_violation,
        )
        routing_decision = (
            "finalize"
            if hard_violation is not None and hard_violation.proven_hard_violation
            else "fan_out"
        )
        trace_entered = self._graph_trace_event(
            node_name="preflight",
            event_kind=GraphTraceEventKind.NODE_ENTERED,
            started_at=started,
            completed_at=None,
        )
        trace_completed = self._graph_trace_event(
            node_name="preflight",
            event_kind=GraphTraceEventKind.NODE_COMPLETED,
            started_at=started,
            completed_at=datetime.now(UTC),
            routing_decision=routing_decision,
            outcome=deterministic_resolution.state.value,
            details=(deterministic_resolution.reason,),
        )
        update: dict[str, Any] = {
            "hard_violation": hard_violation,
            "deterministic_resolution": deterministic_resolution,
            "graph_trace": [trace_entered, trace_completed],
        }
        if routing_decision == "finalize":
            return Command(update=update, goto="finalize")
        return Command(
            update=update,
            goto=[Send(node_name, {**state, **update}) for node_name in SPECIALIST_NODE_NAMES],
        )

    def _specialist_node(
        self,
        node_name: str,
    ) -> Callable[[LangGraphCandidateState], dict[str, Any]]:
        specialist = self._specialist_by_domain[SPECIALIST_DOMAIN_BY_NODE_NAME[node_name]]

        def node(state: LangGraphCandidateState) -> dict[str, Any]:
            started = datetime.now(UTC)
            rerun = state.get("replan_iteration", 0) > 0
            entered_kind = (
                GraphTraceEventKind.SPECIALIST_RERUN_STARTED
                if rerun
                else GraphTraceEventKind.NODE_ENTERED
            )
            completed_kind = (
                GraphTraceEventKind.SPECIALIST_RERUN_COMPLETED
                if rerun
                else GraphTraceEventKind.NODE_COMPLETED
            )
            outcome = runtime_module._run_specialist_invocation(
                specialist,
                runtime_module._specialist_input(
                    scenario=state["scenario"],
                    planning_state=state["planning_state"],
                    candidate_resources=state["candidate_resources"],
                    candidate_resource_ids=state["candidate_resource_ids"],
                    specialist=specialist,
                ),
            )
            completed = datetime.now(UTC)
            details = (
                f"specialist_id={outcome.trace.specialist_id}",
                f"validation_succeeded={outcome.trace.validation_succeeded}",
                f"planning_revision={state['planning_state'].revision_number}",
                f"replan_iteration={state.get('replan_iteration', 0)}",
                f"rerun={rerun}",
            )
            return {
                "specialist_outcomes_by_domain": {node_name: outcome},
                "execution_traces_by_domain": {node_name: outcome.trace},
                "graph_trace": [
                    self._graph_trace_event(
                        node_name=node_name,
                        event_kind=entered_kind,
                        started_at=started,
                        completed_at=None,
                    ),
                    self._graph_trace_event(
                        node_name=node_name,
                        event_kind=completed_kind,
                        started_at=started,
                        completed_at=completed,
                        latency_ms=outcome.trace.latency_ms,
                        failure_kind=(
                            outcome.failure_kind.value if outcome.failure_kind is not None else None
                        ),
                        outcome=(
                            outcome.trace.recommendation_status.value
                            if outcome.trace.recommendation_status is not None
                            else "failed"
                        ),
                        details=details,
                    ),
                ],
            }

        return node

    def _coordinator_node(self, state: LangGraphCandidateState) -> Command[Any]:
        started = datetime.now(UTC)
        current_outcomes = self._current_specialist_outcomes(state)
        candidate_run = self._build_candidate_run(
            scenario=state["scenario"],
            planning_state=state["planning_state"],
            candidate_resources=state["candidate_resources"],
            candidate_resource_ids=state["candidate_resource_ids"],
            specialist_outcomes=current_outcomes,
        )
        completed = datetime.now(UTC)
        route = self._route_for_candidate_run(state, candidate_run)
        if any(
            outcome.decision is not None
            and outcome.decision.status == ArbitrationOutcome.REPLAN_REQUIRED
            for outcome in current_outcomes
        ):
            route = (
                "replan"
                if state.get("replan_iteration", 0) < state["max_replan_iterations"]
                else "human_review"
            )
        return Command(
            update={
                "candidate_run": candidate_run,
                "coordinator_route": route,
                "graph_trace": [
                    self._graph_trace_event(
                        node_name="coordinator",
                        event_kind=GraphTraceEventKind.NODE_ENTERED,
                        started_at=started,
                        completed_at=None,
                    ),
                    self._graph_trace_event(
                        node_name="coordinator",
                        event_kind=GraphTraceEventKind.NODE_COMPLETED,
                        started_at=started,
                        completed_at=completed,
                        latency_ms=candidate_run.coordinated_result.latency_ms,
                        routing_decision=route,
                        outcome=candidate_run.coordinated_result.feasibility_outcome.value,
                        details=(
                            f"specialist_calls={candidate_run.decision_count}",
                            (
                                f"selected_resources="
                                f"{','.join(candidate_run.selected_resource_ids) or 'none'}"
                            ),
                            f"planning_revision={state['planning_state'].revision_number}",
                            f"replan_iteration={state.get('replan_iteration', 0)}",
                        ),
                    ),
                ],
            },
            goto=route,
        )

    def _route_after_coordinator(self, state: LangGraphCandidateState) -> str:
        candidate_run = state.get("candidate_run")
        if candidate_run is None:
            return "finalize"
        route = self._route_for_candidate_run(state, candidate_run)
        return route

    def _route_after_replan(self, state: LangGraphCandidateState) -> str:
        if state.get("targeted_specialist_domains"):
            return "specialists"
        return "human_review"

    def _build_human_review_request(
        self,
        *,
        state: LangGraphCandidateState,
        candidate_run: CandidateRun,
    ) -> HumanReviewRequest:
        execution_id = state["execution_id"]
        review_reason = (
            candidate_run.arbitration.reasons[0]
            if candidate_run.arbitration.reasons
            else candidate_run.coordinated_result.failure_stage or "Human review required."
        )
        unresolved_issues = candidate_run.arbitration.unresolved_uncertainties or (review_reason,)
        targeted_domains = state.get("targeted_specialist_domains", ())
        return HumanReviewRequest(
            execution_id=execution_id,
            scenario_id=state["scenario"].scenario.scenario_id,
            planning_revision=state["planning_state"].revision_number,
            review_reason=review_reason,
            selected_resource_ids=candidate_run.selected_resource_ids,
            controlling_evidence_ids=candidate_run.arbitration.controlling_evidence_ids,
            unresolved_issues=unresolved_issues,
            targeted_domains=targeted_domains,
            notes=(
                f"feasibility={candidate_run.coordinated_result.feasibility_outcome.value}",
                f"planning_revision={state['planning_state'].revision_number}",
                f"candidate_calls={candidate_run.decision_count}",
            ),
        )

    def _review_trace_events(
        self,
        *,
        state: LangGraphCandidateState,
        review_request: HumanReviewRequest,
        review_response: HumanReviewResponse | None,
        review_action: HumanReviewAction | None,
        route: str,
        stale: bool = False,
    ) -> tuple[GraphTraceEvent, ...]:
        now = datetime.now(UTC)
        selected_resources = ",".join(review_request.selected_resource_ids) or "none"
        targeted_domains = ",".join(review_request.targeted_domains) or "none"
        events: list[GraphTraceEvent] = [
            self._graph_trace_event(
                node_name="human_review",
                event_kind=GraphTraceEventKind.NODE_ENTERED,
                started_at=now,
                completed_at=None,
                execution_id=review_request.execution_id,
                review_revision=review_request.planning_revision,
                details=(
                    f"scenario_id={review_request.scenario_id}",
                    f"selected_resources={selected_resources}",
                    f"targeted_domains={targeted_domains}",
                ),
            )
        ]
        if stale:
            response_candidate_resource_ids = (
                ",".join(review_response.candidate_resource_ids) if review_response else "n/a"
            )
            events.append(
                self._graph_trace_event(
                    node_name="human_review",
                    event_kind=GraphTraceEventKind.STALE_REVIEW_REJECTED,
                    started_at=now,
                    completed_at=now,
                    execution_id=review_request.execution_id,
                    review_revision=review_request.planning_revision,
                    details=(
                        f"expected_revision={state['planning_state'].revision_number}",
                        (
                            f"response_revision="
                            f"{review_response.planning_revision if review_response else 'n/a'}"
                        ),
                        f"response_candidate_resource_ids={response_candidate_resource_ids}",
                    ),
                )
            )
        elif review_action is not None:
            events.append(
                self._graph_trace_event(
                    node_name="human_review",
                    event_kind=GraphTraceEventKind.GRAPH_RESUMED,
                    started_at=now,
                    completed_at=now,
                    execution_id=review_request.execution_id,
                    review_revision=review_request.planning_revision,
                )
            )
            events.append(
                self._graph_trace_event(
                    node_name="human_review",
                    event_kind=GraphTraceEventKind.REVIEW_ACTION,
                    started_at=now,
                    completed_at=now,
                    execution_id=review_request.execution_id,
                    review_revision=review_request.planning_revision,
                    review_action=review_action,
                    details=(
                        f"candidate_resource_ids={selected_resources}",
                        f"route={route}",
                    ),
                )
            )
        events.append(
            self._graph_trace_event(
                node_name="human_review",
                event_kind=GraphTraceEventKind.POST_REVIEW_ROUTE,
                started_at=now,
                completed_at=now,
                execution_id=review_request.execution_id,
                review_revision=review_request.planning_revision,
                review_action=review_action,
                routing_decision=route,
            )
        )
        return tuple(events)

    def _finalize_node(self, state: LangGraphCandidateState) -> dict[str, Any]:
        started = datetime.now(UTC)
        candidate_run = state.get("candidate_run")
        if candidate_run is None:
            candidate_run = self._build_terminal_candidate_run(
                scenario=state["scenario"],
                planning_state=state["planning_state"],
                candidate_resources=state["candidate_resources"],
                candidate_resource_ids=state["candidate_resource_ids"],
                hard_violation=state.get("hard_violation"),
            )
        completed = datetime.now(UTC)
        return {
            "candidate_run": candidate_run,
            "graph_trace": [
                self._graph_trace_event(
                    node_name="finalize",
                    event_kind=GraphTraceEventKind.NODE_ENTERED,
                    started_at=started,
                    completed_at=None,
                ),
                self._graph_trace_event(
                    node_name="finalize",
                    event_kind=GraphTraceEventKind.NODE_COMPLETED,
                    started_at=started,
                    completed_at=completed,
                    latency_ms=candidate_run.coordinated_result.latency_ms,
                    routing_decision="end",
                    outcome=candidate_run.coordinated_result.feasibility_outcome.value,
                    details=(
                        f"specialist_calls={candidate_run.decision_count}",
                        f"failure_stage={candidate_run.coordinated_result.failure_stage or 'none'}",
                    ),
                ),
            ],
        }

    def _human_review_node(self, state: LangGraphCandidateState) -> Command[Any]:
        candidate_run = state.get("candidate_run")
        if candidate_run is None:
            candidate_run = self._build_terminal_candidate_run(
                scenario=state["scenario"],
                planning_state=state["planning_state"],
                candidate_resources=state["candidate_resources"],
                candidate_resource_ids=state["candidate_resource_ids"],
                hard_violation=state.get("hard_violation"),
            )
        review_request = self._build_human_review_request(state=state, candidate_run=candidate_run)
        resume_payload = interrupt(review_request.model_dump(mode="json"))
        try:
            review_response = HumanReviewResponse.model_validate(resume_payload)
        except ValidationError:
            return Command(
                update={
                    "graph_trace": list(
                        self._review_trace_events(
                            state=state,
                            review_request=review_request,
                            review_response=None,
                            review_action=None,
                            route="finalize",
                            stale=True,
                        )
                    )
                },
                goto="finalize",
            )
        stale = (
            review_response.execution_id != review_request.execution_id
            or review_response.planning_revision != review_request.planning_revision
            or not review_response.candidate_resource_ids
            or tuple(review_response.candidate_resource_ids)
            != tuple(review_request.selected_resource_ids)
        )
        route = "finalize"
        graph_update: dict[str, Any] = {}
        review_action: HumanReviewAction | None = review_response.action

        if stale:
            graph_update["graph_trace"] = list(
                self._review_trace_events(
                    state=state,
                    review_request=review_request,
                    review_response=review_response,
                    review_action=None,
                    route=route,
                    stale=True,
                )
            )
            return Command(update=graph_update, goto=route)

        if review_response.action is HumanReviewAction.REQUEST_REPLAN:
            route = "replan"
        elif review_response.action is HumanReviewAction.REJECT_CURRENT_PLAN:
            arbitration = candidate_run.arbitration.model_copy(
                update={
                    "outcome": ArbitrationOutcome.HUMAN_REVIEW_REQUIRED,
                    "feasibility_outcome": FeasibilityOutcome.HUMAN_REVIEW_REQUIRED,
                    "reasons": (
                        *candidate_run.arbitration.reasons,
                        "Human reviewer rejected the current plan.",
                    ),
                }
            )
            coordinated_result = candidate_run.coordinated_result.model_copy(
                update={
                    "feasibility_outcome": FeasibilityOutcome.HUMAN_REVIEW_REQUIRED,
                    "arbitration": arbitration,
                    "failure_stage": "human_review_rejected",
                }
            )
            graph_update["candidate_run"] = candidate_run.model_copy(
                update={
                    "arbitration": arbitration,
                    "coordinated_result": coordinated_result,
                }
            )
        elif (
            review_response.action is HumanReviewAction.APPROVE_CURRENT_PLAN
            and candidate_run.coordinated_result.feasibility_outcome is FeasibilityOutcome.FEASIBLE
        ):
            route = "finalize"
        else:
            route = "finalize"

        graph_update["graph_trace"] = list(
            self._review_trace_events(
                state=state,
                review_request=review_request,
                review_response=review_response,
                review_action=review_action,
                route=route,
            )
        )
        if route == "replan":
            return Command(update=graph_update, goto="replan")
        return Command(update=graph_update, goto="finalize")

    def _replan_node(self, state: LangGraphCandidateState) -> Command[Any]:
        started = datetime.now(UTC)
        current_outcomes = self._current_specialist_outcomes(state)
        replan_required_outcomes = tuple(
            outcome
            for outcome in current_outcomes
            if outcome.decision is not None
            and outcome.decision.status == ArbitrationOutcome.REPLAN_REQUIRED
        )
        targeted_domains = self._targeted_domains_from_replan(
            planning_state=state["planning_state"],
            outcomes=current_outcomes,
        )
        update = self._replan_update_for_outcomes(replan_required_outcomes)
        invalidation = apply_updates(state["planning_state"], (update,))
        next_revision = invalidation.updated_state.revision_number
        targeted_domains = targeted_domains or tuple(
            outcome.trace.specialist_id for outcome in replan_required_outcomes
        )
        trace_entered = self._graph_trace_event(
            node_name="replan",
            event_kind=GraphTraceEventKind.NODE_ENTERED,
            started_at=started,
            completed_at=None,
        )
        trace_completed = self._graph_trace_event(
            node_name="replan",
            event_kind=GraphTraceEventKind.REPLAN_PLANNED,
            started_at=started,
            completed_at=datetime.now(UTC),
            routing_decision=(
                "specialists"
                if targeted_domains
                and state.get("replan_iteration", 0) < state["max_replan_iterations"]
                else "human_review"
            ),
            outcome=invalidation.updated_state.revision_number.__str__(),
            details=(
                f"replan_reason={self._replan_reason_for_outcomes(replan_required_outcomes)}",
                f"targeted_domains={','.join(targeted_domains) or 'none'}",
                f"planning_revision={next_revision}",
                f"replan_iteration={state.get('replan_iteration', 0) + 1}",
                (
                    "invalidated_decisions="
                    f"{','.join(invalidation.invalidated_decision_ids) or 'none'}"
                ),
            ),
        )
        replan_iteration = state.get("replan_iteration", 0) + 1
        if (
            not targeted_domains
            or state.get("replan_iteration", 0) >= state["max_replan_iterations"]
        ):
            return Command(
                update={
                    "planning_state": invalidation.updated_state,
                    "planning_invalidation": invalidation,
                    "replan_reason": self._replan_reason_for_outcomes(replan_required_outcomes),
                    "targeted_specialist_domains": targeted_domains,
                    "replan_iteration": replan_iteration,
                    "graph_trace": [trace_entered, trace_completed],
                },
                goto="human_review",
            )
        rerun_specialist_outcomes: dict[str, SpecialistExecutionOutcome] = {}
        rerun_execution_traces: dict[str, SpecialistExecutionTrace] = {}
        rerun_trace_events: list[GraphTraceEvent] = []
        for domain in targeted_domains:
            if domain not in SPECIALIST_NODE_NAMES:
                continue
            specialist = self._specialist_by_domain[SPECIALIST_DOMAIN_BY_NODE_NAME[domain]]
            rerun_started = datetime.now(UTC)
            rerun_trace_events.append(
                self._graph_trace_event(
                    node_name=domain,
                    event_kind=GraphTraceEventKind.SPECIALIST_RERUN_STARTED,
                    started_at=rerun_started,
                    completed_at=None,
                )
            )
            rerun_outcome = runtime_module._run_specialist_invocation(
                specialist,
                runtime_module._specialist_input(
                    scenario=state["scenario"],
                    planning_state=invalidation.updated_state,
                    candidate_resources=state["candidate_resources"],
                    candidate_resource_ids=state["candidate_resource_ids"],
                    specialist=specialist,
                ),
            )
            rerun_completed = datetime.now(UTC)
            rerun_specialist_outcomes[domain] = rerun_outcome
            rerun_execution_traces[domain] = rerun_outcome.trace
            rerun_trace_events.append(
                self._graph_trace_event(
                    node_name=domain,
                    event_kind=GraphTraceEventKind.SPECIALIST_RERUN_COMPLETED,
                    started_at=rerun_started,
                    completed_at=rerun_completed,
                    latency_ms=rerun_outcome.trace.latency_ms,
                    failure_kind=(
                        rerun_outcome.failure_kind.value if rerun_outcome.failure_kind else None
                    ),
                    outcome=(
                        rerun_outcome.trace.recommendation_status.value
                        if rerun_outcome.trace.recommendation_status is not None
                        else "failed"
                    ),
                    details=(
                        f"specialist_id={rerun_outcome.trace.specialist_id}",
                        f"validation_succeeded={rerun_outcome.trace.validation_succeeded}",
                        f"planning_revision={invalidation.updated_state.revision_number}",
                        f"replan_iteration={replan_iteration}",
                        "rerun=True",
                    ),
                )
            )
        return Command(
            update={
                "planning_state": invalidation.updated_state,
                "planning_invalidation": invalidation,
                "replan_reason": self._replan_reason_for_outcomes(replan_required_outcomes),
                "targeted_specialist_domains": targeted_domains,
                "replan_iteration": replan_iteration,
                "specialist_outcomes_by_domain": rerun_specialist_outcomes,
                "execution_traces_by_domain": rerun_execution_traces,
                "graph_trace": [trace_entered, trace_completed, *rerun_trace_events],
            },
            goto="coordinator",
        )

    def _current_specialist_outcomes(
        self, state: LangGraphCandidateState
    ) -> tuple[SpecialistExecutionOutcome, ...]:
        outcomes_by_domain = state.get("specialist_outcomes_by_domain", {})
        return self._ordered_specialist_outcomes(tuple(outcomes_by_domain.values()))

    def _route_for_candidate_run(
        self, state: LangGraphCandidateState, candidate_run: CandidateRun
    ) -> str:
        outcome = candidate_run.arbitration.outcome
        if outcome == ArbitrationOutcome.REPLAN_REQUIRED:
            if state.get("replan_iteration", 0) < state["max_replan_iterations"]:
                return "replan"
            return "human_review"
        if outcome == ArbitrationOutcome.HUMAN_REVIEW_REQUIRED:
            return "human_review"
        return "finalize"

    def _replan_reason_for_outcomes(self, outcomes: tuple[SpecialistExecutionOutcome, ...]) -> str:
        for outcome in outcomes:
            if (
                outcome.decision is not None
                and outcome.decision.status == ArbitrationOutcome.REPLAN_REQUIRED
            ):
                if outcome.trace.failure_reason:
                    return outcome.trace.failure_reason
                return outcome.decision.recommendation
        return "Coordinator requested replanning."

    def _replan_update_for_outcomes(
        self, outcomes: tuple[SpecialistExecutionOutcome, ...]
    ) -> PlanningUpdate:
        evidence_ids = tuple(
            dict.fromkeys(
                evidence.evidence_id
                for outcome in outcomes
                if outcome.decision is not None
                for evidence in outcome.decision.evidence_references
            )
        )
        if evidence_ids:
            replan_description = "Targeted replanning after conflicting specialist output."
            return PlanningUpdate(
                update_id=f"replan-evidence-{len(evidence_ids)}",
                kind=PlanningUpdateKind.NEW_EVIDENCE_DISCOVERED,
                description=replan_description,
                evidence_document_ids=evidence_ids,
            )
        return PlanningUpdate(
            update_id="replan-noop",
            kind=PlanningUpdateKind.NO_OP,
            description="Coordinator requested targeted replanning without new evidence.",
        )

    def _targeted_domains_from_replan(
        self,
        *,
        planning_state: PlanningState,
        outcomes: tuple[SpecialistExecutionOutcome, ...],
    ) -> tuple[str, ...]:
        decision_by_id = {decision.decision_id: decision for decision in planning_state.decisions}
        dependency_kind_by_id = {
            dependency.dependency_id: dependency.kind
            for dependency in planning_state.dependency_relationships
        }
        domains: list[str] = []
        for outcome in outcomes:
            if (
                outcome.decision is None
                or outcome.decision.status != ArbitrationOutcome.REPLAN_REQUIRED
            ):
                continue
            for dependency_id in outcome.decision.dependency_decision_ids:
                kind = dependency_kind_by_id.get(dependency_id)
                if kind is not None:
                    domains.extend(self._domains_for_dependency_kind(kind))
            for decision_id in outcome.decision.dependency_decision_ids:
                if decision_id in decision_by_id:
                    domains.extend(self._domains_for_decision(decision_by_id[decision_id]))
        if not domains:
            domains.extend(
                outcome.trace.specialist_id
                for outcome in outcomes
                if outcome.decision is not None
                and outcome.decision.status == ArbitrationOutcome.REPLAN_REQUIRED
            )
        return tuple(dict.fromkeys(domain for domain in domains if domain in SPECIALIST_NODE_NAMES))

    @staticmethod
    def _domains_for_dependency_kind(kind: PlanningDependencyKind) -> tuple[str, ...]:
        mapping = {
            PlanningDependencyKind.GUEST_COUNT_TO_VENUE_CAPACITY: ("venue",),
            PlanningDependencyKind.GUEST_COUNT_TO_CATERING_QUANTITY: ("catering",),
            PlanningDependencyKind.GUEST_COUNT_TO_CATERING_COST: ("catering",),
            PlanningDependencyKind.GUEST_COUNT_TO_SEATING: ("scheduling",),
            PlanningDependencyKind.GUEST_COUNT_TO_PARKING: ("accessibility",),
            PlanningDependencyKind.VENUE_TO_APPROVED_CATERERS: ("venue", "catering"),
            PlanningDependencyKind.VENUE_TO_ACTIVITY_SPACE: ("venue",),
            PlanningDependencyKind.ACCESSIBILITY_TO_VENUE: ("accessibility",),
            PlanningDependencyKind.ACCESSIBILITY_TO_PATH: ("accessibility",),
            PlanningDependencyKind.ACCESSIBILITY_TO_ROOM: ("accessibility",),
            PlanningDependencyKind.ACCESSIBILITY_TO_RESTROOM: ("accessibility",),
            PlanningDependencyKind.DIETARY_TO_CATERING_EVIDENCE: ("catering",),
            PlanningDependencyKind.SCHEDULE_TO_VENDOR_AVAILABILITY: ("scheduling",),
            PlanningDependencyKind.SCHEDULE_TO_SETUP_WINDOW: ("scheduling",),
            PlanningDependencyKind.BUDGET_TO_RESOURCE_SELECTION: ("budget",),
            PlanningDependencyKind.BUDGET_TO_TOTAL_COST: ("budget",),
            PlanningDependencyKind.FEES_TO_TOTAL_COST: ("budget",),
            PlanningDependencyKind.NEW_EVIDENCE_TO_POLICY_VALIDITY: ("venue", "catering"),
        }
        return mapping.get(kind, ())

    @staticmethod
    def _domains_for_decision(decision: Any) -> tuple[str, ...]:
        category = getattr(decision, "category", None)
        if category is PlanningDecisionCategory.BUDGET:
            return ("budget",)
        if category is PlanningDecisionCategory.ACCESSIBILITY:
            return ("accessibility",)
        if category is PlanningDecisionCategory.DIETARY:
            return ("catering",)
        if category is PlanningDecisionCategory.SCHEDULE:
            return ("scheduling",)
        if category is PlanningDecisionCategory.RESOURCE_SELECTION:
            resource_ids = tuple(getattr(decision, "resource_ids", ()))
            domains: list[str] = []
            if any("venue" in resource_id for resource_id in resource_ids):
                domains.append("venue")
            if any("cater" in resource_id or "food" in resource_id for resource_id in resource_ids):
                domains.append("catering")
            if any("activity" in resource_id for resource_id in resource_ids):
                domains.append("scheduling")
            return tuple(domains)
        return ()

    def _build_terminal_candidate_run(
        self,
        *,
        scenario: CapabilityBoundaryScenario,
        planning_state: PlanningState,
        candidate_resources: tuple[Resource, ...],
        candidate_resource_ids: tuple[str, ...],
        hard_violation: Any,
    ) -> CandidateRun:
        if hard_violation is None:
            hard_violation = runtime_module.GuardrailAssessment(
                reason="Deterministic preflight terminated the candidate.",
                controlling_evidence_ids=(),
                proven_hard_violation=True,
            )
        arbitration = ArbitrationTrace(
            outcome=ArbitrationOutcome.REJECT,
            feasibility_outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN,
            selected_resource_ids=candidate_resource_ids,
            accepted_specialist_ids=(),
            rejected_specialist_ids=(),
            overridden_specialist_ids=(),
            controlling_evidence_ids=hard_violation.controlling_evidence_ids,
            dependency_conflicts=(),
            unresolved_uncertainties=(hard_violation.reason,),
            reasons=(hard_violation.reason,),
            global_score=runtime_module._candidate_total_cost_from_resources(
                scenario, candidate_resources
            ),
            coordination_steps=("preflight:terminal",),
        )
        coordinated_result = runtime_module._coordinated_result_from_arbitration(
            scenario=scenario,
            candidate_resources=candidate_resources,
            arbitration=arbitration,
            specialist_outcomes=(),
            decision_count=0,
            elapsed_ms=0.0,
            failure_kind=None,
            failure_stage_override="hard_constraints",
        )
        return CandidateRun(
            coordinated_result=coordinated_result,
            arbitration=arbitration,
            selected_resource_ids=candidate_resource_ids,
            total_cost=coordinated_result.total_cost,
            execution_traces=(),
            specialist_outcomes=(),
            decision_count=0,
            planning_state=planning_state,
        )

    @staticmethod
    def _ordered_specialist_outcomes(
        specialist_outcomes: tuple[SpecialistExecutionOutcome, ...],
    ) -> tuple[SpecialistExecutionOutcome, ...]:
        order = {node_name: index for index, node_name in enumerate(SPECIALIST_NODE_NAMES)}
        return tuple(
            sorted(specialist_outcomes, key=lambda outcome: order[outcome.trace.specialist_id])
        )

    @staticmethod
    def _graph_trace_event(
        *,
        node_name: str,
        event_kind: GraphTraceEventKind,
        started_at: datetime,
        completed_at: datetime | None,
        latency_ms: float | None = None,
        routing_decision: str | None = None,
        failure_kind: str | None = None,
        outcome: str | None = None,
        execution_id: str | None = None,
        review_revision: int | None = None,
        review_action: HumanReviewAction | None = None,
        details: tuple[str, ...] = (),
    ) -> GraphTraceEvent:
        return GraphTraceEvent(
            node_name=node_name,
            event_kind=event_kind,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            routing_decision=routing_decision,
            failure_kind=failure_kind,
            outcome=outcome,
            execution_id=execution_id,
            review_revision=review_revision,
            review_action=review_action,
            details=details,
        )


__all__ = [
    "SPECIALIST_NODE_NAMES",
    "CandidateGraphExecutionResult",
    "CandidateGraphExecutionStatus",
    "GraphTraceEvent",
    "GraphTraceEventKind",
    "HumanReviewSuspendedError",
    "LangGraphCandidateState",
    "LangGraphMultiAgentPlanningRuntime",
]
