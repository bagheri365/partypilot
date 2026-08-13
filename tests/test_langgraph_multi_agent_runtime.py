from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from partypilot.application import multi_agent_runtime as runtime_module
from partypilot.application import v04_multi_agent as v04
from partypilot.application.review_workflow import (
    HumanReviewAction,
    HumanReviewRequest,
    HumanReviewResponse,
)
from partypilot.composition import multi_agent_runtime as composition_runtime
from partypilot.composition.langgraph_multi_agent_runtime import (
    CandidateGraphExecutionStatus,
    GraphTraceEventKind,
    LangGraphCandidateState,
    LangGraphMultiAgentPlanningRuntime,
)
from partypilot.domain import (
    ArbitrationOutcome,
    CapabilityBoundaryScenario,
    FeasibilityOutcome,
    SpecialistDecision,
    SpecialistDomain,
    SpecialistExecutionOutcome,
    SpecialistExecutionTrace,
    SpecialistFailureKind,
    canonical_specialist_id,
    canonical_specialist_name,
)


def _dummy_specialists() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            specialist_id=canonical_specialist_id(domain),
            specialist_name=canonical_specialist_name(domain),
            domain=domain,
        )
        for domain in (
            SpecialistDomain.VENUE,
            SpecialistDomain.CATERING_SAFETY,
            SpecialistDomain.ACCESSIBILITY,
            SpecialistDomain.SCHEDULING_OPERATIONS,
            SpecialistDomain.BUDGET,
        )
    )


def _decision(domain: SpecialistDomain) -> SpecialistDecision:
    return SpecialistDecision(
        specialist_id=canonical_specialist_id(domain),
        domain=domain,
        recommendation=f"{domain.value} decision",
        status=ArbitrationOutcome.ACCEPT,
        hard_constraints_considered=(),
        evidence_references=(),
        assumptions=(),
        unresolved_uncertainties=(),
        local_score=1.0,
        local_rank=1,
        recommended_resource_ids=(),
        reasons_for_rejection=(),
        dependency_decision_ids=(),
        notes=(),
    )


def _outcome(
    *,
    domain: SpecialistDomain,
    failure_kind: SpecialistFailureKind | None = None,
) -> SpecialistExecutionOutcome:
    started_at = datetime.now(UTC)
    completed_at = datetime.now(UTC)
    decision = None if failure_kind is not None else _decision(domain)
    trace = SpecialistExecutionTrace(
        run_id=f"run-{domain.value}",
        specialist_id=canonical_specialist_id(domain),
        specialist_name=canonical_specialist_name(domain),
        domain=domain,
        started_at=started_at,
        completed_at=completed_at,
        latency_ms=1.0,
        validation_succeeded=failure_kind is None,
        retry_count=0,
        failure_kind=failure_kind,
        failure_error_type=None,
        failure_reason=None,
    )
    return SpecialistExecutionOutcome(
        decision=decision,
        trace=trace,
        failure_kind=failure_kind,
    )


def _replan_outcome(
    *,
    domain: SpecialistDomain,
    revision_number: int,
    replan_iteration: int,
) -> SpecialistExecutionOutcome:
    started_at = datetime.now(UTC)
    completed_at = datetime.now(UTC)
    status = (
        ArbitrationOutcome.REPLAN_REQUIRED
        if domain is SpecialistDomain.SCHEDULING_OPERATIONS and replan_iteration == 0
        else ArbitrationOutcome.ACCEPT
    )
    decision = SpecialistDecision(
        specialist_id=canonical_specialist_id(domain),
        domain=domain,
        recommendation=f"{domain.value} decision",
        status=status,
        hard_constraints_considered=(),
        evidence_references=(),
        assumptions=(),
        unresolved_uncertainties=(),
        local_score=1.0,
        local_rank=1,
        recommended_resource_ids=(),
        reasons_for_rejection=(),
        dependency_decision_ids=(),
        notes=(),
    )
    trace = SpecialistExecutionTrace(
        run_id=f"run-{domain.value}-r{revision_number}-i{replan_iteration}",
        specialist_id=canonical_specialist_id(domain),
        specialist_name=canonical_specialist_name(domain),
        domain=domain,
        started_at=started_at,
        completed_at=completed_at,
        latency_ms=1.0,
        validation_succeeded=True,
        retry_count=0,
        failure_kind=None,
        failure_error_type=None,
        failure_reason=None,
    )
    return SpecialistExecutionOutcome(
        decision=decision,
        trace=trace,
        failure_kind=None,
    )


def _scenario() -> CapabilityBoundaryScenario:
    return next(
        scenario
        for scenario in v04.load_v04_multi_agent_benchmark()
        if scenario.scenario.scenario_id == "cap-boundary-41-venue-caterer-dependency"
    )


def test_build_live_multi_agent_runtime_can_select_langgraph_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composition_runtime,
        "build_specialist_agents",
        lambda *args, **kwargs: _dummy_specialists(),
    )

    runtime = cast(
        LangGraphMultiAgentPlanningRuntime,
        composition_runtime.build_live_multi_agent_runtime(
            provider=None,
            orchestration_backend=composition_runtime.OrchestrationBackend.LANGGRAPH,
            model_name="fake-model",
        ),
    )

    assert isinstance(runtime, LangGraphMultiAgentPlanningRuntime)
    mermaid = runtime.graph_mermaid()
    assert "preflight" in mermaid
    assert "venue" in mermaid
    assert "coordinator" in mermaid
    assert "finalize" in mermaid


def test_build_live_multi_agent_runtime_defaults_to_imperative_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composition_runtime,
        "build_specialist_agents",
        lambda *args, **kwargs: _dummy_specialists(),
    )

    runtime = composition_runtime.build_live_multi_agent_runtime(
        provider=None,
        model_name="fake-model",
    )

    assert isinstance(runtime, runtime_module.MultiAgentPlanningRuntime)


def test_langgraph_runtime_shortcuts_terminal_deterministic_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LangGraphMultiAgentPlanningRuntime(_dummy_specialists(), model_name="fake-model")
    scenario = _scenario()
    candidate_ids = tuple(resource.resource_id for resource in scenario.structured_resources)

    monkeypatch.setattr(
        v04,
        "_candidate_combinations",
        lambda scenario: (candidate_ids,),
    )
    monkeypatch.setattr(
        runtime_module,
        "_deterministic_hard_violation",
        lambda scenario, candidate_resources: runtime_module.GuardrailAssessment(
            reason="terminal",
            controlling_evidence_ids=(),
            proven_hard_violation=True,
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "_run_specialist_invocation",
        lambda specialist, agent_input: pytest.fail("specialists should not run"),
    )

    result = runtime.plan_scenario(scenario)

    assert result.final_result.feasibility_outcome is FeasibilityOutcome.NO_FEASIBLE_PLAN
    assert runtime.last_graph_trace
    assert {event.node_name for event in runtime.last_graph_trace} == {"preflight", "finalize"}
    assert all(
        event.node_name not in {"venue", "catering", "accessibility", "scheduling", "budget"}
        for event in runtime.last_graph_trace
    )


def test_langgraph_runtime_fans_out_and_joins_all_five_specialists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LangGraphMultiAgentPlanningRuntime(_dummy_specialists(), model_name="fake-model")
    scenario = _scenario()
    candidate_ids = tuple(resource.resource_id for resource in scenario.structured_resources)
    call_log: list[str] = []

    monkeypatch.setattr(
        v04,
        "_candidate_combinations",
        lambda scenario: (candidate_ids,),
    )
    monkeypatch.setattr(
        runtime_module,
        "_deterministic_hard_violation",
        lambda scenario, candidate_resources: None,
    )

    def _run_specialist_invocation(specialist: Any, agent_input: Any) -> SpecialistExecutionOutcome:
        call_log.append(specialist.specialist_id)
        failure_kind = (
            SpecialistFailureKind.PROVIDER_TIMEOUT
            if specialist.specialist_id == "accessibility"
            else None
        )
        return _outcome(
            domain=specialist.domain,
            failure_kind=failure_kind,
        )

    monkeypatch.setattr(runtime_module, "_run_specialist_invocation", _run_specialist_invocation)

    result = runtime.plan_scenario(scenario)

    assert sorted(call_log) == [
        "accessibility",
        "budget",
        "catering",
        "scheduling",
        "venue",
    ]
    assert result.final_result.feasibility_outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    assert result.candidate_results[0].specialist_outcomes
    assert result.final_result.specialist_call_count == 4
    assert runtime.last_graph_trace
    assert {"preflight", "coordinator", "finalize"}.issubset(
        {event.node_name for event in runtime.last_graph_trace}
    )
    assert {"venue", "catering", "accessibility", "scheduling", "budget"}.issubset(
        {event.node_name for event in runtime.last_graph_trace}
    )


def test_orchestration_backends_match_on_offline_fake_specialists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario()
    candidate_ids = tuple(resource.resource_id for resource in scenario.structured_resources)

    monkeypatch.setattr(
        composition_runtime,
        "build_specialist_agents",
        lambda *args, **kwargs: _dummy_specialists(),
    )
    monkeypatch.setattr(
        v04,
        "_candidate_combinations",
        lambda scenario: (candidate_ids,),
    )

    def _run_specialist_invocation(specialist: Any, agent_input: Any) -> SpecialistExecutionOutcome:
        return _outcome(domain=specialist.domain)

    monkeypatch.setattr(runtime_module, "_run_specialist_invocation", _run_specialist_invocation)
    monkeypatch.setattr(
        runtime_module,
        "_deterministic_hard_violation",
        lambda scenario, candidate_resources: None,
    )

    imperative_runtime = composition_runtime.build_live_multi_agent_runtime(
        provider=None,
        orchestration_backend=composition_runtime.OrchestrationBackend.IMPERATIVE,
        model_name="fake-model",
    )
    langgraph_runtime = cast(
        LangGraphMultiAgentPlanningRuntime,
        composition_runtime.build_live_multi_agent_runtime(
            provider=None,
            orchestration_backend=composition_runtime.OrchestrationBackend.LANGGRAPH,
            model_name="fake-model",
        ),
    )

    imperative_result = imperative_runtime.plan_scenario(scenario)
    langgraph_result = langgraph_runtime.plan_scenario(scenario)

    assert (
        imperative_result.final_result.feasibility_outcome
        is langgraph_result.final_result.feasibility_outcome
    )
    assert (
        imperative_result.final_result.selected_resource_ids
        == langgraph_result.final_result.selected_resource_ids
    )
    assert imperative_result.final_result.arbitration is not None
    assert langgraph_result.final_result.arbitration is not None
    assert (
        imperative_result.final_result.arbitration.outcome
        is langgraph_result.final_result.arbitration.outcome
    )
    assert len(imperative_result.candidate_results) == len(langgraph_result.candidate_results)
    assert len(imperative_result.candidate_results[0].specialist_outcomes) == len(
        langgraph_result.candidate_results[0].specialist_outcomes
    )
    assert len(imperative_result.execution_traces) == len(langgraph_result.execution_traces)


def test_langgraph_runtime_targets_only_replanned_specialists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composition_runtime,
        "build_specialist_agents",
        lambda *args, **kwargs: _dummy_specialists(),
    )
    runtime = cast(
        LangGraphMultiAgentPlanningRuntime,
        composition_runtime.build_live_multi_agent_runtime(
            provider=None,
            orchestration_backend=composition_runtime.OrchestrationBackend.LANGGRAPH,
            model_name="fake-model",
        ),
    )
    scenario = _scenario()
    candidate_ids = tuple(resource.resource_id for resource in scenario.structured_resources)
    call_log: list[tuple[str, int]] = []

    monkeypatch.setattr(
        v04,
        "_candidate_combinations",
        lambda scenario: (candidate_ids,),
    )
    monkeypatch.setattr(
        runtime_module,
        "_deterministic_hard_violation",
        lambda scenario, candidate_resources: None,
    )

    def _run_specialist_invocation(specialist: Any, agent_input: Any) -> SpecialistExecutionOutcome:
        call_log.append((specialist.specialist_id, agent_input.planning_state.revision_number))
        return _replan_outcome(
            domain=specialist.domain,
            revision_number=agent_input.planning_state.revision_number,
            replan_iteration=agent_input.planning_state.revision_number,
        )

    monkeypatch.setattr(runtime_module, "_run_specialist_invocation", _run_specialist_invocation)

    result = runtime.plan_scenario(scenario)

    assert result.final_result.feasibility_outcome is FeasibilityOutcome.FEASIBLE
    assert sorted(call_log) == [
        ("accessibility", 0),
        ("budget", 0),
        ("catering", 0),
        ("scheduling", 0),
        ("scheduling", 1),
        ("venue", 0),
    ]
    assert len(result.candidate_results[0].specialist_outcomes) == 5
    assert [
        outcome.trace.specialist_id for outcome in result.candidate_results[0].specialist_outcomes
    ] == [
        "venue",
        "catering",
        "accessibility",
        "scheduling",
        "budget",
    ]
    assert any(
        event.node_name == "replan" and event.event_kind is GraphTraceEventKind.REPLAN_PLANNED
        for event in runtime.last_graph_trace
    )
    assert any(
        event.node_name == "scheduling"
        and event.event_kind is GraphTraceEventKind.SPECIALIST_RERUN_STARTED
        for event in runtime.last_graph_trace
    )


def test_langgraph_runtime_bounded_replan_loop_suspends_for_human_review_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composition_runtime,
        "build_specialist_agents",
        lambda *args, **kwargs: _dummy_specialists(),
    )
    runtime = cast(
        LangGraphMultiAgentPlanningRuntime,
        composition_runtime.build_live_multi_agent_runtime(
            provider=None,
            orchestration_backend=composition_runtime.OrchestrationBackend.LANGGRAPH,
            model_name="fake-model",
        ),
    )
    scenario = _scenario()
    candidate_ids = tuple(resource.resource_id for resource in scenario.structured_resources)
    monkeypatch.setattr(
        v04,
        "_candidate_combinations",
        lambda scenario: (candidate_ids,),
    )
    monkeypatch.setattr(
        runtime_module,
        "_deterministic_hard_violation",
        lambda scenario, candidate_resources: None,
    )
    call_log: list[str] = []

    def _run_specialist_invocation(specialist: Any, agent_input: Any) -> SpecialistExecutionOutcome:
        call_log.append(specialist.specialist_id)
        if specialist.specialist_id == "scheduling":
            return _replan_outcome(
                domain=specialist.domain,
                revision_number=agent_input.planning_state.revision_number,
                replan_iteration=0,
            )
        return _outcome(domain=specialist.domain)

    monkeypatch.setattr(runtime_module, "_run_specialist_invocation", _run_specialist_invocation)

    suspended = runtime.run_reviewable_candidate(
        scenario=scenario,
        candidate_resource_ids=candidate_ids,
        execution_id="review-session-1",
    )

    assert suspended.status is CandidateGraphExecutionStatus.SUSPENDED_FOR_HUMAN_REVIEW
    assert suspended.review_request is not None
    assert suspended.review_request.execution_id == "review-session-1"
    assert any(
        event.node_name == "human_review"
        and event.event_kind is GraphTraceEventKind.HUMAN_REVIEW_REQUESTED
        and event.execution_id == "review-session-1"
        for event in suspended.graph_trace
    )
    assert any(
        event.node_name == "human_review"
        and event.event_kind is GraphTraceEventKind.GRAPH_SUSPENDED
        for event in suspended.graph_trace
    )

    resumed = runtime.resume_reviewable_candidate(
        execution_id="review-session-1",
        review_response=HumanReviewResponse(
            execution_id="review-session-1",
            planning_revision=suspended.planning_revision,
            action=HumanReviewAction.REJECT_CURRENT_PLAN,
            candidate_resource_ids=candidate_ids,
        ),
    )

    assert resumed.status is CandidateGraphExecutionStatus.COMPLETED
    assert resumed.candidate_run is not None
    assert resumed.candidate_run.coordinated_result.feasibility_outcome is (
        FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    )
    assert isinstance(resumed.review_request, HumanReviewRequest)
    assert resumed.review_response is not None
    assert resumed.review_response.action is HumanReviewAction.REJECT_CURRENT_PLAN
    assert any(
        event.node_name == "human_review" and event.event_kind is GraphTraceEventKind.GRAPH_RESUMED
        for event in resumed.graph_trace
    )
    assert any(
        event.node_name == "human_review"
        and event.event_kind is GraphTraceEventKind.REVIEW_ACTION
        and event.review_action is HumanReviewAction.REJECT_CURRENT_PLAN
        for event in resumed.graph_trace
    )
    assert call_log.count("scheduling") == 2


def test_langgraph_review_smoke_strict_msgpack_has_no_unregistered_type_warnings() -> None:
    env = os.environ.copy()
    env["LANGGRAPH_STRICT_MSGPACK"] = "true"
    result = subprocess.run(
        [sys.executable, "-m", "partypilot.cli.smoke_langgraph_review"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0
    assert "Review smoke passed." in result.stdout
    assert "Resumed status: completed" in result.stdout
    assert "Deserializing unregistered type" not in output
    assert "Blocked deserialization" not in output


def test_langgraph_reviewable_candidate_rejects_wrong_execution_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composition_runtime,
        "build_specialist_agents",
        lambda *args, **kwargs: _dummy_specialists(),
    )
    runtime = cast(
        LangGraphMultiAgentPlanningRuntime,
        composition_runtime.build_live_multi_agent_runtime(
            provider=None,
            orchestration_backend=composition_runtime.OrchestrationBackend.LANGGRAPH,
            model_name="fake-model",
        ),
    )
    scenario = _scenario()
    candidate_ids = tuple(resource.resource_id for resource in scenario.structured_resources)
    monkeypatch.setattr(
        v04,
        "_candidate_combinations",
        lambda scenario: (candidate_ids,),
    )
    monkeypatch.setattr(
        runtime_module,
        "_deterministic_hard_violation",
        lambda scenario, candidate_resources: None,
    )

    def _run_specialist_invocation(specialist: Any, agent_input: Any) -> SpecialistExecutionOutcome:
        if specialist.specialist_id == "scheduling":
            return _replan_outcome(
                domain=specialist.domain,
                revision_number=agent_input.planning_state.revision_number,
                replan_iteration=0,
            )
        return _outcome(domain=specialist.domain)

    monkeypatch.setattr(runtime_module, "_run_specialist_invocation", _run_specialist_invocation)

    suspended = runtime.run_reviewable_candidate(
        scenario=scenario,
        candidate_resource_ids=candidate_ids,
        execution_id="review-session-2",
    )

    assert suspended.status is CandidateGraphExecutionStatus.SUSPENDED_FOR_HUMAN_REVIEW
    with pytest.raises(ValueError, match="execution_id"):
        runtime.resume_reviewable_candidate(
            execution_id="wrong-session",
            review_response=HumanReviewResponse(
                execution_id="wrong-session",
                planning_revision=suspended.planning_revision,
                action=HumanReviewAction.REJECT_CURRENT_PLAN,
                candidate_resource_ids=candidate_ids,
            ),
        )


def test_langgraph_reviewable_candidate_rejects_stale_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composition_runtime,
        "build_specialist_agents",
        lambda *args, **kwargs: _dummy_specialists(),
    )
    runtime = cast(
        LangGraphMultiAgentPlanningRuntime,
        composition_runtime.build_live_multi_agent_runtime(
            provider=None,
            orchestration_backend=composition_runtime.OrchestrationBackend.LANGGRAPH,
            model_name="fake-model",
        ),
    )
    scenario = _scenario()
    candidate_ids = tuple(resource.resource_id for resource in scenario.structured_resources)
    monkeypatch.setattr(
        v04,
        "_candidate_combinations",
        lambda scenario: (candidate_ids,),
    )
    monkeypatch.setattr(
        runtime_module,
        "_deterministic_hard_violation",
        lambda scenario, candidate_resources: None,
    )

    def _run_specialist_invocation(specialist: Any, agent_input: Any) -> SpecialistExecutionOutcome:
        if specialist.specialist_id == "scheduling":
            return _replan_outcome(
                domain=specialist.domain,
                revision_number=agent_input.planning_state.revision_number,
                replan_iteration=0,
            )
        return _outcome(domain=specialist.domain)

    monkeypatch.setattr(runtime_module, "_run_specialist_invocation", _run_specialist_invocation)

    suspended = runtime.run_reviewable_candidate(
        scenario=scenario,
        candidate_resource_ids=candidate_ids,
        execution_id="review-session-3",
    )

    resumed = runtime.resume_reviewable_candidate(
        execution_id="review-session-3",
        review_response=HumanReviewResponse(
            execution_id="review-session-3",
            planning_revision=suspended.planning_revision + 1,
            action=HumanReviewAction.REJECT_CURRENT_PLAN,
            candidate_resource_ids=candidate_ids,
        ),
    )

    assert resumed.status is CandidateGraphExecutionStatus.COMPLETED
    assert any(
        event.node_name == "human_review"
        and event.event_kind is GraphTraceEventKind.STALE_REVIEW_REJECTED
        for event in resumed.graph_trace
    )


def test_human_review_node_approval_routes_to_finalize_when_plan_is_feasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LangGraphMultiAgentPlanningRuntime(_dummy_specialists(), model_name="fake-model")
    scenario = _scenario()
    candidate_ids = tuple(resource.resource_id for resource in scenario.structured_resources)
    candidate_resources = tuple(scenario.structured_resources)
    planning_state = runtime_module._build_planning_state(scenario, candidate_resources)
    base_candidate_run = runtime._build_candidate_run(
        scenario=scenario,
        planning_state=planning_state,
        candidate_resources=candidate_resources,
        candidate_resource_ids=candidate_ids,
        specialist_outcomes=tuple(
            _outcome(domain=domain)
            for domain in (
                SpecialistDomain.VENUE,
                SpecialistDomain.CATERING_SAFETY,
                SpecialistDomain.ACCESSIBILITY,
                SpecialistDomain.SCHEDULING_OPERATIONS,
                SpecialistDomain.BUDGET,
            )
        ),
    )
    candidate_run = base_candidate_run.model_copy(
        update={
            "coordinated_result": base_candidate_run.coordinated_result.model_copy(
                update={"feasibility_outcome": FeasibilityOutcome.FEASIBLE}
            )
        }
    )
    state: LangGraphCandidateState = {
        "execution_id": "review-approval-feasible",
        "scenario": scenario,
        "planning_state": planning_state,
        "candidate_resources": candidate_resources,
        "candidate_resource_ids": candidate_ids,
        "candidate_run": candidate_run,
        "replan_iteration": 1,
        "max_replan_iterations": 1,
        "targeted_specialist_domains": ("scheduling",),
        "specialist_outcomes_by_domain": {},
        "execution_traces_by_domain": {},
    }

    monkeypatch.setattr(
        "partypilot.composition.langgraph_multi_agent_runtime.interrupt",
        lambda payload: {
            "execution_id": "review-approval-feasible",
            "planning_revision": planning_state.revision_number,
            "action": HumanReviewAction.APPROVE_CURRENT_PLAN.value,
            "candidate_resource_ids": candidate_ids,
        },
    )

    command = runtime._human_review_node(state)

    assert command.goto == "finalize"
    update = command.update
    assert isinstance(update, dict)
    assert "candidate_run" not in update
    assert any(
        event.event_kind is GraphTraceEventKind.REVIEW_ACTION
        and event.review_action is HumanReviewAction.APPROVE_CURRENT_PLAN
        for event in update["graph_trace"]
    )
    assert candidate_run.coordinated_result.feasibility_outcome is FeasibilityOutcome.FEASIBLE


def test_human_review_node_approval_routes_to_finalize_without_mutating_feasibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LangGraphMultiAgentPlanningRuntime(_dummy_specialists(), model_name="fake-model")
    scenario = _scenario()
    candidate_ids = tuple(resource.resource_id for resource in scenario.structured_resources)
    candidate_resources = tuple(scenario.structured_resources)
    planning_state = runtime_module._build_planning_state(scenario, candidate_resources)
    base_candidate_run = runtime._build_candidate_run(
        scenario=scenario,
        planning_state=planning_state,
        candidate_resources=candidate_resources,
        candidate_resource_ids=candidate_ids,
        specialist_outcomes=tuple(
            _outcome(domain=domain)
            for domain in (
                SpecialistDomain.VENUE,
                SpecialistDomain.CATERING_SAFETY,
                SpecialistDomain.ACCESSIBILITY,
                SpecialistDomain.SCHEDULING_OPERATIONS,
                SpecialistDomain.BUDGET,
            )
        ),
    )
    candidate_run = base_candidate_run.model_copy(
        update={
            "coordinated_result": base_candidate_run.coordinated_result.model_copy(
                update={"feasibility_outcome": FeasibilityOutcome.NO_FEASIBLE_PLAN}
            )
        }
    )
    state: LangGraphCandidateState = {
        "execution_id": "review-approval",
        "scenario": scenario,
        "planning_state": planning_state,
        "candidate_resources": candidate_resources,
        "candidate_resource_ids": candidate_ids,
        "candidate_run": candidate_run,
        "replan_iteration": 1,
        "max_replan_iterations": 1,
        "targeted_specialist_domains": ("scheduling",),
        "specialist_outcomes_by_domain": {},
        "execution_traces_by_domain": {},
    }

    monkeypatch.setattr(
        "partypilot.composition.langgraph_multi_agent_runtime.interrupt",
        lambda payload: {
            "execution_id": "review-approval",
            "planning_revision": planning_state.revision_number,
            "action": HumanReviewAction.APPROVE_CURRENT_PLAN.value,
            "candidate_resource_ids": candidate_ids,
        },
    )

    command = runtime._human_review_node(state)

    assert command.goto == "finalize"
    update = command.update
    assert isinstance(update, dict)
    assert "candidate_run" not in update
    assert any(
        event.event_kind is GraphTraceEventKind.REVIEW_ACTION
        and event.review_action is HumanReviewAction.APPROVE_CURRENT_PLAN
        for event in update["graph_trace"]
    )
    assert (
        candidate_run.coordinated_result.feasibility_outcome is FeasibilityOutcome.NO_FEASIBLE_PLAN
    )


def test_human_review_node_request_replan_routes_back_to_replan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LangGraphMultiAgentPlanningRuntime(_dummy_specialists(), model_name="fake-model")
    scenario = _scenario()
    candidate_ids = tuple(resource.resource_id for resource in scenario.structured_resources)
    candidate_resources = tuple(scenario.structured_resources)
    planning_state = runtime_module._build_planning_state(scenario, candidate_resources)
    candidate_run = runtime._build_candidate_run(
        scenario=scenario,
        planning_state=planning_state,
        candidate_resources=candidate_resources,
        candidate_resource_ids=candidate_ids,
        specialist_outcomes=tuple(
            _outcome(domain=domain)
            for domain in (
                SpecialistDomain.VENUE,
                SpecialistDomain.CATERING_SAFETY,
                SpecialistDomain.ACCESSIBILITY,
                SpecialistDomain.SCHEDULING_OPERATIONS,
                SpecialistDomain.BUDGET,
            )
        ),
    )
    state: LangGraphCandidateState = {
        "execution_id": "review-replan",
        "scenario": scenario,
        "planning_state": planning_state,
        "candidate_resources": candidate_resources,
        "candidate_resource_ids": candidate_ids,
        "candidate_run": candidate_run,
        "replan_iteration": 1,
        "max_replan_iterations": 2,
        "targeted_specialist_domains": ("scheduling",),
        "specialist_outcomes_by_domain": {},
        "execution_traces_by_domain": {},
    }

    monkeypatch.setattr(
        "partypilot.composition.langgraph_multi_agent_runtime.interrupt",
        lambda payload: {
            "execution_id": "review-replan",
            "planning_revision": planning_state.revision_number,
            "action": HumanReviewAction.REQUEST_REPLAN.value,
            "candidate_resource_ids": candidate_ids,
        },
    )

    command = runtime._human_review_node(state)

    assert command.goto == "replan"
    update = command.update
    assert isinstance(update, dict)
    assert any(
        event.event_kind is GraphTraceEventKind.REVIEW_ACTION
        and event.review_action is HumanReviewAction.REQUEST_REPLAN
        for event in update["graph_trace"]
    )


def test_human_review_node_rejects_invalid_resume_action_without_mutating_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LangGraphMultiAgentPlanningRuntime(_dummy_specialists(), model_name="fake-model")
    scenario = _scenario()
    candidate_ids = tuple(resource.resource_id for resource in scenario.structured_resources)
    candidate_resources = tuple(scenario.structured_resources)
    planning_state = runtime_module._build_planning_state(scenario, candidate_resources)
    candidate_run = runtime._build_candidate_run(
        scenario=scenario,
        planning_state=planning_state,
        candidate_resources=candidate_resources,
        candidate_resource_ids=candidate_ids,
        specialist_outcomes=tuple(
            _outcome(domain=domain)
            for domain in (
                SpecialistDomain.VENUE,
                SpecialistDomain.CATERING_SAFETY,
                SpecialistDomain.ACCESSIBILITY,
                SpecialistDomain.SCHEDULING_OPERATIONS,
                SpecialistDomain.BUDGET,
            )
        ),
    )
    state: LangGraphCandidateState = {
        "execution_id": "review-invalid",
        "scenario": scenario,
        "planning_state": planning_state,
        "candidate_resources": candidate_resources,
        "candidate_resource_ids": candidate_ids,
        "candidate_run": candidate_run,
        "replan_iteration": 1,
        "max_replan_iterations": 1,
        "targeted_specialist_domains": ("scheduling",),
        "specialist_outcomes_by_domain": {},
        "execution_traces_by_domain": {},
    }

    monkeypatch.setattr(
        "partypilot.composition.langgraph_multi_agent_runtime.interrupt",
        lambda payload: {
            "execution_id": "review-invalid",
            "planning_revision": planning_state.revision_number,
            "action": "not-a-real-action",
            "candidate_resource_ids": candidate_ids,
        },
    )

    command = runtime._human_review_node(state)

    assert command.goto == "finalize"
    update = command.update
    assert isinstance(update, dict)
    assert "candidate_run" not in update
    assert any(
        event.event_kind is GraphTraceEventKind.STALE_REVIEW_REJECTED
        for event in update["graph_trace"]
    )
