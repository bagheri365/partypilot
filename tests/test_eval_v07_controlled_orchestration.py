from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from partypilot.application.review_workflow import HumanReviewAction
from partypilot.cli import eval_v07_controlled_orchestration as eval_v07
from partypilot.cli import v07_controlled_orchestration_evaluation_core as core
from partypilot.composition.multi_agent_runtime import OrchestrationBackend
from partypilot.domain import (
    ArbitrationOutcome,
    ArbitrationTrace,
    CandidateEvaluationResult,
    CapabilityBoundaryScenario,
    CoordinatedPlanResult,
    FeasibilityOutcome,
    MultiAgentPlanningRuntimeResult,
    PlanningStateSummary,
    SpecialistDecision,
    SpecialistDomain,
    SpecialistExecutionOutcome,
    SpecialistExecutionTrace,
    SpecialistFailureKind,
    canonical_specialist_id,
    canonical_specialist_name,
)


def test_run_order_blocks_are_balanced() -> None:
    assert core.V07RunOrderBlocks == (
        (
            OrchestrationBackend.IMPERATIVE,
            OrchestrationBackend.LANGGRAPH,
        ),
        (
            OrchestrationBackend.LANGGRAPH,
            OrchestrationBackend.IMPERATIVE,
        ),
        (
            OrchestrationBackend.IMPERATIVE,
            OrchestrationBackend.LANGGRAPH,
        ),
    )


def test_v07_controlled_evaluation_uses_balanced_order_and_frozen_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = core.load_v07_controlled_scenarios()
    call_log: list[tuple[int, int, str]] = []
    start_snapshot = core.GitSnapshot(
        git_sha="abc123", working_tree_dirty=False, git_metadata_error=None
    )

    monkeypatch.setattr(core, "_git_snapshot", lambda: start_snapshot)

    report = core.run_v07_controlled_evaluation(
        benchmark,
        model="fake-model",
        base_url="http://localhost:11434",
        timeout_seconds=30.0,
        num_ctx=8192,
        max_retries=0,
        timestamp=datetime(2026, 8, 13, tzinfo=UTC),
        backend_runner=lambda **kwargs: _fake_run_report(call_log=call_log, **kwargs),
    )

    assert report.benchmark_version == core.V07_BENCHMARK_VERSION
    assert report.run_order_blocks == (
        ("imperative", "langgraph"),
        ("langgraph", "imperative"),
        ("imperative", "langgraph"),
    )
    assert len(report.runs) == 6
    assert [run.backend for run in report.runs] == [
        "imperative",
        "langgraph",
        "langgraph",
        "imperative",
        "imperative",
        "langgraph",
    ]
    assert [run.repetition_index for run in report.runs] == [1, 1, 2, 2, 3, 3]
    assert [run.order_position for run in report.runs] == [1, 2, 1, 2, 1, 2]
    assert call_log == [
        (1, 1, "imperative"),
        (1, 2, "langgraph"),
        (2, 1, "langgraph"),
        (2, 2, "imperative"),
        (3, 1, "imperative"),
        (3, 2, "langgraph"),
    ]
    assert report.provenance.experiment_start_git_sha == "abc123"
    assert report.provenance.canonical_start_guard_enforced is True
    assert report.backend_summaries[0].backend == "imperative"
    assert report.backend_summaries[0].disposition == "BASELINE"
    assert report.backend_summaries[1].backend == "langgraph"
    assert report.backend_summaries[1].disposition == "RETAIN"
    assert report.backend_summaries[1].graph_executions == 30
    assert report.backend_summaries[1].coordinator_node_executions == 33
    assert report.backend_summaries[1].finalize_executions == 30
    assert report.backend_summaries[1].targeted_specialist_rerun_count > 0
    assert report.backend_summaries[1].interrupt_count is not None
    assert report.backend_summaries[1].interrupt_count > 0
    assert report.backend_summaries[1].resume_count is not None
    assert report.backend_summaries[1].resume_count > 0
    assert report.orchestration_sub_benchmark.passed is True
    assert report.human_review_sub_benchmark.passed is True
    assert report.retention_rule_passed is True


def test_v07_controlled_evaluation_rejects_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        core,
        "_git_snapshot",
        lambda: core.GitSnapshot(
            git_sha="abc123",
            working_tree_dirty=True,
            git_metadata_error=None,
        ),
    )

    with pytest.raises(ValueError, match="clean working tree"):
        core.run_v07_controlled_evaluation(
            (),
            model="fake-model",
            base_url="http://localhost:11434",
            timeout_seconds=30.0,
            num_ctx=8192,
            max_retries=0,
        )


def test_v07_controlled_evaluation_saves_artifacts_and_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = core.load_v07_controlled_scenarios()
    start_snapshot = core.GitSnapshot(
        git_sha="abc123", working_tree_dirty=False, git_metadata_error=None
    )
    artifact_snapshot = core.GitSnapshot(
        git_sha="def456", working_tree_dirty=False, git_metadata_error=None
    )

    monkeypatch.setattr(core, "_git_snapshot", lambda: start_snapshot)

    report = core.run_v07_controlled_evaluation(
        benchmark,
        model="fake-model",
        base_url="http://localhost:11434",
        timeout_seconds=30.0,
        num_ctx=8192,
        max_retries=0,
        timestamp=datetime(2026, 8, 13, tzinfo=UTC),
        backend_runner=_fake_run_report,
    )

    (
        aggregate_json_path,
        aggregate_markdown_path,
        run_paths,
        orchestration_paths,
        human_review_paths,
    ) = core.save_v07_controlled_evaluation_reports(
        report,
        tmp_path,
        artifact_snapshot=artifact_snapshot,
    )

    assert aggregate_json_path.exists()
    assert aggregate_markdown_path.exists()
    assert len(run_paths) == 6
    assert orchestration_paths[0].exists()
    assert human_review_paths[0].exists()
    for json_path, markdown_path in run_paths:
        assert json_path.exists()
        assert markdown_path.exists()
        assert json_path.parent.parent.name in {"imperative", "langgraph"}
    saved_report = json.loads(aggregate_json_path.read_text(encoding="utf-8"))
    assert saved_report["provenance"]["artifact_git_sha"] == "def456"


def test_v07_controlled_orchestration_cli_reports_backend_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    fake_report = SimpleNamespace(
        benchmark_name="PartyPilot v0.7d controlled orchestration evaluation",
        benchmark_version="1.0",
        scenario_count=10,
        run_order_blocks=(
            ("imperative", "langgraph"),
            ("langgraph", "imperative"),
            ("imperative", "langgraph"),
        ),
        retention_rule_passed=True,
        backend_summaries=(
            SimpleNamespace(
                backend="imperative",
                final_decision_accuracy=SimpleNamespace(mean=1.0),
                evidence_grounded_arbitration_accuracy=SimpleNamespace(mean=1.0),
                specialist_success_rate=SimpleNamespace(mean=1.0),
                provider_attempt_count=150,
                graph_executions=None,
                disposition="BASELINE",
            ),
            SimpleNamespace(
                backend="langgraph",
                final_decision_accuracy=SimpleNamespace(mean=1.0),
                evidence_grounded_arbitration_accuracy=SimpleNamespace(mean=1.0),
                specialist_success_rate=SimpleNamespace(mean=1.0),
                provider_attempt_count=150,
                graph_executions=30,
                disposition="RETAIN",
            ),
        ),
    )
    fake_paths = (
        tmp_path / "aggregate.json",
        tmp_path / "aggregate.md",
        tuple((tmp_path / f"run-{index}.json", tmp_path / f"run-{index}.md") for index in range(6)),
        (tmp_path / "orchestration.json", tmp_path / "orchestration.md"),
        (tmp_path / "human-review.json", tmp_path / "human-review.md"),
    )
    monkeypatch.setattr(
        eval_v07,
        "_ollama_config",
        lambda **kwargs: SimpleNamespace(
            base_url="http://localhost:11434",
            model="fake-model",
            timeout_seconds=30.0,
            num_ctx=8192,
            max_retries=0,
        ),
    )
    monkeypatch.setattr(eval_v07, "load_v07_controlled_scenarios", lambda scenario_ids: ())
    monkeypatch.setattr(
        eval_v07, "run_v07_controlled_evaluation", lambda *args, **kwargs: fake_report
    )
    monkeypatch.setattr(
        eval_v07,
        "save_v07_controlled_evaluation_reports",
        lambda report, output_dir: fake_paths,
    )

    exit_code = eval_v07.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "PartyPilot v0.7d Controlled Orchestration Evaluation" in captured.out
    assert "Backend: imperative" in captured.out
    assert "Backend: langgraph" in captured.out
    assert "Retention rule passed: True" in captured.out


def test_build_run_metrics_uses_selected_candidate_and_langgraph_metrics() -> None:
    scenario_result = _fake_scenario_result(
        scenario_id="cap-boundary-43-setup-scheduling-chain",
        backend=OrchestrationBackend.LANGGRAPH,
        graph_trace=_fake_graph_trace("cap-boundary-43-setup-scheduling-chain"),
    )
    metrics = core._build_run_metrics((scenario_result,), backend=OrchestrationBackend.LANGGRAPH)

    assert metrics.top_level_specialist_invocations == 5
    assert metrics.successful_top_level_specialist_invocations == 5
    assert metrics.specialist_timeout_outcomes == 0
    assert metrics.provider_attempt_count == 5
    assert metrics.graph_executions == 1
    assert metrics.coordinator_node_executions == 2
    assert metrics.finalize_executions == 1
    assert metrics.targeted_specialist_rerun_count >= 1
    assert (metrics.interrupt_count or 0) == 0
    assert (metrics.resume_count or 0) == 0


def _fake_run_report(
    *,
    benchmark: Sequence[CapabilityBoundaryScenario],
    backend: OrchestrationBackend,
    config: Any,
    experiment_start_snapshot: core.GitSnapshot,
    order_block_index: int,
    order_position: int,
    repetition_index: int,
    timestamp: datetime,
    call_log: list[tuple[int, int, str]] | None = None,
) -> core.V07RunReport:
    if call_log is not None:
        call_log.append((order_block_index, order_position, backend.value))
    scenario_results = tuple(
        _fake_scenario_result(
            scenario_id=scenario.scenario.scenario_id,
            backend=backend,
            graph_trace=_fake_graph_trace(scenario.scenario.scenario_id)
            if backend is OrchestrationBackend.LANGGRAPH
            else (),
        )
        for scenario in benchmark
    )
    metrics = core._build_run_metrics(scenario_results, backend=backend)
    environment = core._environment_from_config(
        config=config,
        backend=backend,
        structured_output_strategy="with_structured_output",
    )
    return core.V07RunReport(
        run_id=f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{order_block_index}-{order_position}-{backend.value}",
        backend=backend.value,
        repetition_index=repetition_index,
        order_block_index=order_block_index,
        order_position=order_position,
        scenario_count=len(benchmark),
        provenance=core.V07EvaluationProvenance(
            experiment_start_git_sha=experiment_start_snapshot.git_sha,
            experiment_start_working_tree_dirty=experiment_start_snapshot.working_tree_dirty,
            experiment_start_git_metadata_error=experiment_start_snapshot.git_metadata_error,
            canonical_start_guard_enforced=True,
            exploratory_mode=False,
        ),
        environment=environment,
        scenarios=scenario_results,
        metrics=metrics,
        notes=("fake",),
    )


def _fake_scenario_result(
    *,
    scenario_id: str,
    backend: OrchestrationBackend,
    graph_trace: tuple[core.GraphTraceEvent, ...],
) -> core.V07ScenarioResult:
    selected_ids = (f"{scenario_id}-selected",)
    alternate_ids = (f"{scenario_id}-alternate",)
    selected_outcomes = tuple(
        _fake_outcome(domain=domain)
        for domain in (
            SpecialistDomain.VENUE,
            SpecialistDomain.CATERING_SAFETY,
            SpecialistDomain.ACCESSIBILITY,
            SpecialistDomain.SCHEDULING_OPERATIONS,
            SpecialistDomain.BUDGET,
        )
    )
    alternate_outcomes = tuple(
        _fake_outcome(
            domain=domain,
            failure_kind=SpecialistFailureKind.PROVIDER_TIMEOUT
            if domain is SpecialistDomain.SCHEDULING_OPERATIONS
            else None,
        )
        for domain in (
            SpecialistDomain.VENUE,
            SpecialistDomain.CATERING_SAFETY,
            SpecialistDomain.ACCESSIBILITY,
            SpecialistDomain.SCHEDULING_OPERATIONS,
            SpecialistDomain.BUDGET,
        )
    )
    selected_candidate = _fake_candidate_result(
        candidate_resource_ids=selected_ids,
        selected_resource_ids=selected_ids,
        outcomes=selected_outcomes,
        feasibility_outcome=FeasibilityOutcome.FEASIBLE,
        architecture=backend.value,
    )
    alternate_candidate = _fake_candidate_result(
        candidate_resource_ids=alternate_ids,
        selected_resource_ids=alternate_ids,
        outcomes=alternate_outcomes,
        feasibility_outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN,
        architecture=backend.value,
    )
    runtime = MultiAgentPlanningRuntimeResult(
        architecture=backend.value,
        planning_state=PlanningStateSummary(
            revision_number=1,
            selected_resource_ids=selected_ids,
            invalidated_decision_ids=(),
            preserved_decision_ids=(),
            unresolved_uncertainties=(),
            notes=(),
        ),
        candidate_results=(alternate_candidate, selected_candidate),
        final_result=selected_candidate.coordinated_result,
        execution_traces=tuple(
            outcome.trace for outcome in (alternate_outcomes + selected_outcomes)
        ),
        wall_clock_latency_ms=42.0,
        notes=("fake runtime",),
    )
    live_result = selected_candidate.coordinated_result
    return core.V07ScenarioResult(
        scenario_id=scenario_id,
        title=f"Title for {scenario_id}",
        description=f"Description for {scenario_id}",
        capability_tags=("global_optimization",) if scenario_id.endswith("48") else (),
        requires_evidence=True,
        requires_global_optimum=scenario_id.endswith("48"),
        expected_feasibility=FeasibilityOutcome.FEASIBLE,
        live_result=live_result,
        runtime=runtime,
        graph_trace=graph_trace,
        notes=("fake",),
    )


def _fake_candidate_result(
    *,
    candidate_resource_ids: tuple[str, ...],
    selected_resource_ids: tuple[str, ...],
    outcomes: tuple[SpecialistExecutionOutcome, ...],
    feasibility_outcome: FeasibilityOutcome,
    architecture: str,
) -> CandidateEvaluationResult:
    decisions = tuple(outcome.decision for outcome in outcomes if outcome.decision is not None)
    coordinated_result = CoordinatedPlanResult(
        architecture=architecture,
        feasibility_outcome=feasibility_outcome,
        selected_resource_ids=selected_resource_ids,
        total_cost=100.0,
        latency_ms=5.0,
        hard_constraint_validity=True,
        cross_domain_compatibility=True,
        evidence_grounded_arbitration=True,
        global_optimum=feasibility_outcome is FeasibilityOutcome.FEASIBLE,
        human_review_calibrated=feasibility_outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED,
        disagreement_resolved_correctly=False,
        disagreement_resolved_incorrectly=False,
        specialist_call_count=len(outcomes),
        coordination_overhead_count=5,
        arbitration=ArbitrationTrace(
            outcome=ArbitrationOutcome.ACCEPT,
            feasibility_outcome=feasibility_outcome,
            selected_resource_ids=selected_resource_ids,
            accepted_specialist_ids=tuple(
                outcome.trace.specialist_id for outcome in outcomes if outcome.decision is not None
            ),
            rejected_specialist_ids=(),
            overridden_specialist_ids=(),
            controlling_evidence_ids=(),
            dependency_conflicts=(),
            unresolved_uncertainties=(),
            reasons=("fake",),
            global_score=100.0,
            coordination_steps=("coordinator:fake",),
        ),
        specialist_decisions=decisions,
        notes=("fake",),
        failure_stage=None,
    )
    return CandidateEvaluationResult(
        candidate_resource_ids=candidate_resource_ids,
        specialist_outcomes=outcomes,
        selected_resource_ids=selected_resource_ids,
        arbitration_outcome=ArbitrationOutcome.ACCEPT,
        coordinated_result=coordinated_result,
        total_cost=100.0,
        latency_ms=5.0,
    )


def _fake_outcome(
    *,
    domain: SpecialistDomain,
    failure_kind: SpecialistFailureKind | None = None,
) -> SpecialistExecutionOutcome:
    now = datetime.now(UTC)
    decision = (
        None
        if failure_kind is not None
        else SpecialistDecision(
            specialist_id=canonical_specialist_id(domain),
            domain=domain,
            recommendation=f"{domain.value} recommendation",
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
    )
    trace = SpecialistExecutionTrace(
        run_id=f"run-{domain.value}",
        specialist_id=canonical_specialist_id(domain),
        specialist_name=canonical_specialist_name(domain),
        domain=domain,
        started_at=now,
        completed_at=now,
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


def _fake_graph_trace(scenario_id: str) -> tuple[core.GraphTraceEvent, ...]:
    now = datetime.now(UTC)
    scenario_label = scenario_id.casefold()
    events: list[core.GraphTraceEvent] = [
        core.GraphTraceEvent(
            node_name="preflight",
            event_kind=core.GraphTraceEventKind.NODE_ENTERED,
            started_at=now,
            completed_at=None,
            execution_id=scenario_id,
        ),
        core.GraphTraceEvent(
            node_name="preflight",
            event_kind=core.GraphTraceEventKind.NODE_COMPLETED,
            started_at=now,
            completed_at=now,
            routing_decision="fan_out",
            outcome="PROVEN_FEASIBLE",
            execution_id=scenario_id,
        ),
    ]
    for node_name in ("venue", "catering", "accessibility", "scheduling", "budget"):
        events.extend(
            [
                core.GraphTraceEvent(
                    node_name=node_name,
                    event_kind=core.GraphTraceEventKind.NODE_ENTERED,
                    started_at=now,
                    completed_at=None,
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name=node_name,
                    event_kind=core.GraphTraceEventKind.NODE_COMPLETED,
                    started_at=now,
                    completed_at=now,
                    execution_id=scenario_id,
                    outcome="accept",
                ),
            ]
        )
    if "48" in scenario_label or "global-optimum" in scenario_label:
        events.extend(
            [
                core.GraphTraceEvent(
                    node_name="coordinator",
                    event_kind=core.GraphTraceEventKind.NODE_ENTERED,
                    started_at=now,
                    completed_at=None,
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="coordinator",
                    event_kind=core.GraphTraceEventKind.NODE_COMPLETED,
                    started_at=now,
                    completed_at=now,
                    routing_decision="finalize",
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="finalize",
                    event_kind=core.GraphTraceEventKind.NODE_ENTERED,
                    started_at=now,
                    completed_at=None,
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="finalize",
                    event_kind=core.GraphTraceEventKind.NODE_COMPLETED,
                    started_at=now,
                    completed_at=now,
                    routing_decision="end",
                    execution_id=scenario_id,
                ),
            ]
        )
    elif "59" in scenario_label or "evidence" in scenario_label:
        events.extend(
            [
                core.GraphTraceEvent(
                    node_name="coordinator",
                    event_kind=core.GraphTraceEventKind.NODE_ENTERED,
                    started_at=now,
                    completed_at=None,
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="coordinator",
                    event_kind=core.GraphTraceEventKind.NODE_COMPLETED,
                    started_at=now,
                    completed_at=now,
                    routing_decision="human_review",
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="human_review",
                    event_kind=core.GraphTraceEventKind.HUMAN_REVIEW_REQUESTED,
                    started_at=now,
                    completed_at=now,
                    execution_id=scenario_id,
                    review_revision=1,
                ),
                core.GraphTraceEvent(
                    node_name="human_review",
                    event_kind=core.GraphTraceEventKind.GRAPH_RESUMED,
                    started_at=now,
                    completed_at=now,
                    execution_id=scenario_id,
                    review_revision=1,
                ),
                core.GraphTraceEvent(
                    node_name="human_review",
                    event_kind=core.GraphTraceEventKind.REVIEW_ACTION,
                    started_at=now,
                    completed_at=now,
                    execution_id=scenario_id,
                    review_revision=1,
                    review_action=HumanReviewAction.REJECT_CURRENT_PLAN,
                ),
                core.GraphTraceEvent(
                    node_name="human_review",
                    event_kind=core.GraphTraceEventKind.POST_REVIEW_ROUTE,
                    started_at=now,
                    completed_at=now,
                    execution_id=scenario_id,
                    review_revision=1,
                    routing_decision="finalize",
                ),
                core.GraphTraceEvent(
                    node_name="finalize",
                    event_kind=core.GraphTraceEventKind.NODE_ENTERED,
                    started_at=now,
                    completed_at=None,
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="finalize",
                    event_kind=core.GraphTraceEventKind.NODE_COMPLETED,
                    started_at=now,
                    completed_at=now,
                    routing_decision="end",
                    execution_id=scenario_id,
                ),
            ]
        )
    elif "43" in scenario_label or "scheduling" in scenario_label:
        events.extend(
            [
                core.GraphTraceEvent(
                    node_name="coordinator",
                    event_kind=core.GraphTraceEventKind.NODE_ENTERED,
                    started_at=now,
                    completed_at=None,
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="coordinator",
                    event_kind=core.GraphTraceEventKind.NODE_COMPLETED,
                    started_at=now,
                    completed_at=now,
                    routing_decision="replan",
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="replan",
                    event_kind=core.GraphTraceEventKind.NODE_ENTERED,
                    started_at=now,
                    completed_at=None,
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="replan",
                    event_kind=core.GraphTraceEventKind.REPLAN_PLANNED,
                    started_at=now,
                    completed_at=now,
                    routing_decision="specialists",
                    execution_id=scenario_id,
                    review_revision=2,
                ),
                core.GraphTraceEvent(
                    node_name="venue",
                    event_kind=core.GraphTraceEventKind.SPECIALIST_RERUN_STARTED,
                    started_at=now,
                    completed_at=None,
                    execution_id=scenario_id,
                    review_revision=2,
                ),
                core.GraphTraceEvent(
                    node_name="venue",
                    event_kind=core.GraphTraceEventKind.SPECIALIST_RERUN_COMPLETED,
                    started_at=now,
                    completed_at=now,
                    execution_id=scenario_id,
                    review_revision=2,
                    review_action=None,
                ),
                core.GraphTraceEvent(
                    node_name="coordinator",
                    event_kind=core.GraphTraceEventKind.NODE_ENTERED,
                    started_at=now,
                    completed_at=None,
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="coordinator",
                    event_kind=core.GraphTraceEventKind.NODE_COMPLETED,
                    started_at=now,
                    completed_at=now,
                    routing_decision="finalize",
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="finalize",
                    event_kind=core.GraphTraceEventKind.NODE_ENTERED,
                    started_at=now,
                    completed_at=None,
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="finalize",
                    event_kind=core.GraphTraceEventKind.NODE_COMPLETED,
                    started_at=now,
                    completed_at=now,
                    routing_decision="end",
                    execution_id=scenario_id,
                ),
            ]
        )
    else:
        events.extend(
            [
                core.GraphTraceEvent(
                    node_name="coordinator",
                    event_kind=core.GraphTraceEventKind.NODE_ENTERED,
                    started_at=now,
                    completed_at=None,
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="coordinator",
                    event_kind=core.GraphTraceEventKind.NODE_COMPLETED,
                    started_at=now,
                    completed_at=now,
                    routing_decision="human_review",
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="replan",
                    event_kind=core.GraphTraceEventKind.REPLAN_PLANNED,
                    started_at=now,
                    completed_at=now,
                    routing_decision="human_review",
                    execution_id=scenario_id,
                    review_revision=1,
                ),
                core.GraphTraceEvent(
                    node_name="replan",
                    event_kind=core.GraphTraceEventKind.LOOP_BOUND_EXHAUSTED,
                    started_at=now,
                    completed_at=now,
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="finalize",
                    event_kind=core.GraphTraceEventKind.NODE_ENTERED,
                    started_at=now,
                    completed_at=None,
                    execution_id=scenario_id,
                ),
                core.GraphTraceEvent(
                    node_name="finalize",
                    event_kind=core.GraphTraceEventKind.NODE_COMPLETED,
                    started_at=now,
                    completed_at=now,
                    routing_decision="end",
                    execution_id=scenario_id,
                ),
            ]
        )
    return tuple(events)
