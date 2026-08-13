"""Offline smoke test for PartyPilot's resumable LangGraph human review."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from partypilot.application import multi_agent_runtime as runtime_module
from partypilot.application import v04_multi_agent as v04
from partypilot.application.review_workflow import HumanReviewAction, HumanReviewResponse
from partypilot.composition.langgraph_multi_agent_runtime import (
    CandidateGraphExecutionStatus,
    LangGraphMultiAgentPlanningRuntime,
)
from partypilot.domain import (
    ArbitrationOutcome,
    SpecialistDecision,
    SpecialistDomain,
    SpecialistExecutionOutcome,
    SpecialistExecutionTrace,
    canonical_specialist_id,
    canonical_specialist_name,
)

DEFAULT_REVIEW_SCENARIO_ID = "cap-boundary-59-conflicting-agents-evidence"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tiny offline LangGraph human-review smoke test."
    )
    parser.add_argument(
        "--scenario-id",
        default=DEFAULT_REVIEW_SCENARIO_ID,
        help="Scenario ID to review.",
    )
    parser.add_argument(
        "--execution-id",
        default="smoke-review-execution",
        help="Stable execution/thread ID to use for the review smoke.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        scenario = _load_scenario(args.scenario_id)
        candidate_resource_ids = tuple(v04._candidate_combinations(scenario)[0])
        runtime = LangGraphMultiAgentPlanningRuntime(
            _dummy_specialists(),
            model_name="fake-model",
        )
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        print(f"ERROR: v0.7c LangGraph review smoke setup failed. Details: {exc}", file=sys.stderr)
        return 1

    original_run_specialist_invocation = runtime_module._run_specialist_invocation
    runtime_module._run_specialist_invocation = _fake_run_specialist_invocation
    try:
        suspended = runtime.run_reviewable_candidate(
            scenario=scenario,
            candidate_resource_ids=candidate_resource_ids,
            execution_id=args.execution_id,
        )
        print("# PartyPilot v0.7c LangGraph Review Smoke Test")
        print(f"Scenario: {scenario.scenario.scenario_id}")
        print(f"Execution ID: {suspended.execution_id}")
        print(f"Status: {suspended.status.value}")
        print(f"Planning revision: {suspended.planning_revision}")
        print("Review payload:")
        review_request = suspended.review_request
        if review_request is None:
            print("ERROR: review request missing from suspended result.", file=sys.stderr)
            return 1
        print(json.dumps(review_request.model_dump(mode="json"), indent=2, sort_keys=True))
        if suspended.status is not CandidateGraphExecutionStatus.SUSPENDED_FOR_HUMAN_REVIEW:
            print("Unexpected terminal result before review.", file=sys.stderr)
            return 1
        response = HumanReviewResponse(
            execution_id=suspended.execution_id,
            planning_revision=suspended.planning_revision,
            action=HumanReviewAction.REJECT_CURRENT_PLAN,
            candidate_resource_ids=candidate_resource_ids,
        )
        resumed = runtime.resume_reviewable_candidate(
            execution_id=suspended.execution_id,
            review_response=response,
        )
        print(f"Resumed status: {resumed.status.value}")
        final_feasibility = (
            resumed.candidate_run.coordinated_result.feasibility_outcome.value
            if resumed.candidate_run
            else "n/a"
        )
        print(f"Final feasibility: {final_feasibility}")
        print(f"Graph events: {len(resumed.graph_trace)}")
        if resumed.status is not CandidateGraphExecutionStatus.COMPLETED:
            print("Review smoke did not complete after resume.", file=sys.stderr)
            return 1
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        print(f"ERROR: v0.7c LangGraph review smoke failed. Details: {exc}", file=sys.stderr)
        return 1
    finally:
        runtime_module._run_specialist_invocation = original_run_specialist_invocation

    print("Review smoke passed.")
    return 0


def _load_scenario(scenario_id: str) -> Any:
    benchmark = v04.load_v04_multi_agent_benchmark()
    scenarios = {scenario.scenario.scenario_id: scenario for scenario in benchmark}
    scenario = scenarios.get(scenario_id)
    if scenario is None:
        raise ValueError(f"unknown review smoke scenario ID: {scenario_id}")
    return scenario


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


def _fake_run_specialist_invocation(
    specialist: Any, agent_input: Any
) -> SpecialistExecutionOutcome:
    revision_number = agent_input.planning_state.revision_number
    if specialist.specialist_id == "scheduling" and revision_number < 2:
        return _replan_outcome(domain=specialist.domain, revision_number=revision_number)
    return _outcome(domain=specialist.domain)


def _outcome(*, domain: SpecialistDomain) -> SpecialistExecutionOutcome:
    started_at = datetime.now(UTC)
    completed_at = datetime.now(UTC)
    decision = SpecialistDecision(
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
    trace = SpecialistExecutionTrace(
        run_id=f"run-{domain.value}",
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


def _replan_outcome(
    *, domain: SpecialistDomain, revision_number: int
) -> SpecialistExecutionOutcome:
    started_at = datetime.now(UTC)
    completed_at = datetime.now(UTC)
    decision = SpecialistDecision(
        specialist_id=canonical_specialist_id(domain),
        domain=domain,
        recommendation=f"{domain.value} decision",
        status=ArbitrationOutcome.REPLAN_REQUIRED,
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
        run_id=f"run-{domain.value}-r{revision_number}",
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


if __name__ == "__main__":
    raise SystemExit(main())
