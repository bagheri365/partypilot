"""Small live smoke test for PartyPilot's LangChain tool-using specialist agents."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from partypilot.adapters.ollama import OllamaConfig
from partypilot.application import multi_agent_runtime as runtime_module
from partypilot.application import v04_multi_agent as v04
from partypilot.cli.eval_baseline import _ollama_config
from partypilot.cli.smoke_multi_agent import DEFAULT_SMOKE_SCENARIO_IDS, _smoke_scenarios
from partypilot.composition.multi_agent_runtime import (
    OrchestrationBackend,
    SpecialistAdapterKind,
    build_live_multi_agent_runtime,
    build_specialist_agents,
    resolve_orchestration_backend,
)
from partypilot.domain import (
    CandidateEvaluationResult,
    MultiAgentPlanningRuntimeResult,
    MultiAgentSmokeRow,
    SpecialistExecutionOutcome,
)

AGENT_EXECUTION_BOUND = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tiny live LangChain create_agent multi-agent smoke test."
    )
    parser.add_argument("--base-url", default=None, help="Override PARTYPILOT_OLLAMA_BASE_URL.")
    parser.add_argument("--model", default=None, help="Override PARTYPILOT_OLLAMA_MODEL.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Override PARTYPILOT_OLLAMA_TIMEOUT_SECONDS.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Override PARTYPILOT_OLLAMA_NUM_CTX.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override PARTYPILOT_OLLAMA_MAX_RETRIES.",
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        default=None,
        help="Scenario ID to smoke test. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--orchestration-backend",
        choices=[backend.value for backend in OrchestrationBackend],
        default=None,
        help="Override PARTYPILOT_ORCHESTRATION_BACKEND.",
    )
    parser.add_argument(
        "--diagnostic-scenario-id",
        default=None,
        help=(
            "Optional scenario ID for a focused tool-necessity diagnostic. "
            "When supplied with --diagnostic-specialist-id, runs one specialist."
        ),
    )
    parser.add_argument(
        "--diagnostic-specialist-id",
        default=None,
        choices=["venue", "catering", "accessibility", "scheduling", "budget"],
        help="Canonical specialist_id to run in the focused tool-necessity diagnostic.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        orchestration_backend = resolve_orchestration_backend(args.orchestration_backend)
        config = _ollama_config(
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            num_ctx=args.num_ctx,
        )
    except ValueError as exc:  # pragma: no cover - exercised via CLI tests
        print(
            f"ERROR: v0.6c LangChain agent smoke configuration failed. Details: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        runtime = build_live_multi_agent_runtime(
            timeout_seconds=config.timeout_seconds,
            model_name=config.model,
            adapter_kind=SpecialistAdapterKind.LANGCHAIN_AGENT,
            orchestration_backend=orchestration_backend,
            ollama_config=config,
            chat_model_factory=None,
        )
        scenarios = _smoke_scenarios(args.scenario_id or DEFAULT_SMOKE_SCENARIO_IDS)
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        print(
            f"ERROR: v0.6c LangChain agent smoke setup failed. Details: {exc}",
            file=sys.stderr,
        )
        return 1

    print("# PartyPilot v0.6c LangChain Agent Smoke Test")
    print(f"Adapter kind: {SpecialistAdapterKind.LANGCHAIN_AGENT.value}")
    print(f"Provider I/O timeout: {config.timeout_seconds:.1f}s")
    print(f"Ollama context budget: {getattr(config, 'num_ctx', 8192)} tokens")
    print(f"Agent execution bound: {AGENT_EXECUTION_BOUND}")
    print(f"Model: {config.model}")
    print(f"Orchestration backend: {orchestration_backend.value}")
    print(f"Scenario count: {len(scenarios)}")

    exit_code = 0
    for scenario in scenarios:
        try:
            result = runtime.plan_scenario(scenario)
        except Exception as exc:  # pragma: no cover - exercised via CLI tests
            print(
                f"ERROR: smoke scenario {scenario.scenario.scenario_id} failed. Details: {exc}",
                file=sys.stderr,
            )
            return 1

        print(f"Scenario: {scenario.scenario.scenario_id}")
        print(f"Expected feasibility: {scenario.scenario.expected_feasibility.value}")
        print(f"Predicted feasibility: {result.final_result.feasibility_outcome.value}")
        print(
            f"Selected resources: {', '.join(result.final_result.selected_resource_ids) or 'none'}"
        )

        candidate = _selected_candidate(result)
        if candidate is None:
            print(
                f"ERROR: {scenario.scenario.scenario_id} produced no selected candidate.",
                file=sys.stderr,
            )
            return 1

        print(f"Tools invoked: {_scenario_uses_tools(candidate)}")
        for outcome in candidate.specialist_outcomes:
            row = _smoke_row(outcome)
            adapter_variant = outcome.trace.adapter_variant.value
            tool_names = _tool_names(outcome)
            tool_failures = _tool_failures(outcome)
            agent_limit_hit = outcome.trace.agent_execution_limit_hit
            if outcome.decision is None:
                print(
                    "  - "
                    f"{row.specialist_name} | adapter={adapter_variant} | FAILED | "
                    f"{row.status} | {outcome.trace.failure_reason or 'unknown failure'} | "
                    f"evidence={', '.join(row.evidence_ids) or 'none'} | "
                    f"validated={row.validation_succeeded} | retries={row.retry_count} | "
                    f"latency_ms={row.latency_ms:.1f}, "
                    f"tool_calls={outcome.trace.tool_call_count}, "
                    f"tool_names={tool_names} | tool_failures={tool_failures} | "
                    f"agent_limit_hit={agent_limit_hit}"
                )
            else:
                print(
                    "  - "
                    f"{row.specialist_name}: adapter={adapter_variant}, "
                    f"status={row.status}, "
                    f"recommendations={row.recommendation_count}, "
                    f"evidence={', '.join(row.evidence_ids) or 'none'}, "
                    f"latency_ms={row.latency_ms:.1f}, "
                    f"validated={row.validation_succeeded}, "
                    f"retries={row.retry_count}, "
                    f"tool_calls={outcome.trace.tool_call_count}, "
                    f"tool_names={tool_names}, "
                    f"tool_failures={tool_failures}, "
                    f"agent_limit_hit={agent_limit_hit}"
                )
            if not row.validation_succeeded or outcome.decision is None:
                exit_code = 1

    if args.diagnostic_scenario_id is not None or args.diagnostic_specialist_id is not None:
        if args.diagnostic_scenario_id is None or args.diagnostic_specialist_id is None:
            print(
                "ERROR: focused diagnostic requires both --diagnostic-scenario-id and "
                "--diagnostic-specialist-id.",
                file=sys.stderr,
            )
            return 1
        diagnostic_exit_code = _run_tool_necessity_diagnostic(
            config=config,
            scenario_id=args.diagnostic_scenario_id,
            specialist_id=args.diagnostic_specialist_id,
        )
        exit_code = max(exit_code, diagnostic_exit_code)

    if exit_code == 0:
        print("Smoke test passed.")
    return exit_code


def _scenario_uses_tools(candidate: CandidateEvaluationResult) -> str:
    return (
        "yes"
        if any(outcome.trace.tool_call_count > 0 for outcome in candidate.specialist_outcomes)
        else "no"
    )


def _selected_candidate(
    result: MultiAgentPlanningRuntimeResult,
) -> CandidateEvaluationResult | None:
    if not isinstance(result, MultiAgentPlanningRuntimeResult):  # pragma: no cover - defensive
        return None
    selected = tuple(result.final_result.selected_resource_ids)
    for candidate in result.candidate_results:
        if tuple(candidate.candidate_resource_ids) == selected:
            return candidate
    return result.candidate_results[0] if result.candidate_results else None


def _smoke_row(outcome: SpecialistExecutionOutcome) -> MultiAgentSmokeRow:
    return MultiAgentSmokeRow(
        specialist_name=outcome.trace.specialist_name,
        status=(
            outcome.decision.status.value
            if outcome.decision is not None
            else (outcome.failure_kind.name if outcome.failure_kind is not None else "unknown")
        ),
        recommendation_count=(
            len(outcome.decision.recommended_resource_ids) if outcome.decision is not None else 0
        ),
        evidence_ids=(
            tuple(evidence.evidence_id for evidence in outcome.decision.evidence_references)
            if outcome.decision is not None
            else outcome.trace.evidence_document_ids
        ),
        latency_ms=outcome.trace.latency_ms,
        validation_succeeded=outcome.trace.validation_succeeded,
        retry_count=outcome.trace.retry_count,
    )


def _tool_names(outcome: SpecialistExecutionOutcome) -> str:
    names = tuple(dict.fromkeys(trace.tool_name for trace in outcome.trace.tool_call_traces))
    return ", ".join(names) if names else "none"


def _tool_failures(outcome: SpecialistExecutionOutcome) -> str:
    failures = [
        f"{trace.tool_name}:{trace.error_kind}"
        for trace in outcome.trace.tool_call_traces
        if not trace.success
    ]
    return ", ".join(failures) if failures else "none"


def _run_tool_necessity_diagnostic(
    *,
    config: OllamaConfig,
    scenario_id: str,
    specialist_id: str,
) -> int:
    print("# Tool-Necessity Diagnostic")
    print(f"Diagnostic scenario: {scenario_id}")
    print(f"Diagnostic specialist_id: {specialist_id}")

    scenarios = _smoke_scenarios([scenario_id])
    if not scenarios:
        print(f"ERROR: diagnostic scenario {scenario_id!r} was not found.", file=sys.stderr)
        return 1
    scenario = scenarios[0]

    specialists = build_specialist_agents(
        adapter_kind=SpecialistAdapterKind.LANGCHAIN_AGENT,
        timeout_seconds=config.timeout_seconds,
        model_name=config.model,
        ollama_config=config,
    )
    specialist = next(
        (item for item in specialists if item.specialist_id == specialist_id),
        None,
    )
    if specialist is None:
        print(
            f"ERROR: diagnostic specialist_id {specialist_id!r} was not found.",
            file=sys.stderr,
        )
        return 1

    candidate_resources = scenario.structured_resources
    resources_by_id = {resource.resource_id: resource for resource in candidate_resources}
    candidate_ids = next(iter(v04._candidate_combinations(scenario)), None)
    if candidate_ids is None:
        print("ERROR: diagnostic scenario did not produce any candidates.", file=sys.stderr)
        return 1
    selected_candidate_resources = tuple(
        resources_by_id[resource_id] for resource_id in candidate_ids
    )
    planning_state = runtime_module._build_planning_state(scenario, selected_candidate_resources)
    agent_input = runtime_module._specialist_input(
        scenario=scenario,
        planning_state=planning_state,
        candidate_resources=selected_candidate_resources,
        candidate_resource_ids=candidate_ids,
        specialist=specialist,
    )

    outcome = specialist.run(agent_input)
    print("Diagnostic expected tool use: yes")
    print(f"Structured response present: {outcome.raw_structured_output is not None}")
    print(f"Validation succeeded: {outcome.trace.validation_succeeded}")
    print(f"Agent execution limit hit: {outcome.trace.agent_execution_limit_hit}")
    print(f"Tool calls: {outcome.trace.tool_call_count}")
    for trace in outcome.trace.tool_call_traces:
        print(
            "  - "
            f"{trace.tool_name} | request={trace.request_summary or 'none'} | "
            f"success={trace.success} | failure={trace.error_kind or 'none'}"
        )

    if (
        outcome.decision is None
        or not outcome.trace.validation_succeeded
        or outcome.trace.agent_execution_limit_hit
        or outcome.trace.tool_call_count == 0
    ):
        print("Diagnostic failed.", file=sys.stderr)
        return 1

    evidence_ids = (
        ", ".join(evidence.evidence_id for evidence in outcome.decision.evidence_references)
        or "none"
    )
    outcome_status = outcome.decision.status.value
    print(f"Diagnostic specialist outcome: {outcome_status} | evidence={evidence_ids}")
    print("Diagnostic passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
