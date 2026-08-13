"""Small live smoke test for PartyPilot's LangChain multi-agent runtime."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from partypilot.cli.eval_baseline import _ollama_config
from partypilot.cli.smoke_multi_agent import DEFAULT_SMOKE_SCENARIO_IDS, _smoke_scenarios
from partypilot.composition.multi_agent_runtime import (
    OrchestrationBackend,
    SpecialistAdapterKind,
    build_live_multi_agent_runtime,
    resolve_orchestration_backend,
)
from partypilot.domain import (
    CandidateEvaluationResult,
    MultiAgentPlanningRuntimeResult,
    MultiAgentSmokeRow,
    SpecialistExecutionOutcome,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tiny live LangChain v0.6b multi-agent smoke test."
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
            f"ERROR: v0.6b LangChain multi-agent smoke configuration failed. Details: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        runtime = build_live_multi_agent_runtime(
            timeout_seconds=config.timeout_seconds,
            model_name=config.model,
            adapter_kind=SpecialistAdapterKind.LANGCHAIN,
            orchestration_backend=orchestration_backend,
            ollama_config=config,
        )
        scenarios = _smoke_scenarios(args.scenario_id or DEFAULT_SMOKE_SCENARIO_IDS)
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        print(
            f"ERROR: v0.6b LangChain multi-agent smoke setup failed. Details: {exc}",
            file=sys.stderr,
        )
        return 1

    print("# PartyPilot v0.6b LangChain Multi-Agent Smoke Test")
    print(f"Adapter kind: {SpecialistAdapterKind.LANGCHAIN.value}")
    print(f"Provider I/O timeout: {config.timeout_seconds:.1f}s")
    print(f"Ollama context budget: {getattr(config, 'num_ctx', 8192)} tokens")
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

        for outcome in candidate.specialist_outcomes:
            row = _smoke_row(outcome)
            adapter_variant = outcome.trace.adapter_variant.value
            if outcome.decision is None:
                print(
                    "  - "
                    f"{row.specialist_name} | adapter={adapter_variant} | FAILED | {row.status} | "
                    f"{outcome.trace.failure_reason or 'unknown failure'}"
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
                    f"retries={row.retry_count}"
                )
            if not row.validation_succeeded or outcome.decision is None:
                exit_code = 1

    if exit_code == 0:
        print("Smoke test passed.")
    return exit_code


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


if __name__ == "__main__":
    raise SystemExit(main())
