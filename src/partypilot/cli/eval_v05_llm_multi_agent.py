"""Live PartyPilot v0.5 multi-agent experiment CLI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from partypilot.adapters import OllamaAdapter, UrllibHttpTransport
from partypilot.application.multi_agent_runtime import (
    default_output_dir,
    load_v05_multi_agent_benchmark,
    run_v05_multi_agent_experiment,
    save_v05_multi_agent_reports,
)
from partypilot.cli.eval_baseline import _ollama_config
from partypilot.composition.multi_agent_runtime import (
    OrchestrationBackend,
    build_live_multi_agent_runtime,
    resolve_orchestration_backend,
)
from partypilot.domain import CapabilityBoundaryScenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the live PartyPilot v0.5 multi-agent experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON and Markdown artifacts.",
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
        help="Scenario ID to include in the live evaluation. Can be supplied multiple times.",
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
    timestamp = datetime.now(UTC)

    try:
        orchestration_backend = resolve_orchestration_backend(args.orchestration_backend)
        config = _ollama_config(
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            num_ctx=args.num_ctx,
        )
        provider = OllamaAdapter(config, UrllibHttpTransport())
        runtime = build_live_multi_agent_runtime(
            provider,
            timeout_seconds=config.timeout_seconds,
            model_name=config.model,
            orchestration_backend=orchestration_backend,
        )
        scenarios = _filter_scenarios(load_v05_multi_agent_benchmark(), args.scenario_id)
        report = run_v05_multi_agent_experiment(
            scenarios,
            runtime=runtime,
            orchestration_backend=orchestration_backend.value,
            timestamp=timestamp,
        )
        output_dir = args.output_dir or default_output_dir(timestamp)
        json_path, markdown_path = save_v05_multi_agent_reports(report, output_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        print(f"ERROR: v0.5 multi-agent experiment failed. Details: {exc}", file=sys.stderr)
        return 1

    print("# PartyPilot v0.5 Live Multi-Agent Runtime Experiment")
    print(f"Benchmark: {report.benchmark_name} ({report.benchmark_version})")
    print(f"Orchestration backend: {orchestration_backend.value}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    print(f"Scenario count: {report.metrics.scenario_count}")
    print(f"Retention rule passed: {report.metrics.retention_rule_passed}")
    print(
        f"Baseline final decision accuracy: {report.metrics.baseline.final_decision_accuracy:.3f}"
    )
    print(f"Live final decision accuracy: {report.metrics.live.final_decision_accuracy:.3f}")
    print(
        f"Baseline evidence-grounded arbitration: "
        f"{report.metrics.baseline.evidence_grounded_arbitration_accuracy:.3f}"
    )
    print(
        f"Live evidence-grounded arbitration: "
        f"{report.metrics.live.evidence_grounded_arbitration_accuracy:.3f}"
    )
    print(f"Live specialist success rate: {report.metrics.runtime.specialist_success_rate:.3f}")
    print(
        "Live validation failure rate: "
        f"{report.metrics.runtime.structured_output_validation_failure_rate:.3f}"
    )
    print(f"Live mean latency (ms): {report.metrics.runtime.mean_latency_ms:.3f}")
    if report.terminal_outcome_mismatch_scenario_ids:
        print("Terminal outcome mismatches:")
        for scenario_id in report.terminal_outcome_mismatch_scenario_ids:
            print(f"- {scenario_id}")
    else:
        print("Terminal outcome mismatches: none")
    if report.diagnostic_failure_stage_scenario_ids:
        print("Diagnostic failure-stage cases:")
        for scenario_id in report.diagnostic_failure_stage_scenario_ids:
            print(f"- {scenario_id}")
    else:
        print("Diagnostic failure-stage cases: none")
    return 0


def _filter_scenarios(
    scenarios: Sequence[CapabilityBoundaryScenario],
    scenario_ids: Sequence[str] | None,
) -> tuple[CapabilityBoundaryScenario, ...]:
    if not scenario_ids:
        return tuple(scenarios)
    scenarios_by_id = {scenario.scenario.scenario_id: scenario for scenario in scenarios}
    filtered = []
    for scenario_id in scenario_ids:
        scenario = scenarios_by_id.get(scenario_id)
        if scenario is None:
            raise ValueError(f"unknown evaluation scenario ID: {scenario_id}")
        filtered.append(scenario)
    return tuple(filtered)


if __name__ == "__main__":
    raise SystemExit(main())
