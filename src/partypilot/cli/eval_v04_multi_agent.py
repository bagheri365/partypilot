"""Offline PartyPilot v0.4 minimal specialist coordination experiment CLI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from partypilot.application.v04_multi_agent import (
    default_output_dir,
    load_v04_multi_agent_benchmark,
    run_v04_multi_agent_experiment,
    save_v04_multi_agent_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline PartyPilot v0.4 minimal specialist coordination experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON and Markdown artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    timestamp = datetime.now(UTC)

    try:
        scenarios = load_v04_multi_agent_benchmark()
        report = run_v04_multi_agent_experiment(scenarios, timestamp=timestamp)
        output_dir = args.output_dir or default_output_dir(timestamp)
        json_path, markdown_path = save_v04_multi_agent_reports(report, output_dir)
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        print(f"ERROR: v0.4 multi-agent experiment failed. Details: {exc}", file=sys.stderr)
        return 1

    print("# PartyPilot v0.4 Multi-Agent Coordination Experiment")
    print(f"Benchmark: {report.benchmark_name} ({report.benchmark_version})")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    print(f"Scenario count: {report.metrics.scenario_count}")
    print(f"Retention rule passed: {report.metrics.retention_rule_passed}")
    print(
        f"Baseline final decision accuracy: {report.metrics.baseline.final_decision_accuracy:.3f}"
    )
    print(
        "Coordinated final decision accuracy: "
        f"{report.metrics.coordinated.final_decision_accuracy:.3f}"
    )
    print(
        f"Baseline global optimum accuracy: {report.metrics.baseline.global_optimum_accuracy:.3f}"
    )
    print(
        "Coordinated global optimum accuracy: "
        f"{report.metrics.coordinated.global_optimum_accuracy:.3f}"
    )
    ratio = report.metrics.coordination_overhead_ratio
    print(
        f"Coordination overhead ratio: {ratio:.3f}"
        if ratio is not None
        else "Coordination overhead ratio: N/A"
    )
    failure_cases = [
        f"{scenario.scenario_id} ({scenario.coordinated.failure_stage})"
        for scenario in report.scenarios
        if scenario.coordinated.failure_stage is not None
    ]
    if failure_cases:
        print("Coordinated failure cases:")
        for failure_case in failure_cases:
            print(f"- {failure_case}")
    else:
        print("Coordinated failure cases: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
