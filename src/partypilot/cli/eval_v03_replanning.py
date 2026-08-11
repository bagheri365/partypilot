"""Offline PartyPilot v0.3 replanning experiment CLI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from partypilot.application.v03_replanning import (
    build_replanning_metadata,
    default_output_dir,
    load_v03_replanning_benchmark,
    run_v03_replanning_experiment,
    save_v03_replanning_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline PartyPilot v0.3 replanning experiment."
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
        scenarios = load_v03_replanning_benchmark()
        report = run_v03_replanning_experiment(scenarios, timestamp=timestamp)
        # The report already embeds metadata, but we keep the call here for symmetry with
        # the release CLIs and to make the reproducibility intent explicit in tests.
        report = report.model_copy(
            update={"metadata": build_replanning_metadata(timestamp=timestamp)}
        )
        output_dir = args.output_dir or default_output_dir(timestamp)
        json_path, markdown_path = save_v03_replanning_reports(report, output_dir)
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        print(f"ERROR: v0.3 replanning experiment failed. Details: {exc}", file=sys.stderr)
        return 1

    print("# PartyPilot v0.3 Replanning Experiment")
    print(f"Benchmark: {report.benchmark_name} ({report.benchmark_version})")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    print(f"Scenario count: {report.metrics.scenario_count}")
    print(f"Retention rule passed: {report.metrics.retention_rule_passed}")
    print(
        "Full replanning recomputed decisions: "
        f"{report.metrics.full_replan.recomputed_decision_count}"
    )
    print(
        "Targeted replanning recomputed decisions: "
        f"{report.metrics.targeted_replan.recomputed_decision_count}"
    )
    print(
        "Targeted-vs-full recomputation reduction ratio: "
        f"{report.metrics.recomputation_reduction_ratio:.3f}"
    )
    print(
        "Targeted final-state correctness: "
        f"{report.metrics.targeted_replan.final_state_correctness:.3f}"
    )
    print(f"Full final-state correctness: {report.metrics.full_replan.final_state_correctness:.3f}")
    failure_cases = [
        f"{scenario.scenario_id} ({scenario.failure_stage})"
        for scenario in report.scenarios
        if scenario.failure_stage is not None
    ]
    if failure_cases:
        print("Failure cases:")
        for failure_case in failure_cases:
            print(f"- {failure_case}")
    else:
        print("Failure cases: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
