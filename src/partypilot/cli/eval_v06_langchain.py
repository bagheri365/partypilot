"""PartyPilot v0.6d three-way controlled LangChain evaluation CLI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from partypilot.cli.eval_baseline import _ollama_config
from partypilot.cli.v06_langchain_controlled_evaluation_core import (
    default_output_dir,
    load_v06_controlled_scenarios,
    run_v06_controlled_evaluation,
    save_v06_controlled_run_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PartyPilot v0.6d three-way LangChain controlled evaluation."
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
        help="Scenario ID to include in the evaluation. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--allow-dirty-tree",
        action="store_true",
        help="Allow exploratory evaluation from a dirty working tree.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    timestamp = datetime.now(UTC)

    try:
        config = _ollama_config(
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            num_ctx=args.num_ctx,
        )
        scenarios = load_v06_controlled_scenarios(args.scenario_id)
        report = run_v06_controlled_evaluation(
            scenarios,
            model=config.model,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
            num_ctx=config.num_ctx,
            max_retries=config.max_retries,
            allow_dirty_tree=args.allow_dirty_tree,
            timestamp=timestamp,
        )
        output_dir = args.output_dir or default_output_dir(timestamp)
        aggregate_json_path, aggregate_markdown_path, run_paths = save_v06_controlled_run_reports(
            report,
            output_dir,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        print(
            f"ERROR: v0.6d controlled LangChain evaluation failed. Details: {exc}",
            file=sys.stderr,
        )
        return 1

    print("# PartyPilot v0.6d Three-Way LangChain Controlled Evaluation")
    print(f"Benchmark: {report.benchmark_name} ({report.benchmark_version})")
    print(f"JSON: {aggregate_json_path}")
    print(f"Markdown: {aggregate_markdown_path}")
    print(f"Scenario count: {report.scenario_count}")
    print(f"Run order blocks: {', '.join(' -> '.join(block) for block in report.run_order_blocks)}")
    print(f"Run artifacts: {len(run_paths)}")
    for summary in report.variant_summaries:
        print(f"Variant: {summary.variant}")
        print(f"  Final decision accuracy: {summary.final_decision_accuracy.mean:.3f}")
        print(
            "  Evidence-grounded arbitration: "
            f"{summary.evidence_grounded_arbitration_accuracy.mean:.3f}"
        )
        print(f"  Specialist success rate: {summary.specialist_success_rate.mean:.3f}")
        print(f"  Provider timeout count: {summary.provider_timeout_count}")
        print(f"  Structured-output failures: {summary.structured_output_validation_failure_count}")
        print(f"  Disposition: {summary.disposition or 'n/a'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
