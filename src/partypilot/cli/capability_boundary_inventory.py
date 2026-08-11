"""CLI for inspecting the expanded capability-boundary benchmark inventory."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from partypilot.application.capability_boundary_benchmark import load_capability_boundary_scenarios
from partypilot.application.capability_boundary_inventory import (
    build_capability_boundary_inventory_report,
    render_capability_boundary_inventory_markdown,
    write_capability_boundary_inventory_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the capability-boundary benchmark inventory."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for JSON and Markdown inventory artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        scenarios = load_capability_boundary_scenarios()
        report = build_capability_boundary_inventory_report(scenarios)
        markdown = render_capability_boundary_inventory_markdown(report)
        if args.output_dir is not None:
            json_path, markdown_path = write_capability_boundary_inventory_reports(
                report, args.output_dir
            )
            print(f"JSON: {json_path}")
            print(f"Markdown: {markdown_path}")
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        print(f"ERROR: capability-boundary inventory failed. Details: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
