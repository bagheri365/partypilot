from __future__ import annotations

from pathlib import Path

import pytest

from partypilot.application.capability_boundary_benchmark import load_capability_boundary_scenarios
from partypilot.application.capability_boundary_inventory import (
    build_capability_boundary_inventory_report,
    render_capability_boundary_inventory_markdown,
    write_capability_boundary_inventory_reports,
)
from partypilot.cli import capability_boundary_inventory as inventory_cli


def test_inventory_report_summarizes_benchmark_surface() -> None:
    scenarios = load_capability_boundary_scenarios()
    report = build_capability_boundary_inventory_report(scenarios)

    assert report.total_scenario_count == 50
    assert sum(report.category_counts.values()) == report.total_scenario_count
    assert report.requires_evidence_count > 0
    assert report.temporal_version_behavior_count > 0
    assert report.cross_domain_dependency_count > 0
    assert report.adversarial_count > 0
    assert report.complexity_trap_count > 0
    assert report.dynamic_replanning_count > 0
    assert report.capability_tag_counts["requires_evidence"] > 0
    assert report.capability_tag_counts["replanning"] > 0
    assert report.capability_tag_counts["global_optimization"] > 0
    assert report.milestone_counts["v0.4-capability-boundary"] > 0
    assert "cap-boundary-26-hidden-service-fees" in report.scenario_ids_by_category["budget"]
    assert "cap-boundary-51-incremental-replanning" in report.scenario_ids_by_category["capacity"]
    assert "cap-boundary-60-full-boss-battle" in report.scenario_ids_by_category["other"]

    markdown = render_capability_boundary_inventory_markdown(report)
    assert "does not evaluate architecture performance" in markdown
    assert "Scenario IDs by Category" in markdown
    assert "cap-boundary-60-full-boss-battle" in markdown


def test_inventory_reports_are_machine_readable_and_markdown(tmp_path: Path) -> None:
    report = build_capability_boundary_inventory_report(load_capability_boundary_scenarios())
    json_path, markdown_path = write_capability_boundary_inventory_reports(report, tmp_path)

    json_text = json_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")

    assert json_path.name == "capability_boundary_inventory.json"
    assert markdown_path.name == "capability_boundary_inventory.md"
    assert '"total_scenario_count": 50' in json_text
    assert "Scenario IDs by Category" in markdown
    assert render_capability_boundary_inventory_markdown(report) == markdown


def test_inventory_cli_prints_summary_and_writes_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = inventory_cli.main(["--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Total scenarios: **50**" in captured.out
    assert "does not evaluate architecture performance" in captured.out
    assert (tmp_path / "capability_boundary_inventory.json").exists()
    assert (tmp_path / "capability_boundary_inventory.md").exists()
