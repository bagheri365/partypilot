"""Inventory reporting for the expanded capability-boundary benchmark."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from partypilot.domain.evaluation import CapabilityBoundaryScenario
from partypilot.domain.evidence_corpus import EvidenceDocumentStatus


class CapabilityBoundaryInventoryReport(BaseModel):
    """Summary of the capability-boundary benchmark surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_name: str
    total_scenario_count: int = Field(ge=0)
    category_counts: dict[str, int]
    scenario_ids_by_category: dict[str, tuple[str, ...]]
    capability_tag_counts: dict[str, int]
    milestone_counts: dict[str, int]
    requires_evidence_count: int = Field(ge=0)
    temporal_version_behavior_count: int = Field(ge=0)
    cross_domain_dependency_count: int = Field(ge=0)
    adversarial_count: int = Field(ge=0)
    complexity_trap_count: int = Field(ge=0)
    dynamic_replanning_count: int = Field(ge=0)
    notes: tuple[str, ...] = ()


def build_capability_boundary_inventory_report(
    scenarios: Sequence[CapabilityBoundaryScenario],
    *,
    benchmark_name: str = "Capability-boundary benchmark",
) -> CapabilityBoundaryInventoryReport:
    category_counts: Counter[str] = Counter()
    category_ids: defaultdict[str, list[str]] = defaultdict(list)
    capability_tag_counts: Counter[str] = Counter()
    milestone_counts: Counter[str] = Counter()
    requires_evidence_count = 0
    temporal_version_behavior_count = 0
    cross_domain_dependency_count = 0
    adversarial_count = 0
    complexity_trap_count = 0
    dynamic_replanning_count = 0

    for item in scenarios:
        scenario = item.scenario
        category = scenario.scenario_category.value
        category_counts[category] += 1
        category_ids[category].append(scenario.scenario_id)
        capability_tag_counts.update(tag.casefold() for tag in item.metadata.capability_tags)
        milestone_counts[item.metadata.milestone_introduced] += 1

        requires_evidence_count += int(item.metadata.requires_evidence)
        temporal_version_behavior_count += int(_contains_temporal_version_behavior(item))
        cross_domain_dependency_count += int(item.metadata.cross_domain_dependency_count > 0)
        adversarial_count += int(item.metadata.adversarial_flag)
        complexity_trap_count += int(item.metadata.complexity_trap_flag)
        dynamic_replanning_count += int(_contains_dynamic_replanning_behavior(item))

    return CapabilityBoundaryInventoryReport(
        benchmark_name=benchmark_name,
        total_scenario_count=len(scenarios),
        category_counts=dict(sorted(category_counts.items())),
        scenario_ids_by_category={
            category: tuple(scenario_ids) for category, scenario_ids in sorted(category_ids.items())
        },
        capability_tag_counts=dict(sorted(capability_tag_counts.items())),
        milestone_counts=dict(sorted(milestone_counts.items())),
        requires_evidence_count=requires_evidence_count,
        temporal_version_behavior_count=temporal_version_behavior_count,
        cross_domain_dependency_count=cross_domain_dependency_count,
        adversarial_count=adversarial_count,
        complexity_trap_count=complexity_trap_count,
        dynamic_replanning_count=dynamic_replanning_count,
        notes=(
            "This inventory summarizes the expanded capability-boundary benchmark only.",
            (
                "It does not run the canonical v0.2 release evaluation or claim "
                "architecture performance."
            ),
        ),
    )


def render_capability_boundary_inventory_markdown(report: CapabilityBoundaryInventoryReport) -> str:
    lines = [
        "# Capability-Boundary Inventory",
        "",
        f"Benchmark name: `{report.benchmark_name}`",
        f"Total scenarios: **{report.total_scenario_count}**",
        f"Requires evidence: **{report.requires_evidence_count}**",
        f"Temporal/version behavior: **{report.temporal_version_behavior_count}**",
        f"Cross-domain dependencies: **{report.cross_domain_dependency_count}**",
        f"Adversarial scenarios: **{report.adversarial_count}**",
        f"Complexity-trap scenarios: **{report.complexity_trap_count}**",
        f"Dynamic/replanning scenarios: **{report.dynamic_replanning_count}**",
        "",
        "This inventory summarizes the expanded benchmark surface only.",
        "It does not evaluate architecture performance or the canonical v0.2 release metrics.",
        "",
        "## Counts by Capability Tag",
    ]
    for tag, count in sorted(
        report.capability_tag_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"- `{tag}`: {count}")

    lines.extend(["", "## Counts by Milestone Introduced"])
    for milestone, count in sorted(report.milestone_counts.items()):
        lines.append(f"- `{milestone}`: {count}")

    lines.extend(["", "## Counts by Scenario Category"])
    for category, count in sorted(report.category_counts.items()):
        lines.append(f"- `{category}`: {count}")

    lines.extend(["", "## Scenario IDs by Category"])
    for category, scenario_ids in sorted(report.scenario_ids_by_category.items()):
        lines.append(f"- `{category}`")
        lines.extend(f"  - `{scenario_id}`" for scenario_id in scenario_ids)

    if report.notes:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in report.notes)

    return "\n".join(lines) + "\n"


def write_capability_boundary_inventory_reports(
    report: CapabilityBoundaryInventoryReport, output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "capability_boundary_inventory.json"
    markdown_path = output_dir / "capability_boundary_inventory.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_capability_boundary_inventory_markdown(report), encoding="utf-8"
    )
    return json_path, markdown_path


def _contains_temporal_version_behavior(item: CapabilityBoundaryScenario) -> bool:
    tags = {tag.casefold() for tag in item.metadata.capability_tags}
    if any("temporal" in tag or "version" in tag for tag in tags):
        return True
    return any(
        document.metadata.status is not EvidenceDocumentStatus.CURRENT
        for document in item.evidence_documents
    )


def _contains_dynamic_replanning_behavior(item: CapabilityBoundaryScenario) -> bool:
    tags = {tag.casefold() for tag in item.metadata.capability_tags}
    return item.metadata.requires_state_replanning or bool(
        tags.intersection({"replanning", "incremental_update", "state_replanning"})
    )
