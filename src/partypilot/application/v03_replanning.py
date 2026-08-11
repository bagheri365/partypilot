"""Offline v0.3 replanning experiment for explicit state and targeted invalidation."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application.targeted_replanning import (
    PlanningReplanResult,
    ReplanningStrategy,
    apply_full_replanning,
    apply_targeted_replanning,
)
from partypilot.domain import (
    AccessibilityAttribute,
    Activity,
    Caterer,
    ExperimentConfig,
    ExperimentResultMetadata,
    PartyRequest,
    PlanningDecision,
    PlanningDecisionCategory,
    PlanningDependency,
    PlanningDependencyKind,
    PlanningState,
    PlanningUpdate,
    PlanningUpdateKind,
    Resource,
    Venue,
)

BENCHMARK_NAME = "v0.3 replanning benchmark"
BENCHMARK_VERSION = "1.0"
ARCHITECTURE_VARIANT = "stateful_decomposition_and_targeted_replanning"
DEFAULT_OUTPUT_ROOT = Path("evals") / "results" / "v0_3" / "replanning"
TARGETED_RECOMPUTATION_REDUCTION_MIN = 0.25
TARGETED_PRESERVED_DECISION_ACCURACY_MIN = 0.9


class ReplanningBenchmarkScenario(BaseModel):
    """Deterministic offline benchmark fixture for a replanning experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    capability_tags: tuple[str, ...] = ()
    initial_state: PlanningState
    updates: tuple[PlanningUpdate, ...] = Field(min_length=1)
    expected_invalidated_decision_ids: tuple[str, ...] = ()
    expected_preserved_decision_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class ReplanningMetricDefinition(BaseModel):
    """Human-readable definition for a deterministic replanning metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)


class ReplanningStrategyMetrics(BaseModel):
    """Objective metrics for one replanning strategy on a benchmark suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: ReplanningStrategy
    scenario_count: int = Field(ge=0)
    invalidation_accuracy: float = Field(ge=0, le=1)
    preserved_decision_accuracy: float = Field(ge=0, le=1)
    final_state_correctness: float = Field(ge=0, le=1)
    recomputed_decision_count: int = Field(ge=0)
    unnecessary_recomputation_count: int = Field(ge=0)
    missed_recomputation_count: int = Field(ge=0)
    mean_latency_ms: float = Field(ge=0)
    cycle_detected_count: int = Field(ge=0)


class ReplanningScenarioStrategyMetrics(BaseModel):
    """Per-scenario metrics for one replanning strategy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: ReplanningStrategy
    invalidated_decision_ids: tuple[str, ...]
    preserved_decision_ids: tuple[str, ...]
    recomputed_decision_ids: tuple[str, ...]
    recomputed_decision_count: int = Field(ge=0)
    invalidation_accuracy: float = Field(ge=0, le=1)
    preserved_decision_accuracy: float = Field(ge=0, le=1)
    final_state_correctness: float = Field(ge=0, le=1)
    missed_recomputation_count: int = Field(ge=0)
    unnecessary_recomputation_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    cycle_detected: bool = False
    cycle_decision_ids: tuple[str, ...] = ()
    cycle_error: str | None = None


class ReplanningScenarioResult(BaseModel):
    """Machine-readable result for a single replanning benchmark scenario."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    capability_tags: tuple[str, ...] = ()
    expected_invalidated_decision_ids: tuple[str, ...] = ()
    expected_preserved_decision_ids: tuple[str, ...] = ()
    full_replan: ReplanningScenarioStrategyMetrics
    targeted_replan: ReplanningScenarioStrategyMetrics
    failure_stage: str | None = None
    notes: tuple[str, ...] = ()


class ReplanningExperimentMetrics(BaseModel):
    """Aggregate metrics for the v0.3 replanning experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_count: int = Field(ge=0)
    full_replan: ReplanningStrategyMetrics
    targeted_replan: ReplanningStrategyMetrics
    recomputation_reduction_ratio: float = Field(ge=0, le=1)
    retention_rule_passed: bool


class V03ReplanningReport(BaseModel):
    """Complete offline v0.3 replanning experiment report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    system_name: str = "PartyPilot v0.3 replanning experiment"
    benchmark_name: str = BENCHMARK_NAME
    benchmark_version: str = BENCHMARK_VERSION
    evaluation_variant: str = "full_replan_vs_dependency_aware_targeted_replan"
    metrics: ReplanningExperimentMetrics
    scenarios: tuple[ReplanningScenarioResult, ...]
    metric_definitions: tuple[ReplanningMetricDefinition, ...]
    metadata: ExperimentResultMetadata | None = None
    notes: tuple[str, ...] = ()


def load_v03_replanning_benchmark() -> tuple[ReplanningBenchmarkScenario, ...]:
    """Return the frozen offline fixtures for the v0.3 replanning experiment."""

    return (
        _incremental_guest_count_scenario(),
        _new_sesame_allergy_scenario(),
        _cascading_failure_scenario(),
        _no_op_control_scenario(),
        _broad_update_control_scenario(),
    )


def run_v03_replanning_experiment(
    scenarios: Sequence[ReplanningBenchmarkScenario] | None = None,
    *,
    timestamp: datetime | None = None,
) -> V03ReplanningReport:
    """Run the offline replanning comparison against the deterministic benchmark."""

    benchmark = tuple(scenarios) if scenarios is not None else load_v03_replanning_benchmark()
    scenario_results: list[ReplanningScenarioResult] = []
    for scenario in benchmark:
        full = apply_full_replanning(scenario.initial_state, scenario.updates)
        targeted = apply_targeted_replanning(scenario.initial_state, scenario.updates)
        full_metrics = _scenario_strategy_metrics(
            full,
            scenario.expected_invalidated_decision_ids,
            scenario.expected_preserved_decision_ids,
        )
        targeted_metrics = _scenario_strategy_metrics(
            targeted,
            scenario.expected_invalidated_decision_ids,
            scenario.expected_preserved_decision_ids,
        )
        scenario_results.append(
            ReplanningScenarioResult(
                scenario_id=scenario.scenario_id,
                title=scenario.title,
                description=scenario.description,
                capability_tags=scenario.capability_tags,
                expected_invalidated_decision_ids=scenario.expected_invalidated_decision_ids,
                expected_preserved_decision_ids=scenario.expected_preserved_decision_ids,
                full_replan=full_metrics,
                targeted_replan=targeted_metrics,
                failure_stage=_failure_stage(targeted_metrics),
                notes=scenario.notes,
            )
        )

    full_aggregate = _aggregate_strategy_metrics(
        ReplanningStrategy.FULL_REPLAN, tuple(item.full_replan for item in scenario_results)
    )
    targeted_aggregate = _aggregate_strategy_metrics(
        ReplanningStrategy.TARGETED_REPLAN,
        tuple(item.targeted_replan for item in scenario_results),
    )
    recomputation_reduction_ratio = (
        1.0
        - (targeted_aggregate.recomputed_decision_count / full_aggregate.recomputed_decision_count)
        if full_aggregate.recomputed_decision_count > 0
        else 0.0
    )
    retention_rule_passed = _retention_rule_passed(full_aggregate, targeted_aggregate)

    return V03ReplanningReport(
        metrics=ReplanningExperimentMetrics(
            scenario_count=len(benchmark),
            full_replan=full_aggregate,
            targeted_replan=targeted_aggregate,
            recomputation_reduction_ratio=recomputation_reduction_ratio,
            retention_rule_passed=retention_rule_passed,
        ),
        scenarios=tuple(scenario_results),
        metric_definitions=_metric_definitions(),
        metadata=build_replanning_metadata(timestamp=timestamp or datetime.now(UTC)),
        notes=(
            "This experiment is fully offline and deterministic.",
            "It compares full replanning against dependency-aware targeted replanning.",
            (
                "Scenario IDs 51, 52, and 55 align with the predeclared capability-boundary "
                "research fixtures."
            ),
            "No agents, orchestration framework, semantic retrieval, RRF, or reranking are used.",
        ),
    )


def render_v03_replanning_markdown(report: V03ReplanningReport) -> str:
    """Render a reproducible Markdown summary for the offline replanning experiment."""

    lines = [
        "# PartyPilot v0.3 Replanning Experiment",
        "",
        f"Benchmark name: `{report.benchmark_name}`",
        f"Benchmark version: `{report.benchmark_version}`",
        f"Evaluation variant: `{report.evaluation_variant}`",
        f"Scenario count: **{report.metrics.scenario_count}**",
        "",
    ]
    if report.metadata is not None:
        config = report.metadata.config
        lines.extend(
            [
                "## Reproducibility Metadata",
                "",
                f"- Git SHA: `{config.code_commit_sha or 'n/a'}`",
                f"- Working tree dirty: `{config.working_tree_dirty}`",
                f"- Dataset version: `{config.dataset_version}`",
                f"- Architecture variant: `{config.architecture_variant}`",
                f"- Evaluation split: `{report.metadata.evaluation_split or 'n/a'}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Metric Definitions",
            "",
        ]
    )
    for definition in report.metric_definitions:
        lines.append(f"- `{definition.name}`: {definition.definition}")

    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            (
                "| Strategy | Invalidation Accuracy | Preserved Accuracy | "
                "Final-State Correctness | Recomputed Decisions | Unnecessary "
                "Recomputation | Missed Recomputation | Mean Latency (ms) | "
                "Cycle Detections |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            _render_strategy_row(report.metrics.full_replan),
            _render_strategy_row(report.metrics.targeted_replan),
            "",
            (
                "- Targeted-vs-full recomputation reduction ratio: "
                f"{report.metrics.recomputation_reduction_ratio:.3f}"
            ),
            f"- Retention rule passed: {report.metrics.retention_rule_passed}",
            "",
            "## Per-Scenario Results",
            "",
        ]
    )
    for scenario in report.scenarios:
        capability_tags = ", ".join(f"`{tag}`" for tag in scenario.capability_tags) or "n/a"
        expected_invalidated = (
            ", ".join(f"`{item}`" for item in scenario.expected_invalidated_decision_ids) or "none"
        )
        expected_preserved = (
            ", ".join(f"`{item}`" for item in scenario.expected_preserved_decision_ids) or "none"
        )
        lines.extend(
            [
                f"### {scenario.scenario_id}",
                scenario.title,
                "",
                f"- Capability tags: {capability_tags}",
                f"- Expected invalidated decisions: {expected_invalidated}",
                f"- Expected preserved decisions: {expected_preserved}",
                f"- Failure stage: `{scenario.failure_stage or 'none'}`",
                "",
                (
                    "| Strategy | Invalidation Accuracy | Preserved Accuracy | "
                    "Final-State Correctness | Recomputed Decisions | Unnecessary "
                    "Recomputation | Missed Recomputation | Mean Latency (ms) | "
                    "Cycle Detections |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                _render_scenario_strategy_row(scenario.full_replan),
                _render_scenario_strategy_row(scenario.targeted_replan),
                "",
            ]
        )
    if report.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines).strip() + "\n"


def save_v03_replanning_reports(
    report: V03ReplanningReport,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown artifacts for the replanning experiment."""

    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "v0_3_replanning.json"
    markdown_path = output_directory / "v0_3_replanning.md"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_v03_replanning_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def default_output_dir(timestamp: datetime) -> Path:
    """Return the canonical timestamped output directory for the experiment."""

    return DEFAULT_OUTPUT_ROOT / timestamp.strftime("%Y%m%dT%H%M%SZ")


def build_replanning_metadata(*, timestamp: datetime) -> ExperimentResultMetadata:
    """Build reproducibility metadata for the offline replanning experiment."""

    commit_sha, working_tree_dirty, git_metadata_error = _git_metadata()
    config = ExperimentConfig(
        experiment_id=f"v0.3-replanning-{timestamp.strftime('%Y%m%dT%H%M%SZ')}",
        code_commit_sha=commit_sha,
        working_tree_dirty=working_tree_dirty,
        git_metadata_error=git_metadata_error,
        dataset_version="v0.3",
        architecture_variant=ARCHITECTURE_VARIANT,
        timestamp=timestamp,
    )
    return ExperimentResultMetadata(config=config, evaluation_split="benchmark")


def _scenario_strategy_metrics(
    result: PlanningReplanResult,
    expected_invalidated_decision_ids: Sequence[str],
    expected_preserved_decision_ids: Sequence[str],
) -> ReplanningScenarioStrategyMetrics:
    invalidated = result.invalidation.invalidated_decision_ids
    preserved = result.invalidation.preserved_decision_ids
    recomputed = result.recomputed_decision_ids
    expected_invalidated = tuple(expected_invalidated_decision_ids)
    expected_preserved = tuple(expected_preserved_decision_ids)
    expected_invalidated_set = set(expected_invalidated)
    expected_preserved_set = set(expected_preserved)
    recomputed_set = set(recomputed)
    return ReplanningScenarioStrategyMetrics(
        strategy=result.strategy,
        invalidated_decision_ids=invalidated,
        preserved_decision_ids=preserved,
        recomputed_decision_ids=recomputed,
        recomputed_decision_count=result.recomputed_decision_count,
        invalidation_accuracy=_accuracy(invalidated, expected_invalidated),
        preserved_decision_accuracy=_accuracy(preserved, expected_preserved),
        final_state_correctness=float(
            set(invalidated) == expected_invalidated_set
            and set(preserved) == expected_preserved_set
            and result.invalidation.updated_state.revision_number
            == result.invalidation.previous_state.revision_number + 1
        ),
        missed_recomputation_count=len(expected_invalidated_set - recomputed_set),
        unnecessary_recomputation_count=len(recomputed_set - expected_invalidated_set),
        latency_ms=result.latency_ms,
        cycle_detected=result.cycle_detected,
        cycle_decision_ids=result.cycle_decision_ids,
        cycle_error=result.cycle_error,
    )


def _aggregate_strategy_metrics(
    strategy: ReplanningStrategy,
    metrics: Sequence[ReplanningScenarioStrategyMetrics],
) -> ReplanningStrategyMetrics:
    if not metrics:
        return ReplanningStrategyMetrics(
            strategy=strategy,
            scenario_count=0,
            invalidation_accuracy=0.0,
            preserved_decision_accuracy=0.0,
            final_state_correctness=0.0,
            recomputed_decision_count=0,
            unnecessary_recomputation_count=0,
            missed_recomputation_count=0,
            mean_latency_ms=0.0,
            cycle_detected_count=0,
        )
    count = len(metrics)
    return ReplanningStrategyMetrics(
        strategy=strategy,
        scenario_count=count,
        invalidation_accuracy=sum(item.invalidation_accuracy for item in metrics) / count,
        preserved_decision_accuracy=sum(item.preserved_decision_accuracy for item in metrics)
        / count,
        final_state_correctness=sum(item.final_state_correctness for item in metrics) / count,
        recomputed_decision_count=sum(item.recomputed_decision_count for item in metrics),
        unnecessary_recomputation_count=sum(
            item.unnecessary_recomputation_count for item in metrics
        ),
        missed_recomputation_count=sum(item.missed_recomputation_count for item in metrics),
        mean_latency_ms=sum(item.latency_ms for item in metrics) / count,
        cycle_detected_count=sum(int(item.cycle_detected) for item in metrics),
    )


def _failure_stage(metrics: ReplanningScenarioStrategyMetrics) -> str | None:
    if metrics.cycle_detected:
        return "cycle_detected"
    if metrics.invalidation_accuracy < 1.0:
        return "invalidation"
    if metrics.preserved_decision_accuracy < 1.0:
        return "preservation"
    if metrics.final_state_correctness < 1.0:
        return "final_state"
    return None


def _retention_rule_passed(
    full: ReplanningStrategyMetrics, targeted: ReplanningStrategyMetrics
) -> bool:
    return (
        targeted.final_state_correctness >= full.final_state_correctness
        and targeted.invalidation_accuracy == 1.0
        and targeted.missed_recomputation_count == 0
        and targeted.preserved_decision_accuracy >= TARGETED_PRESERVED_DECISION_ACCURACY_MIN
        and targeted.unnecessary_recomputation_count < full.unnecessary_recomputation_count
        and (
            1.0
            - (
                targeted.recomputed_decision_count / full.recomputed_decision_count
                if full.recomputed_decision_count > 0
                else 0.0
            )
        )
        >= TARGETED_RECOMPUTATION_REDUCTION_MIN
    )


def _metric_definitions() -> tuple[ReplanningMetricDefinition, ...]:
    unnecessary_recomputation_definition = (
        "Count of recomputed decisions beyond the minimal required set."
    )
    return (
        ReplanningMetricDefinition(
            name="invalidation_accuracy",
            definition=(
                "Fraction of required recomputations that are reflected in the final invalidated "
                "decision set."
            ),
        ),
        ReplanningMetricDefinition(
            name="preserved_decision_accuracy",
            definition=(
                "Fraction of benchmark-preserved decisions that remain preserved in the "
                "resulting state."
            ),
        ),
        ReplanningMetricDefinition(
            name="final_state_correctness",
            definition=(
                "Exact match between the expected final decision statuses and the strategy's "
                "resulting state, independent of recomputation volume."
            ),
        ),
        ReplanningMetricDefinition(
            name="recomputed_decision_count",
            definition="Total number of decisions recomputed by the strategy across scenarios.",
        ),
        ReplanningMetricDefinition(
            name="unnecessary_recomputation_count",
            definition=unnecessary_recomputation_definition,
        ),
        ReplanningMetricDefinition(
            name="missed_recomputation_count",
            definition=("Total count of decisions that should have been recomputed but were not."),
        ),
        ReplanningMetricDefinition(
            name="recomputation_reduction_ratio",
            definition=(
                "Targeted replanning reduction relative to full replanning, computed from "
                "aggregate recomputed decision counts."
            ),
        ),
        ReplanningMetricDefinition(
            name="mean_latency_ms",
            definition="Mean wall-clock latency per scenario for the strategy.",
        ),
    )


def _git_metadata() -> tuple[str | None, bool | None, str | None]:
    project_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        working_tree_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except Exception as exc:
        return None, None, f"Git metadata unavailable: {type(exc).__name__}: {exc}"
    return commit or None, working_tree_dirty, None


def _resources() -> tuple[Resource, ...]:
    return (
        Venue(
            resource_id="venue-brooklyn-loft",
            name="Brooklyn Loft",
            location="Brooklyn, NY",
            price=Decimal("1200.00"),
            capacity=90,
            accessibility_attributes=frozenset(
                {
                    AccessibilityAttribute.WHEELCHAIR_ACCESSIBLE,
                    AccessibilityAttribute.ACCESSIBLE_RESTROOM,
                    AccessibilityAttribute.STEP_FREE_ACCESS,
                }
            ),
        ),
        Caterer(
            resource_id="caterer-family-table",
            name="Family Table",
            location="Brooklyn, NY",
            price=Decimal("600.00"),
            capacity=90,
        ),
        Activity(
            resource_id="activity-craft-party",
            name="Craft Party",
            location="Brooklyn, NY",
            price=Decimal("200.00"),
            capacity=90,
        ),
    )


def _request(
    *,
    guest_count: int,
    total_budget: Decimal,
    event_date: date,
    event_time: time | None = None,
    allergies: tuple[str, ...] = (),
    dietary_restrictions: tuple[str, ...] = (),
    accessibility_needs: tuple[str, ...] = (),
    theme_preferences: tuple[str, ...] = (),
    other_constraints: tuple[str, ...] = (),
) -> PartyRequest:
    return PartyRequest(
        location="Brooklyn, NY",
        event_date=event_date,
        event_time=event_time,
        guest_count=guest_count,
        total_budget=total_budget,
        allergies=allergies,
        dietary_restrictions=dietary_restrictions,
        accessibility_needs=accessibility_needs,
        theme_preferences=theme_preferences,
        other_constraints=other_constraints,
    )


def _dependency(
    *,
    dependency_id: str,
    kind: PlanningDependencyKind,
    source: str,
    target: str,
    description: str,
) -> PlanningDependency:
    return PlanningDependency(
        dependency_id=dependency_id,
        kind=kind,
        source=source,
        target=target,
        description=description,
    )


def _decision(
    *,
    decision_id: str,
    category: PlanningDecisionCategory,
    summary: str,
    dependency_ids: tuple[str, ...] = (),
    prerequisite_decision_ids: tuple[str, ...] = (),
    resource_ids: tuple[str, ...] = (),
) -> PlanningDecision:
    return PlanningDecision(
        decision_id=decision_id,
        category=category,
        summary=summary,
        dependency_ids=dependency_ids,
        prerequisite_decision_ids=prerequisite_decision_ids,
        resource_ids=resource_ids,
    )


def _state(
    *,
    request: PartyRequest,
    decisions: tuple[PlanningDecision, ...],
    dependencies: tuple[PlanningDependency, ...],
    selected_resources: tuple[Resource, ...] | None = None,
    assumptions: tuple[str, ...] = (),
    unresolved_uncertainties: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> PlanningState:
    return PlanningState(
        revision_number=1,
        request=request,
        selected_resources=selected_resources if selected_resources is not None else _resources(),
        decisions=decisions,
        dependency_relationships=dependencies,
        assumptions=assumptions,
        unresolved_uncertainties=unresolved_uncertainties,
        notes=notes,
    )


def _incremental_guest_count_scenario() -> ReplanningBenchmarkScenario:
    dependencies = (
        _dependency(
            dependency_id="dep-guest-capacity",
            kind=PlanningDependencyKind.GUEST_COUNT_TO_VENUE_CAPACITY,
            source="guest_count",
            target="venue_capacity",
            description="Guest count affects venue capacity",
        ),
        _dependency(
            dependency_id="dep-guest-quantity",
            kind=PlanningDependencyKind.GUEST_COUNT_TO_CATERING_QUANTITY,
            source="guest_count",
            target="catering_quantity",
            description="Guest count affects catering quantity",
        ),
        _dependency(
            dependency_id="dep-guest-seating",
            kind=PlanningDependencyKind.GUEST_COUNT_TO_SEATING,
            source="guest_count",
            target="seating",
            description="Guest count affects seating",
        ),
        _dependency(
            dependency_id="dep-guest-parking",
            kind=PlanningDependencyKind.GUEST_COUNT_TO_PARKING,
            source="guest_count",
            target="parking",
            description="Guest count affects parking",
        ),
        _dependency(
            dependency_id="dep-budget-total-cost",
            kind=PlanningDependencyKind.BUDGET_TO_TOTAL_COST,
            source="total_budget",
            target="total_cost",
            description="Budget affects total cost",
        ),
    )
    decisions = (
        _decision(
            decision_id="venue_capacity",
            category=PlanningDecisionCategory.RESOURCE_SELECTION,
            summary="Venue capacity remains sufficient for the initial guest count",
            dependency_ids=("dep-guest-capacity",),
            resource_ids=("venue-brooklyn-loft",),
        ),
        _decision(
            decision_id="catering_quantity",
            category=PlanningDecisionCategory.RESOURCE_SELECTION,
            summary="Catering quantity fits the initial guest count",
            dependency_ids=("dep-guest-quantity",),
            resource_ids=("caterer-family-table",),
        ),
        _decision(
            decision_id="seating",
            category=PlanningDecisionCategory.RESOURCE_SELECTION,
            summary="Seating plan matches the guest count",
            dependency_ids=("dep-guest-seating",),
            resource_ids=("venue-brooklyn-loft",),
        ),
        _decision(
            decision_id="parking",
            category=PlanningDecisionCategory.RESOURCE_SELECTION,
            summary="Parking plan matches the guest count",
            dependency_ids=("dep-guest-parking",),
            resource_ids=("venue-brooklyn-loft",),
        ),
        _decision(
            decision_id="total_cost",
            category=PlanningDecisionCategory.BUDGET,
            summary="Total cost stays within the available budget",
            dependency_ids=("dep-budget-total-cost",),
            resource_ids=("venue-brooklyn-loft", "caterer-family-table"),
        ),
        _decision(
            decision_id="theme",
            category=PlanningDecisionCategory.PREFERENCE,
            summary="Garden theme preference remains intact",
        ),
        _decision(
            decision_id="dietary_policies",
            category=PlanningDecisionCategory.DIETARY,
            summary="Dietary policies remain unchanged",
        ),
        _decision(
            decision_id="entertainment",
            category=PlanningDecisionCategory.PREFERENCE,
            summary="Entertainment preference remains intact",
        ),
        _decision(
            decision_id="accessibility_requirements",
            category=PlanningDecisionCategory.ACCESSIBILITY,
            summary="Accessibility requirements remain satisfied",
            resource_ids=("venue-brooklyn-loft",),
        ),
    )
    request = _request(
        guest_count=60,
        total_budget=Decimal("2500.00"),
        event_date=date(2027, 1, 15),
        theme_preferences=("garden",),
        accessibility_needs=("wheelchair_accessible",),
        dietary_restrictions=("vegan",),
    )
    state = _state(
        request=request,
        decisions=decisions,
        dependencies=dependencies,
        unresolved_uncertainties=("Guest count increase has not yet been applied.",),
        notes=("Incremental guest-count update benchmark fixture.",),
    )
    update = PlanningUpdate(
        update_id="update-guest-count-85",
        kind=PlanningUpdateKind.GUEST_COUNT_CHANGED,
        description="Guest count increases from 60 to 85",
        guest_count=85,
    )
    return ReplanningBenchmarkScenario(
        scenario_id="cap-boundary-51-incremental-replanning",
        title="Incremental replanning after guest-count increase",
        description=(
            "Guest count changes after the initial plan, so only guest-sensitive decisions "
            "should be invalidated."
        ),
        capability_tags=("replanning", "incremental_update", "guest_count", "cross_domain"),
        initial_state=state,
        updates=(update,),
        expected_invalidated_decision_ids=(
            "venue_capacity",
            "catering_quantity",
            "seating",
            "parking",
            "total_cost",
        ),
        expected_preserved_decision_ids=(
            "theme",
            "dietary_policies",
            "entertainment",
            "accessibility_requirements",
        ),
        notes=(
            "Guest-count-sensitive decisions must be rechecked.",
            "Theme and accessibility should survive the update.",
        ),
    )


def _new_sesame_allergy_scenario() -> ReplanningBenchmarkScenario:
    dependencies = (
        _dependency(
            dependency_id="dep-dietary-evidence",
            kind=PlanningDependencyKind.DIETARY_TO_CATERING_EVIDENCE,
            source="dietary_restrictions",
            target="catering_safety_conclusion",
            description="Dietary restrictions affect catering safety evidence",
        ),
        _dependency(
            dependency_id="dep-policy-validity",
            kind=PlanningDependencyKind.NEW_EVIDENCE_TO_POLICY_VALIDITY,
            source="new_evidence",
            target="dietary_evidence_review",
            description="New evidence affects policy validity",
        ),
    )
    decisions = (
        _decision(
            decision_id="catering_safety_conclusion",
            category=PlanningDecisionCategory.DIETARY,
            summary="Catering remains safe for the original dietary profile",
            dependency_ids=("dep-dietary-evidence",),
            resource_ids=("caterer-family-table",),
        ),
        _decision(
            decision_id="dietary_evidence_review",
            category=PlanningDecisionCategory.REVIEW,
            summary="Dietary evidence review is current",
            dependency_ids=("dep-policy-validity",),
            resource_ids=("caterer-family-table",),
        ),
        _decision(
            decision_id="venue_choice",
            category=PlanningDecisionCategory.RESOURCE_SELECTION,
            summary="Venue choice is independent of the new allergy",
            resource_ids=("venue-brooklyn-loft",),
        ),
        _decision(
            decision_id="theme",
            category=PlanningDecisionCategory.PREFERENCE,
            summary="Theme remains intact",
        ),
        _decision(
            decision_id="entertainment",
            category=PlanningDecisionCategory.PREFERENCE,
            summary="Entertainment remains intact",
        ),
        _decision(
            decision_id="accessibility",
            category=PlanningDecisionCategory.ACCESSIBILITY,
            summary="Accessibility remains intact",
            resource_ids=("venue-brooklyn-loft",),
        ),
    )
    request = _request(
        guest_count=36,
        total_budget=Decimal("1850.00"),
        event_date=date(2027, 1, 16),
        allergies=("sesame",),
        dietary_restrictions=("vegan",),
        accessibility_needs=("wheelchair_accessible",),
    )
    state = _state(
        request=request,
        decisions=decisions,
        dependencies=dependencies,
        unresolved_uncertainties=("A severe sesame allergy has just been introduced.",),
        notes=("Safety update invalidation benchmark fixture.",),
    )
    update = PlanningUpdate(
        update_id="update-new-sesame-allergy",
        kind=PlanningUpdateKind.NEW_ALLERGY_ADDED,
        description="A severe sesame allergy is introduced after the draft plan",
        added_allergies=("sesame",),
    )
    return ReplanningBenchmarkScenario(
        scenario_id="cap-boundary-52-new-safety-constraint-after-planning",
        title="New safety constraint after planning",
        description=(
            "A newly introduced severe allergy should invalidate prior catering safety "
            "conclusions while preserving unrelated decisions."
        ),
        capability_tags=("replanning", "safety_update", "allergy", "cross_domain"),
        initial_state=state,
        updates=(update,),
        expected_invalidated_decision_ids=(
            "catering_safety_conclusion",
            "dietary_evidence_review",
        ),
        expected_preserved_decision_ids=(
            "venue_choice",
            "theme",
            "entertainment",
            "accessibility",
        ),
        notes=(
            "The new allergy should invalidate catering assumptions.",
            "Venue and preference decisions should remain stable.",
        ),
    )


def _cascading_failure_scenario() -> ReplanningBenchmarkScenario:
    dependencies = (
        _dependency(
            dependency_id="dep-schedule-setup",
            kind=PlanningDependencyKind.SCHEDULE_TO_SETUP_WINDOW,
            source="event_time",
            target="rain_contingency",
            description="Schedule changes affect the setup window",
        ),
        _dependency(
            dependency_id="dep-budget-total-cost",
            kind=PlanningDependencyKind.BUDGET_TO_TOTAL_COST,
            source="total_budget",
            target="budget_confirmation",
            description="Budget affects the final budget confirmation",
        ),
    )
    decisions = (
        _decision(
            decision_id="rain_contingency",
            category=PlanningDecisionCategory.SCHEDULE,
            summary="Rain contingency is active",
            dependency_ids=("dep-schedule-setup",),
            resource_ids=("venue-brooklyn-loft",),
        ),
        _decision(
            decision_id="indoor_move",
            category=PlanningDecisionCategory.SCHEDULE,
            summary="Indoor move is feasible",
            prerequisite_decision_ids=("rain_contingency",),
            resource_ids=("venue-brooklyn-loft",),
        ),
        _decision(
            decision_id="indoor_setup_space",
            category=PlanningDecisionCategory.RESOURCE_SELECTION,
            summary="Indoor setup space remains available",
            prerequisite_decision_ids=("indoor_move",),
            resource_ids=("venue-brooklyn-loft",),
        ),
        _decision(
            decision_id="staffing_adjustment",
            category=PlanningDecisionCategory.SCHEDULE,
            summary="Staffing can adjust to the indoor setup",
            prerequisite_decision_ids=("indoor_setup_space",),
        ),
        _decision(
            decision_id="cost_recalculation",
            category=PlanningDecisionCategory.BUDGET,
            summary="Updated staffing and setup cost remains feasible",
            prerequisite_decision_ids=("staffing_adjustment",),
            dependency_ids=("dep-budget-total-cost",),
        ),
        _decision(
            decision_id="budget_confirmation",
            category=PlanningDecisionCategory.BUDGET,
            summary="Budget remains valid after the cascading changes",
            prerequisite_decision_ids=("cost_recalculation",),
            dependency_ids=("dep-budget-total-cost",),
        ),
        _decision(
            decision_id="theme",
            category=PlanningDecisionCategory.PREFERENCE,
            summary="Theme preference is preserved",
        ),
        _decision(
            decision_id="dietary_policy",
            category=PlanningDecisionCategory.DIETARY,
            summary="Dietary policy is preserved",
        ),
        _decision(
            decision_id="accessibility",
            category=PlanningDecisionCategory.ACCESSIBILITY,
            summary="Accessibility remains intact",
            resource_ids=("venue-brooklyn-loft",),
        ),
    )
    request = _request(
        guest_count=58,
        total_budget=Decimal("1750.00"),
        event_date=date(2027, 1, 19),
        event_time=time(18, 0),
        theme_preferences=("garden",),
        dietary_restrictions=("vegan",),
        accessibility_needs=("wheelchair_accessible",),
        other_constraints=("rain may force an indoor move",),
    )
    state = _state(
        request=request,
        decisions=decisions,
        dependencies=dependencies,
        unresolved_uncertainties=("Rain contingency has not been resolved yet.",),
        assumptions=("The event starts outdoors unless weather changes.",),
        notes=("Cascading schedule and budget dependency benchmark fixture.",),
    )
    update = PlanningUpdate(
        update_id="update-rain-contingency",
        kind=PlanningUpdateKind.DATE_TIME_CHANGED,
        description="Rain forces an indoor move and later schedule adjustment",
        event_date=date(2027, 1, 19),
        event_time=time(19, 0),
    )
    return ReplanningBenchmarkScenario(
        scenario_id="cap-boundary-55-cascading-failure",
        title="Cascading failure after a rain-triggered schedule change",
        description=(
            "A schedule change should invalidate the rain contingency and propagate through "
            "the downstream setup, staffing, cost, and budget decisions."
        ),
        capability_tags=("replanning", "cascading_failure", "temporal_dependency", "cross_domain"),
        initial_state=state,
        updates=(update,),
        expected_invalidated_decision_ids=(
            "rain_contingency",
            "indoor_move",
            "indoor_setup_space",
            "staffing_adjustment",
            "cost_recalculation",
            "budget_confirmation",
        ),
        expected_preserved_decision_ids=(
            "theme",
            "dietary_policy",
            "accessibility",
        ),
        notes=(
            "Invalidation must propagate through prerequisite edges.",
            "Budget should fail only after the chain reaches cost recalculation.",
        ),
    )


def _no_op_control_scenario() -> ReplanningBenchmarkScenario:
    dependencies = (
        _dependency(
            dependency_id="dep-guest-capacity",
            kind=PlanningDependencyKind.GUEST_COUNT_TO_VENUE_CAPACITY,
            source="guest_count",
            target="venue_capacity",
            description="Guest count affects venue capacity",
        ),
    )
    decisions = (
        _decision(
            decision_id="venue_capacity",
            category=PlanningDecisionCategory.RESOURCE_SELECTION,
            summary="Venue capacity remains sufficient",
            dependency_ids=("dep-guest-capacity",),
        ),
        _decision(
            decision_id="theme",
            category=PlanningDecisionCategory.PREFERENCE,
            summary="Theme preference is stable",
        ),
        _decision(
            decision_id="dietary",
            category=PlanningDecisionCategory.DIETARY,
            summary="Dietary constraints are stable",
        ),
    )
    request = _request(
        guest_count=40,
        total_budget=Decimal("1200.00"),
        event_date=date(2027, 1, 21),
        theme_preferences=("playful",),
    )
    state = _state(
        request=request,
        decisions=decisions,
        dependencies=dependencies,
        notes=("No-op update control fixture.",),
    )
    update = PlanningUpdate(
        update_id="update-no-op",
        kind=PlanningUpdateKind.NO_OP,
        description="No change to the plan",
    )
    return ReplanningBenchmarkScenario(
        scenario_id="v0-3-control-no-op-update",
        title="No-op update control",
        description="A no-op update should preserve the full plan.",
        capability_tags=("control", "no_op", "replanning"),
        initial_state=state,
        updates=(update,),
        expected_invalidated_decision_ids=(),
        expected_preserved_decision_ids=("venue_capacity", "theme", "dietary"),
        notes=("Targeted replanning should recompute nothing.",),
    )


def _broad_update_control_scenario() -> ReplanningBenchmarkScenario:
    dependencies = (
        _dependency(
            dependency_id="dep-schedule-availability",
            kind=PlanningDependencyKind.SCHEDULE_TO_VENDOR_AVAILABILITY,
            source="event_time",
            target="vendor_availability",
            description="Schedule affects vendor availability",
        ),
        _dependency(
            dependency_id="dep-schedule-setup",
            kind=PlanningDependencyKind.SCHEDULE_TO_SETUP_WINDOW,
            source="event_time",
            target="setup_window",
            description="Schedule affects the setup window",
        ),
        _dependency(
            dependency_id="dep-budget-total-cost",
            kind=PlanningDependencyKind.BUDGET_TO_TOTAL_COST,
            source="total_budget",
            target="budget",
            description="Budget affects total cost",
        ),
    )
    decisions = (
        _decision(
            decision_id="venue_availability",
            category=PlanningDecisionCategory.SCHEDULE,
            summary="Venue availability depends on the changed schedule",
            dependency_ids=("dep-schedule-availability",),
            resource_ids=("venue-brooklyn-loft",),
        ),
        _decision(
            decision_id="vendor_availability",
            category=PlanningDecisionCategory.SCHEDULE,
            summary="Vendor availability depends on the changed schedule",
            dependency_ids=("dep-schedule-availability",),
            resource_ids=("caterer-family-table",),
        ),
        _decision(
            decision_id="setup_window",
            category=PlanningDecisionCategory.SCHEDULE,
            summary="Setup window depends on the changed schedule",
            dependency_ids=("dep-schedule-setup",),
            resource_ids=("venue-brooklyn-loft",),
        ),
        _decision(
            decision_id="parking",
            category=PlanningDecisionCategory.RESOURCE_SELECTION,
            summary="Parking depends on the changed schedule",
            dependency_ids=("dep-schedule-availability",),
            resource_ids=("venue-brooklyn-loft",),
        ),
        _decision(
            decision_id="budget",
            category=PlanningDecisionCategory.BUDGET,
            summary="Budget depends on the changed schedule and setup",
            dependency_ids=("dep-schedule-setup",),
        ),
        _decision(
            decision_id="accessibility",
            category=PlanningDecisionCategory.ACCESSIBILITY,
            summary="Accessibility remains structurally stable",
            resource_ids=("venue-brooklyn-loft",),
        ),
        _decision(
            decision_id="theme",
            category=PlanningDecisionCategory.PREFERENCE,
            summary="Theme preference remains stable",
        ),
    )
    request = _request(
        guest_count=72,
        total_budget=Decimal("2300.00"),
        event_date=date(2027, 2, 12),
        event_time=time(17, 30),
        theme_preferences=("elegant",),
        accessibility_needs=("wheelchair_accessible",),
        other_constraints=("event time has shifted",),
    )
    state = _state(
        request=request,
        decisions=decisions,
        dependencies=dependencies,
        assumptions=("The current schedule is valid.",),
        notes=("Broad update control fixture.",),
    )
    update = PlanningUpdate(
        update_id="update-time-shift",
        kind=PlanningUpdateKind.DATE_TIME_CHANGED,
        description="Event time shifts enough to affect most schedule-sensitive decisions",
        event_date=date(2027, 2, 12),
        event_time=time(19, 0),
    )
    return ReplanningBenchmarkScenario(
        scenario_id="v0-3-control-broad-update",
        title="Broad schedule update control",
        description=(
            "A broad schedule change should legitimately invalidate most schedule-sensitive "
            "decisions."
        ),
        capability_tags=("control", "broad_update", "replanning", "cross_domain"),
        initial_state=state,
        updates=(update,),
        expected_invalidated_decision_ids=(
            "venue_availability",
            "vendor_availability",
            "setup_window",
            "parking",
            "budget",
        ),
        expected_preserved_decision_ids=("accessibility", "theme"),
        notes=("This control should not artificially preserve decisions that depend on time.",),
    )


def _accuracy(actual: Sequence[str], expected: Sequence[str]) -> float:
    expected_set = set(expected)
    actual_set = set(actual)
    if not expected_set:
        return 1.0 if not actual_set else 0.0
    return len(actual_set & expected_set) / len(expected_set)


def _render_strategy_row(metrics: ReplanningStrategyMetrics) -> str:
    return (
        f"| `{metrics.strategy.value}` | {metrics.invalidation_accuracy:.3f} | "
        f"{metrics.preserved_decision_accuracy:.3f} | {metrics.final_state_correctness:.3f} | "
        f"{metrics.recomputed_decision_count} | {metrics.unnecessary_recomputation_count} | "
        f"{metrics.missed_recomputation_count} | {metrics.mean_latency_ms:.3f} | "
        f"{metrics.cycle_detected_count} |"
    )


def _render_scenario_strategy_row(metrics: ReplanningScenarioStrategyMetrics) -> str:
    return (
        f"| `{metrics.strategy.value}` | {metrics.invalidation_accuracy:.3f} | "
        f"{metrics.preserved_decision_accuracy:.3f} | {metrics.final_state_correctness:.3f} | "
        f"{metrics.recomputed_decision_count} | {metrics.unnecessary_recomputation_count} | "
        f"{metrics.missed_recomputation_count} | {metrics.latency_ms:.3f} | "
        f"{1 if metrics.cycle_detected else 0} |"
    )
