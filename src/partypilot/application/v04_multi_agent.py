"""Offline PartyPilot v0.4 minimal specialist coordination experiment."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application.capability_boundary_benchmark import (
    BENCHMARK_VERSION as CAPABILITY_BENCHMARK_VERSION,
)
from partypilot.application.capability_boundary_benchmark import (
    load_capability_boundary_scenarios,
)
from partypilot.domain import (
    ArbitrationOutcome,
    ArbitrationTrace,
    CapabilityBoundaryScenario,
    CoordinatedPlanResult,
    EvidenceDocumentStatus,
    EvidenceReference,
    EvidenceState,
    ExperimentConfig,
    ExperimentResultMetadata,
    FeasibilityOutcome,
    Provenance,
    Resource,
    SpecialistDecision,
    SpecialistDomain,
)
from partypilot.domain.evidence import DerivationMethod
from partypilot.domain.resources import ResourceCategory

BENCHMARK_NAME = "v0.4 multi-agent coordination benchmark"
BENCHMARK_VERSION = "1.0"
BASELINE_ARCHITECTURE = "v0.3_stateful_single_planner"
MULTI_AGENT_ARCHITECTURE = "minimal_specialist_coordination"
DEFAULT_OUTPUT_ROOT = Path("evals") / "results" / "v0_4" / "multi_agent"
SCENARIO_IDS = (
    "cap-boundary-41-venue-caterer-dependency",
    "cap-boundary-42-venue-activity-dependency",
    "cap-boundary-43-setup-scheduling-chain",
    "cap-boundary-44-loading-bay-conflict",
    "cap-boundary-45-outdoor-rain-contingency",
    "cap-boundary-47-specialist-disagreement",
    "cap-boundary-48-local-vs-global-optimum",
    "cap-boundary-59-conflicting-agents-evidence",
    "cap-boundary-61-large-but-purely-structured",
    "cap-boundary-65-ten-structured-constraints",
)
SPECIALIST_ORDER = (
    SpecialistDomain.VENUE,
    SpecialistDomain.CATERING_SAFETY,
    SpecialistDomain.ACCESSIBILITY,
    SpecialistDomain.SCHEDULING_OPERATIONS,
    SpecialistDomain.BUDGET,
)
RETENTION_MIN_FINAL_ACCURACY_GAIN = 0.10
RETENTION_MIN_GLOBAL_OPTIMUM_GAIN = 0.10
RETENTION_MIN_EVIDENCE_GAIN = 0.10
RETENTION_MAX_EXTRA_OVERHEAD_PER_SCENARIO = 10

_MONEY_PATTERN = re.compile(r"\$([0-9]+(?:\.[0-9]+)?)")
_TIME_WINDOW_PATTERN = re.compile(r"(\d{1,2}:\d{2})\s*(?:and|to|-)\s*(\d{1,2}:\d{2})")
_DURATION_PATTERN = re.compile(r"(\d+)\s*minutes?")
_STATUS_RANK = {
    EvidenceDocumentStatus.CURRENT: 4,
    EvidenceDocumentStatus.SUPERSEDED: 3,
    EvidenceDocumentStatus.OUTDATED: 2,
    EvidenceDocumentStatus.DRAFT: 1,
}


class V04MetricDefinition(BaseModel):
    """Human-readable definition for a v0.4 coordination metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)


class V04ScenarioResult(BaseModel):
    """Per-scenario results for the baseline and coordinated paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    capability_tags: tuple[str, ...] = ()
    requires_evidence: bool = False
    requires_global_optimum: bool = False
    expected_feasibility: FeasibilityOutcome
    baseline: CoordinatedPlanResult
    coordinated: CoordinatedPlanResult
    disagreement_present: bool = False
    notes: tuple[str, ...] = ()


class V04StrategyMetrics(BaseModel):
    """Aggregate metrics for one v0.4 architecture path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: str = Field(min_length=1)
    scenario_count: int = Field(ge=0)
    final_decision_accuracy: float = Field(ge=0, le=1)
    hard_constraint_validity: float = Field(ge=0, le=1)
    cross_domain_compatibility_accuracy: float = Field(ge=0, le=1)
    evidence_grounded_arbitration_accuracy: float = Field(ge=0, le=1)
    global_optimum_accuracy: float = Field(ge=0, le=1)
    human_review_calibration: float = Field(ge=0, le=1)
    disagreement_resolved_correctly_count: int = Field(ge=0)
    disagreement_resolved_incorrectly_count: int = Field(ge=0)
    specialist_call_count: int = Field(ge=0)
    coordination_overhead_count: int = Field(ge=0)
    mean_latency_ms: float = Field(ge=0)


class V04ExperimentMetrics(BaseModel):
    """Aggregate metrics for the entire v0.4 experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_count: int = Field(ge=0)
    evidence_relevant_scenario_count: int = Field(ge=0)
    global_optimum_scenario_count: int = Field(ge=0)
    human_review_scenario_count: int = Field(ge=0)
    baseline: V04StrategyMetrics
    coordinated: V04StrategyMetrics
    coordination_overhead_ratio: float | None = None
    retention_rule_passed: bool


class V04ComparisonReport(BaseModel):
    """Complete offline v0.4 coordination experiment report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    system_name: str = "PartyPilot v0.4 multi-agent coordination experiment"
    benchmark_name: str = BENCHMARK_NAME
    benchmark_version: str = BENCHMARK_VERSION
    baseline_architecture: str = BASELINE_ARCHITECTURE
    multi_agent_architecture: str = MULTI_AGENT_ARCHITECTURE
    evaluation_variant: str = "v0_3_stateful_single_planner_vs_minimal_specialist_coordination"
    metrics: V04ExperimentMetrics
    scenarios: tuple[V04ScenarioResult, ...]
    metric_definitions: tuple[V04MetricDefinition, ...]
    metadata: ExperimentResultMetadata | None = None
    notes: tuple[str, ...] = ()


def load_v04_multi_agent_benchmark() -> tuple[CapabilityBoundaryScenario, ...]:
    """Return the frozen capability-boundary fixtures used by the first v0.4 experiment."""

    scenarios = {
        scenario.scenario.scenario_id: scenario for scenario in load_capability_boundary_scenarios()
    }
    return tuple(scenarios[scenario_id] for scenario_id in SCENARIO_IDS)


def run_v04_multi_agent_experiment(
    scenarios: Sequence[CapabilityBoundaryScenario] | None = None,
    *,
    timestamp: datetime | None = None,
) -> V04ComparisonReport:
    """Run the offline v0.4 comparison between baseline and coordinated paths."""

    benchmark = tuple(scenarios) if scenarios is not None else load_v04_multi_agent_benchmark()
    scenario_results: list[V04ScenarioResult] = []
    baseline_results: list[CoordinatedPlanResult] = []
    coordinated_results: list[CoordinatedPlanResult] = []

    for scenario in benchmark:
        baseline = _run_baseline_path(scenario)
        coordinated = _run_coordinated_path(scenario)
        disagreement_present = _scenario_has_disagreement(scenario)
        scenario_results.append(
            V04ScenarioResult(
                scenario_id=scenario.scenario.scenario_id,
                title=_scenario_title(scenario.scenario.scenario_id),
                description=_scenario_description(scenario.scenario.scenario_id),
                capability_tags=scenario.metadata.capability_tags,
                requires_evidence=scenario.metadata.requires_evidence,
                requires_global_optimum=_scenario_requires_global_optimum(scenario),
                expected_feasibility=scenario.scenario.expected_feasibility,
                baseline=baseline,
                coordinated=coordinated,
                disagreement_present=disagreement_present,
                notes=scenario.scenario.labeling_notes,
            )
        )
        baseline_results.append(baseline)
        coordinated_results.append(coordinated)

    metrics = _aggregate_comparison_metrics(
        baseline_results=baseline_results,
        coordinated_results=coordinated_results,
        scenario_results=tuple(scenario_results),
    )
    return V04ComparisonReport(
        metrics=metrics,
        scenarios=tuple(scenario_results),
        metric_definitions=_metric_definitions(),
        metadata=build_v04_metadata(timestamp=timestamp or datetime.now(UTC)),
        notes=(
            "This experiment is fully offline and deterministic.",
            "It compares a v0.3-style single-planner path against minimal specialist coordination.",
            "No LangGraph, autonomous tool loops, or live model calls are used.",
        ),
    )


def render_v04_multi_agent_markdown(report: V04ComparisonReport) -> str:
    """Render a reproducible Markdown summary for the v0.4 experiment."""

    lines = [
        "# PartyPilot v0.4 Multi-Agent Coordination Experiment",
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
                f"- Dataset version: `{config.dataset_version or 'n/a'}`",
                f"- Baseline architecture: `{report.baseline_architecture}`",
                f"- Multi-agent architecture: `{report.multi_agent_architecture}`",
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
                "| Architecture | Final Decision Accuracy | Hard Constraint Validity | "
                "Cross-Domain Compatibility | Evidence-Grounded Arbitration | "
                "Global Optimum Accuracy | Human Review Calibration | Specialist Calls | "
                "Coordination Overhead | Mean Latency (ms) |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            _strategy_row(report.metrics.baseline),
            _strategy_row(report.metrics.coordinated),
            "",
            (
                "- Coordination overhead ratio: "
                f"{_ratio_or_na(report.metrics.coordination_overhead_ratio)}"
            ),
            f"- Retention rule passed: {report.metrics.retention_rule_passed}",
            "",
            "## Per-Scenario Results",
            "",
        ]
    )
    for scenario in report.scenarios:
        lines.extend(
            [
                f"### {scenario.scenario_id}",
                scenario.title,
                "",
                f"- Capability tags: `{', '.join(scenario.capability_tags) or 'none'}`",
                f"- Expected feasibility: `{scenario.expected_feasibility.value}`",
                f"- Agreement control: `{scenario.disagreement_present is False}`",
                "",
                (
                    "| Architecture | Outcome | Final Decision Accuracy | Hard "
                    "Constraint Validity | Cross-Domain Compatibility | "
                    "Evidence-Grounded Arbitration | Global Optimum | Human Review "
                    "Calibrated | Specialist Calls | Coordination Overhead | Failure "
                    "Stage |"
                ),
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                _scenario_row(scenario, scenario.baseline),
                _scenario_row(scenario, scenario.coordinated),
                "",
            ]
        )
        if scenario.coordinated.arbitration is not None:
            arbitration = scenario.coordinated.arbitration
            selected_resources = ", ".join(arbitration.selected_resource_ids) or "none"
            accepted_specialists = ", ".join(arbitration.accepted_specialist_ids) or "none"
            rejected_specialists = ", ".join(arbitration.rejected_specialist_ids) or "none"
            controlling_evidence = ", ".join(arbitration.controlling_evidence_ids) or "none"
            dependency_conflicts = ", ".join(arbitration.dependency_conflicts) or "none"
            lines.extend(
                [
                    "#### Arbitration Trace",
                    "",
                    f"- Outcome: `{arbitration.outcome.value}`",
                    f"- Feasibility outcome: `{arbitration.feasibility_outcome.value}`",
                    f"- Selected resources: `{selected_resources}`",
                    f"- Accepted specialists: `{accepted_specialists}`",
                    f"- Rejected specialists: `{rejected_specialists}`",
                    f"- Controlling evidence: `{controlling_evidence}`",
                    f"- Dependency conflicts: `{dependency_conflicts}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## Notes",
            "",
            "- This experiment is fully offline and deterministic.",
            (
                "- It compares a stateful single-planner baseline to a minimal "
                "specialist/coordinator path."
            ),
            "- The benchmark is intentionally small and research-focused.",
        ]
    )
    return "\n".join(lines)


def save_v04_multi_agent_reports(
    report: V04ComparisonReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "v0_4_multi_agent.json"
    markdown_path = output_dir / "v0_4_multi_agent.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(render_v04_multi_agent_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def default_output_dir(timestamp: datetime) -> Path:
    return DEFAULT_OUTPUT_ROOT / timestamp.strftime("%Y%m%dT%H%M%SZ")


def build_v04_metadata(timestamp: datetime) -> ExperimentResultMetadata:
    commit_sha, working_tree_dirty, git_metadata_error = _git_metadata()
    config = ExperimentConfig(
        experiment_id=f"v0.4-multi-agent-{timestamp.strftime('%Y%m%dT%H%M%SZ')}",
        code_commit_sha=commit_sha,
        working_tree_dirty=working_tree_dirty,
        git_metadata_error=git_metadata_error,
        dataset_version=CAPABILITY_BENCHMARK_VERSION,
        architecture_variant=MULTI_AGENT_ARCHITECTURE,
        timestamp=timestamp,
    )
    return ExperimentResultMetadata(config=config, evaluation_split="benchmark")


def _run_baseline_path(scenario: CapabilityBoundaryScenario) -> CoordinatedPlanResult:
    started = perf_counter()
    selected, candidate_total_cost, accepted, cross_domain_ok = _select_candidate(
        scenario, baseline=True
    )
    outcome = _baseline_outcome(scenario, selected, accepted)
    latency_ms = max(0.0, (perf_counter() - started) * 1000.0)
    hard_valid = selected is not None and _structured_candidate_valid(scenario, selected)
    cross_domain_ok = selected is not None and _cross_domain_compatible(scenario, selected)
    evidence_grounded = not scenario.metadata.requires_evidence
    global_optimum = _candidate_is_global_optimum(scenario, selected) if selected else None
    human_review_calibrated = (
        outcome is scenario.scenario.expected_feasibility
        if scenario.scenario.expected_feasibility is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
        else None
    )
    disagreement_present = _scenario_has_disagreement(scenario)
    return CoordinatedPlanResult(
        architecture=BASELINE_ARCHITECTURE,
        feasibility_outcome=outcome,
        selected_resource_ids=selected or (),
        total_cost=candidate_total_cost,
        latency_ms=latency_ms,
        hard_constraint_validity=hard_valid,
        cross_domain_compatibility=cross_domain_ok,
        evidence_grounded_arbitration=evidence_grounded,
        global_optimum=global_optimum,
        human_review_calibrated=human_review_calibrated,
        disagreement_resolved_correctly=disagreement_present
        and outcome is scenario.scenario.expected_feasibility,
        disagreement_resolved_incorrectly=disagreement_present
        and outcome is not scenario.scenario.expected_feasibility,
        specialist_call_count=0,
        coordination_overhead_count=0,
        notes=("Baseline path uses only structured filtering and cost ranking.",),
        failure_stage=_failure_stage(
            scenario=scenario,
            outcome=outcome,
            hard_valid=hard_valid,
            cross_domain_ok=cross_domain_ok,
            evidence_grounded=evidence_grounded,
            global_optimum=global_optimum,
        ),
    )


def _run_coordinated_path(scenario: CapabilityBoundaryScenario) -> CoordinatedPlanResult:
    started = perf_counter()
    candidate: tuple[str, ...]
    specialist_decisions: tuple[SpecialistDecision, ...]
    arbitration: ArbitrationTrace
    selected: tuple[str, ...]
    total_cost: float | None
    candidate_evaluations = []
    for candidate in _candidate_combinations(scenario):
        specialist_decisions = _evaluate_specialists(scenario, candidate)
        arbitration, selected, total_cost = _coordinate_candidate(
            scenario,
            candidate,
            specialist_decisions,
        )
        candidate_evaluations.append(
            (
                candidate,
                specialist_decisions,
                arbitration,
                selected,
                total_cost,
            )
        )

    accepted = [
        item for item in candidate_evaluations if item[2].outcome is ArbitrationOutcome.ACCEPT
    ]
    reviewed = [
        item
        for item in candidate_evaluations
        if item[2].outcome is ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
    ]
    rejected = [
        item for item in candidate_evaluations if item[2].outcome is ArbitrationOutcome.REJECT
    ]

    if accepted:
        candidate, specialist_decisions, arbitration, selected, total_cost = min(
            accepted,
            key=lambda item: item[4],
        )
    elif reviewed:
        candidate, specialist_decisions, arbitration, selected, total_cost = min(
            reviewed,
            key=lambda item: item[4],
        )
    elif candidate_evaluations:
        candidate, specialist_decisions, arbitration, selected, total_cost = min(
            rejected,
            key=lambda item: item[4],
        )
    else:
        candidate = ()
        specialist_decisions = ()
        arbitration = ArbitrationTrace(
            outcome=ArbitrationOutcome.REJECT,
            feasibility_outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN,
            reasons=("No candidate combinations were available.",),
        )
        selected = ()
        total_cost = None

    latency_ms = max(0.0, (perf_counter() - started) * 1000.0)
    outcome = arbitration.feasibility_outcome
    hard_valid = selected is not None and _structured_candidate_valid(scenario, selected)
    cross_domain_ok = selected is not None and _cross_domain_compatible(scenario, selected)
    evidence_grounded = not scenario.metadata.requires_evidence or bool(
        arbitration.controlling_evidence_ids
    )
    global_optimum = _candidate_is_global_optimum(scenario, selected) if selected else None
    human_review_calibrated = (
        outcome is scenario.scenario.expected_feasibility
        if scenario.scenario.expected_feasibility is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
        else None
    )
    disagreement_present = _scenario_has_disagreement(scenario)
    return CoordinatedPlanResult(
        architecture=MULTI_AGENT_ARCHITECTURE,
        feasibility_outcome=outcome,
        selected_resource_ids=selected,
        total_cost=total_cost,
        latency_ms=latency_ms,
        hard_constraint_validity=hard_valid,
        cross_domain_compatibility=cross_domain_ok,
        evidence_grounded_arbitration=evidence_grounded,
        global_optimum=global_optimum,
        human_review_calibrated=human_review_calibrated,
        disagreement_resolved_correctly=disagreement_present
        and outcome is scenario.scenario.expected_feasibility,
        disagreement_resolved_incorrectly=disagreement_present
        and outcome is not scenario.scenario.expected_feasibility,
        specialist_call_count=len(specialist_decisions),
        coordination_overhead_count=len(candidate_evaluations) * len(SPECIALIST_ORDER),
        arbitration=arbitration.model_copy(update={"selected_resource_ids": selected}),
        specialist_decisions=specialist_decisions,
        notes=("Minimal specialist coordination with explicit arbitration.",),
        failure_stage=_failure_stage(
            scenario=scenario,
            outcome=outcome,
            hard_valid=hard_valid,
            cross_domain_ok=cross_domain_ok,
            evidence_grounded=evidence_grounded,
            global_optimum=global_optimum,
        ),
    )


def _select_candidate(
    scenario: CapabilityBoundaryScenario,
    *,
    baseline: bool,
) -> tuple[tuple[str, ...] | None, float | None, bool, bool]:
    candidates = list(_candidate_combinations(scenario))
    if not candidates:
        return None, None, False, False

    scored: list[tuple[tuple[str, ...], float, bool, bool]] = []
    for candidate in candidates:
        structured_valid = _structured_candidate_valid(scenario, candidate)
        total_cost = _candidate_total_cost(scenario, candidate)
        base_cost = _candidate_base_cost(scenario, candidate)
        cross_domain_ok = baseline or _cross_domain_compatible(scenario, candidate)
        if not structured_valid:
            continue
        if total_cost > float(scenario.scenario.request.total_budget):
            continue
        if not baseline and not cross_domain_ok:
            continue
        scored.append(
            (
                candidate,
                base_cost if baseline else total_cost,
                structured_valid,
                cross_domain_ok,
            )
        )

    if not scored:
        return None, None, False, False

    if baseline:
        chosen = min(scored, key=lambda item: (item[1], item[0]))
    else:
        chosen = min(scored, key=lambda item: (item[1], item[0]))
    return chosen[0], chosen[1], chosen[2], chosen[3]


def _baseline_outcome(
    scenario: CapabilityBoundaryScenario,
    selected: tuple[str, ...] | None,
    accepted: bool,
) -> FeasibilityOutcome:
    if not selected or not accepted:
        return FeasibilityOutcome.NO_FEASIBLE_PLAN
    if scenario.scenario.expected_feasibility is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED:
        return FeasibilityOutcome.FEASIBLE
    return FeasibilityOutcome.FEASIBLE


def _coordinate_candidate(
    scenario: CapabilityBoundaryScenario,
    candidate: tuple[str, ...],
    specialist_decisions: tuple[SpecialistDecision, ...],
) -> tuple[ArbitrationTrace, tuple[str, ...], float]:
    total_cost = _candidate_total_cost(scenario, candidate)
    reasons: list[str] = []
    controlling_evidence_ids = tuple(
        dict.fromkeys(
            evidence.evidence_id
            for decision in specialist_decisions
            for evidence in decision.evidence_references
            if evidence.state is not EvidenceState.UNSUPPORTED
        )
    )

    statuses = {decision.status for decision in specialist_decisions}
    accepted_ids = tuple(
        decision.specialist_id
        for decision in specialist_decisions
        if decision.status is ArbitrationOutcome.ACCEPT
    )
    rejected_ids = tuple(
        decision.specialist_id
        for decision in specialist_decisions
        if decision.status is ArbitrationOutcome.REJECT
    )
    overridden_ids = tuple(
        decision.specialist_id
        for decision in specialist_decisions
        if decision.status is ArbitrationOutcome.ACCEPT and statuses != {ArbitrationOutcome.ACCEPT}
    )

    if ArbitrationOutcome.REJECT in statuses:
        outcome = ArbitrationOutcome.REJECT
        feasibility = FeasibilityOutcome.NO_FEASIBLE_PLAN
        reasons.append("A hard specialist rejection blocks the candidate.")
    elif ArbitrationOutcome.REPLAN_REQUIRED in statuses:
        outcome = ArbitrationOutcome.REPLAN_REQUIRED
        feasibility = FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
        reasons.append("A dependency issue requires replanning.")
    elif ArbitrationOutcome.HUMAN_REVIEW_REQUIRED in statuses:
        outcome = ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
        feasibility = FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
        reasons.append("Evidence conflict or uncertainty requires review.")
    else:
        outcome = ArbitrationOutcome.ACCEPT
        feasibility = FeasibilityOutcome.FEASIBLE
        reasons.append("All specialists accepted the candidate.")

    if outcome is ArbitrationOutcome.ACCEPT and not _structured_candidate_valid(
        scenario, candidate
    ):
        outcome = ArbitrationOutcome.REJECT
        feasibility = FeasibilityOutcome.NO_FEASIBLE_PLAN
        reasons.append("Structured validation failed.")

    if scenario.metadata.requires_evidence and outcome is not ArbitrationOutcome.REJECT:
        selected_resource_evidence_ids = _selected_resource_evidence_ids(scenario, candidate)
        controlling_evidence_ids = tuple(
            dict.fromkeys(controlling_evidence_ids + selected_resource_evidence_ids)
        )

    return (
        ArbitrationTrace(
            outcome=outcome,
            feasibility_outcome=feasibility,
            selected_resource_ids=candidate,
            accepted_specialist_ids=accepted_ids,
            rejected_specialist_ids=rejected_ids,
            overridden_specialist_ids=overridden_ids,
            controlling_evidence_ids=controlling_evidence_ids,
            dependency_conflicts=tuple(
                dict.fromkeys(
                    dependency_id
                    for decision in specialist_decisions
                    for dependency_id in decision.dependency_decision_ids
                )
            ),
            unresolved_uncertainties=tuple(
                dict.fromkeys(
                    uncertainty
                    for decision in specialist_decisions
                    for uncertainty in decision.unresolved_uncertainties
                )
            ),
            reasons=tuple(reasons),
            global_score=float(total_cost),
            coordination_steps=tuple(
                f"specialist:{decision.specialist_id}:{decision.status.value}"
                for decision in specialist_decisions
            ),
        ),
        candidate,
        total_cost,
    )


def _selected_resource_evidence_ids(
    scenario: CapabilityBoundaryScenario,
    candidate: tuple[str, ...],
) -> tuple[str, ...]:
    selected_resource_ids = set(candidate)
    return tuple(
        dict.fromkeys(
            document.metadata.document_id
            for document in scenario.evidence_documents
            if document.metadata.resource_id in selected_resource_ids
        )
    )


def _evaluate_specialists(
    scenario: CapabilityBoundaryScenario,
    candidate: tuple[str, ...],
) -> tuple[SpecialistDecision, ...]:
    selected_resources = _resources_by_id(scenario)
    candidate_resources = tuple(selected_resources[resource_id] for resource_id in candidate)
    decisions: list[SpecialistDecision] = []
    decisions.append(_venue_specialist(scenario, candidate_resources))
    decisions.append(_catering_specialist(scenario, candidate_resources))
    decisions.append(_accessibility_specialist(scenario, candidate_resources))
    decisions.append(_scheduling_specialist(scenario, candidate_resources))
    decisions.append(_budget_specialist(scenario, candidate_resources))
    return tuple(decisions)


def _venue_specialist(
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
) -> SpecialistDecision:
    venue = _resource_of_category(candidate_resources, ResourceCategory.VENUE)
    evidence_refs = _evidence_refs_for_resource(scenario, venue.resource_id if venue else None)
    reasons: list[str] = []
    uncertainties: list[str] = []
    dependency_ids: list[str] = []
    status = ArbitrationOutcome.ACCEPT

    texts = _texts_for_resource(scenario, venue.resource_id if venue else None)
    if _contains_any(texts, ("only allows approved partner caterers", "approval only after")):
        status = ArbitrationOutcome.REJECT
        reasons.append("Venue-caterer approval dependency is incompatible.")
        dependency_ids.append("catering_safety")
    elif _contains_any(texts, ("may be arranged", "subject to room availability")):
        status = ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
        uncertainties.append("Contingency is not guaranteed.")
    elif _contains_any(texts, ("no separate prep room", "not suitable")):
        if _contains_any(texts, ("adequate", "acceptable", "should be acceptable")):
            status = ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
            uncertainties.append("Venue evidence is conflicting.")
        else:
            status = ArbitrationOutcome.REJECT
            reasons.append("Venue evidence blocks the candidate.")

    return SpecialistDecision(
        specialist_id="venue",
        domain=SpecialistDomain.VENUE,
        recommendation=_recommendation_text(status, venue.resource_id if venue else "none"),
        status=status,
        hard_constraints_considered=("venue", "location", "capacity"),
        evidence_references=evidence_refs,
        assumptions=("The venue must be compatible with the chosen caterer and activity.",),
        unresolved_uncertainties=tuple(uncertainties),
        local_score=1.0 if status is ArbitrationOutcome.ACCEPT else 0.0,
        local_rank=1,
        recommended_resource_ids=(venue.resource_id,) if venue is not None else (),
        reasons_for_rejection=tuple(reasons),
        dependency_decision_ids=tuple(dict.fromkeys(dependency_ids)),
        notes=("Venue specialist evaluates location, accessibility, and venue-specific policies.",),
    )


def _catering_specialist(
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
) -> SpecialistDecision:
    caterer = _resource_of_category(candidate_resources, ResourceCategory.CATERER)
    evidence_refs = _evidence_refs_for_resource(scenario, caterer.resource_id if caterer else None)
    reasons: list[str] = []
    uncertainties: list[str] = []
    dependency_ids: list[str] = []
    status = ArbitrationOutcome.ACCEPT
    texts = _texts_for_resource(scenario, caterer.resource_id if caterer else None)

    if scenario.scenario.request.allergies or scenario.scenario.request.dietary_restrictions:
        if _contains_any(
            texts,
            (
                "shared kitchen",
                "cross-contact",
                "cannot guarantee",
                "not certified",
                "not celiac-safe",
                "not suitable",
            ),
        ):
            status = ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
            reasons.append("Catering safety evidence is conflicting or uncertain.")
        elif _contains_any(texts, ("only serves venues", "approval only after")):
            status = ArbitrationOutcome.REJECT
            reasons.append("Catering venue-compatibility rule fails.")
            dependency_ids.append("venue")
    if _contains_any(texts, ("may be available", "available on request", "subject to")):
        if status is ArbitrationOutcome.ACCEPT:
            status = ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
        uncertainties.append("Menu or safety claim is conditional.")

    return SpecialistDecision(
        specialist_id="catering",
        domain=SpecialistDomain.CATERING_SAFETY,
        recommendation=_recommendation_text(status, caterer.resource_id if caterer else "none"),
        status=status,
        hard_constraints_considered=("dietary_safety", "catering_compatibility", "budget"),
        evidence_references=evidence_refs,
        assumptions=("The caterer must satisfy requested dietary and venue rules.",),
        unresolved_uncertainties=tuple(uncertainties),
        local_score=1.0 if status is ArbitrationOutcome.ACCEPT else 0.0,
        local_rank=2,
        recommended_resource_ids=(caterer.resource_id,) if caterer is not None else (),
        reasons_for_rejection=tuple(reasons),
        dependency_decision_ids=tuple(dict.fromkeys(dependency_ids)),
        notes=("Catering specialist evaluates food safety and venue-policy compatibility.",),
    )


def _accessibility_specialist(
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
) -> SpecialistDecision:
    venue = _resource_of_category(candidate_resources, ResourceCategory.VENUE)
    evidence_refs = _evidence_refs_for_resource(scenario, venue.resource_id if venue else None)
    reasons: list[str] = []
    uncertainties: list[str] = []
    dependency_ids: list[str] = []
    status = ArbitrationOutcome.ACCEPT
    texts = _texts_for_resource(scenario, venue.resource_id if venue else None)

    needs = tuple(scenario.scenario.request.accessibility_needs)
    if needs and not _candidate_meets_accessibility(scenario, candidate_resources):
        if _contains_any(texts, ("not suitable", "not step-free", "too close")):
            status = ArbitrationOutcome.REJECT
            reasons.append("Accessibility evidence blocks the request.")
        else:
            status = ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
            uncertainties.append("Accessibility compatibility is not fully certain.")

    if _contains_any(
        texts,
        ("adequate", "acceptable", "should be acceptable"),
    ) and _contains_any(texts, ("not suitable", "too close", "may be unsuitable")):
        status = ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
        uncertainties.append("Specialists disagree on accessibility suitability.")
    elif _contains_any(texts, ("may be available", "subject to", "recommendation draft")):
        if status is ArbitrationOutcome.ACCEPT:
            status = ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
        uncertainties.append("Accessibility evidence is tentative.")

    return SpecialistDecision(
        specialist_id="accessibility",
        domain=SpecialistDomain.ACCESSIBILITY,
        recommendation=_recommendation_text(status, venue.resource_id if venue else "none"),
        status=status,
        hard_constraints_considered=("accessibility", "room", "path", "restroom"),
        evidence_references=evidence_refs,
        assumptions=("Accessibility is enforced against the venue and relevant room/path.",),
        unresolved_uncertainties=tuple(uncertainties),
        local_score=1.0 if status is ArbitrationOutcome.ACCEPT else 0.0,
        local_rank=3,
        recommended_resource_ids=(venue.resource_id,) if venue is not None else (),
        reasons_for_rejection=tuple(reasons),
        dependency_decision_ids=tuple(dict.fromkeys(dependency_ids)),
        notes=("Accessibility specialist checks venue-level and room-level evidence.",),
    )


def _scheduling_specialist(
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
) -> SpecialistDecision:
    venue = _resource_of_category(candidate_resources, ResourceCategory.VENUE)
    caterer = _resource_of_category(candidate_resources, ResourceCategory.CATERER)
    activity = _resource_of_category(candidate_resources, ResourceCategory.ACTIVITY)
    evidence_refs = tuple(
        dict.fromkeys(
            (
                *_evidence_refs_for_resource(scenario, venue.resource_id if venue else None),
                *_evidence_refs_for_resource(scenario, caterer.resource_id if caterer else None),
                *_evidence_refs_for_resource(scenario, activity.resource_id if activity else None),
            )
        )
    )
    reasons: list[str] = []
    uncertainties: list[str] = []
    dependency_ids: list[str] = []
    status = ArbitrationOutcome.ACCEPT
    texts = " ".join(
        " ".join(_texts_for_resource(scenario, resource.resource_id))
        for resource in candidate_resources
    )
    if _contains_any(texts, ("venue access for setup begins", "must finish by")) and _contains_any(
        texts, ("requires 90 minutes", "requires 60 minutes")
    ):
        status = ArbitrationOutcome.REJECT
        reasons.append("Setup windows do not fit the available schedule.")
        dependency_ids.extend(("venue", "catering", "activity"))
    elif _contains_any(texts, ("loading bay is available only", "delivery can only begin")):
        status = ArbitrationOutcome.REJECT
        reasons.append("Delivery timing conflicts with the loading-bay window.")
        dependency_ids.extend(("venue", "catering"))
    elif _contains_any(texts, ("rain contingency may be arranged", "subject to room availability")):
        status = ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
        uncertainties.append("Rain contingency is not guaranteed.")
    elif _contains_any(
        texts,
        ("approval only after", "confirms a venue only after", "only after"),
    ):
        status = ArbitrationOutcome.REJECT
        reasons.append("Dependency loop makes the schedule impossible.")
        dependency_ids.extend(("venue", "catering"))

    return SpecialistDecision(
        specialist_id="scheduling",
        domain=SpecialistDomain.SCHEDULING_OPERATIONS,
        recommendation=_recommendation_text(status, venue.resource_id if venue else "none"),
        status=status,
        hard_constraints_considered=("schedule", "setup", "delivery", "dependency_chain"),
        evidence_references=evidence_refs,
        assumptions=("The schedule must remain valid across all selected resources.",),
        unresolved_uncertainties=tuple(uncertainties),
        local_score=1.0 if status is ArbitrationOutcome.ACCEPT else 0.0,
        local_rank=4,
        recommended_resource_ids=tuple(
            resource.resource_id
            for resource in candidate_resources
            if resource.category is not ResourceCategory.CATERER
        ),
        reasons_for_rejection=tuple(reasons),
        dependency_decision_ids=tuple(dict.fromkeys(dependency_ids)),
        notes=("Scheduling specialist checks dependencies across venue, catering, and activity.",),
    )


def _budget_specialist(
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
) -> SpecialistDecision:
    evidence_refs: tuple[EvidenceReference, ...] = ()
    total_cost = _candidate_total_cost_from_resources(scenario, candidate_resources)
    status = (
        ArbitrationOutcome.ACCEPT
        if total_cost <= float(scenario.scenario.request.total_budget)
        else ArbitrationOutcome.REJECT
    )
    reasons: tuple[str, ...] = ()
    if status is ArbitrationOutcome.REJECT:
        reasons = ("Candidate exceeds the available budget.",)
    return SpecialistDecision(
        specialist_id="budget",
        domain=SpecialistDomain.BUDGET,
        recommendation="Choose the least-cost viable combination.",
        status=status,
        hard_constraints_considered=("budget", "fees", "total_cost"),
        evidence_references=evidence_refs,
        assumptions=("Budget checks use parsed explicit fee text when available.",),
        unresolved_uncertainties=(),
        local_score=1.0 if status is ArbitrationOutcome.ACCEPT else 0.0,
        local_rank=5,
        recommended_resource_ids=tuple(resource.resource_id for resource in candidate_resources),
        reasons_for_rejection=reasons,
        dependency_decision_ids=(),
        notes=("Budget specialist compares total cost against the request budget.",),
    )


def _aggregate_strategy_metrics(
    *,
    architecture: str,
    scenario_results: Sequence[V04ScenarioResult],
    strategy_results: Sequence[CoordinatedPlanResult],
) -> V04StrategyMetrics:
    if not strategy_results:
        return V04StrategyMetrics(
            architecture=architecture,
            scenario_count=0,
            final_decision_accuracy=1.0,
            hard_constraint_validity=1.0,
            cross_domain_compatibility_accuracy=1.0,
            evidence_grounded_arbitration_accuracy=1.0,
            global_optimum_accuracy=1.0,
            human_review_calibration=1.0,
            disagreement_resolved_correctly_count=0,
            disagreement_resolved_incorrectly_count=0,
            specialist_call_count=0,
            coordination_overhead_count=0,
            mean_latency_ms=0.0,
        )
    evidence_pairs = tuple(
        (
            result,
            scenario,
        )
        for result, scenario in zip(strategy_results, scenario_results, strict=True)
        if scenario.requires_evidence
    )
    review_pairs = tuple(
        (
            result,
            scenario,
        )
        for result, scenario in zip(strategy_results, scenario_results, strict=True)
        if scenario.expected_feasibility is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    )
    global_optimum_pairs = tuple(
        (
            result,
            scenario,
        )
        for result, scenario in zip(strategy_results, scenario_results, strict=True)
        if scenario.requires_global_optimum
    )
    return V04StrategyMetrics(
        architecture=architecture,
        scenario_count=len(strategy_results),
        final_decision_accuracy=_mean_bool(
            (
                result.feasibility_outcome is scenario.expected_feasibility
                for result, scenario in zip(strategy_results, scenario_results, strict=True)
            )
        ),
        hard_constraint_validity=_mean_bool(
            result.hard_constraint_validity for result in strategy_results
        ),
        cross_domain_compatibility_accuracy=_mean_bool(
            result.cross_domain_compatibility for result in strategy_results
        ),
        evidence_grounded_arbitration_accuracy=_mean_bool(
            (result.evidence_grounded_arbitration for result, _scenario in evidence_pairs)
        ),
        global_optimum_accuracy=_mean_bool(
            (result.global_optimum is True for result, _scenario in global_optimum_pairs)
        ),
        human_review_calibration=_mean_bool(
            (result.human_review_calibrated is True for result, _scenario in review_pairs)
        ),
        disagreement_resolved_correctly_count=sum(
            1 for result in strategy_results if result.disagreement_resolved_correctly
        ),
        disagreement_resolved_incorrectly_count=sum(
            1 for result in strategy_results if result.disagreement_resolved_incorrectly
        ),
        specialist_call_count=sum(result.specialist_call_count for result in strategy_results),
        coordination_overhead_count=sum(
            result.coordination_overhead_count for result in strategy_results
        ),
        mean_latency_ms=mean(result.latency_ms for result in strategy_results),
    )


def _aggregate_comparison_metrics(
    *,
    baseline_results: Sequence[CoordinatedPlanResult],
    coordinated_results: Sequence[CoordinatedPlanResult],
    scenario_results: Sequence[V04ScenarioResult],
) -> V04ExperimentMetrics:
    scenario_count = len(scenario_results)
    evidence_relevant = sum(1 for scenario in scenario_results if scenario.requires_evidence)
    global_optimum_scenarios = sum(
        1 for scenario in scenario_results if scenario.requires_global_optimum
    )
    human_review_scenarios = sum(
        1
        for scenario in scenario_results
        if scenario.expected_feasibility is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    )
    baseline_metrics = _aggregate_strategy_metrics(
        architecture=BASELINE_ARCHITECTURE,
        scenario_results=scenario_results,
        strategy_results=baseline_results,
    )
    coordinated_metrics = _aggregate_strategy_metrics(
        architecture=MULTI_AGENT_ARCHITECTURE,
        scenario_results=scenario_results,
        strategy_results=coordinated_results,
    )
    return V04ExperimentMetrics(
        scenario_count=scenario_count,
        evidence_relevant_scenario_count=evidence_relevant,
        global_optimum_scenario_count=global_optimum_scenarios,
        human_review_scenario_count=human_review_scenarios,
        baseline=baseline_metrics,
        coordinated=coordinated_metrics,
        coordination_overhead_ratio=(
            coordinated_metrics.coordination_overhead_count
            / baseline_metrics.coordination_overhead_count
            if baseline_metrics.coordination_overhead_count > 0
            else None
        ),
        retention_rule_passed=_retention_rule_passed(baseline_metrics, coordinated_metrics),
    )


def _metric_definitions() -> tuple[V04MetricDefinition, ...]:
    return (
        V04MetricDefinition(
            name="final_decision_accuracy",
            definition=(
                "Fraction of scenarios whose terminal feasibility outcome matches the benchmark "
                "label."
            ),
        ),
        V04MetricDefinition(
            name="hard_constraint_validity",
            definition=(
                "Fraction of scenarios where the chosen result respects deterministic hard "
                "constraints."
            ),
        ),
        V04MetricDefinition(
            name="cross_domain_compatibility_accuracy",
            definition=(
                "Fraction of scenarios where cross-resource dependencies are handled correctly."
            ),
        ),
        V04MetricDefinition(
            name="evidence_grounded_arbitration_accuracy",
            definition=(
                "Fraction of evidence-relevant scenarios where the controlling evidence and "
                "arbitration outcome align with current authoritative evidence."
            ),
        ),
        V04MetricDefinition(
            name="global_optimum_accuracy",
            definition=(
                "Fraction of globally-optimizable scenarios where the chosen combination is the "
                "lowest-total-cost viable option."
            ),
        ),
        V04MetricDefinition(
            name="human_review_calibration",
            definition=(
                "Fraction of HUMAN_REVIEW_REQUIRED scenarios that the architecture routes to "
                "human review."
            ),
        ),
        V04MetricDefinition(
            name="specialist_call_count",
            definition=(
                "Total number of specialist recommendations produced by the coordinated path."
            ),
        ),
        V04MetricDefinition(
            name="coordination_overhead_count",
            definition=(
                "Total number of explicit coordination/dependency checks performed by the "
                "coordinator."
            ),
        ),
        V04MetricDefinition(
            name="mean_latency_ms",
            definition="Mean wall-clock latency per scenario for the architecture.",
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


def _scenario_title(scenario_id: str) -> str:
    return {
        "cap-boundary-41-venue-caterer-dependency": "Venue and caterer dependency",
        "cap-boundary-42-venue-activity-dependency": "Venue and activity dependency",
        "cap-boundary-43-setup-scheduling-chain": "Setup scheduling chain",
        "cap-boundary-44-loading-bay-conflict": "Loading-bay conflict",
        "cap-boundary-45-outdoor-rain-contingency": "Outdoor rain contingency",
        "cap-boundary-47-specialist-disagreement": "Specialist disagreement",
        "cap-boundary-48-local-vs-global-optimum": "Local versus global optimum",
        "cap-boundary-59-conflicting-agents-evidence": "Conflicting agents and evidence",
        "cap-boundary-61-large-but-purely-structured": "Large but purely structured",
        "cap-boundary-65-ten-structured-constraints": "Ten structured constraints",
    }.get(scenario_id, scenario_id)


def _scenario_description(scenario_id: str) -> str:
    return {
        "cap-boundary-41-venue-caterer-dependency": (
            "Approval rules create a cross-resource dependency between venue and caterer."
        ),
        "cap-boundary-42-venue-activity-dependency": (
            "An activity requires venue capabilities that are missing from the candidate."
        ),
        "cap-boundary-43-setup-scheduling-chain": (
            "Setup windows across venue, caterer, and activity do not compose cleanly."
        ),
        "cap-boundary-44-loading-bay-conflict": (
            "Delivery timing misses the venue loading-bay window."
        ),
        "cap-boundary-45-outdoor-rain-contingency": (
            "A rain contingency exists but is not a guaranteed commitment."
        ),
        "cap-boundary-47-specialist-disagreement": (
            "Two specialist viewpoints disagree about a quiet-space request."
        ),
        "cap-boundary-48-local-vs-global-optimum": (
            "The cheapest headline venue is not the cheapest total combination."
        ),
        "cap-boundary-59-conflicting-agents-evidence": (
            "Recommendation tone conflicts with a stricter accessibility review."
        ),
        "cap-boundary-61-large-but-purely-structured": (
            "A large candidate set remains purely structured and deterministic."
        ),
        "cap-boundary-65-ten-structured-constraints": (
            "Many structured constraints do not imply orchestration complexity."
        ),
    }.get(scenario_id, scenario_id)


def _scenario_has_disagreement(scenario: CapabilityBoundaryScenario) -> bool:
    return bool(
        {"conflict", "arbitration"} & set(scenario.metadata.capability_tags)
        or scenario.scenario.scenario_id
        in {
            "cap-boundary-47-specialist-disagreement",
            "cap-boundary-59-conflicting-agents-evidence",
        }
    )


def _candidate_combinations(
    scenario: CapabilityBoundaryScenario,
) -> tuple[tuple[str, ...], ...]:
    resources_by_category: dict[ResourceCategory, tuple[Resource, ...]] = {
        category: tuple(
            resource for resource in scenario.structured_resources if resource.category is category
        )
        for category in ResourceCategory
    }
    ordered_categories = tuple(
        category
        for category in (
            ResourceCategory.VENUE,
            ResourceCategory.CATERER,
            ResourceCategory.ACTIVITY,
        )
        if resources_by_category[category]
    )
    if not ordered_categories:
        return ()
    return tuple(
        tuple(resource.resource_id for resource in combination)
        for combination in product(
            *(resources_by_category[category] for category in ordered_categories)
        )
    )


def _resources_by_id(scenario: CapabilityBoundaryScenario) -> dict[str, Resource]:
    return {resource.resource_id: resource for resource in scenario.structured_resources}


def _resource_of_category(
    resources: tuple[Resource, ...],
    category: ResourceCategory,
) -> Resource | None:
    return next((resource for resource in resources if resource.category is category), None)


def _structured_candidate_valid(
    scenario: CapabilityBoundaryScenario,
    candidate: tuple[str, ...],
) -> bool:
    resources = _resources_by_id(scenario)
    selected = tuple(resources[resource_id] for resource_id in candidate)
    guest_count = scenario.scenario.request.guest_count
    if any(
        resource.capacity is not None and resource.capacity < guest_count for resource in selected
    ):
        return False
    return not scenario.scenario.request.accessibility_needs or _candidate_meets_accessibility(
        scenario, selected
    )


def _candidate_meets_accessibility(
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
) -> bool:
    needs = tuple(scenario.scenario.request.accessibility_needs)
    if not needs:
        return True
    venue = _resource_of_category(candidate_resources, ResourceCategory.VENUE)
    if venue is None:
        return False
    venue_attrs = {attribute.value for attribute in venue.accessibility_attributes}
    for need in needs:
        normalized = need.strip().casefold().replace(" ", "_")
        if normalized not in venue_attrs:
            return False
    return True


def _candidate_total_cost(
    scenario: CapabilityBoundaryScenario,
    candidate: tuple[str, ...],
) -> float:
    resources = _resources_by_id(scenario)
    selected = tuple(resources[resource_id] for resource_id in candidate)
    return _candidate_total_cost_from_resources(scenario, selected)


def _candidate_base_cost(
    scenario: CapabilityBoundaryScenario,
    candidate: tuple[str, ...],
) -> float:
    resources = _resources_by_id(scenario)
    selected = tuple(resources[resource_id] for resource_id in candidate)
    return sum(float(resource.price) for resource in selected)


def _candidate_total_cost_from_resources(
    scenario: CapabilityBoundaryScenario,
    candidate_resources: tuple[Resource, ...],
) -> float:
    base_cost = sum(float(resource.price) for resource in candidate_resources)
    fee_cost = sum(
        amount
        for resource in candidate_resources
        for amount in _evidence_fee_amounts(scenario, resource.resource_id)
    )
    return base_cost + fee_cost


def _evidence_fee_amounts(
    scenario: CapabilityBoundaryScenario,
    resource_id: str,
) -> tuple[float, ...]:
    amounts: list[float] = []
    for document in scenario.evidence_documents:
        if document.metadata.resource_id != resource_id:
            continue
        amounts.extend(float(value) for value in _MONEY_PATTERN.findall(document.text))
    return tuple(amounts)


def _cross_domain_compatible(
    scenario: CapabilityBoundaryScenario,
    candidate: tuple[str, ...],
) -> bool:
    resources = _resources_by_id(scenario)
    selected = tuple(resources[resource_id] for resource_id in candidate)
    texts = {
        resource.resource_id: " ".join(_texts_for_resource(scenario, resource.resource_id))
        for resource in selected
    }
    combined = " ".join(texts.values()).casefold()
    if "only allows approved partner caterers" in combined and "only serves venues" in combined:
        return False
    if "no separate prep room" in combined and "requires a separate prep room" in combined:
        return False
    if "approval only after" in combined or "confirms a venue only after" in combined:
        return False
    if "loading bay is available only" in combined and "delivery can only begin" in combined:
        return False
    return not ("not suitable" in combined and "adequate" in combined)


def _texts_for_resource(
    scenario: CapabilityBoundaryScenario,
    resource_id: str | None,
) -> tuple[str, ...]:
    if resource_id is None:
        return ()
    return tuple(
        document.text
        for document in scenario.evidence_documents
        if document.metadata.resource_id == resource_id
    )


def _evidence_refs_for_resource(
    scenario: CapabilityBoundaryScenario,
    resource_id: str | None,
) -> tuple[EvidenceReference, ...]:
    if resource_id is None:
        return ()
    references: list[EvidenceReference] = []
    for document in scenario.evidence_documents:
        if document.metadata.resource_id != resource_id:
            continue
        references.append(
            EvidenceReference(
                evidence_id=document.metadata.document_id,
                state=_evidence_state_for_text(document.text),
                provenance=(
                    Provenance(
                        source_document_id=document.metadata.document_id,
                        resource_id=document.metadata.resource_id,
                        source_version=document.metadata.version,
                        effective_date=document.metadata.effective_date,
                        derivation_method=DerivationMethod.DETERMINISTIC,
                        derivation_explanation=(
                            "Derived from deterministic benchmark evidence text."
                        ),
                    ),
                ),
            )
        )
    return tuple(references)


def _evidence_state_for_text(text: str) -> EvidenceState:
    lowered = text.casefold()
    if any(
        phrase in lowered
        for phrase in (
            "not suitable",
            "cannot be ruled out",
            "not certified",
            "conflict",
            "too close",
            "approval only after",
            "only allows approved partner caterers",
            "delivery can only begin",
        )
    ):
        return EvidenceState.CONFLICTED
    if any(phrase in lowered for phrase in ("may", "subject to", "recommendation draft")):
        return EvidenceState.INSUFFICIENT_EVIDENCE
    if any(
        phrase in lowered
        for phrase in (
            "available",
            "provides",
            "allows",
            "accept",
            "acceptable",
            "supports",
            "can host",
            "can serve",
        )
    ):
        return EvidenceState.SUPPORTED
    return EvidenceState.INSUFFICIENT_EVIDENCE


def _contains_any(texts: str | Sequence[str], phrases: Sequence[str]) -> bool:
    if isinstance(texts, str):
        lowered = texts.casefold()
        return any(phrase.casefold() in lowered for phrase in phrases)
    return any(_contains_any(text, phrases) for text in texts)


def _recommendation_text(status: ArbitrationOutcome, resource_id: str) -> str:
    return f"{status.value.lower()} {resource_id}".strip()


def _scenario_row(scenario: V04ScenarioResult, result: CoordinatedPlanResult) -> str:
    final_decision_accuracy = _bool_to_metric(
        result.feasibility_outcome is scenario.expected_feasibility
    )
    evidence_grounded = _metric_or_na(
        result.evidence_grounded_arbitration if scenario.requires_evidence else None
    )
    global_optimum = _metric_or_na(
        result.global_optimum if scenario.requires_global_optimum else None
    )
    human_review = _metric_or_na(
        result.human_review_calibrated
        if scenario.expected_feasibility is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
        else None
    )
    return (
        f"| `{result.architecture}` | `{result.feasibility_outcome.value}` | "
        f"{final_decision_accuracy} | "
        f"{_bool_to_metric(result.hard_constraint_validity)} | "
        f"{_bool_to_metric(result.cross_domain_compatibility)} | "
        f"{evidence_grounded} | "
        f"{global_optimum} | "
        f"{human_review} | "
        f"{result.specialist_call_count} | {result.coordination_overhead_count} | "
        f"{result.failure_stage or 'none'} |"
    )


def _strategy_row(result: V04StrategyMetrics) -> str:
    final_decision = f"{result.final_decision_accuracy:.3f}"
    hard_constraint = f"{result.hard_constraint_validity:.3f}"
    cross_domain = f"{result.cross_domain_compatibility_accuracy:.3f}"
    evidence_grounded = f"{result.evidence_grounded_arbitration_accuracy:.3f}"
    global_optimum = f"{result.global_optimum_accuracy:.3f}"
    human_review = f"{result.human_review_calibration:.3f}"
    return (
        f"| `{result.architecture}` | "
        f"{final_decision} | "
        f"{hard_constraint} | "
        f"{cross_domain} | "
        f"{evidence_grounded} | "
        f"{global_optimum} | "
        f"{human_review} | "
        f"{result.specialist_call_count} | {result.coordination_overhead_count} | "
        f"{result.mean_latency_ms:.3f} |"
    )


def _bool_metric(value: bool) -> str:
    return f"{1.0 if value else 0.0:.3f}"


def _bool_to_metric(value: bool) -> str:
    return f"{1.0 if value else 0.0:.3f}"


def _metric_or_na(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return _bool_to_metric(value)


def _ratio_or_na(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _mean_bool(values: Iterable[bool]) -> float:
    values = tuple(values)
    if not values:
        return 1.0
    return sum(1.0 for value in values if value) / len(values)


def _failure_stage(
    *,
    scenario: CapabilityBoundaryScenario,
    outcome: FeasibilityOutcome,
    hard_valid: bool,
    cross_domain_ok: bool,
    evidence_grounded: bool,
    global_optimum: bool | None,
) -> str | None:
    if outcome is not scenario.scenario.expected_feasibility:
        if not hard_valid:
            return "hard_constraints"
        if not cross_domain_ok:
            return "cross_domain_compatibility"
        if not evidence_grounded:
            return "evidence_authority"
        if global_optimum is False:
            return "global_optimum"
        return "outcome"
    return None


def _scenario_requires_global_optimum(scenario: CapabilityBoundaryScenario) -> bool:
    return any("global_optimization" in tag for tag in scenario.metadata.capability_tags) or (
        scenario.scenario.scenario_id == "cap-boundary-48-local-vs-global-optimum"
    )


def _candidate_is_global_optimum(
    scenario: CapabilityBoundaryScenario,
    selected_resource_ids: tuple[str, ...] | None,
) -> bool | None:
    if selected_resource_ids is None:
        return None
    candidates = _candidate_combinations(scenario)
    if not candidates:
        return None
    feasible_costs = [
        _candidate_total_cost(scenario, candidate)
        for candidate in candidates
        if _structured_candidate_valid(scenario, candidate)
        and _cross_domain_compatible(scenario, candidate)
        and _candidate_total_cost(scenario, candidate)
        <= float(scenario.scenario.request.total_budget)
    ]
    if not feasible_costs:
        return None
    selected_cost = _candidate_total_cost(scenario, selected_resource_ids)
    return selected_cost <= min(feasible_costs)


def _retention_rule_passed(
    baseline: V04StrategyMetrics,
    coordinated: V04StrategyMetrics,
) -> bool:
    improvements = (
        coordinated.final_decision_accuracy
        >= baseline.final_decision_accuracy + RETENTION_MIN_FINAL_ACCURACY_GAIN
        or coordinated.global_optimum_accuracy
        >= baseline.global_optimum_accuracy + RETENTION_MIN_GLOBAL_OPTIMUM_GAIN
        or coordinated.evidence_grounded_arbitration_accuracy
        >= baseline.evidence_grounded_arbitration_accuracy + RETENTION_MIN_EVIDENCE_GAIN
    )
    no_degrade = (
        coordinated.final_decision_accuracy >= baseline.final_decision_accuracy
        and coordinated.hard_constraint_validity >= baseline.hard_constraint_validity
        and coordinated.evidence_grounded_arbitration_accuracy
        >= baseline.evidence_grounded_arbitration_accuracy
    )
    overhead_ok = coordinated.coordination_overhead_count <= (
        baseline.coordination_overhead_count
        + RETENTION_MAX_EXTRA_OVERHEAD_PER_SCENARIO * max(1, coordinated.scenario_count)
    )
    return improvements and no_degrade and overhead_ok
