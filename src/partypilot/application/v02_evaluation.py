"""Objective v0.2 evaluation for the evidence-grounded planning flow."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from fractions import Fraction
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application.citation_validation import (
    CitationViolationCode,
    validate_citations,
)
from partypilot.application.evidence_grounded_planner import (
    EvidenceGroundedPlanCandidate,
    EvidenceGroundedPlanningResult,
)
from partypilot.domain.evaluation import EvaluationScenario
from partypilot.domain.evidence import EvidenceState
from partypilot.domain.evidence_corpus import EvidenceDocument
from partypilot.domain.experiment import ExperimentResultMetadata
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest


class EvidenceGroundedScenarioPlanner(Protocol):
    """Minimal planner surface required by the v0.2 evaluator."""

    def plan(self, request: PartyRequest) -> EvidenceGroundedPlanningResult: ...


class V02ScenarioEvaluation(BaseModel):
    """Measured evidence-grounded behavior for one benchmark scenario."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    expected_outcome: FeasibilityOutcome
    predicted_outcome: FeasibilityOutcome
    outcome_correct: bool
    hard_constraints_valid: bool
    grounded_decision_correct: bool | None = None
    expected_evidence_document_ids: tuple[str, ...] = ()
    attributed_evidence_document_ids: tuple[str, ...] = ()
    source_attributions_correct: int = Field(default=0, ge=0)
    source_attributions_expected: int = Field(default=0, ge=0)
    supported_claim_count: int = Field(default=0, ge=0)
    invalid_supported_claim_count: int = Field(default=0, ge=0)
    citation_count: int = Field(default=0, ge=0)
    wrong_source_or_version_count: int = Field(default=0, ge=0)
    derived_constraint_count: int = Field(default=0, ge=0)
    correct_derived_constraint_count: int = Field(default=0, ge=0)
    latency_ms: float = Field(ge=0)


class V02EvaluationMetrics(BaseModel):
    """Aggregate objective metrics for the evidence-grounded v0.2 system."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_count: int = Field(ge=0)
    feasibility_accuracy: float = Field(ge=0, le=1)
    hard_constraint_validity: float = Field(ge=0, le=1)
    grounded_decision_accuracy: float | None = Field(default=None, ge=0, le=1)
    source_attribution_accuracy: float | None = Field(default=None, ge=0, le=1)
    derived_constraint_accuracy: float | None = Field(default=None, ge=0, le=1)
    unsupported_claim_rate: float | None = Field(default=None, ge=0, le=1)
    wrong_source_version_rate: float | None = Field(default=None, ge=0, le=1)
    no_feasible_plan_accuracy: float | None = Field(default=None, ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class BaselineMetricsSnapshot(BaseModel):
    """Measured v0.1 baseline metrics available for comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    feasibility_accuracy: float = Field(ge=0, le=1)
    hard_constraint_validity: float = Field(ge=0, le=1)
    no_feasible_plan_accuracy: float | None = Field(default=None, ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)


class RetrievalMetricsSnapshot(BaseModel):
    """Separately reported retained-retrieval metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: str
    top_k: int = Field(gt=0)
    query_count: int = Field(ge=0)
    recall_at_k: float = Field(ge=0, le=1)
    precision_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    correct_policy_retrieval: float = Field(ge=0, le=1)
    correct_version_retrieval: float = Field(ge=0, le=1)
    wrong_vendor_retrieval_rate: float = Field(ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)


class V02EvaluationReport(BaseModel):
    """Complete v0.2 evaluation with planning and retrieval metrics kept separate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    system_name: str = "PartyPilot v0.2 evidence-grounded system"
    evaluation_variant: str
    metrics: V02EvaluationMetrics
    scenarios: tuple[V02ScenarioEvaluation, ...]
    metadata: ExperimentResultMetadata | None = None
    v01_baselines: tuple[BaselineMetricsSnapshot, ...] = ()
    retrieval_metrics: tuple[RetrievalMetricsSnapshot, ...] = ()
    notes: tuple[str, ...] = ()


class V02EvaluationRunner:
    """Measure evidence-grounded planning behavior against labeled scenarios."""

    def __init__(
        self,
        planner: EvidenceGroundedScenarioPlanner,
        *,
        corpus: tuple[EvidenceDocument, ...],
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._planner = planner
        self._corpus = corpus
        self._clock = clock

    def run(
        self, scenarios: Sequence[EvaluationScenario]
    ) -> tuple[V02EvaluationMetrics, tuple[V02ScenarioEvaluation, ...]]:
        measured: list[V02ScenarioEvaluation] = []
        for scenario in scenarios:
            started = self._clock()
            result = self._planner.plan(scenario.request)
            latency_ms = max(0.0, (self._clock() - started) * 1000.0)
            measured.append(self._evaluate_scenario(scenario, result, latency_ms))
        frozen = tuple(measured)
        return _aggregate_metrics(frozen), frozen

    def _evaluate_scenario(
        self,
        scenario: EvaluationScenario,
        result: EvidenceGroundedPlanningResult,
        latency_ms: float,
    ) -> V02ScenarioEvaluation:
        expected_docs = tuple(
            document_id
            for label in scenario.retrieval_ground_truth
            for document_id in label.expected_document_ids
        )
        attributed_docs = tuple(
            sorted(
                {
                    provenance.source_document_id
                    for reference in result.evidence_references
                    for provenance in reference.provenance
                    if provenance.source_document_id is not None
                }
            )
        )
        attributed_set = set(attributed_docs)
        correct_attributions = sum(doc_id in attributed_set for doc_id in expected_docs)

        derived = tuple(
            item for candidate in result.candidates for item in candidate.derived_constraints
        )
        citation_validation = validate_citations(
            corpus=self._corpus,
            evidence_references=result.evidence_references,
            derived_constraints=derived,
        )
        wrong_codes = {
            CitationViolationCode.NONEXISTENT_SOURCE,
            CitationViolationCode.WRONG_RESOURCE,
            CitationViolationCode.VERSION_MISMATCH,
            CitationViolationCode.OUTDATED_VERSION,
        }
        wrong_source_or_version = sum(
            violation.code in wrong_codes for violation in citation_validation.violations
        )
        invalid_supported_ids = {
            violation.evidence_id
            for violation in citation_validation.violations
            if violation.code is CitationViolationCode.UNSUPPORTED_AS_SUPPORTED
            and violation.evidence_id is not None
        }
        supported_claims = sum(
            reference.state is EvidenceState.SUPPORTED for reference in result.evidence_references
        )
        citation_count = sum(
            len(reference.provenance) for reference in result.evidence_references
        ) + sum(len(item.provenance) for item in derived)

        derived_total = 0
        derived_correct = 0
        for candidate in result.candidates:
            total, correct = _derived_accuracy_counts(candidate, scenario.request)
            derived_total += total
            derived_correct += correct

        hard_valid = all(candidate.validation.feasible for candidate in result.candidates)
        grounded = None
        if scenario.retrieval_ground_truth:
            grounded = result.outcome is scenario.expected_feasibility

        return V02ScenarioEvaluation(
            scenario_id=scenario.scenario_id,
            expected_outcome=scenario.expected_feasibility,
            predicted_outcome=result.outcome,
            outcome_correct=result.outcome is scenario.expected_feasibility,
            hard_constraints_valid=hard_valid,
            grounded_decision_correct=grounded,
            expected_evidence_document_ids=expected_docs,
            attributed_evidence_document_ids=attributed_docs,
            source_attributions_correct=correct_attributions,
            source_attributions_expected=len(expected_docs),
            supported_claim_count=supported_claims,
            invalid_supported_claim_count=len(invalid_supported_ids),
            citation_count=citation_count,
            wrong_source_or_version_count=wrong_source_or_version,
            derived_constraint_count=derived_total,
            correct_derived_constraint_count=derived_correct,
            latency_ms=latency_ms,
        )


def _derived_accuracy_counts(
    candidate: EvidenceGroundedPlanCandidate,
    request: PartyRequest,
) -> tuple[int, int]:
    if request.child_age is None and request.child_age_range is None:
        return 0, 0

    ratios: list[Fraction] = []
    for extracted in candidate.extracted_constraints:
        constraint = extracted.constraint
        if constraint.key == "adult_child_ratio" and isinstance(constraint.value, str):
            try:
                ratio = Fraction(constraint.value)
            except (ValueError, ZeroDivisionError):
                continue
            if ratio > 0:
                ratios.append(ratio)
    if not ratios:
        return 0, 0

    expected = max(ceil(request.guest_count * ratio) for ratio in ratios)
    relevant = [
        derived
        for derived in candidate.derived_constraints
        if derived.constraint.key == "minimum_adults"
    ]
    if not relevant:
        return 1, 0
    values = [item.constraint.value for item in relevant]
    return 1, int(all(isinstance(value, int) for value in values) and max(values) == expected)


def _aggregate_metrics(results: tuple[V02ScenarioEvaluation, ...]) -> V02EvaluationMetrics:
    if not results:
        return V02EvaluationMetrics(
            scenario_count=0,
            feasibility_accuracy=0.0,
            hard_constraint_validity=0.0,
            mean_latency_ms=0.0,
        )

    count = len(results)
    grounded = [item for item in results if item.grounded_decision_correct is not None]
    expected_attributions = sum(item.source_attributions_expected for item in results)
    correct_attributions = sum(item.source_attributions_correct for item in results)
    derived_total = sum(item.derived_constraint_count for item in results)
    derived_correct = sum(item.correct_derived_constraint_count for item in results)
    supported_total = sum(item.supported_claim_count for item in results)
    invalid_supported = sum(item.invalid_supported_claim_count for item in results)
    citation_total = sum(item.citation_count for item in results)
    wrong_citations = sum(item.wrong_source_or_version_count for item in results)
    no_plan = [
        item for item in results if item.expected_outcome is FeasibilityOutcome.NO_FEASIBLE_PLAN
    ]

    return V02EvaluationMetrics(
        scenario_count=count,
        feasibility_accuracy=sum(item.outcome_correct for item in results) / count,
        hard_constraint_validity=sum(item.hard_constraints_valid for item in results) / count,
        grounded_decision_accuracy=(
            sum(bool(item.grounded_decision_correct) for item in grounded) / len(grounded)
            if grounded
            else None
        ),
        source_attribution_accuracy=(
            correct_attributions / expected_attributions if expected_attributions else None
        ),
        derived_constraint_accuracy=(derived_correct / derived_total if derived_total else None),
        unsupported_claim_rate=(invalid_supported / supported_total if supported_total else None),
        wrong_source_version_rate=(wrong_citations / citation_total if citation_total else None),
        no_feasible_plan_accuracy=(
            sum(item.predicted_outcome is FeasibilityOutcome.NO_FEASIBLE_PLAN for item in no_plan)
            / len(no_plan)
            if no_plan
            else None
        ),
        mean_latency_ms=sum(item.latency_ms for item in results) / count,
    )


def load_v01_baseline_snapshot(path: Path) -> BaselineMetricsSnapshot:
    """Load the measured deterministic v0.1 baseline report."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    return BaselineMetricsSnapshot(
        name="v0.1 deterministic baseline",
        feasibility_accuracy=metrics["feasibility_accuracy"],
        hard_constraint_validity=metrics["hard_constraint_validity"],
        no_feasible_plan_accuracy=metrics["no_feasible_plan_accuracy"],
        mean_latency_ms=metrics["mean_latency_ms"],
    )


def load_retrieval_snapshots(path: Path) -> tuple[RetrievalMetricsSnapshot, ...]:
    """Load measured retrieval metrics without mixing them into planning metrics."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    snapshots: list[RetrievalMetricsSnapshot] = []
    for variant in payload["variants"]:
        metrics = variant["metrics"]
        snapshots.append(
            RetrievalMetricsSnapshot(
                variant=variant["variant"],
                top_k=variant["top_k"],
                query_count=variant["query_count"],
                recall_at_k=metrics["recall_at_k"],
                precision_at_k=metrics["precision_at_k"],
                mrr=metrics["mrr"],
                correct_policy_retrieval=metrics["correct_policy_retrieval"],
                correct_version_retrieval=metrics["correct_version_retrieval"],
                wrong_vendor_retrieval_rate=metrics["wrong_vendor_retrieval_rate"],
                mean_latency_ms=metrics["mean_latency_ms"],
            )
        )
    return tuple(snapshots)


def save_v02_evaluation_reports(
    report: V02EvaluationReport,
    output_directory: Path,
    *,
    stem: str = "v0_2_evidence_grounded_evaluation",
) -> tuple[Path, Path]:
    """Write machine-readable and Markdown v0.2 evaluation outputs."""
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / f"{stem}.json"
    markdown_path = output_directory / f"{stem}.md"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_v02_evaluation_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def render_v02_evaluation_markdown(report: V02EvaluationReport) -> str:
    """Render a concise report keeping retrieval metrics visibly separate."""
    metrics = report.metrics

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        "# PartyPilot v0.2 Evidence-Grounded Evaluation",
        "",
        f"Evaluation variant: `{report.evaluation_variant}`",
        "",
    ]
    if report.metadata is not None:
        metadata = report.metadata.config
        working_tree_dirty = (
            metadata.working_tree_dirty if metadata.working_tree_dirty is not None else "unknown"
        )
        lines.extend(
            [
                "## Reproducibility metadata",
                "",
                f"- Experiment ID: {metadata.experiment_id}",
                f"- Evaluation split: {report.metadata.evaluation_split or 'n/a'}",
                f"- Timestamp: {metadata.timestamp.isoformat()}",
                f"- Commit SHA: {metadata.code_commit_sha or 'unavailable'}",
                f"- Working tree dirty: {working_tree_dirty}",
                f"- Git metadata error: {metadata.git_metadata_error or 'none'}",
                f"- Model provider: {metadata.model_provider or 'n/a'}",
                f"- Model name: {metadata.model_name or 'n/a'}",
                f"- Architecture variant: {metadata.architecture_variant}",
                "",
            ]
        )

    lines.extend(
        [
            "## Planning and grounding metrics",
            "",
            f"- Scenarios: {metrics.scenario_count}",
            f"- Feasibility accuracy: {metrics.feasibility_accuracy:.3f}",
            f"- Hard-constraint validity: {metrics.hard_constraint_validity:.3f}",
            f"- Grounded-decision accuracy: {fmt(metrics.grounded_decision_accuracy)}",
            f"- Source-attribution accuracy: {fmt(metrics.source_attribution_accuracy)}",
            f"- Derived-constraint accuracy: {fmt(metrics.derived_constraint_accuracy)}",
            f"- Unsupported-claim rate: {fmt(metrics.unsupported_claim_rate)}",
            f"- Wrong-source/version rate: {fmt(metrics.wrong_source_version_rate)}",
            f"- No-feasible-plan accuracy: {fmt(metrics.no_feasible_plan_accuracy)}",
            f"- Mean latency: {metrics.mean_latency_ms:.3f} ms",
            f"- Tokens: {metrics.total_tokens if metrics.total_tokens is not None else 'n/a'}",
            (
                "- Estimated model cost: "
                f"{metrics.estimated_cost_usd if metrics.estimated_cost_usd is not None else 'n/a'}"
            ),
            "",
            "## v0.1 measured baseline comparison",
            "",
        ]
    )
    if report.v01_baselines:
        for baseline in report.v01_baselines:
            lines.extend(
                [
                    f"### {baseline.name}",
                    f"- Feasibility accuracy: {baseline.feasibility_accuracy:.3f}",
                    f"- Hard-constraint validity: {baseline.hard_constraint_validity:.3f}",
                    f"- No-feasible-plan accuracy: {fmt(baseline.no_feasible_plan_accuracy)}",
                    f"- Mean latency: {baseline.mean_latency_ms:.3f} ms",
                    "",
                ]
            )
    else:
        lines.extend(["No measured v0.1 baseline report was supplied.", ""])

    lines.extend(["## Retrieval metrics (separate)", ""])
    for retrieval in report.retrieval_metrics:
        lines.extend(
            [
                f"### {retrieval.variant}",
                f"- Recall@{retrieval.top_k}: {retrieval.recall_at_k:.3f}",
                f"- Precision@{retrieval.top_k}: {retrieval.precision_at_k:.3f}",
                f"- MRR: {retrieval.mrr:.3f}",
                f"- Correct-policy retrieval: {retrieval.correct_policy_retrieval:.3f}",
                f"- Correct-version retrieval: {retrieval.correct_version_retrieval:.3f}",
                f"- Wrong-vendor retrieval rate: {retrieval.wrong_vendor_retrieval_rate:.3f}",
                f"- Mean retrieval latency: {retrieval.mean_latency_ms:.3f} ms",
                "",
            ]
        )

    if report.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
        lines.append("")
    return "\n".join(lines)
