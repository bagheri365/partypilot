"""Controlled v0.2 end-to-end comparison with and without conditional rewriting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application.v02_evaluation import (
    BaselineMetricsSnapshot,
    EvidenceGroundedScenarioPlanner,
    RetrievalMetricsSnapshot,
    V02EvaluationMetrics,
    V02EvaluationRunner,
    V02ScenarioEvaluation,
    load_retrieval_snapshots,
    load_v01_baseline_snapshot,
)
from partypilot.domain.evaluation import EvaluationScenario
from partypilot.domain.evidence_corpus import EvidenceDocument
from partypilot.domain.feasibility import FeasibilityOutcome


class V02ComparisonVariantResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: str
    metrics: V02EvaluationMetrics
    scenarios: tuple[V02ScenarioEvaluation, ...]


class V02ComparisonScenarioResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    expected_outcome: FeasibilityOutcome
    expected_evidence_document_ids: tuple[str, ...] = ()
    baseline: V02ScenarioEvaluation
    conditional: V02ScenarioEvaluation


class V02QueryRewritingComparisonReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_name: str = "v0.2 evidence-grounded query rewriting comparison"
    retained_retriever: str = "bm25"
    top_k: int = Field(gt=0)
    variants: tuple[V02ComparisonVariantResult, ...]
    evidence_labeled_scenarios: tuple[V02ComparisonScenarioResult, ...]
    decision: str
    decision_explanation: str
    v01_baselines: tuple[BaselineMetricsSnapshot, ...] = ()
    retrieval_metrics: tuple[RetrievalMetricsSnapshot, ...] = ()
    notes: tuple[str, ...] = ()


def run_v02_query_rewriting_comparison(
    *,
    baseline_planner: EvidenceGroundedScenarioPlanner,
    conditional_planner: EvidenceGroundedScenarioPlanner,
    corpus: tuple[EvidenceDocument, ...],
    scenarios: Sequence[EvaluationScenario],
    top_k: int,
    clock: Callable[[], float] = perf_counter,
) -> V02QueryRewritingComparisonReport:
    baseline_metrics, baseline_results = V02EvaluationRunner(
        baseline_planner,
        corpus=corpus,
        clock=clock,
    ).run(scenarios)
    conditional_metrics, conditional_results = V02EvaluationRunner(
        conditional_planner,
        corpus=corpus,
        clock=clock,
    ).run(scenarios)

    baseline = V02ComparisonVariantResult(
        variant="bm25 + live_ollama_constraint_extractor",
        metrics=baseline_metrics,
        scenarios=baseline_results,
    )
    conditional = V02ComparisonVariantResult(
        variant="bm25 + conditional_query_rewriting + live_ollama_constraint_extractor",
        metrics=conditional_metrics,
        scenarios=conditional_results,
    )
    evidence_labeled_scenarios = _evidence_labeled_scenarios(
        baseline_results=baseline_results,
        conditional_results=conditional_results,
    )
    decision, explanation = _make_decision(baseline.metrics, conditional.metrics)
    return V02QueryRewritingComparisonReport(
        retained_retriever="bm25",
        top_k=top_k,
        variants=(baseline, conditional),
        evidence_labeled_scenarios=evidence_labeled_scenarios,
        decision=decision,
        decision_explanation=explanation,
        v01_baselines=(
            load_v01_baseline_snapshot(Path("evals/results/v0_1/deterministic_baseline.json")),
        ),
        retrieval_metrics=load_retrieval_snapshots(
            Path("evals/results/v0_2/retrieval_benchmark.json")
        ),
        notes=(
            "Both variants use the same corpus, planner semantics, citation validation, and live "
            "constraint extractor; only the evidence-retrieval query text differs.",
            "The conditional variant uses the retained query rewriter only through an "
            "EvidenceRetriever decorator.",
            "Conditional rewriting is retained only if it improves a meaningful downstream "
            "metric without degrading hard-constraint validity, unsupported-claim rate, or "
            "wrong-source/version rate.",
            "If downstream metrics are identical, plain BM25 is preferred.",
        ),
    )


def save_v02_query_rewriting_comparison_reports(
    report: V02QueryRewritingComparisonReport,
    output_directory: Path,
    *,
    stem: str = "v0_2_query_rewriting_comparison",
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / f"{stem}.json"
    markdown_path = output_directory / f"{stem}.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_v02_query_rewriting_comparison_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path


def render_v02_query_rewriting_comparison_markdown(
    report: V02QueryRewritingComparisonReport,
) -> str:
    baseline, conditional = report.variants
    lines = [
        "# PartyPilot v0.2 Query Rewriting Comparison",
        "",
        f"Retained retriever: `{report.retained_retriever}`",
        f"Top-k: **{report.top_k}**",
        "",
        "## Decision",
        "",
        f"**{report.decision}** — {report.decision_explanation}",
        "",
        "## Planning and grounding metrics",
        "",
        (
            "| Variant | Feasibility | Hard validity | Grounded | Source attribution | "
            "Derived | Unsupported claim | Wrong source/version | No-feasible-plan | "
            "Mean latency (ms) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metrics_row(baseline),
        _metrics_row(conditional),
        "",
        "## Evidence-labeled scenarios",
        "",
        (
            "| Scenario | Expected | Expected docs | BM25 predicted | BM25 attributed "
            "docs | BM25 grounded | Conditional predicted | Conditional attributed docs | "
            "Conditional grounded |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for scenario in report.evidence_labeled_scenarios:
        lines.append(_scenario_row(scenario))
    lines.extend(
        [
            "",
            "## v0.1 measured baseline comparison",
            "",
        ]
    )
    if report.v01_baselines:
        for baseline_snapshot in report.v01_baselines:
            lines.extend(
                [
                    f"### {baseline_snapshot.name}",
                    f"- Feasibility accuracy: {baseline_snapshot.feasibility_accuracy:.3f}",
                    f"- Hard-constraint validity: {baseline_snapshot.hard_constraint_validity:.3f}",
                    (
                        "- No-feasible-plan accuracy: "
                        f"{_fmt(baseline_snapshot.no_feasible_plan_accuracy)}"
                    ),
                    f"- Mean latency: {baseline_snapshot.mean_latency_ms:.3f} ms",
                    "",
                ]
            )
    lines.extend(
        [
            "## Retrieval metrics (separate)",
            "",
        ]
    )
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


def _metrics_row(variant: V02ComparisonVariantResult) -> str:
    m = variant.metrics
    return (
        f"| {variant.variant} | {m.feasibility_accuracy:.3f} | {m.hard_constraint_validity:.3f} | "
        f"{_fmt(m.grounded_decision_accuracy)} | {_fmt(m.source_attribution_accuracy)} | "
        f"{_fmt(m.derived_constraint_accuracy)} | {_fmt(m.unsupported_claim_rate)} | "
        f"{_fmt(m.wrong_source_version_rate)} | {_fmt(m.no_feasible_plan_accuracy)} | "
        f"{m.mean_latency_ms:.3f} |"
    )


def _scenario_row(scenario: V02ComparisonScenarioResult) -> str:
    return (
        f"| {scenario.scenario_id} | {scenario.expected_outcome.value} | "
        f"{', '.join(scenario.expected_evidence_document_ids) or 'none'} | "
        f"{scenario.baseline.predicted_outcome.value} | "
        f"{', '.join(scenario.baseline.attributed_evidence_document_ids) or 'none'} | "
        f"{_fmt_bool(scenario.baseline.grounded_decision_correct)} | "
        f"{scenario.conditional.predicted_outcome.value} | "
        f"{', '.join(scenario.conditional.attributed_evidence_document_ids) or 'none'} | "
        f"{_fmt_bool(scenario.conditional.grounded_decision_correct)} |"
    )


def _evidence_labeled_scenarios(
    *,
    baseline_results: tuple[V02ScenarioEvaluation, ...],
    conditional_results: tuple[V02ScenarioEvaluation, ...],
) -> tuple[V02ComparisonScenarioResult, ...]:
    conditional_by_id = {item.scenario_id: item for item in conditional_results}
    labeled: list[V02ComparisonScenarioResult] = []
    for baseline in baseline_results:
        if not baseline.expected_evidence_document_ids:
            continue
        conditional = conditional_by_id[baseline.scenario_id]
        labeled.append(
            V02ComparisonScenarioResult(
                scenario_id=baseline.scenario_id,
                expected_outcome=baseline.expected_outcome,
                expected_evidence_document_ids=baseline.expected_evidence_document_ids,
                baseline=baseline,
                conditional=conditional,
            )
        )
    return tuple(labeled)


def _make_decision(
    baseline: V02EvaluationMetrics,
    conditional: V02EvaluationMetrics,
) -> tuple[str, str]:
    meaningful = (
        "grounded_decision_accuracy",
        "source_attribution_accuracy",
        "feasibility_accuracy",
        "no_feasible_plan_accuracy",
    )
    guardrails = (
        "hard_constraint_validity",
        "unsupported_claim_rate",
        "wrong_source_version_rate",
    )

    improvements: list[str] = []
    degradations: list[str] = []
    for metric in meaningful:
        baseline_value = getattr(baseline, metric)
        conditional_value = getattr(conditional, metric)
        if baseline_value is None or conditional_value is None:
            continue
        if conditional_value > baseline_value:
            improvements.append(metric)
        elif conditional_value < baseline_value:
            degradations.append(metric)

    for metric in guardrails:
        baseline_value = getattr(baseline, metric)
        conditional_value = getattr(conditional, metric)
        if baseline_value is None or conditional_value is None:
            continue
        if conditional_value < baseline_value:
            degradations.append(metric)

    if improvements and not degradations:
        return (
            "retain_conditional_rewriting",
            (
                "Conditional rewriting improved a downstream metric without degrading "
                "the guardrails."
            ),
        )

    if not improvements:
        return (
            "reject_conditional_rewriting",
            (
                "Conditional rewriting did not improve any meaningful downstream metric, "
                "so plain BM25 is preferred."
            ),
        )

    return (
        "reject_conditional_rewriting",
        (
            "Conditional rewriting improved a downstream metric but degraded at least "
            "one guardrail metric."
        ),
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_bool(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"
