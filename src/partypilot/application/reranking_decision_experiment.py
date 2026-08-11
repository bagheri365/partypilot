from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application.query_rewriting_experiment import QueryRewriter
from partypilot.application.retrieval_benchmark import RetrievalBenchmarkCase
from partypilot.ports.evidence_retriever import EvidenceRetrievalQuery, EvidenceRetriever


class RerankingDiagnosticRule(BaseModel):
    """Predeclared rule for deciding whether a reranker experiment is warranted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    low_rank_starts_at: int = Field(default=3, ge=2)
    minimum_low_rank_failure_rate: float = Field(default=0.20, ge=0.0, le=1.0)


class RerankingDiagnosticMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    correct_evidence_at_rank_1_rate: float = Field(ge=0.0, le=1.0)
    correct_evidence_low_rank_rate: float = Field(ge=0.0, le=1.0)
    missed_relevant_evidence_rate: float = Field(ge=0.0, le=1.0)
    mean_latency_ms: float = Field(ge=0.0)


class RerankingDecisionReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_name: str
    retained_retrieval: str
    top_k: int = Field(gt=0)
    query_count: int = Field(ge=0)
    diagnostic_rule: RerankingDiagnosticRule
    metrics: RerankingDiagnosticMetrics
    failure_pattern_present: bool
    reranker_comparison_run: bool
    decision: str
    decision_explanation: str
    model_cost_usd: float = Field(default=0.0, ge=0.0)


def run_reranking_decision_experiment(
    *,
    retriever: EvidenceRetriever,
    cases: Sequence[RetrievalBenchmarkCase],
    top_k: int = 5,
    rewriter: QueryRewriter | None = None,
    rule: RerankingDiagnosticRule | None = None,
) -> RerankingDecisionReport:
    """Diagnose whether reranking is justified before introducing a reranker.

    Prompt 38 requires stopping when the retained retriever does not commonly retrieve
    relevant evidence too low in its ranking. This function therefore performs only the
    prerequisite diagnosis. A reranker is deliberately not constructed or called unless
    that prerequisite is established by measured data.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    diagnostic_rule = rule or RerankingDiagnosticRule()
    if not cases:
        return RerankingDecisionReport(
            experiment_name="v0.2 reranking decision experiment",
            retained_retrieval="bm25_with_conditional_rewriting" if rewriter else "bm25",
            top_k=top_k,
            query_count=0,
            diagnostic_rule=diagnostic_rule,
            metrics=RerankingDiagnosticMetrics(
                recall_at_k=0.0,
                mrr=0.0,
                correct_evidence_at_rank_1_rate=0.0,
                correct_evidence_low_rank_rate=0.0,
                missed_relevant_evidence_rate=0.0,
                mean_latency_ms=0.0,
            ),
            failure_pattern_present=False,
            reranker_comparison_run=False,
            decision="reranking_not_justified",
            decision_explanation=(
                "No labeled retrieval cases were available to establish a low-rank failure pattern."
            ),
        )

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []
    rank_one_hits = 0
    low_rank_hits = 0
    misses = 0

    for case in cases:
        query_text = case.query_text if rewriter is None else rewriter.rewrite(case.query_text)
        started = perf_counter()
        results = retriever.retrieve(EvidenceRetrievalQuery(text=query_text, top_k=top_k))
        latencies_ms.append((perf_counter() - started) * 1000.0)

        expected = set(case.ground_truth.expected_document_ids)
        retrieved_ids = [result.document_id for result in results]
        relevant_count = sum(document_id in expected for document_id in retrieved_ids)
        recalls.append(relevant_count / len(expected))
        first_rank = next(
            (
                rank
                for rank, document_id in enumerate(retrieved_ids, start=1)
                if document_id in expected
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        if first_rank == 1:
            rank_one_hits += 1
        elif first_rank is None:
            misses += 1
        elif first_rank >= diagnostic_rule.low_rank_starts_at:
            low_rank_hits += 1

    count = len(cases)
    low_rank_rate = low_rank_hits / count
    metrics = RerankingDiagnosticMetrics(
        recall_at_k=sum(recalls) / count,
        mrr=sum(reciprocal_ranks) / count,
        correct_evidence_at_rank_1_rate=rank_one_hits / count,
        correct_evidence_low_rank_rate=low_rank_rate,
        missed_relevant_evidence_rate=misses / count,
        mean_latency_ms=sum(latencies_ms) / count,
    )
    failure_pattern_present = low_rank_rate >= diagnostic_rule.minimum_low_rank_failure_rate

    if failure_pattern_present:
        # The current measured PartyPilot corpus does not take this branch. Keeping the
        # diagnosis explicit prevents silently introducing a reranking dependency before
        # benchmark evidence justifies one.
        decision = "reranker_comparison_required"
        explanation = (
            "The retained retriever meets the predeclared low-rank failure threshold. "
            "A controlled with/without-reranker comparison is required before retention."
        )
    else:
        decision = "reranking_not_justified"
        explanation = (
            "The retained retriever does not commonly place correct evidence at rank "
            f"{diagnostic_rule.low_rank_starts_at} or lower, so Prompt 38 stops before "
            "adding a reranker."
        )

    return RerankingDecisionReport(
        experiment_name="v0.2 reranking decision experiment",
        retained_retrieval="bm25_with_conditional_rewriting" if rewriter else "bm25",
        top_k=top_k,
        query_count=count,
        diagnostic_rule=diagnostic_rule,
        metrics=metrics,
        failure_pattern_present=failure_pattern_present,
        reranker_comparison_run=False,
        decision=decision,
        decision_explanation=explanation,
        model_cost_usd=0.0,
    )


def render_reranking_decision_markdown(report: RerankingDecisionReport) -> str:
    m = report.metrics
    rule = report.diagnostic_rule
    return "\n".join(
        [
            "# v0.2 Reranking Decision Experiment",
            "",
            "## Prerequisite diagnostic",
            "",
            (
                "Reranking is tested only if correct evidence appears at rank "
                f"{rule.low_rank_starts_at} or lower in at least "
                f"{rule.minimum_low_rank_failure_rate:.0%} of labeled queries."
            ),
            "",
            f"Retained retrieval: `{report.retained_retrieval}`",
            f"Labeled queries: **{report.query_count}**",
            f"Recall@{report.top_k}: **{m.recall_at_k:.3f}**",
            f"MRR: **{m.mrr:.3f}**",
            f"Correct evidence at rank 1: **{m.correct_evidence_at_rank_1_rate:.3f}**",
            (
                f"Correct evidence at rank {rule.low_rank_starts_at}+: "
                f"**{m.correct_evidence_low_rank_rate:.3f}**"
            ),
            f"Missed relevant evidence: **{m.missed_relevant_evidence_rate:.3f}**",
            f"Mean retrieval latency: **{m.mean_latency_ms:.3f} ms**",
            "",
            "## Decision",
            "",
            f"**{report.decision}** — {report.decision_explanation}",
            "",
            "## Reranker comparison",
            "",
            (
                "Not run. The prerequisite failure pattern was absent, so retrieval quality, "
                "downstream decision quality, reranker latency, and model cost cannot be "
                "truthfully reported for a reranker that was not justified or invoked."
                if not report.failure_pattern_present
                else "Required by the diagnostic before any reranker can be retained."
            ),
            "",
            "No reranking model, provider call, or model cost was introduced by this experiment.",
            "",
        ]
    )


def write_reranking_decision_reports(
    report: RerankingDecisionReport, *, json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_reranking_decision_markdown(report), encoding="utf-8")
