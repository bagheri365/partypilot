from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application.retrieval_benchmark import RetrievalBenchmarkCase
from partypilot.ports.evidence_retriever import EvidenceRetrievalQuery, EvidenceRetriever

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


class QueryRewriteDecisionRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum_mrr_improvement: float = Field(default=0.02, ge=0.0)
    minimum_correct_policy_improvement: float = Field(default=0.0, ge=0.0)
    maximum_query_drift_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    maximum_latency_increase_ms: float = Field(default=2.0, ge=0.0)
    maximum_model_cost_usd: float = Field(default=0.0, ge=0.0)


class QueryRewriteMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    correct_policy_retrieval: float = Field(ge=0.0, le=1.0)
    query_drift_rate: float = Field(ge=0.0, le=1.0)
    grounded_decision_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_latency_ms: float = Field(ge=0.0)
    model_input_tokens: int = Field(ge=0)
    model_output_tokens: int = Field(ge=0)
    model_cost_usd: float = Field(ge=0.0)


class QueryRewriteVariantResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: str
    query_count: int = Field(ge=0)
    rewritten_query_count: int = Field(ge=0)
    metrics: QueryRewriteMetrics


class QueryRewriteExperimentReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_name: str
    retained_retriever: str
    top_k: int = Field(gt=0)
    decision_rule: QueryRewriteDecisionRule
    variants: tuple[QueryRewriteVariantResult, ...]
    decision: str
    decision_explanation: str


class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> str: ...


class LexicalSignalPreservingRewriter:
    """Deterministic append-only rewrite used for the controlled offline experiment."""

    _EXPANSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("allergen", ("allergy", "cross-contact", "food safety")),
        ("gluten-free", ("gluten", "dietary", "cross-contact")),
        ("accessibility", ("accessible", "wheelchair", "restroom", "accommodation")),
        ("venue", ("rules", "requirements", "current terms")),
        ("supervision", ("adult", "children", "ratio")),
        ("cancellation", ("refund", "notice", "terms")),
    )

    def rewrite(self, query: str) -> str:
        normalized = query.casefold()
        additions: list[str] = []
        for trigger, expansion in self._EXPANSIONS:
            if trigger in normalized:
                additions.extend(expansion)
        if not additions:
            additions.extend(("current", "requirements"))
        # The original query is retained verbatim as a prefix. Rewriting can enrich but
        # cannot delete vendor names, allergens, dates, or policy identifiers.
        return f"{query} {' '.join(dict.fromkeys(additions))}".strip()


class ConditionalQueryRewriter:
    """Rewrite only complex/broad queries; short lexical queries stay untouched."""

    def __init__(self, rewriter: QueryRewriter, *, minimum_tokens: int = 8) -> None:
        if minimum_tokens <= 0:
            raise ValueError("minimum_tokens must be positive")
        self._rewriter = rewriter
        self._minimum_tokens = minimum_tokens

    def rewrite(self, query: str) -> str:
        if len(_tokenize(query)) < self._minimum_tokens:
            return query
        return self._rewriter.rewrite(query)


def run_query_rewriting_experiment(
    *,
    retriever: EvidenceRetriever,
    cases: Sequence[RetrievalBenchmarkCase],
    top_k: int = 5,
    decision_rule: QueryRewriteDecisionRule | None = None,
) -> QueryRewriteExperimentReport:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rule = decision_rule or QueryRewriteDecisionRule()
    base_rewriter = LexicalSignalPreservingRewriter()
    variants = (
        _evaluate_variant("direct_bm25", retriever, cases, top_k, rewriter=None),
        _evaluate_variant("always_on_rewriting", retriever, cases, top_k, rewriter=base_rewriter),
        _evaluate_variant(
            "conditional_rewriting",
            retriever,
            cases,
            top_k,
            rewriter=ConditionalQueryRewriter(base_rewriter),
        ),
    )
    decision, explanation = _apply_decision_rule(variants, rule)
    return QueryRewriteExperimentReport(
        experiment_name="v0.2 query rewriting experiment",
        retained_retriever="bm25",
        top_k=top_k,
        decision_rule=rule,
        variants=variants,
        decision=decision,
        decision_explanation=explanation,
    )


def _evaluate_variant(
    name: str,
    retriever: EvidenceRetriever,
    cases: Sequence[RetrievalBenchmarkCase],
    top_k: int,
    *,
    rewriter: QueryRewriter | None,
) -> QueryRewriteVariantResult:
    if not cases:
        return QueryRewriteVariantResult(
            variant=name,
            query_count=0,
            rewritten_query_count=0,
            metrics=QueryRewriteMetrics(
                recall_at_k=0.0,
                mrr=0.0,
                correct_policy_retrieval=0.0,
                query_drift_rate=0.0,
                mean_latency_ms=0.0,
                model_input_tokens=0,
                model_output_tokens=0,
                model_cost_usd=0.0,
            ),
        )

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    policy_hits = 0
    drifted = 0
    rewritten_count = 0
    elapsed_ms: list[float] = []

    for case in cases:
        start = perf_counter()
        rewritten = case.query_text if rewriter is None else rewriter.rewrite(case.query_text)
        rewritten_count += int(rewritten != case.query_text)
        drifted += int(_has_query_drift(case.query_text, rewritten))
        results = retriever.retrieve(EvidenceRetrievalQuery(text=rewritten, top_k=top_k))
        elapsed_ms.append((perf_counter() - start) * 1000.0)

        expected = set(case.ground_truth.expected_document_ids)
        retrieved_ids = [result.document_id for result in results]
        relevant = sum(document_id in expected for document_id in retrieved_ids)
        recalls.append(relevant / len(expected))
        rank = next(
            (
                index
                for index, document_id in enumerate(retrieved_ids, start=1)
                if document_id in expected
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        policy_hits += int(rank is not None)

    count = len(cases)
    return QueryRewriteVariantResult(
        variant=name,
        query_count=count,
        rewritten_query_count=rewritten_count,
        metrics=QueryRewriteMetrics(
            recall_at_k=sum(recalls) / count,
            mrr=sum(reciprocal_ranks) / count,
            correct_policy_retrieval=policy_hits / count,
            query_drift_rate=drifted / count,
            grounded_decision_accuracy=None,
            mean_latency_ms=sum(elapsed_ms) / count,
            # This controlled experiment uses deterministic rewriting, not a model call.
            model_input_tokens=0,
            model_output_tokens=0,
            model_cost_usd=0.0,
        ),
    )


def _has_query_drift(original: str, rewritten: str) -> bool:
    """Drift means the rewrite dropped any lexical signal from the original query."""
    return not set(_tokenize(original)).issubset(set(_tokenize(rewritten)))


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.casefold()))


def _apply_decision_rule(
    variants: tuple[QueryRewriteVariantResult, ...], rule: QueryRewriteDecisionRule
) -> tuple[str, str]:
    direct = variants[0]
    eligible: list[QueryRewriteVariantResult] = []
    for candidate in variants[1:]:
        quality_improved = (
            candidate.metrics.mrr - direct.metrics.mrr >= rule.minimum_mrr_improvement
            or candidate.metrics.correct_policy_retrieval - direct.metrics.correct_policy_retrieval
            > rule.minimum_correct_policy_improvement
        )
        safe = (
            candidate.metrics.query_drift_rate <= rule.maximum_query_drift_rate
            and candidate.metrics.mean_latency_ms - direct.metrics.mean_latency_ms
            <= rule.maximum_latency_increase_ms
            and candidate.metrics.model_cost_usd <= rule.maximum_model_cost_usd
        )
        if quality_improved and safe:
            eligible.append(candidate)

    if not eligible:
        return (
            "reject_rewriting",
            "Neither rewriting variant met the predeclared minimum retrieval-quality improvement "
            "while satisfying drift, latency, and model-cost limits; retain direct BM25.",
        )
    best = max(eligible, key=lambda item: (item.metrics.mrr, -item.metrics.mean_latency_ms))
    return (
        f"retain_{best.variant}",
        "A rewriting variant met the predeclared quality and safety/cost thresholds.",
    )


def render_query_rewriting_markdown(report: QueryRewriteExperimentReport) -> str:
    rule = report.decision_rule
    lines = [
        "# v0.2 Query Rewriting Experiment",
        "",
        "## Predeclared decision rule",
        "",
        (
            "Retain rewriting only if it improves MRR by at least "
            f"{rule.minimum_mrr_improvement:.3f} or improves correct-policy retrieval, "
            f"keeps query drift <= {rule.maximum_query_drift_rate:.3f}, adds no more than "
            f"{rule.maximum_latency_increase_ms:.3f} ms mean latency, and stays within "
            f"${rule.maximum_model_cost_usd:.4f} model cost for this offline experiment."
        ),
        "",
        "High-value lexical signals are protected by an append-only rewrite: the original query "
        "remains verbatim and expansions are appended.",
        "",
        (
            "| Variant | Recall@k | MRR | Correct policy | Query drift | "
            "Mean latency (ms) | Rewritten | Model tokens | Model cost |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in report.variants:
        m = variant.metrics
        model_tokens = m.model_input_tokens + m.model_output_tokens
        lines.append(
            f"| {variant.variant} | {m.recall_at_k:.3f} | {m.mrr:.3f} | "
            f"{m.correct_policy_retrieval:.3f} | {m.query_drift_rate:.3f} | "
            f"{m.mean_latency_ms:.3f} | {variant.rewritten_query_count}/{variant.query_count} | "
            f"{model_tokens} | ${m.model_cost_usd:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Grounded-decision accuracy",
            "",
            "Not yet available: Prompt 43 evidence-grounded planning has not been implemented, "
            "so this experiment does not fabricate a downstream decision metric.",
            "",
            "## Decision",
            "",
            f"**{report.decision}** — {report.decision_explanation}",
            "",
            "The rewrite variants are deterministic and make no model calls, so measured model "
            "tokens and model cost are zero. This does not estimate the cost of an LLM rewriter.",
            "",
        ]
    )
    return "\n".join(lines)


def write_query_rewriting_reports(
    report: QueryRewriteExperimentReport, *, json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_query_rewriting_markdown(report), encoding="utf-8")
