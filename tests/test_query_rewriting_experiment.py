from __future__ import annotations

from datetime import date

from partypilot.application.query_rewriting_experiment import (
    ConditionalQueryRewriter,
    LexicalSignalPreservingRewriter,
    QueryRewriteDecisionRule,
    _has_query_drift,
    render_query_rewriting_markdown,
    run_query_rewriting_experiment,
)
from partypilot.application.retrieval_benchmark import RetrievalBenchmarkCase
from partypilot.domain.evaluation import RetrievalGroundTruthLabel
from partypilot.domain.evidence_corpus import EvidenceDocumentStatus, EvidenceDocumentType
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalQuery,
    EvidenceRetrievalResult,
    EvidenceVersionMetadata,
    RetrievalMethod,
)


class FakeRetriever:
    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
        expected = "doc-1"
        result = EvidenceRetrievalResult(
            document_id=expected,
            chunk_id=f"{expected}#chunk-1",
            resource_id="vendor-alpha",
            version=EvidenceVersionMetadata(
                version="2.0",
                effective_date=date(2026, 1, 1),
                status=EvidenceDocumentStatus.CURRENT,
            ),
            text="Peanut policy",
            score=1.0,
            rank=1,
            retrieval_method=RetrievalMethod.BM25,
        )
        return (result,)


def _case(
    query: str = "vendor-alpha peanuts 2026 policy-17 allergen policy",
) -> RetrievalBenchmarkCase:
    return RetrievalBenchmarkCase(
        scenario_id="scenario-1",
        query_text=query,
        ground_truth=RetrievalGroundTruthLabel(
            expected_document_ids=("doc-1",),
            resource_id="vendor-alpha",
            expected_version="2.0",
            expected_status=EvidenceDocumentStatus.CURRENT,
            policy_type=EvidenceDocumentType.ALLERGEN_POLICY,
        ),
    )


def test_append_only_rewrite_preserves_high_value_lexical_signals() -> None:
    original = "vendor-alpha peanuts 2026 policy-17 allergen policy"
    rewritten = LexicalSignalPreservingRewriter().rewrite(original)
    assert rewritten.startswith(original)
    assert not _has_query_drift(original, rewritten)


def test_conditional_rewriter_leaves_short_query_untouched() -> None:
    rewriter = ConditionalQueryRewriter(LexicalSignalPreservingRewriter(), minimum_tokens=8)
    assert (
        rewriter.rewrite("vendor-alpha peanuts allergen policy")
        == "vendor-alpha peanuts allergen policy"
    )


def test_conditional_rewriter_rewrites_complex_query() -> None:
    query = "vendor-alpha peanuts tree nuts vegan wheelchair quiet room allergen policy"
    rewriter = ConditionalQueryRewriter(LexicalSignalPreservingRewriter(), minimum_tokens=8)
    assert rewriter.rewrite(query) != query


def test_experiment_records_zero_model_cost_and_no_grounded_metric() -> None:
    report = run_query_rewriting_experiment(retriever=FakeRetriever(), cases=(_case(),), top_k=1)
    assert len(report.variants) == 3
    for variant in report.variants:
        assert variant.metrics.model_cost_usd == 0.0
        assert variant.metrics.model_input_tokens == 0
        assert variant.metrics.model_output_tokens == 0
        assert variant.metrics.grounded_decision_accuracy is None


def test_decision_rejects_rewriting_without_quality_improvement() -> None:
    report = run_query_rewriting_experiment(
        retriever=FakeRetriever(),
        cases=(_case(),),
        top_k=1,
        decision_rule=QueryRewriteDecisionRule(minimum_mrr_improvement=0.02),
    )
    assert report.decision == "reject_rewriting"


def test_markdown_contains_predeclared_rule_and_decision() -> None:
    report = run_query_rewriting_experiment(retriever=FakeRetriever(), cases=(_case(),), top_k=1)
    markdown = render_query_rewriting_markdown(report)
    assert "Predeclared decision rule" in markdown
    assert "Grounded-decision accuracy" in markdown
    assert report.decision in markdown
