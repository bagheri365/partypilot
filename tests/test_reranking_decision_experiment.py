from __future__ import annotations

from datetime import date

from partypilot.application.reranking_decision_experiment import (
    RerankingDiagnosticRule,
    render_reranking_decision_markdown,
    run_reranking_decision_experiment,
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


class RankedFakeRetriever:
    def __init__(self, relevant_rank: int | None) -> None:
        self._relevant_rank = relevant_rank

    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
        results: list[EvidenceRetrievalResult] = []
        for rank in range(1, query.top_k + 1):
            is_relevant = rank == self._relevant_rank
            document_id = "doc-relevant" if is_relevant else f"doc-distractor-{rank}"
            results.append(
                EvidenceRetrievalResult(
                    document_id=document_id,
                    chunk_id=f"{document_id}#chunk-1",
                    resource_id="vendor-alpha" if is_relevant else "vendor-other",
                    version=EvidenceVersionMetadata(
                        version="1.0",
                        effective_date=date(2026, 1, 1),
                        status=EvidenceDocumentStatus.CURRENT,
                    ),
                    text=document_id,
                    score=float(query.top_k - rank + 1),
                    rank=rank,
                    retrieval_method=RetrievalMethod.BM25,
                )
            )
        return tuple(results)


def _case() -> RetrievalBenchmarkCase:
    return RetrievalBenchmarkCase(
        scenario_id="scenario-1",
        query_text="vendor-alpha allergen policy",
        ground_truth=RetrievalGroundTruthLabel(
            expected_document_ids=("doc-relevant",),
            resource_id="vendor-alpha",
            expected_version="1.0",
            expected_status=EvidenceDocumentStatus.CURRENT,
            policy_type=EvidenceDocumentType.ALLERGEN_POLICY,
        ),
    )


def test_stops_when_low_rank_failure_pattern_is_absent() -> None:
    report = run_reranking_decision_experiment(
        retriever=RankedFakeRetriever(relevant_rank=1), cases=(_case(),), top_k=5
    )
    assert report.failure_pattern_present is False
    assert report.reranker_comparison_run is False
    assert report.decision == "reranking_not_justified"
    assert report.metrics.correct_evidence_at_rank_1_rate == 1.0


def test_detects_when_low_rank_failure_pattern_requires_followup() -> None:
    report = run_reranking_decision_experiment(
        retriever=RankedFakeRetriever(relevant_rank=3),
        cases=(_case(),),
        top_k=5,
        rule=RerankingDiagnosticRule(minimum_low_rank_failure_rate=0.20),
    )
    assert report.failure_pattern_present is True
    assert report.decision == "reranker_comparison_required"
    assert report.metrics.correct_evidence_low_rank_rate == 1.0


def test_missing_relevant_evidence_is_not_mislabeled_as_reranking_problem() -> None:
    report = run_reranking_decision_experiment(
        retriever=RankedFakeRetriever(relevant_rank=None), cases=(_case(),), top_k=5
    )
    assert report.failure_pattern_present is False
    assert report.metrics.missed_relevant_evidence_rate == 1.0
    assert report.metrics.correct_evidence_low_rank_rate == 0.0


def test_empty_cases_stop_without_claiming_comparison() -> None:
    report = run_reranking_decision_experiment(
        retriever=RankedFakeRetriever(relevant_rank=1), cases=(), top_k=5
    )
    assert report.query_count == 0
    assert report.reranker_comparison_run is False
    assert report.decision == "reranking_not_justified"


def test_markdown_documents_stop_condition() -> None:
    report = run_reranking_decision_experiment(
        retriever=RankedFakeRetriever(relevant_rank=1), cases=(_case(),), top_k=5
    )
    markdown = render_reranking_decision_markdown(report)
    assert "Prerequisite diagnostic" in markdown
    assert "Not run" in markdown
    assert "reranking_not_justified" in markdown
