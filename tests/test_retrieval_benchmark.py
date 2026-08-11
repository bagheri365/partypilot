from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from partypilot.application.retrieval_benchmark import (
    DeterministicHashEmbeddingProvider,
    RetrievalBenchmarkCase,
    RetrievalBenchmarkReport,
    evaluate_retriever,
    render_markdown_report,
    write_retrieval_benchmark_reports,
)
from partypilot.domain.evaluation import RetrievalGroundTruthLabel
from partypilot.domain.evidence_corpus import EvidenceDocumentStatus, EvidenceDocumentType
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalQuery,
    EvidenceRetrievalResult,
    EvidenceVersionMetadata,
    RetrievalMethod,
)


def _label() -> RetrievalGroundTruthLabel:
    return RetrievalGroundTruthLabel(
        expected_document_ids=("doc-good",),
        resource_id="vendor-a",
        expected_version="2.0",
        expected_status=EvidenceDocumentStatus.CURRENT,
        policy_type=EvidenceDocumentType.ALLERGEN_POLICY,
    )


def _result(
    document_id: str,
    *,
    rank: int,
    resource_id: str = "vendor-a",
    version: str = "2.0",
) -> EvidenceRetrievalResult:
    return EvidenceRetrievalResult(
        document_id=document_id,
        chunk_id=f"{document_id}#chunk-1",
        resource_id=resource_id,
        version=EvidenceVersionMetadata(
            version=version,
            effective_date=date(2026, 1, 1),
            status=EvidenceDocumentStatus.CURRENT,
        ),
        text="policy text",
        score=1.0,
        rank=rank,
        retrieval_method=RetrievalMethod.FAKE,
    )


class _FakeRetriever:
    def __init__(self, results: tuple[EvidenceRetrievalResult, ...]) -> None:
        self.results = results

    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
        return self.results[: query.top_k]


def test_metrics_include_recall_precision_mrr_version_vendor_and_latency() -> None:
    case = RetrievalBenchmarkCase("scenario", "peanut policy", _label())
    retriever = _FakeRetriever(
        (
            _result("doc-wrong", rank=1, resource_id="vendor-b"),
            _result("doc-good", rank=2),
        )
    )
    ticks = iter((10.0, 10.002))

    result = evaluate_retriever(
        variant="fake", retriever=retriever, cases=(case,), top_k=2, clock=lambda: next(ticks)
    )

    assert result.metrics.recall_at_k == 1.0
    assert result.metrics.precision_at_k == 0.5
    assert result.metrics.mrr == 0.5
    assert result.metrics.correct_policy_retrieval == 1.0
    assert result.metrics.correct_version_retrieval == 1.0
    assert result.metrics.wrong_vendor_retrieval_rate == 0.5
    assert result.metrics.mean_latency_ms == pytest.approx(2.0)


def test_wrong_version_does_not_count_as_correct_version() -> None:
    case = RetrievalBenchmarkCase("scenario", "policy", _label())
    retriever = _FakeRetriever((_result("doc-good", rank=1, version="1.0"),))
    result = evaluate_retriever(variant="fake", retriever=retriever, cases=(case,), top_k=1)
    assert result.metrics.correct_policy_retrieval == 1.0
    assert result.metrics.correct_version_retrieval == 0.0


def test_empty_cases_produce_zero_metrics() -> None:
    result = evaluate_retriever(variant="fake", retriever=_FakeRetriever(()), cases=(), top_k=3)
    assert result.query_count == 0
    assert result.metrics.recall_at_k == 0.0


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="top_k"):
        evaluate_retriever(variant="fake", retriever=_FakeRetriever(()), cases=(), top_k=0)


def test_hash_embeddings_are_deterministic_and_validate_timeout() -> None:
    provider = DeterministicHashEmbeddingProvider(dimensions=16)
    first = provider.embed(("peanut allergen",), timeout_seconds=1.0)
    second = provider.embed(("peanut allergen",), timeout_seconds=1.0)
    assert first == second
    assert len(first[0]) == 16
    with pytest.raises(ValueError, match="timeout_seconds"):
        provider.embed(("text",), timeout_seconds=0)


def test_invalid_embedding_dimensions_rejected() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        DeterministicHashEmbeddingProvider(dimensions=0)


def test_reports_are_machine_readable_and_markdown(tmp_path: Path) -> None:
    variant = evaluate_retriever(
        variant="fake",
        retriever=_FakeRetriever((_result("doc-good", rank=1),)),
        cases=(RetrievalBenchmarkCase("scenario", "policy", _label()),),
        top_k=1,
    )
    report = RetrievalBenchmarkReport(
        benchmark_name="Retrieval benchmark",
        embedding_backend="test",
        variants=(variant,),
    )
    json_path = tmp_path / "results.json"
    markdown_path = tmp_path / "results.md"
    write_retrieval_benchmark_reports(report, json_path=json_path, markdown_path=markdown_path)

    assert '"variant": "fake"' in json_path.read_text()
    markdown = markdown_path.read_text()
    assert "Recall@k" in markdown
    assert "does not select" in markdown
    assert render_markdown_report(report) == markdown
