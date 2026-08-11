from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from partypilot.adapters.query_rewriting_retriever import (
    ConditionalQueryRewritingEvidenceRetriever,
)
from partypilot.application.constraint_engine import ConstraintEngineResult
from partypilot.application.evidence_grounded_planner import (
    EvidenceGroundedPlanCandidate,
    EvidenceGroundedPlanningResult,
)
from partypilot.application.query_rewriting_experiment import (
    ConditionalQueryRewriter,
    LexicalSignalPreservingRewriter,
)
from partypilot.application.v02_query_rewriting_comparison import (
    V02QueryRewritingComparisonReport,
    run_v02_query_rewriting_comparison,
    save_v02_query_rewriting_comparison_reports,
)
from partypilot.domain.evaluation import (
    ComplexityMetadata,
    DatasetSplit,
    EvaluationScenario,
    RetrievalGroundTruthLabel,
    ScenarioCategory,
)
from partypilot.domain.evidence import (
    DerivationMethod,
    EvidenceReference,
    EvidenceState,
    Provenance,
)
from partypilot.domain.evidence_corpus import (
    EvidenceDocument,
    EvidenceDocumentMetadata,
    EvidenceDocumentStatus,
    EvidenceDocumentType,
)
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest
from partypilot.domain.resources import Resource, ResourceCategory
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalFilters,
    EvidenceRetrievalQuery,
    EvidenceRetrievalResult,
)


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[EvidenceRetrievalQuery] = []

    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
        self.queries.append(query)
        return ()


class StaticPlanner:
    def __init__(self, result: EvidenceGroundedPlanningResult) -> None:
        self._result = result

    def plan(self, request: PartyRequest) -> EvidenceGroundedPlanningResult:
        return self._result


def _query() -> EvidenceRetrievalQuery:
    return EvidenceRetrievalQuery(
        text="vendor-alpha peanuts allergen policy wheelchair accessible restroom current",
        top_k=7,
        filters=EvidenceRetrievalFilters(
            resource_id="venue-1",
            document_type=EvidenceDocumentType.ALLERGEN_POLICY,
            status=EvidenceDocumentStatus.CURRENT,
        ),
    )


def _scenario(expected: FeasibilityOutcome) -> EvaluationScenario:
    return EvaluationScenario(
        scenario_id="scenario-1",
        request=PartyRequest(
            location="Brooklyn, NY",
            event_date=date(2026, 9, 20),
            guest_count=10,
            total_budget=Decimal("500"),
        ),
        expected_feasibility=expected,
        retrieval_ground_truth=(
            RetrievalGroundTruthLabel(
                expected_document_ids=("doc-1",),
                resource_id="venue-1",
                expected_version="1.0",
                expected_status=EvidenceDocumentStatus.CURRENT,
                policy_type=EvidenceDocumentType.VENUE_POLICY,
            ),
        ),
        scenario_category=ScenarioCategory.FEASIBLE,
        complexity=ComplexityMetadata(),
        dataset_split=DatasetSplit.DEVELOPMENT,
    )


def _corpus() -> tuple[EvidenceDocument, ...]:
    return (
        EvidenceDocument(
            metadata=EvidenceDocumentMetadata(
                document_id="doc-1",
                resource_id="venue-1",
                document_type=EvidenceDocumentType.VENUE_POLICY,
                version="1.0",
                effective_date=date(2026, 1, 1),
                status=EvidenceDocumentStatus.CURRENT,
            ),
            text="Current venue policy.",
        ),
    )


def _evidence_reference(document_id: str) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=f"constraint:{document_id}",
        state=EvidenceState.SUPPORTED,
        provenance=(
            Provenance(
                source_document_id=document_id,
                source_chunk_id=f"{document_id}#chunk-1",
                resource_id="venue-1",
                source_version="1.0",
                effective_date=date(2026, 1, 1),
                derivation_method=DerivationMethod.LLM_EXTRACTED,
                derivation_explanation="Extracted from policy text.",
            ),
        ),
    )


def _feasible_result(document_id: str) -> EvidenceGroundedPlanningResult:
    candidate = EvidenceGroundedPlanCandidate(
        resources=(
            Resource(
                resource_id="venue-1",
                name="Venue",
                location="Brooklyn, NY",
                price=Decimal("100"),
                capacity=10,
                category=ResourceCategory.VENUE,
            ),
        ),
        total_cost=Decimal("100"),
        validation=ConstraintEngineResult(
            feasible=True,
            satisfied_constraint_ids=("request-location",),
            violations=(),
            unresolved_constraint_ids=(),
        ),
        evidence_references=(_evidence_reference(document_id),),
    )
    return EvidenceGroundedPlanningResult(
        outcome=FeasibilityOutcome.FEASIBLE,
        candidates=(candidate,),
        evidence_references=(_evidence_reference(document_id),),
    )


def test_conditional_rewriting_retriever_preserves_query_fields() -> None:
    inner = RecordingRetriever()
    wrapper = ConditionalQueryRewritingEvidenceRetriever(
        inner,
        ConditionalQueryRewriter(LexicalSignalPreservingRewriter()),
    )

    wrapper.retrieve(_query())

    assert len(inner.queries) == 1
    forwarded = inner.queries[0]
    assert forwarded.top_k == 7
    assert forwarded.filters == _query().filters
    assert forwarded.text != _query().text
    assert forwarded.model_dump(exclude={"text"}) == _query().model_dump(exclude={"text"})


def test_query_rewriting_comparison_prefers_plain_bm25_when_metrics_are_identical(
    tmp_path: Path,
) -> None:
    result = _feasible_result("doc-1")
    report = run_v02_query_rewriting_comparison(
        baseline_planner=StaticPlanner(result),
        conditional_planner=StaticPlanner(result),
        corpus=_corpus(),
        scenarios=(_scenario(FeasibilityOutcome.FEASIBLE),),
        top_k=5,
        clock=lambda: 1.0,
    )

    assert isinstance(report, V02QueryRewritingComparisonReport)
    assert report.variants[0].variant == "bm25 + live_ollama_constraint_extractor"
    assert (
        report.variants[1].variant
        == "bm25 + conditional_query_rewriting + live_ollama_constraint_extractor"
    )
    assert report.decision == "reject_conditional_rewriting"
    assert report.evidence_labeled_scenarios[0].expected_evidence_document_ids == ("doc-1",)

    json_path, md_path = save_v02_query_rewriting_comparison_reports(report, tmp_path)
    assert json_path.exists()
    assert md_path.exists()


def test_query_rewriting_comparison_retains_rewriting_when_downstream_metrics_improve() -> None:
    baseline = EvidenceGroundedPlanningResult(outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN)
    conditional = _feasible_result("doc-1")
    report = run_v02_query_rewriting_comparison(
        baseline_planner=StaticPlanner(baseline),
        conditional_planner=StaticPlanner(conditional),
        corpus=_corpus(),
        scenarios=(_scenario(FeasibilityOutcome.FEASIBLE),),
        top_k=5,
        clock=lambda: 1.0,
    )

    assert report.decision == "retain_conditional_rewriting"
    assert report.evidence_labeled_scenarios[0].conditional.predicted_outcome is (
        FeasibilityOutcome.FEASIBLE
    )
