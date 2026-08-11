from __future__ import annotations

from datetime import date
from decimal import Decimal
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from partypilot.adapters.bm25_evidence_retriever import BM25EvidenceRetriever
from partypilot.adapters.llm_constraint_extractor import (
    LLMConstraintExtractor,
    LLMConstraintExtractorOutputError,
    LLMConstraintExtractorProviderError,
)
from partypilot.application.evidence_grounded_planner import EvidenceGroundedPlanner
from partypilot.application.v02_evaluation import V02EvaluationMetrics
from partypilot.domain.evidence_corpus import (
    EvidenceDocument,
    EvidenceDocumentMetadata,
    EvidenceDocumentStatus,
    EvidenceDocumentType,
)
from partypilot.domain.party_request import PartyRequest
from partypilot.ports.constraint_extractor import (
    ConstraintExtractionContext,
    ConstraintExtractionInput,
)
from partypilot.ports.llm_provider import (
    FailingFakeLLMProvider,
    FakeLLMProvider,
    GenerationResponse,
)

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "evals" / "run_v0_2_evaluation.py"
_SPEC = spec_from_file_location("run_v0_2_evaluation", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
v02_eval = module_from_spec(_SPEC)
_SPEC.loader.exec_module(v02_eval)


def _corpus() -> tuple[EvidenceDocument, ...]:
    return (
        EvidenceDocument(
            metadata=EvidenceDocumentMetadata(
                document_id="doc-venue-1",
                resource_id="venue-1",
                document_type=EvidenceDocumentType.VENUE_POLICY,
                version="1.0",
                effective_date=date(2026, 1, 1),
                status=EvidenceDocumentStatus.CURRENT,
            ),
            text="Venue policy for the live composition test.",
        ),
    )


def _extraction_input() -> ConstraintExtractionInput:
    metadata = _corpus()[0].metadata
    return ConstraintExtractionInput(
        evidence_text="One adult is required for every five children.",
        evidence_metadata=metadata,
        chunk_id=f"{metadata.document_id}#chunk-1",
        planning_context=ConstraintExtractionContext(
            request=PartyRequest(
                location="Boston",
                event_date=date(2026, 9, 1),
                guest_count=24,
                child_age=8,
                total_budget=Decimal("1200.00"),
            ),
            resource_id=metadata.resource_id,
        ),
    )


def test_build_v02_planner_uses_live_constraint_extractor_and_bm25_retriever() -> None:
    provider = FakeLLMProvider([GenerationResponse(text="", structured_output={"constraints": []})])

    planner = v02_eval.build_v02_planner(corpus=_corpus(), provider=provider)

    assert isinstance(planner, EvidenceGroundedPlanner)
    assert isinstance(planner._constraint_extractor, LLMConstraintExtractor)
    assert isinstance(planner._evidence_retriever, BM25EvidenceRetriever)
    assert planner._constraint_extractor._provider is provider


def test_build_v02_evaluation_report_identifies_live_llm_configuration() -> None:
    metrics = V02EvaluationMetrics(
        scenario_count=0,
        feasibility_accuracy=0.0,
        hard_constraint_validity=0.0,
        grounded_decision_accuracy=None,
        source_attribution_accuracy=None,
        derived_constraint_accuracy=None,
        unsupported_claim_rate=None,
        wrong_source_version_rate=None,
        no_feasible_plan_accuracy=None,
        mean_latency_ms=0.0,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        estimated_cost_usd=None,
    )
    report = v02_eval.build_v02_evaluation_report(
        root=Path(__file__).resolve().parents[1],
        metrics=metrics,
        scenario_results=(),
    )

    assert report.evaluation_variant == "bm25 + live_ollama_constraint_extractor"
    assert any("live Ollama-backed constraint extractor" in note for note in report.notes)
    assert all("offline" not in note.casefold() for note in report.notes)


def test_live_constraint_extractor_surfaces_typed_provider_failures() -> None:
    provider = FailingFakeLLMProvider(TimeoutError("slow"))
    extractor = v02_eval.build_live_constraint_extractor(provider=provider)

    with pytest.raises(LLMConstraintExtractorProviderError, match="TimeoutError"):
        extractor.extract(_extraction_input())

    assert len(provider.requests) == 1


def test_live_constraint_extractor_rejects_missing_structured_output() -> None:
    extractor = v02_eval.build_live_constraint_extractor(
        provider=FakeLLMProvider([GenerationResponse(text="no structured output")])
    )

    with pytest.raises(LLMConstraintExtractorOutputError, match="no structured output"):
        extractor.extract(_extraction_input())
