from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from partypilot.domain.constraints import Constraint, ConstraintOperator, ConstraintType
from partypilot.domain.evidence import DerivationMethod, Provenance
from partypilot.domain.evidence_corpus import (
    EvidenceDocumentMetadata,
    EvidenceDocumentStatus,
    EvidenceDocumentType,
)
from partypilot.domain.party_request import PartyRequest
from partypilot.ports.constraint_extractor import (
    ConstraintExtractionContext,
    ConstraintExtractionInput,
    ConstraintExtractionResult,
    ConstraintExtractor,
    ExtractedConstraint,
    FailingFakeConstraintExtractor,
    FakeConstraintExtractor,
)


def _request() -> PartyRequest:
    return PartyRequest(
        location="Raleigh",
        event_date=date(2026, 9, 5),
        guest_count=24,
        total_budget=Decimal("1200"),
    )


def _metadata() -> EvidenceDocumentMetadata:
    return EvidenceDocumentMetadata(
        document_id="doc-supervision-current",
        resource_id="venue-sunrise",
        document_type=EvidenceDocumentType.SUPERVISION_REQUIREMENTS,
        version="2.0",
        effective_date=date(2026, 1, 1),
        status=EvidenceDocumentStatus.CURRENT,
    )


def _input() -> ConstraintExtractionInput:
    return ConstraintExtractionInput(
        evidence_text="One adult is required for every five children.",
        evidence_metadata=_metadata(),
        chunk_id="doc-supervision-current#0",
        planning_context=ConstraintExtractionContext(
            request=_request(),
            resource_id="venue-sunrise",
        ),
    )


def _result() -> ConstraintExtractionResult:
    constraint = Constraint(
        identifier="extracted-adult-child-ratio",
        key="adult_child_ratio",
        operator=ConstraintOperator.EQ,
        value="1/5",
        constraint_type=ConstraintType.HARD,
        description="One adult is required for every five children.",
    )
    provenance = Provenance(
        source_document_id="doc-supervision-current",
        source_chunk_id="doc-supervision-current#0",
        resource_id="venue-sunrise",
        source_version="2.0",
        effective_date=date(2026, 1, 1),
        derivation_method=DerivationMethod.LLM_EXTRACTED,
        derivation_explanation="Extracted directly from supervision policy text.",
    )
    return ConstraintExtractionResult(
        constraints=(
            ExtractedConstraint(
                constraint=constraint,
                provenance=provenance,
                confidence=0.94,
            ),
        )
    )


def test_extraction_input_carries_evidence_metadata_and_planning_context() -> None:
    extraction_input = _input()

    assert extraction_input.evidence_metadata.document_id == "doc-supervision-current"
    assert extraction_input.planning_context.request.guest_count == 24
    assert extraction_input.planning_context.resource_id == "venue-sunrise"


def test_extracted_constraint_preserves_typed_constraint_provenance_and_confidence() -> None:
    extracted = _result().constraints[0]

    assert extracted.constraint.key == "adult_child_ratio"
    assert extracted.provenance.source_chunk_id == "doc-supervision-current#0"
    assert extracted.confidence == pytest.approx(0.94)


def test_confidence_is_optional() -> None:
    base = _result().constraints[0]
    extracted = ExtractedConstraint(
        constraint=base.constraint,
        provenance=base.provenance,
    )
    assert extracted.confidence is None


def test_confidence_must_be_bounded() -> None:
    base = _result().constraints[0]
    with pytest.raises(ValidationError):
        ExtractedConstraint(
            constraint=base.constraint,
            provenance=base.provenance,
            confidence=1.01,
        )


def test_fake_extractor_is_structurally_compatible_with_port() -> None:
    extractor: ConstraintExtractor = FakeConstraintExtractor([_result()])
    extraction_input = _input()

    result = extractor.extract(extraction_input)

    assert result.constraints[0].constraint.identifier == "extracted-adult-child-ratio"
    assert isinstance(extractor, FakeConstraintExtractor)
    assert extractor.inputs == [extraction_input]


def test_fake_extractor_uses_results_in_deterministic_queue_order() -> None:
    empty = ConstraintExtractionResult()
    extractor = FakeConstraintExtractor([empty, _result()])

    assert extractor.extract(_input()) == empty
    assert len(extractor.extract(_input()).constraints) == 1


def test_fake_extractor_raises_when_queue_is_exhausted() -> None:
    extractor = FakeConstraintExtractor([])
    with pytest.raises(RuntimeError, match="no queued result"):
        extractor.extract(_input())


def test_failing_fake_extractor_reproduces_failure() -> None:
    error = TimeoutError("extractor timed out")
    extractor = FailingFakeConstraintExtractor(error)
    extraction_input = _input()

    with pytest.raises(TimeoutError, match="extractor timed out"):
        extractor.extract(extraction_input)
    assert extractor.inputs == [extraction_input]


def test_extraction_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ConstraintExtractionContext(request=_request(), unexpected=True)  # type: ignore[call-arg]
