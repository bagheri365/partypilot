from datetime import date

import pytest
from pydantic import ValidationError

from partypilot.domain import (
    DerivationMethod,
    EvidenceReference,
    EvidenceState,
    Provenance,
)


def test_evidence_states_are_stable() -> None:
    assert {state.value for state in EvidenceState} == {
        "SUPPORTED",
        "CONFLICTED",
        "INSUFFICIENT_EVIDENCE",
        "UNSUPPORTED",
    }


def test_derivation_methods_are_stable() -> None:
    assert {method.value for method in DerivationMethod} == {
        "deterministic",
        "llm_extracted",
        "llm_inferred",
    }


def test_provenance_supports_document_chunk_and_version_metadata() -> None:
    provenance = Provenance(
        source_document_id="venue-policy-v3",
        source_chunk_id="chunk-17",
        source_version="3.2",
        effective_date=date(2026, 7, 1),
        derivation_method=DerivationMethod.DETERMINISTIC,
        derivation_explanation="Parsed from the structured capacity field.",
    )

    assert provenance.source_document_id == "venue-policy-v3"
    assert provenance.source_chunk_id == "chunk-17"
    assert provenance.source_version == "3.2"
    assert provenance.effective_date == date(2026, 7, 1)


def test_provenance_supports_resource_vendor_id() -> None:
    provenance = Provenance(
        resource_id="vendor-caterer-42",
        derivation_method=DerivationMethod.LLM_EXTRACTED,
        derivation_explanation="Extracted the allergy policy from vendor text.",
    )

    assert provenance.resource_id == "vendor-caterer-42"


def test_provenance_requires_at_least_one_source_reference() -> None:
    with pytest.raises(ValidationError, match="must reference"):
        Provenance(
            derivation_method=DerivationMethod.DETERMINISTIC,
            derivation_explanation="Calculated from known structured values.",
        )


def test_chunk_reference_requires_document_reference() -> None:
    with pytest.raises(ValidationError, match="requires source_document_id"):
        Provenance(
            source_chunk_id="chunk-7",
            derivation_method=DerivationMethod.LLM_INFERRED,
            derivation_explanation="Inferred from a source passage.",
        )


def test_evidence_reference_requires_provenance() -> None:
    with pytest.raises(ValidationError):
        EvidenceReference(
            evidence_id="evidence-1",
            state=EvidenceState.INSUFFICIENT_EVIDENCE,
            provenance=(),
        )


def test_evidence_reference_can_retain_multiple_sources() -> None:
    evidence = EvidenceReference(
        evidence_id="evidence-availability-1",
        state=EvidenceState.CONFLICTED,
        provenance=(
            Provenance(
                source_document_id="availability-sheet",
                derivation_method=DerivationMethod.DETERMINISTIC,
                derivation_explanation="Structured availability marks the date open.",
            ),
            Provenance(
                resource_id="venue-12",
                derivation_method=DerivationMethod.LLM_EXTRACTED,
                derivation_explanation="Vendor policy text states the date is blocked.",
            ),
        ),
    )

    assert evidence.state is EvidenceState.CONFLICTED
    assert len(evidence.provenance) == 2


def test_evidence_models_are_frozen_and_reject_unknown_fields() -> None:
    provenance = Provenance(
        resource_id="venue-1",
        derivation_method=DerivationMethod.DETERMINISTIC,
        derivation_explanation="Copied from structured venue data.",
    )

    with pytest.raises(ValidationError):
        provenance.resource_id = "venue-2"

    with pytest.raises(ValidationError):
        Provenance.model_validate(
            {
                "resource_id": "venue-1",
                "derivation_method": DerivationMethod.DETERMINISTIC,
                "derivation_explanation": "Copied from structured venue data.",
                "unexpected": "nope",
            }
        )
