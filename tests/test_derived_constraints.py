from __future__ import annotations

from datetime import date
from math import ceil

import pytest

from partypilot.application.derived_constraints import (
    DerivedConstraintContext,
    DerivedConstraintError,
    derive_constraint,
)
from partypilot.domain.constraints import Constraint, ConstraintOperator, ConstraintType
from partypilot.domain.evidence import DerivationMethod, EvidenceState, Provenance
from partypilot.ports.constraint_extractor import ExtractedConstraint


def _extracted_ratio(value: str = "1/5") -> ExtractedConstraint:
    return ExtractedConstraint(
        constraint=Constraint(
            identifier="extracted:supervision-ratio",
            key="adult_child_ratio",
            operator=ConstraintOperator.EQ,
            value=value,
            constraint_type=ConstraintType.HARD,
            description="One adult is required for every five children.",
        ),
        provenance=Provenance(
            source_document_id="doc-supervision-current",
            source_chunk_id="doc-supervision-current:chunk-1",
            resource_id="venue-sunrise",
            source_version="2.0",
            effective_date=date(2026, 1, 1),
            derivation_method=DerivationMethod.LLM_EXTRACTED,
            derivation_explanation="Extracted the stated adult-to-child supervision ratio.",
        ),
        confidence=0.98,
    )


def test_derives_minimum_adults_deterministically() -> None:
    result = derive_constraint(
        _extracted_ratio(),
        evidence_state=EvidenceState.SUPPORTED,
        context=DerivedConstraintContext(child_count=24),
    )

    assert len(result.constraints) == 1
    derived = result.constraints[0]
    assert derived.constraint.key == "minimum_adults"
    assert derived.constraint.operator is ConstraintOperator.GTE
    assert derived.constraint.value == 5
    assert derived.constraint.constraint_type is ConstraintType.DERIVED
    assert derived.constraint.provenance is not None
    assert derived.constraint.provenance.source_constraint_ids == ("extracted:supervision-ratio",)
    assert "24 children" in derived.constraint.provenance.derivation_explanation
    assert derived.provenance[0].source_document_id == "doc-supervision-current"
    assert derived.provenance[0].source_chunk_id == "doc-supervision-current:chunk-1"
    assert derived.provenance[0].source_version == "2.0"
    assert derived.provenance[0].resource_id == "venue-sunrise"
    assert derived.provenance[0].derivation_method is DerivationMethod.DETERMINISTIC


@pytest.mark.parametrize(
    "state",
    [
        EvidenceState.INSUFFICIENT_EVIDENCE,
        EvidenceState.CONFLICTED,
        EvidenceState.UNSUPPORTED,
    ],
)
def test_non_supported_evidence_cannot_produce_definitive_constraint(
    state: EvidenceState,
) -> None:
    result = derive_constraint(
        _extracted_ratio(),
        evidence_state=state,
        context=DerivedConstraintContext(child_count=24),
    )

    assert result.constraints == ()
    assert state.value in result.explanation


def test_property_ratio_one_per_five_always_rounds_up() -> None:
    extracted = _extracted_ratio()
    for child_count in range(1, 201):
        result = derive_constraint(
            extracted,
            evidence_state=EvidenceState.SUPPORTED,
            context=DerivedConstraintContext(child_count=child_count),
        )
        assert result.constraints[0].constraint.value == ceil(child_count / 5)


def test_property_required_adults_is_monotonic() -> None:
    extracted = _extracted_ratio()
    previous = 0
    for child_count in range(1, 201):
        result = derive_constraint(
            extracted,
            evidence_state=EvidenceState.SUPPORTED,
            context=DerivedConstraintContext(child_count=child_count),
        )
        current = result.constraints[0].constraint.value
        assert isinstance(current, int)
        assert current >= previous
        previous = current


@pytest.mark.parametrize("value", ["0", "-1/5", "bogus", "1/0"])
def test_invalid_ratio_is_rejected(value: str) -> None:
    with pytest.raises(DerivedConstraintError):
        derive_constraint(
            _extracted_ratio(value),
            evidence_state=EvidenceState.SUPPORTED,
            context=DerivedConstraintContext(child_count=10),
        )


def test_unsupported_rule_is_rejected() -> None:
    extracted = _extracted_ratio()
    unsupported = extracted.model_copy(
        update={
            "constraint": extracted.constraint.model_copy(update={"key": "minimum_notice_days"})
        }
    )
    with pytest.raises(DerivedConstraintError, match="unsupported"):
        derive_constraint(
            unsupported,
            evidence_state=EvidenceState.SUPPORTED,
            context=DerivedConstraintContext(child_count=10),
        )
