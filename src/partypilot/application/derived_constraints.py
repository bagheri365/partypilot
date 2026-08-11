from __future__ import annotations

from fractions import Fraction
from math import ceil
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from partypilot.domain.constraints import (
    Constraint,
    ConstraintOperator,
    ConstraintProvenance,
    ConstraintType,
)
from partypilot.domain.evidence import DerivationMethod, EvidenceState, Provenance
from partypilot.ports.constraint_extractor import ExtractedConstraint

PositiveChildCount = Annotated[int, Field(gt=0)]


class DerivedConstraintError(ValueError):
    """Raised when a supported extracted rule cannot be derived deterministically."""


class DerivedConstraintContext(BaseModel):
    """Concrete deterministic inputs needed to derive request-specific constraints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    child_count: PositiveChildCount


class DerivedConstraint(BaseModel):
    """A derived PartyPilot constraint plus its retained evidence provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    constraint: Constraint
    provenance: tuple[Provenance, ...] = Field(min_length=1)


class DerivedConstraintResult(BaseModel):
    """Result of deterministic derivation from one evidence-backed extracted rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_state: EvidenceState
    constraints: tuple[DerivedConstraint, ...] = ()
    explanation: str


def derive_constraint(
    extracted: ExtractedConstraint,
    *,
    evidence_state: EvidenceState,
    context: DerivedConstraintContext,
) -> DerivedConstraintResult:
    """Derive request-specific constraints from a supported extracted policy rule.

    Only clearly supported evidence can yield a definitive derived constraint. Other
    evidence states return an explicit empty result instead of guessing.
    """

    if evidence_state is not EvidenceState.SUPPORTED:
        return DerivedConstraintResult(
            evidence_state=evidence_state,
            explanation=(
                f"No definitive derived constraint produced because evidence state is "
                f"{evidence_state.value}."
            ),
        )

    rule = extracted.constraint
    if rule.key != "adult_child_ratio":
        raise DerivedConstraintError(f"unsupported deterministic derivation rule: {rule.key}")
    if rule.operator is not ConstraintOperator.EQ:
        raise DerivedConstraintError("adult_child_ratio must use the eq operator")
    if not isinstance(rule.value, str):
        raise DerivedConstraintError("adult_child_ratio must be encoded as a ratio string")

    ratio = _parse_positive_ratio(rule.value)
    required_adults = ceil(context.child_count * ratio)
    explanation = (
        f"Applied extracted adult-to-child ratio {ratio.numerator}/{ratio.denominator} "
        f"to {context.child_count} children and rounded up deterministically to "
        f"{required_adults} required adults."
    )

    derived = Constraint(
        identifier=f"derived:{rule.identifier}:minimum_adults:{context.child_count}",
        key="minimum_adults",
        operator=ConstraintOperator.GTE,
        value=required_adults,
        constraint_type=ConstraintType.DERIVED,
        description=f"At least {required_adults} adults are required by the supervision policy.",
        provenance=ConstraintProvenance(
            source_constraint_ids=(rule.identifier,),
            derivation_explanation=explanation,
        ),
    )

    source = extracted.provenance
    deterministic_provenance = Provenance(
        source_document_id=source.source_document_id,
        source_chunk_id=source.source_chunk_id,
        resource_id=source.resource_id,
        source_version=source.source_version,
        effective_date=source.effective_date,
        derivation_method=DerivationMethod.DETERMINISTIC,
        derivation_explanation=explanation,
    )

    return DerivedConstraintResult(
        evidence_state=evidence_state,
        constraints=(
            DerivedConstraint(
                constraint=derived,
                provenance=(deterministic_provenance,),
            ),
        ),
        explanation=explanation,
    )


def _parse_positive_ratio(value: str) -> Fraction:
    try:
        ratio = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise DerivedConstraintError(f"invalid adult_child_ratio: {value!r}") from exc
    if ratio <= 0:
        raise DerivedConstraintError("adult_child_ratio must be positive")
    return ratio
