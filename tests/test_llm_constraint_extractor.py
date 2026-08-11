from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from partypilot.adapters.llm_constraint_extractor import (
    LLMConstraintExtractor,
    LLMConstraintExtractorOutputError,
    LLMConstraintExtractorProviderError,
)
from partypilot.domain.constraints import ConstraintOperator, ConstraintType
from partypilot.domain.evidence import DerivationMethod
from partypilot.domain.evidence_corpus import (
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


def _extractor_input(
    *,
    evidence_text: str,
    document_id: str,
    resource_id: str,
    document_type: EvidenceDocumentType,
    version: str = "2.0",
) -> ConstraintExtractionInput:
    return ConstraintExtractionInput(
        evidence_text=evidence_text,
        evidence_metadata=EvidenceDocumentMetadata(
            document_id=document_id,
            resource_id=resource_id,
            document_type=document_type,
            version=version,
            effective_date=date(2026, 1, 1),
            status=EvidenceDocumentStatus.CURRENT,
        ),
        chunk_id=f"{document_id}#chunk-1",
        planning_context=ConstraintExtractionContext(
            request=PartyRequest(
                location="Boston",
                event_date=date(2026, 9, 1),
                guest_count=24,
                child_age=8,
                total_budget=Decimal("1200.00"),
            ),
            resource_id=resource_id,
        ),
    )


def _input() -> ConstraintExtractionInput:
    return _extractor_input(
        evidence_text="One adult is required for every five children.",
        document_id="doc-supervision-v2",
        resource_id="venue-alpha",
        document_type=EvidenceDocumentType.SUPERVISION_REQUIREMENTS,
    )


def test_extracts_ratio_rule_and_preserves_provenance_without_calculating_final_count() -> None:
    provider = FakeLLMProvider(
        [
            GenerationResponse(
                text="",
                structured_output={
                    "constraints": [
                        {
                            "identifier": "policy-adult-child-ratio",
                            "key": "adult_child_ratio",
                            "operator": "eq",
                            "value": "1/5",
                            "constraint_type": "HARD",
                            "description": "One adult is required for every five children.",
                            "confidence": 0.98,
                            "derivation_explanation": "Directly stated supervision ratio.",
                        }
                    ]
                },
            )
        ]
    )
    extractor = LLMConstraintExtractor(provider, timeout_seconds=7.5)

    result = extractor.extract(_input())

    assert len(result.constraints) == 1
    extracted = result.constraints[0]
    assert extracted.constraint.key == "adult_child_ratio"
    assert extracted.constraint.operator is ConstraintOperator.EQ
    assert extracted.constraint.value == "1/5"
    assert extracted.constraint.constraint_type is ConstraintType.HARD
    assert extracted.provenance.source_document_id == "doc-supervision-v2"
    assert extracted.provenance.source_chunk_id == "doc-supervision-v2#chunk-1"
    assert extracted.provenance.source_version == "2.0"
    assert extracted.provenance.resource_id == "venue-alpha"
    assert extracted.provenance.derivation_method is DerivationMethod.LLM_EXTRACTED
    assert extracted.confidence == 0.98
    request, timeout = provider.requests[0]
    assert timeout == 7.5
    assert request.structured_output is not None
    system_prompt = request.system_prompt or ""
    assert 'A "constraint" includes prohibitions, requirements, limitations' in system_prompt
    assert "The only legal constraint_type values are HARD, SOFT, and DERIVED." in system_prompt
    assert "Never emit alternative labels such as RECOMMENDED" in system_prompt
    assert (
        "The only legal operator values are eq, ne, lt, lte, gt, gte, in, not_in," in system_prompt
    )
    assert "and contains." in system_prompt
    assert (
        'The JSON value "constraint_type": "RECOMMENDED" is invalid and must never be emitted.'
        in system_prompt
    )
    assert "Do not return an empty" in system_prompt
    assert "constraints array when the evidence contains a planning-relevant rule." in system_prompt
    assert 'If no explicit planning rule exists, return {"constraints":[]}' in system_prompt
    assert "Example 1 - allergen/shared kitchen" in system_prompt
    assert "Example 2 - vegan advance notice" in system_prompt
    assert "Example 3 - accessibility" in system_prompt
    assert "Example 4 - supervision" in system_prompt
    assert '"constraints"' in (request.system_prompt or "")
    assert "minimum_adults" not in result.model_dump_json()


def test_supports_empty_extraction() -> None:
    extractor = LLMConstraintExtractor(
        FakeLLMProvider([GenerationResponse(text="", structured_output={"constraints": []})])
    )
    assert extractor.extract(_input()).constraints == ()


@pytest.mark.parametrize(
    "evidence_text,document_id,resource_id,document_type,structured_output,expected_key,expected_operator,expected_value,expected_description",
    [
        (
            (
                "Family Table peanut and tree nut allergen policy: foods containing peanuts and "
                "tree nuts are prepared in a shared kitchen."
            ),
            "doc-family-allergen-current",
            "caterer-family-table",
            EvidenceDocumentType.ALLERGEN_POLICY,
            {
                "constraints": [
                    {
                        "identifier": "policy-family-allergen",
                        "key": "allergen_policy",
                        "operator": "eq",
                        "value": (
                            "foods containing peanuts and tree nuts are prepared in a shared "
                            "kitchen"
                        ),
                        "constraint_type": "HARD",
                        "description": (
                            "Foods containing peanuts and tree nuts are prepared in a shared "
                            "kitchen."
                        ),
                        "confidence": 0.94,
                        "derivation_explanation": "Directly stated allergen policy.",
                    }
                ]
            },
            "allergen_policy",
            ConstraintOperator.EQ,
            "foods containing peanuts and tree nuts are prepared in a shared kitchen",
            ("Foods containing peanuts and tree nuts are prepared in a shared kitchen."),
        ),
        (
            (
                "Family Table offers gluten-free menu selections, but food is prepared in a "
                "shared kitchen and is not certified free from gluten cross-contact."
            ),
            "doc-family-gluten-current",
            "caterer-family-table",
            EvidenceDocumentType.ALLERGEN_POLICY,
            {
                "constraints": [
                    {
                        "identifier": "policy-family-gluten-cross-contact",
                        "key": "cross_contact_risk",
                        "operator": "eq",
                        "value": "present",
                        "constraint_type": "HARD",
                        "description": (
                            "Food is prepared in a shared kitchen and is not certified free from "
                            "gluten cross-contact."
                        ),
                        "confidence": 0.91,
                        "derivation_explanation": "Directly stated cross-contact risk.",
                    }
                ]
            },
            "cross_contact_risk",
            ConstraintOperator.EQ,
            "present",
            (
                "Food is prepared in a shared kitchen and is not certified free from gluten "
                "cross-contact."
            ),
        ),
        (
            (
                "Family Table menu includes vegan entree and dessert options when requested at "
                "least seven days in advance."
            ),
            "doc-family-vegan-current",
            "caterer-family-table",
            EvidenceDocumentType.VENUE_POLICY,
            {
                "constraints": [
                    {
                        "identifier": "policy-family-vegan-notice",
                        "key": "vegan_notice_days",
                        "operator": "gte",
                        "value": 7,
                        "constraint_type": "HARD",
                        "description": (
                            "Vegan entree and dessert options require at least seven days of "
                            "advance notice."
                        ),
                        "confidence": 0.9,
                        "derivation_explanation": "Directly stated advance-notice requirement.",
                    }
                ]
            },
            "vegan_notice_days",
            ConstraintOperator.GTE,
            7,
            ("Vegan entree and dessert options require at least seven days of advance notice."),
        ),
        (
            "Brooklyn Loft provides step-free wheelchair access and an accessible restroom.",
            "doc-loft-accessibility-current",
            "venue-brooklyn-loft",
            EvidenceDocumentType.ACCESSIBILITY_GUIDANCE,
            {
                "constraints": [
                    {
                        "identifier": "policy-loft-accessibility",
                        "key": "accessible_restroom",
                        "operator": "eq",
                        "value": True,
                        "constraint_type": "HARD",
                        "description": "Brooklyn Loft provides an accessible restroom.",
                        "confidence": 0.95,
                        "derivation_explanation": "Directly stated accessibility guidance.",
                    }
                ]
            },
            "accessible_restroom",
            ConstraintOperator.EQ,
            True,
            "Brooklyn Loft provides an accessible restroom.",
        ),
        (
            "One adult is required for every five children.",
            "doc-supervision-v2",
            "venue-alpha",
            EvidenceDocumentType.SUPERVISION_REQUIREMENTS,
            {
                "constraints": [
                    {
                        "identifier": "policy-adult-child-ratio",
                        "key": "adult_child_ratio",
                        "operator": "eq",
                        "value": "1/5",
                        "constraint_type": "HARD",
                        "description": "One adult is required for every five children.",
                        "confidence": 0.98,
                        "derivation_explanation": "Directly stated supervision ratio.",
                    }
                ]
            },
            "adult_child_ratio",
            ConstraintOperator.EQ,
            "1/5",
            "One adult is required for every five children.",
        ),
    ],
)
def test_extracts_representative_typed_constraints_with_provenance(
    evidence_text: str,
    document_id: str,
    resource_id: str,
    document_type: EvidenceDocumentType,
    structured_output: Any,
    expected_key: str,
    expected_operator: ConstraintOperator,
    expected_value: object,
    expected_description: str,
) -> None:
    extractor = LLMConstraintExtractor(
        FakeLLMProvider([GenerationResponse(text="", structured_output=structured_output)])
    )
    result = extractor.extract(
        _extractor_input(
            evidence_text=evidence_text,
            document_id=document_id,
            resource_id=resource_id,
            document_type=document_type,
        )
    )

    assert len(result.constraints) == 1
    extracted = result.constraints[0]
    assert extracted.constraint.key == expected_key
    assert extracted.constraint.operator is expected_operator
    assert extracted.constraint.value == expected_value
    assert extracted.constraint.description == expected_description
    assert extracted.provenance.source_document_id == document_id
    assert extracted.provenance.source_chunk_id == f"{document_id}#chunk-1"
    assert extracted.provenance.resource_id == resource_id
    assert extracted.provenance.derivation_method is DerivationMethod.LLM_EXTRACTED
    assert extracted.confidence is not None


def test_ignores_extra_top_level_fields_while_preserving_valid_constraints() -> None:
    provider = FakeLLMProvider(
        [
            GenerationResponse(
                text="",
                structured_output={
                    "constraints": [
                        {
                            "identifier": "policy-adult-child-ratio",
                            "key": "adult_child_ratio",
                            "operator": "eq",
                            "value": "1/5",
                            "constraint_type": "HARD",
                            "description": "One adult is required for every five children.",
                            "confidence": 0.98,
                            "derivation_explanation": "Directly stated supervision ratio.",
                        }
                    ],
                    "outside_food_rules": "only from licensed caterers",
                    "birthday_cake": "may be brought by the host",
                },
            )
        ]
    )
    extractor = LLMConstraintExtractor(provider)

    result = extractor.extract(_input())

    assert len(result.constraints) == 1
    assert result.constraints[0].constraint.key == "adult_child_ratio"
    assert result.constraints[0].provenance.source_document_id == "doc-supervision-v2"
    assert "outside_food_rules" not in result.model_dump_json()
    assert "birthday_cake" not in result.model_dump_json()


def test_narrative_json_without_constraints_does_not_become_trusted_constraints() -> None:
    extractor = LLMConstraintExtractor(
        FakeLLMProvider(
            [
                GenerationResponse(
                    text="",
                    structured_output={
                        "accessibility": "step-free_wheelchair_access",
                        "accessible_restroom": True,
                        "seating_layout_assistance": "contact_staff_in_advance",
                    },
                )
            ]
        )
    )

    result = extractor.extract(
        _extractor_input(
            evidence_text=(
                "Brooklyn Loft provides step-free wheelchair access and an accessible restroom."
            ),
            document_id="doc-loft-accessibility-current",
            resource_id="venue-brooklyn-loft",
            document_type=EvidenceDocumentType.ACCESSIBILITY_GUIDANCE,
        )
    )

    assert result.constraints == ()


def test_rejects_extra_fields_inside_individual_constraint_objects() -> None:
    extractor = LLMConstraintExtractor(
        FakeLLMProvider(
            [
                GenerationResponse(
                    text="",
                    structured_output={
                        "constraints": [
                            {
                                "identifier": "policy-adult-child-ratio",
                                "key": "adult_child_ratio",
                                "operator": "eq",
                                "value": "1/5",
                                "constraint_type": "HARD",
                                "description": "One adult is required for every five children.",
                                "confidence": 0.98,
                                "derivation_explanation": "Directly stated supervision ratio.",
                                "outside_food_rules": "only from licensed caterers",
                            }
                        ]
                    },
                )
            ]
        )
    )

    with pytest.raises(LLMConstraintExtractorOutputError, match="Extra inputs are not permitted"):
        extractor.extract(_input())


def test_rejects_derived_constraint_from_llm() -> None:
    extractor = LLMConstraintExtractor(
        FakeLLMProvider(
            [
                GenerationResponse(
                    text="",
                    structured_output={
                        "constraints": [
                            {
                                "identifier": "derived-adults",
                                "key": "minimum_adults",
                                "operator": "gte",
                                "value": 5,
                                "constraint_type": "DERIVED",
                                "description": "Calculated requirement.",
                                "derivation_explanation": "Calculated from ratio and guest count.",
                            }
                        ]
                    },
                )
            ]
        )
    )
    with pytest.raises(LLMConstraintExtractorOutputError, match="must not emit derived"):
        extractor.extract(_input())


def test_translates_provider_failure() -> None:
    extractor = LLMConstraintExtractor(FailingFakeLLMProvider(TimeoutError("slow")))
    with pytest.raises(LLMConstraintExtractorProviderError, match="TimeoutError"):
        extractor.extract(_input())


def test_rejects_missing_or_invalid_structured_output() -> None:
    missing = LLMConstraintExtractor(FakeLLMProvider([GenerationResponse(text="no json")]))
    with pytest.raises(LLMConstraintExtractorOutputError, match="no structured output"):
        missing.extract(_input())

    invalid = LLMConstraintExtractor(
        FakeLLMProvider([GenerationResponse(text="", structured_output={"constraints": "bad"})])
    )
    with pytest.raises(LLMConstraintExtractorOutputError, match="invalid extraction output"):
        invalid.extract(_input())


def test_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        LLMConstraintExtractor(FakeLLMProvider([]), timeout_seconds=0)
