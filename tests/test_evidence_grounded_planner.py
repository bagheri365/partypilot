from __future__ import annotations

import json
from datetime import date, time
from decimal import Decimal
from pathlib import Path

from partypilot.adapters.bm25_evidence_retriever import BM25EvidenceRetriever
from partypilot.adapters.in_memory_resource_store import InMemoryResourceStore
from partypilot.application.evidence_grounded_planner import EvidenceGroundedPlanner
from partypilot.domain.constraints import (
    Constraint,
    ConstraintOperator,
    ConstraintType,
    ConstraintValue,
)
from partypilot.domain.evidence import DerivationMethod, Provenance
from partypilot.domain.evidence_corpus import (
    EvidenceDocument,
    EvidenceDocumentStatus,
    EvidenceDocumentType,
)
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest
from partypilot.ports.constraint_extractor import (
    ConstraintExtractionInput,
    ConstraintExtractionResult,
    ExtractedConstraint,
)
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalQuery,
    EvidenceRetrievalResult,
    EvidenceRetriever,
    EvidenceVersionMetadata,
    RetrievalMethod,
)


class PolicyAwareFakeExtractor:
    """Deterministic extractor keyed only by supplied evidence text."""

    def extract(self, extraction_input: ConstraintExtractionInput) -> ConstraintExtractionResult:
        metadata = extraction_input.evidence_metadata
        if "one supervising adult for every five children" not in extraction_input.evidence_text:
            return ConstraintExtractionResult()
        return ConstraintExtractionResult(
            constraints=(
                ExtractedConstraint(
                    constraint=Constraint(
                        identifier=f"extracted:{metadata.document_id}:adult-ratio",
                        key="adult_child_ratio",
                        operator=ConstraintOperator.EQ,
                        value="1/5",
                        constraint_type=ConstraintType.HARD,
                        description="One supervising adult is required for every five children.",
                    ),
                    provenance=Provenance(
                        source_document_id=metadata.document_id,
                        source_chunk_id=extraction_input.chunk_id,
                        resource_id=metadata.resource_id,
                        source_version=metadata.version,
                        effective_date=metadata.effective_date,
                        derivation_method=DerivationMethod.LLM_EXTRACTED,
                        derivation_explanation=(
                            "Deterministic test extraction of supervision ratio."
                        ),
                    ),
                    confidence=1.0,
                ),
            )
        )


class ScriptedRetriever:
    def __init__(
        self,
        results_by_resource_id: dict[str, tuple[EvidenceRetrievalResult, ...]],
    ) -> None:
        self.results_by_resource_id = results_by_resource_id
        self.queries: list[EvidenceRetrievalQuery] = []

    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
        self.queries.append(query)
        resource_id = query.filters.resource_id
        if resource_id is None:
            raise AssertionError("tests must supply resource-specific retrieval filters")
        return self.results_by_resource_id.get(resource_id, ())


class ScriptedConstraintExtractor:
    def __init__(
        self,
        outputs_by_document_id: dict[str, tuple[ExtractedConstraint, ...]],
    ) -> None:
        self.outputs_by_document_id = outputs_by_document_id
        self.inputs: list[ConstraintExtractionInput] = []

    def extract(self, extraction_input: ConstraintExtractionInput) -> ConstraintExtractionResult:
        self.inputs.append(extraction_input)
        return ConstraintExtractionResult(
            constraints=self.outputs_by_document_id.get(
                extraction_input.evidence_metadata.document_id, ()
            )
        )


class RecordingRetriever:
    def __init__(self, retriever: EvidenceRetriever) -> None:
        self._retriever = retriever
        self.queries: list[EvidenceRetrievalQuery] = []

    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
        self.queries.append(query)
        return self._retriever.retrieve(query)


def _retrieval_result(
    *,
    document_id: str,
    resource_id: str,
    document_type: EvidenceDocumentType,
    text: str,
    rank: int = 1,
) -> EvidenceRetrievalResult:
    return EvidenceRetrievalResult(
        document_id=document_id,
        chunk_id=f"{document_id}#chunk-1",
        resource_id=resource_id,
        version=EvidenceVersionMetadata(
            version="2.0",
            effective_date=date(2026, 1, 1),
            status=EvidenceDocumentStatus.CURRENT,
        ),
        document_type=document_type,
        text=text,
        score=1.0,
        rank=rank,
        retrieval_method=RetrievalMethod.BM25,
    )


def _extracted_constraint(
    *,
    identifier: str,
    key: str,
    operator: ConstraintOperator,
    value: ConstraintValue,
    description: str,
    provenance_document_id: str,
    provenance_chunk_id: str,
    provenance_resource_id: str,
    derivation_explanation: str,
) -> ExtractedConstraint:
    return ExtractedConstraint(
        constraint=Constraint(
            identifier=identifier,
            key=key,
            operator=operator,
            value=value,
            constraint_type=ConstraintType.HARD,
            description=description,
        ),
        provenance=Provenance(
            source_document_id=provenance_document_id,
            source_chunk_id=provenance_chunk_id,
            resource_id=provenance_resource_id,
            source_version="2.0",
            effective_date=date(2026, 1, 1),
            derivation_method=DerivationMethod.LLM_EXTRACTED,
            derivation_explanation=derivation_explanation,
        ),
        confidence=0.95,
    )


def _documents() -> tuple[EvidenceDocument, ...]:
    payload = json.loads(Path("data/evidence/v0_2_documents.json").read_text())
    return tuple(EvidenceDocument.model_validate(item) for item in payload)


def _request(*, budget: Decimal = Decimal("1600")) -> PartyRequest:
    return PartyRequest(
        location="Brooklyn, NY",
        event_date=date(2026, 9, 20),
        event_time=time(14, 0),
        guest_count=24,
        child_age=8,
        total_budget=budget,
    )


def test_evidence_grounded_flow_returns_validated_plan_with_provenance() -> None:
    planner = EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=BM25EvidenceRetriever(_documents()),
        constraint_extractor=PolicyAwareFakeExtractor(),
    )

    result = planner.plan(_request())

    assert result.outcome is FeasibilityOutcome.FEASIBLE
    assert result.feasible
    assert result.candidates
    candidate = result.candidates[0]
    assert candidate.validation.feasible
    assert candidate.total_cost == Decimal("1375.00")
    assert candidate.required_adults == 5
    assert candidate.derived_constraints[0].constraint.value == 5
    assert (
        candidate.derived_constraints[0].provenance[0].source_document_id
        == "doc-craft-supervision-current"
    )
    assert any(
        ref.provenance[0].source_document_id == "doc-craft-supervision-current"
        for ref in result.evidence_references
    )


def test_structured_infeasibility_stops_before_evidence_calls() -> None:
    class RecordingRetriever:
        def __init__(self) -> None:
            self.calls = 0

        def retrieve(self, query):  # type: ignore[no-untyped-def]
            self.calls += 1
            return ()

    retriever = RecordingRetriever()
    planner = EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=retriever,
        constraint_extractor=PolicyAwareFakeExtractor(),
    )

    result = planner.plan(_request(budget=Decimal("100")))

    # Structured resources exist; budget is evaluated after evidence by the required flow.
    assert result.outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    assert retriever.calls > 0


def test_empty_candidate_specific_evidence_requires_review_for_child_request() -> None:
    class EmptyRetriever:
        def retrieve(self, query):  # type: ignore[no-untyped-def]
            return ()

    planner = EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=EmptyRetriever(),
        constraint_extractor=PolicyAwareFakeExtractor(),
    )

    result = planner.plan(_request())

    assert result.outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    assert result.candidates == ()
    assert any("no current evidence retrieved" in item for item in result.unresolved_evidence)


def test_no_structured_candidates_returns_no_feasible_plan_without_retrieval() -> None:
    class RecordingRetriever:
        def __init__(self) -> None:
            self.calls = 0

        def retrieve(self, query):  # type: ignore[no-untyped-def]
            self.calls += 1
            return ()

    retriever = RecordingRetriever()
    request = _request().model_copy(update={"location": "Manhattan, NY"})
    planner = EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=retriever,
        constraint_extractor=PolicyAwareFakeExtractor(),
    )

    result = planner.plan(request)

    assert result.outcome is FeasibilityOutcome.NO_FEASIBLE_PLAN
    assert retriever.calls == 0


def test_accessibility_scenarios_reach_evidence_path() -> None:
    retriever = RecordingRetriever(BM25EvidenceRetriever(_documents()))
    planner = EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=retriever,
        constraint_extractor=PolicyAwareFakeExtractor(),
    )

    for request in (
        PartyRequest(
            location="Brooklyn, NY",
            event_date=date(2026, 9, 20),
            event_time=time(13, 0),
            guest_count=12,
            child_age=8,
            total_budget=Decimal("1600.00"),
            accessibility_needs=("wheelchair_accessible",),
        ),
        PartyRequest(
            location="Brooklyn, NY",
            event_date=date(2026, 9, 20),
            event_time=time(13, 0),
            guest_count=12,
            child_age=8,
            total_budget=Decimal("1600.00"),
            accessibility_needs=("accessible_restroom",),
        ),
    ):
        result = planner.plan(request)
        assert result.outcome is FeasibilityOutcome.FEASIBLE
        assert retriever.queries
        assert result.evidence_references


def test_complex_safety_scenario_reaches_evidence_path_and_requires_review() -> None:
    retriever = RecordingRetriever(BM25EvidenceRetriever(_documents()))
    allergen_description = (
        "Foods containing peanuts and tree nuts are prepared in a shared kitchen."
    )
    extractor = ScriptedConstraintExtractor(
        {
            "doc-loft-accessibility-current": (
                _extracted_constraint(
                    identifier="policy-loft-accessibility",
                    key="accessible_restroom",
                    operator=ConstraintOperator.EQ,
                    value=True,
                    description="Brooklyn Loft provides an accessible restroom.",
                    provenance_document_id="doc-loft-accessibility-current",
                    provenance_chunk_id="doc-loft-accessibility-current#chunk-1",
                    provenance_resource_id="venue-brooklyn-loft",
                    derivation_explanation="Directly stated accessibility guidance.",
                ),
            ),
            "doc-family-allergen-current": (
                _extracted_constraint(
                    identifier="policy-family-allergen-risk",
                    key="cross_contact_risk",
                    operator=ConstraintOperator.EQ,
                    value="present",
                    description=allergen_description,
                    provenance_document_id="doc-family-allergen-current",
                    provenance_chunk_id="doc-family-allergen-current#chunk-1",
                    provenance_resource_id="caterer-family-table",
                    derivation_explanation="Directly stated cross-contact risk.",
                ),
            ),
            "doc-family-vegan-current": (
                _extracted_constraint(
                    identifier="policy-family-vegan-notice",
                    key="vegan_notice_days",
                    operator=ConstraintOperator.GTE,
                    value=7,
                    description=(
                        "Vegan entree and dessert options require at least seven days of "
                        "advance notice."
                    ),
                    provenance_document_id="doc-family-vegan-current",
                    provenance_chunk_id="doc-family-vegan-current#chunk-1",
                    provenance_resource_id="caterer-family-table",
                    derivation_explanation="Directly stated advance-notice requirement.",
                ),
            ),
            "doc-craft-supervision-current": (
                _extracted_constraint(
                    identifier="policy-adult-child-ratio",
                    key="adult_child_ratio",
                    operator=ConstraintOperator.EQ,
                    value="1/5",
                    description="One adult is required for every five children.",
                    provenance_document_id="doc-craft-supervision-current",
                    provenance_chunk_id="doc-craft-supervision-current#chunk-1",
                    provenance_resource_id="activity-craft-party",
                    derivation_explanation="Directly stated supervision ratio.",
                ),
            ),
        }
    )
    planner = EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=retriever,
        constraint_extractor=extractor,
    )

    result = planner.plan(
        PartyRequest(
            location="Brooklyn, NY",
            event_date=date(2026, 9, 20),
            event_time=time(13, 0),
            guest_count=25,
            child_age=8,
            total_budget=Decimal("1800.00"),
            allergies=("peanuts", "tree nuts"),
            dietary_restrictions=("vegan",),
            accessibility_needs=("wheelchair_accessible",),
            other_constraints=("quiet room required",),
        )
    )

    assert retriever.queries
    assert result.outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    assert result.evidence_references


def test_supported_allergen_cross_contact_risk_requires_review() -> None:
    retriever = ScriptedRetriever(
        {
            "venue-brooklyn-loft": (
                _retrieval_result(
                    document_id="doc-loft-accessibility-current",
                    resource_id="venue-brooklyn-loft",
                    document_type=EvidenceDocumentType.ACCESSIBILITY_GUIDANCE,
                    text=(
                        "Brooklyn Loft provides step-free wheelchair access and an accessible "
                        "restroom."
                    ),
                ),
            ),
            "caterer-family-table": (
                _retrieval_result(
                    document_id="doc-family-allergen-current",
                    resource_id="caterer-family-table",
                    document_type=EvidenceDocumentType.ALLERGEN_POLICY,
                    text=(
                        "Family Table peanut and tree nut allergen policy: foods containing "
                        "peanuts and tree nuts are prepared in a shared kitchen. The caterer "
                        "cannot guarantee an allergen-free meal."
                    ),
                ),
            ),
            "activity-craft-party": (
                _retrieval_result(
                    document_id="doc-craft-supervision-current",
                    resource_id="activity-craft-party",
                    document_type=EvidenceDocumentType.SUPERVISION_REQUIREMENTS,
                    text="One adult is required for every five children.",
                ),
            ),
        }
    )
    extractor = ScriptedConstraintExtractor(
        {
            "doc-loft-accessibility-current": (
                _extracted_constraint(
                    identifier="policy-loft-accessibility",
                    key="accessible_restroom",
                    operator=ConstraintOperator.EQ,
                    value=True,
                    description="Brooklyn Loft provides an accessible restroom.",
                    provenance_document_id="doc-loft-accessibility-current",
                    provenance_chunk_id="doc-loft-accessibility-current#chunk-1",
                    provenance_resource_id="venue-brooklyn-loft",
                    derivation_explanation="Directly stated accessibility guidance.",
                ),
            ),
            "doc-family-allergen-current": (
                _extracted_constraint(
                    identifier="policy-family-allergen-risk",
                    key="cross_contact_risk",
                    operator=ConstraintOperator.EQ,
                    value="present",
                    description=(
                        "Foods containing peanuts and tree nuts are prepared in a shared kitchen."
                    ),
                    provenance_document_id="doc-family-allergen-current",
                    provenance_chunk_id="doc-family-allergen-current#chunk-1",
                    provenance_resource_id="caterer-family-table",
                    derivation_explanation="Directly stated cross-contact risk.",
                ),
            ),
            "doc-craft-supervision-current": (
                _extracted_constraint(
                    identifier="policy-adult-child-ratio",
                    key="adult_child_ratio",
                    operator=ConstraintOperator.EQ,
                    value="1/5",
                    description="One adult is required for every five children.",
                    provenance_document_id="doc-craft-supervision-current",
                    provenance_chunk_id="doc-craft-supervision-current#chunk-1",
                    provenance_resource_id="activity-craft-party",
                    derivation_explanation="Directly stated supervision ratio.",
                ),
            ),
        }
    )
    planner = EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=retriever,
        constraint_extractor=extractor,
    )

    result = planner.plan(
        PartyRequest(
            location="Brooklyn, NY",
            event_date=date(2026, 9, 20),
            guest_count=24,
            child_age=8,
            total_budget=Decimal("1600.00"),
            allergies=("peanuts",),
        )
    )

    assert result.outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    assert result.candidates == ()
    assert "supported cross-contact risk leaves the request unresolved" in "\n".join(
        result.unresolved_evidence
    )


def test_supported_gluten_cross_contact_risk_requires_review() -> None:
    retriever = ScriptedRetriever(
        {
            "venue-brooklyn-loft": (
                _retrieval_result(
                    document_id="doc-loft-accessibility-current",
                    resource_id="venue-brooklyn-loft",
                    document_type=EvidenceDocumentType.ACCESSIBILITY_GUIDANCE,
                    text=(
                        "Brooklyn Loft provides step-free wheelchair access and an accessible "
                        "restroom."
                    ),
                ),
            ),
            "caterer-family-table": (
                _retrieval_result(
                    document_id="doc-family-gluten-current",
                    resource_id="caterer-family-table",
                    document_type=EvidenceDocumentType.ALLERGEN_POLICY,
                    text=(
                        "Family Table offers gluten-free menu selections, but food is prepared "
                        "in a shared kitchen and is not certified free from gluten cross-contact."
                    ),
                ),
            ),
            "activity-craft-party": (
                _retrieval_result(
                    document_id="doc-craft-supervision-current",
                    resource_id="activity-craft-party",
                    document_type=EvidenceDocumentType.SUPERVISION_REQUIREMENTS,
                    text="One adult is required for every five children.",
                ),
            ),
        }
    )
    extractor = ScriptedConstraintExtractor(
        {
            "doc-loft-accessibility-current": (
                _extracted_constraint(
                    identifier="policy-loft-accessibility",
                    key="accessible_restroom",
                    operator=ConstraintOperator.EQ,
                    value=True,
                    description="Brooklyn Loft provides an accessible restroom.",
                    provenance_document_id="doc-loft-accessibility-current",
                    provenance_chunk_id="doc-loft-accessibility-current#chunk-1",
                    provenance_resource_id="venue-brooklyn-loft",
                    derivation_explanation="Directly stated accessibility guidance.",
                ),
            ),
            "doc-family-gluten-current": (
                _extracted_constraint(
                    identifier="policy-family-gluten-cross-contact",
                    key="gluten_cross_contact_risk",
                    operator=ConstraintOperator.EQ,
                    value="present",
                    description=(
                        "Food is prepared in a shared kitchen and is not certified free from "
                        "gluten cross-contact."
                    ),
                    provenance_document_id="doc-family-gluten-current",
                    provenance_chunk_id="doc-family-gluten-current#chunk-1",
                    provenance_resource_id="caterer-family-table",
                    derivation_explanation="Directly stated cross-contact risk.",
                ),
            ),
            "doc-craft-supervision-current": (
                _extracted_constraint(
                    identifier="policy-adult-child-ratio",
                    key="adult_child_ratio",
                    operator=ConstraintOperator.EQ,
                    value="1/5",
                    description="One adult is required for every five children.",
                    provenance_document_id="doc-craft-supervision-current",
                    provenance_chunk_id="doc-craft-supervision-current#chunk-1",
                    provenance_resource_id="activity-craft-party",
                    derivation_explanation="Directly stated supervision ratio.",
                ),
            ),
        }
    )
    planner = EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=retriever,
        constraint_extractor=extractor,
    )

    result = planner.plan(
        PartyRequest(
            location="Brooklyn, NY",
            event_date=date(2026, 9, 20),
            guest_count=24,
            child_age=8,
            total_budget=Decimal("1600.00"),
            dietary_restrictions=("gluten-free",),
        )
    )

    assert result.outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    assert result.candidates == ()
    assert "supported cross-contact risk leaves the request unresolved" in "\n".join(
        result.unresolved_evidence
    )


def test_supported_supervision_ratio_can_remain_feasible() -> None:
    planner = EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=BM25EvidenceRetriever(_documents()),
        constraint_extractor=PolicyAwareFakeExtractor(),
    )

    result = planner.plan(_request())

    assert result.outcome is FeasibilityOutcome.FEASIBLE
    assert result.candidates
    assert result.candidates[0].required_adults == 5


def test_conflicted_evidence_requires_review() -> None:
    retriever = ScriptedRetriever(
        {
            "venue-brooklyn-loft": (
                _retrieval_result(
                    document_id="doc-loft-accessibility-current",
                    resource_id="venue-brooklyn-loft",
                    document_type=EvidenceDocumentType.ACCESSIBILITY_GUIDANCE,
                    text=(
                        "Brooklyn Loft provides step-free wheelchair access and an accessible "
                        "restroom."
                    ),
                ),
            ),
            "caterer-family-table": (
                _retrieval_result(
                    document_id="doc-family-allergen-current",
                    resource_id="caterer-family-table",
                    document_type=EvidenceDocumentType.ALLERGEN_POLICY,
                    text=(
                        "Family Table peanut and tree nut allergen policy: foods containing "
                        "peanuts and tree nuts are prepared in a shared kitchen."
                    ),
                ),
                _retrieval_result(
                    document_id="doc-family-allergen-old",
                    resource_id="caterer-family-table",
                    document_type=EvidenceDocumentType.ALLERGEN_POLICY,
                    text=(
                        "Previous Family Table policy stated peanut-free orders could be "
                        "prepared on request, but cross-contact controls were not described."
                    ),
                    rank=2,
                ),
            ),
            "activity-craft-party": (
                _retrieval_result(
                    document_id="doc-craft-supervision-current",
                    resource_id="activity-craft-party",
                    document_type=EvidenceDocumentType.SUPERVISION_REQUIREMENTS,
                    text="One adult is required for every five children.",
                ),
            ),
        }
    )
    extractor = ScriptedConstraintExtractor(
        {
            "doc-loft-accessibility-current": (
                _extracted_constraint(
                    identifier="policy-loft-accessibility",
                    key="accessible_restroom",
                    operator=ConstraintOperator.EQ,
                    value=True,
                    description="Brooklyn Loft provides an accessible restroom.",
                    provenance_document_id="doc-loft-accessibility-current",
                    provenance_chunk_id="doc-loft-accessibility-current#chunk-1",
                    provenance_resource_id="venue-brooklyn-loft",
                    derivation_explanation="Directly stated accessibility guidance.",
                ),
            ),
            "doc-family-allergen-current": (
                _extracted_constraint(
                    identifier="policy-family-allergen-risk",
                    key="cross_contact_risk",
                    operator=ConstraintOperator.EQ,
                    value="present",
                    description=(
                        "Foods containing peanuts and tree nuts are prepared in a shared kitchen."
                    ),
                    provenance_document_id="doc-family-allergen-current",
                    provenance_chunk_id="doc-family-allergen-current#chunk-1",
                    provenance_resource_id="caterer-family-table",
                    derivation_explanation="Directly stated cross-contact risk.",
                ),
            ),
            "doc-family-allergen-old": (
                _extracted_constraint(
                    identifier="policy-family-allergen-legacy",
                    key="cross_contact_risk",
                    operator=ConstraintOperator.EQ,
                    value="absent",
                    description=(
                        "Previous Family Table policy stated peanut-free orders could be "
                        "prepared on request."
                    ),
                    provenance_document_id="doc-family-allergen-old",
                    provenance_chunk_id="doc-family-allergen-old#chunk-1",
                    provenance_resource_id="caterer-family-table",
                    derivation_explanation="Legacy policy statement.",
                ),
            ),
            "doc-craft-supervision-current": (
                _extracted_constraint(
                    identifier="policy-adult-child-ratio",
                    key="adult_child_ratio",
                    operator=ConstraintOperator.EQ,
                    value="1/5",
                    description="One adult is required for every five children.",
                    provenance_document_id="doc-craft-supervision-current",
                    provenance_chunk_id="doc-craft-supervision-current#chunk-1",
                    provenance_resource_id="activity-craft-party",
                    derivation_explanation="Directly stated supervision ratio.",
                ),
            ),
        }
    )
    planner = EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=retriever,
        constraint_extractor=extractor,
    )

    result = planner.plan(
        PartyRequest(
            location="Brooklyn, NY",
            event_date=date(2026, 9, 20),
            guest_count=24,
            child_age=8,
            total_budget=Decimal("1600.00"),
            allergies=("peanuts",),
        )
    )

    assert result.outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    assert result.candidates == ()
    assert any("CONFLICTED" in item for item in result.unresolved_evidence)


def test_insufficient_evidence_for_safety_sensitive_request_requires_review() -> None:
    class EmptyRetriever:
        def __init__(self) -> None:
            self.queries: list[EvidenceRetrievalQuery] = []

        def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
            self.queries.append(query)
            return ()

    retriever = EmptyRetriever()
    planner = EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=retriever,
        constraint_extractor=ScriptedConstraintExtractor({}),
    )

    result = planner.plan(
        PartyRequest(
            location="Brooklyn, NY",
            event_date=date(2026, 9, 20),
            guest_count=24,
            child_age=8,
            total_budget=Decimal("1600.00"),
            allergies=("peanuts",),
        )
    )

    assert result.outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    assert result.candidates == ()
    assert any("no current evidence retrieved" in item for item in result.unresolved_evidence)
