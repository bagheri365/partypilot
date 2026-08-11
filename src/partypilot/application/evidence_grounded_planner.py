"""Ordinary-Python evidence-grounded planning flow for PartyPilot v0.2."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from itertools import product

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application.budget_validation import CostComponent, validate_budget
from partypilot.application.candidate_filtering import CandidateRequirements, filter_candidates
from partypilot.application.constraint_engine import (
    ConstraintEngineInput,
    ConstraintEngineResult,
    validate_constraints,
)
from partypilot.application.derived_constraints import (
    DerivedConstraint,
    DerivedConstraintContext,
    derive_constraint,
)
from partypilot.application.evidence_state_resolution import (
    EvidenceAssessment,
    EvidenceResolution,
    resolve_evidence_state,
)
from partypilot.domain.constraints import Constraint, ConstraintOperator, ConstraintType
from partypilot.domain.evidence import EvidenceReference, EvidenceState
from partypilot.domain.evidence_corpus import EvidenceDocumentMetadata, EvidenceDocumentStatus
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest
from partypilot.domain.resources import AccessibilityAttribute, Resource, ResourceCategory
from partypilot.domain.temporal import Duration, TimeWindow
from partypilot.ports.constraint_extractor import (
    ConstraintExtractionContext,
    ConstraintExtractionInput,
    ConstraintExtractor,
    ExtractedConstraint,
)
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalFilters,
    EvidenceRetrievalQuery,
    EvidenceRetriever,
)
from partypilot.ports.resource_store import ResourceSearchCriteria, ResourceStore


class EvidenceGroundedPlannerConfig(BaseModel):
    """Small deterministic configuration surface for the v0.2 planning flow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_duration: Duration = Field(default_factory=lambda: Duration.hours(2))
    evidence_top_k: int = Field(default=5, gt=0)
    max_candidates: int = Field(default=5, gt=0)


class EvidenceGroundedPlanCandidate(BaseModel):
    """A deterministically validated resource combination plus evidence-derived requirements."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resources: tuple[Resource, ...]
    total_cost: Decimal = Field(ge=0)
    required_adults: int | None = Field(default=None, ge=0)
    validation: ConstraintEngineResult
    evidence_references: tuple[EvidenceReference, ...] = ()
    extracted_constraints: tuple[ExtractedConstraint, ...] = ()
    derived_constraints: tuple[DerivedConstraint, ...] = ()


class EvidenceGroundedPlanningResult(BaseModel):
    """Terminal result of the evidence-grounded planning use case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: FeasibilityOutcome
    candidates: tuple[EvidenceGroundedPlanCandidate, ...] = ()
    evidence_references: tuple[EvidenceReference, ...] = ()
    unresolved_evidence: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.outcome is FeasibilityOutcome.FEASIBLE and bool(self.candidates)


class EvidenceGroundedPlanner:
    """Simple evidence-grounded orchestration with explicit injected dependencies."""

    def __init__(
        self,
        *,
        resource_store: ResourceStore,
        evidence_retriever: EvidenceRetriever,
        constraint_extractor: ConstraintExtractor,
        config: EvidenceGroundedPlannerConfig | None = None,
    ) -> None:
        self._resource_store = resource_store
        self._evidence_retriever = evidence_retriever
        self._constraint_extractor = constraint_extractor
        self._config = config or EvidenceGroundedPlannerConfig()

    def plan(self, request: PartyRequest) -> EvidenceGroundedPlanningResult:
        event_window = self._event_window(request)
        accessibility, unresolved_accessibility = self._parse_accessibility(request)
        if unresolved_accessibility:
            return EvidenceGroundedPlanningResult(
                outcome=FeasibilityOutcome.HUMAN_REVIEW_REQUIRED,
                unresolved_evidence=unresolved_accessibility,
            )

        candidates_by_category = self._structured_candidates(request, event_window, accessibility)
        if any(not values for values in candidates_by_category.values()):
            return EvidenceGroundedPlanningResult(outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN)

        all_references: dict[str, EvidenceReference] = {}
        unresolved: list[str] = []
        feasible_candidates: list[EvidenceGroundedPlanCandidate] = []

        for combination in product(
            candidates_by_category[ResourceCategory.VENUE],
            candidates_by_category[ResourceCategory.CATERER],
            candidates_by_category[ResourceCategory.ACTIVITY],
        ):
            resources = tuple(combination)
            evidence = self._collect_evidence(request, resources)
            for reference in evidence.references:
                all_references[reference.evidence_id] = reference
            unresolved.extend(evidence.unresolved)
            if evidence.requires_review:
                continue

            components = tuple(
                CostComponent(
                    component_id=resource.resource_id,
                    description=resource.name,
                    amount=resource.price,
                )
                for resource in resources
            )
            budget = validate_budget(request.total_budget, components)
            if not budget.within_budget:
                continue

            hard_constraints = self._structured_hard_constraints(
                request, event_window, accessibility
            )
            validation = validate_constraints(
                ConstraintEngineInput(
                    hard_constraints=hard_constraints,
                    selected_resources=resources,
                    candidate_requirements=CandidateRequirements(
                        location=request.location,
                        guest_count=request.guest_count,
                        child_age=request.child_age,
                        child_age_range=request.child_age_range,
                        availability=event_window,
                    ),
                    budget=request.total_budget,
                    cost_components=components,
                )
            )
            if not validation.feasible:
                continue

            required_adults = self._required_adults(evidence.derived_constraints)
            feasible_candidates.append(
                EvidenceGroundedPlanCandidate(
                    resources=resources,
                    total_cost=budget.total_cost,
                    required_adults=required_adults,
                    validation=validation,
                    evidence_references=evidence.references,
                    extracted_constraints=evidence.extracted_constraints,
                    derived_constraints=evidence.derived_constraints,
                )
            )

        feasible_candidates.sort(
            key=lambda candidate: (
                candidate.total_cost,
                tuple(resource.resource_id for resource in candidate.resources),
            )
        )
        references = tuple(all_references[key] for key in sorted(all_references))
        unresolved_unique = tuple(dict.fromkeys(unresolved))
        if feasible_candidates:
            return EvidenceGroundedPlanningResult(
                outcome=FeasibilityOutcome.FEASIBLE,
                candidates=tuple(feasible_candidates[: self._config.max_candidates]),
                evidence_references=references,
            )
        if unresolved_unique:
            return EvidenceGroundedPlanningResult(
                outcome=FeasibilityOutcome.HUMAN_REVIEW_REQUIRED,
                evidence_references=references,
                unresolved_evidence=unresolved_unique,
            )
        return EvidenceGroundedPlanningResult(
            outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN,
            evidence_references=references,
        )

    def _structured_candidates(
        self,
        request: PartyRequest,
        event_window: TimeWindow | None,
        accessibility: frozenset[AccessibilityAttribute],
    ) -> dict[ResourceCategory, tuple[Resource, ...]]:
        result: dict[ResourceCategory, tuple[Resource, ...]] = {}
        for category in ResourceCategory:
            resources = self._resource_store.search(
                ResourceSearchCriteria(
                    location=request.location,
                    minimum_capacity=request.guest_count,
                    category=category,
                )
            )
            requirements = CandidateRequirements(
                location=request.location,
                guest_count=request.guest_count,
                child_age=request.child_age,
                child_age_range=request.child_age_range,
                availability=event_window,
                accessibility=(
                    accessibility if category is not ResourceCategory.CATERER else frozenset()
                ),
            )
            result[category] = filter_candidates(resources, requirements).eligible
        return result

    def _collect_evidence(
        self,
        request: PartyRequest,
        resources: tuple[Resource, ...],
    ) -> _CandidateEvidence:
        extracted: list[ExtractedConstraint] = []
        references: list[EvidenceReference] = []
        unresolved: list[str] = []
        assessments_by_key: dict[str, list[EvidenceAssessment]] = defaultdict(list)

        for resource in resources:
            results = self._evidence_retriever.retrieve(
                EvidenceRetrievalQuery(
                    text=self._evidence_query(request),
                    top_k=self._config.evidence_top_k,
                    filters=EvidenceRetrievalFilters(
                        resource_id=resource.resource_id,
                        status=EvidenceDocumentStatus.CURRENT,
                    ),
                )
            )
            if not results and self._needs_evidence(request):
                unresolved.append(f"no current evidence retrieved for {resource.resource_id}")
                continue

            for result in results:
                if result.document_type is None:
                    unresolved.append(
                        f"retrieval result {result.document_id} is missing document_type"
                    )
                    continue
                metadata = EvidenceDocumentMetadata(
                    document_id=result.document_id,
                    resource_id=result.resource_id,
                    document_type=result.document_type,
                    version=result.version.version,
                    effective_date=result.version.effective_date,
                    status=result.version.status,
                )
                extraction = self._constraint_extractor.extract(
                    ConstraintExtractionInput(
                        evidence_text=result.text,
                        evidence_metadata=metadata,
                        chunk_id=result.chunk_id,
                        planning_context=ConstraintExtractionContext(
                            request=request,
                            resource_id=resource.resource_id,
                        ),
                    )
                )
                if not extraction.constraints:
                    continue
                for item in extraction.constraints:
                    extracted.append(item)
                    assessments_by_key[item.constraint.key].append(
                        EvidenceAssessment(
                            provenance=item.provenance,
                            document_status=result.version.status,
                            constraint=item.constraint,
                            applicable=True,
                            ambiguous=False,
                            safety_sensitive=self._is_safety_sensitive(request, item.constraint),
                            explanation=f"Extracted from {result.document_id}.",
                        )
                    )

        resolutions: list[EvidenceResolution] = []
        for key in sorted(assessments_by_key):
            resolution = resolve_evidence_state(tuple(assessments_by_key[key]))
            resolutions.append(resolution)
            references.append(
                EvidenceReference(
                    evidence_id=f"constraint:{key}",
                    state=resolution.state,
                    provenance=resolution.provenance,
                )
            )
            if resolution.state is not EvidenceState.SUPPORTED:
                unresolved.append(f"evidence for {key}: {resolution.state.value}")
                continue

            review_reason = self._request_specific_review_reason(request, resolution)
            if review_reason is not None:
                unresolved.append(f"evidence for {key}: {review_reason}")

        derived: list[DerivedConstraint] = []
        if request.child_age is not None or request.child_age_range is not None:
            # PartyRequest does not yet distinguish child attendees from total guests.
            # For child-focused requests the current v0.2 model treats guest_count as
            # the child attendee count and makes that assumption visible in provenance.
            for item in extracted:
                if item.constraint.key != "adult_child_ratio":
                    continue
                state = next(
                    (
                        resolution.state
                        for resolution in resolutions
                        if any(c.key == item.constraint.key for c in resolution.constraints)
                    ),
                    EvidenceState.UNSUPPORTED,
                )
                derivation = derive_constraint(
                    item,
                    evidence_state=state,
                    context=DerivedConstraintContext(child_count=request.guest_count),
                )
                derived.extend(derivation.constraints)

        requires_review = bool(unresolved)
        return _CandidateEvidence(
            extracted_constraints=tuple(extracted),
            derived_constraints=tuple(derived),
            references=tuple(references),
            unresolved=tuple(unresolved),
            requires_review=requires_review,
        )

    @staticmethod
    def _needs_evidence(request: PartyRequest) -> bool:
        return bool(
            request.allergies
            or request.dietary_restrictions
            or request.accessibility_needs
            or request.child_age is not None
            or request.child_age_range is not None
            or request.other_constraints
        )

    @staticmethod
    def _is_safety_sensitive(request: PartyRequest, constraint: Constraint) -> bool:
        return bool(request.allergies) or constraint.key in {"adult_child_ratio", "allergen_policy"}

    @staticmethod
    def _request_specific_review_reason(
        request: PartyRequest,
        resolution: EvidenceResolution,
    ) -> str | None:
        if not (request.allergies or request.dietary_restrictions):
            return None

        for constraint in resolution.constraints:
            reason = EvidenceGroundedPlanner._dietary_risk_reason(request, constraint)
            if reason is not None:
                return reason
        return None

    @staticmethod
    def _dietary_risk_reason(request: PartyRequest, constraint: Constraint) -> str | None:
        del request
        normalized = EvidenceGroundedPlanner._constraint_text(constraint)

        if constraint.key in {"cross_contact_risk", "gluten_cross_contact_risk"}:
            if EvidenceGroundedPlanner._is_positive_risk_value(constraint.value):
                return "supported cross-contact risk leaves the request unresolved"
            return None

        if constraint.key in {"allergen_policy", "allergy_requirement"}:
            risk_markers = (
                "shared kitchen",
                "cannot guarantee",
                "not certified",
                "cross-contact",
                "allergen-free",
                "allergy details",
                "before booking",
            )
            if any(marker in normalized for marker in risk_markers):
                return "supported allergen policy includes a safety limitation"

        return None

    @staticmethod
    def _constraint_text(constraint: Constraint) -> str:
        parts: list[str] = [constraint.key, constraint.description]
        value = constraint.value
        if isinstance(value, tuple):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value))
        return " ".join(parts).casefold()

    @staticmethod
    def _is_positive_risk_value(value: object) -> bool:
        if value is True:
            return True
        if value is False:
            return False

        normalized = str(value).casefold()
        return normalized not in {"", "0", "false", "none", "no", "absent", "not_present"}

    @staticmethod
    def _evidence_query(request: PartyRequest) -> str:
        parts = ["policy safety supervision accessibility allergen outside food"]
        parts.extend(request.allergies)
        parts.extend(request.dietary_restrictions)
        parts.extend(request.accessibility_needs)
        parts.extend(request.other_constraints)
        if request.child_age is not None:
            parts.append(f"child age {request.child_age}")
        if request.child_age_range is not None:
            parts.append(
                f"child ages {request.child_age_range.minimum}-{request.child_age_range.maximum}"
            )
        return " ".join(parts)

    def _event_window(self, request: PartyRequest) -> TimeWindow | None:
        if request.event_time is None:
            return None
        start = datetime.combine(request.event_date, request.event_time)
        return TimeWindow(start=start, end=start + self._config.event_duration.value)

    @staticmethod
    def _parse_accessibility(
        request: PartyRequest,
    ) -> tuple[frozenset[AccessibilityAttribute], tuple[str, ...]]:
        parsed: set[AccessibilityAttribute] = set()
        unresolved: list[str] = []
        for need in request.accessibility_needs:
            normalized = need.strip().casefold().replace(" ", "_")
            try:
                parsed.add(AccessibilityAttribute(normalized))
            except ValueError:
                unresolved.append(f"accessibility:{need}")
        return frozenset(parsed), tuple(unresolved)

    @staticmethod
    def _required_adults(derived: tuple[DerivedConstraint, ...]) -> int | None:
        values = [
            item.constraint.value
            for item in derived
            if item.constraint.key == "minimum_adults" and isinstance(item.constraint.value, int)
        ]
        return max(values) if values else None

    @staticmethod
    def _structured_hard_constraints(
        request: PartyRequest,
        event_window: TimeWindow | None,
        accessibility: frozenset[AccessibilityAttribute],
    ) -> tuple[Constraint, ...]:
        constraints = [
            Constraint(
                identifier="request-location",
                key="location",
                operator=ConstraintOperator.EQ,
                value=request.location,
                constraint_type=ConstraintType.HARD,
                description="Resources must match the requested location.",
            ),
            Constraint(
                identifier="request-capacity",
                key="guest_count",
                operator=ConstraintOperator.GTE,
                value=request.guest_count,
                constraint_type=ConstraintType.HARD,
                description="Resources must support the requested guest count.",
            ),
            Constraint(
                identifier="request-budget",
                key="total_budget",
                operator=ConstraintOperator.LTE,
                value=request.total_budget,
                constraint_type=ConstraintType.HARD,
                description="Combined resource cost must not exceed the total budget.",
            ),
        ]
        if request.child_age is not None or request.child_age_range is not None:
            constraints.append(
                Constraint(
                    identifier="request-child-age",
                    key="age_restrictions",
                    operator=ConstraintOperator.EQ,
                    value=request.child_age if request.child_age is not None else "age_range",
                    constraint_type=ConstraintType.HARD,
                    description="Resources must support the requested child age.",
                )
            )
        if event_window is not None:
            constraints.append(
                Constraint(
                    identifier="request-availability",
                    key="availability",
                    operator=ConstraintOperator.EQ,
                    value=event_window.start.isoformat(),
                    constraint_type=ConstraintType.HARD,
                    description="Resources must be available for the event window.",
                )
            )
        if accessibility:
            constraints.append(
                Constraint(
                    identifier="request-accessibility",
                    key="accessibility",
                    operator=ConstraintOperator.CONTAINS,
                    value=tuple(sorted(item.value for item in accessibility)),
                    constraint_type=ConstraintType.HARD,
                    description="Resources must meet requested accessibility needs.",
                )
            )
        return tuple(constraints)


class _CandidateEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    extracted_constraints: tuple[ExtractedConstraint, ...] = ()
    derived_constraints: tuple[DerivedConstraint, ...] = ()
    references: tuple[EvidenceReference, ...] = ()
    unresolved: tuple[str, ...] = ()
    requires_review: bool = False
