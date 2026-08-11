from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from partypilot.application import v04_multi_agent as v04
from partypilot.cli import eval_v04_multi_agent as eval_v04
from partypilot.domain import (
    AccessibilityAttribute,
    Activity,
    ArbitrationOutcome,
    CapabilityBoundaryScenario,
    CapabilityBoundaryScenarioMetadata,
    Caterer,
    EvidenceDocument,
    EvidenceDocumentMetadata,
    EvidenceDocumentStatus,
    EvidenceDocumentType,
    FeasibilityOutcome,
    PartyRequest,
    ScenarioCategory,
    SpecialistDecision,
    SpecialistDomain,
    Venue,
)
from partypilot.domain.evaluation import ComplexityMetadata, DatasetSplit, EvaluationScenario


def _request(
    *,
    guest_count: int = 50,
    budget: Decimal = Decimal("500"),
    accessibility_needs: tuple[str, ...] = (),
    allergies: tuple[str, ...] = (),
) -> PartyRequest:
    return PartyRequest(
        location="Brooklyn, NY",
        event_date=date(2026, 9, 20),
        guest_count=guest_count,
        total_budget=budget,
        accessibility_needs=accessibility_needs,
        allergies=allergies,
    )


def _venue(
    resource_id: str,
    *,
    price: str,
    capacity: int = 100,
    accessibility_attributes: tuple[AccessibilityAttribute, ...] = (),
) -> Venue:
    return Venue(
        resource_id=resource_id,
        name=resource_id,
        location="Brooklyn, NY",
        price=Decimal(price),
        capacity=capacity,
        accessibility_attributes=frozenset(accessibility_attributes),
    )


def _caterer(resource_id: str, *, price: str = "0") -> Caterer:
    return Caterer(
        resource_id=resource_id,
        name=resource_id,
        location="Brooklyn, NY",
        price=Decimal(price),
    )


def _activity(resource_id: str, *, price: str = "0", capacity: int = 100) -> Activity:
    return Activity(
        resource_id=resource_id,
        name=resource_id,
        location="Brooklyn, NY",
        price=Decimal(price),
        capacity=capacity,
    )


def _doc(
    *,
    document_id: str,
    resource_id: str,
    document_type: EvidenceDocumentType,
    text: str,
    status: EvidenceDocumentStatus = EvidenceDocumentStatus.CURRENT,
    version: str = "1.0",
) -> EvidenceDocument:
    return EvidenceDocument(
        metadata=EvidenceDocumentMetadata(
            document_id=document_id,
            resource_id=resource_id,
            document_type=document_type,
            version=version,
            effective_date=date(2026, 1, 1),
            status=status,
        ),
        text=text,
    )


def _scenario(
    *,
    scenario_id: str,
    request: PartyRequest,
    resources: tuple[Venue | Caterer | Activity, ...],
    evidence_documents: tuple[EvidenceDocument, ...] = (),
    expected_feasibility: FeasibilityOutcome = FeasibilityOutcome.FEASIBLE,
    capability_tags: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
    complexity_trap_flag: bool = False,
    requires_evidence: bool = False,
    requires_semantic_interpretation: bool = False,
    requires_state_replanning: bool = False,
    cross_domain_dependency_count: int = 0,
    milestone_introduced: str = "v0.4-test",
) -> CapabilityBoundaryScenario:
    scenario = EvaluationScenario(
        scenario_id=scenario_id,
        request=request,
        expected_feasibility=expected_feasibility,
        scenario_category=ScenarioCategory.OTHER,
        complexity=ComplexityMetadata(),
        dataset_split=DatasetSplit.DEVELOPMENT,
        labeling_notes=notes,
    )
    metadata = CapabilityBoundaryScenarioMetadata(
        capability_tags=capability_tags,
        requires_evidence=requires_evidence,
        requires_semantic_interpretation=requires_semantic_interpretation,
        requires_state_replanning=requires_state_replanning,
        cross_domain_dependency_count=cross_domain_dependency_count,
        adversarial_flag=False,
        complexity_trap_flag=complexity_trap_flag,
        milestone_introduced=milestone_introduced,
        notes=notes,
    )
    return CapabilityBoundaryScenario(
        scenario=scenario,
        metadata=metadata,
        evidence_documents=evidence_documents,
        structured_resources=resources,
    )


def _decision(
    specialist_id: str,
    domain: SpecialistDomain,
    status: ArbitrationOutcome,
) -> SpecialistDecision:
    return SpecialistDecision(
        specialist_id=specialist_id,
        domain=domain,
        recommendation=f"{status.value.lower()} {specialist_id}",
        status=status,
        hard_constraints_considered=("hard",),
        evidence_references=(),
        assumptions=("assumption",),
        unresolved_uncertainties=(),
        local_score=1.0 if status is ArbitrationOutcome.ACCEPT else 0.0,
        local_rank=1,
        recommended_resource_ids=(f"{specialist_id}-resource",),
        reasons_for_rejection=("reason",) if status is ArbitrationOutcome.REJECT else (),
        dependency_decision_ids=(),
        notes=("note",),
    )


def test_v04_benchmark_loads_expected_fixture_set() -> None:
    scenarios = v04.load_v04_multi_agent_benchmark()

    assert len(scenarios) == 10
    assert {
        "cap-boundary-41-venue-caterer-dependency",
        "cap-boundary-42-venue-activity-dependency",
        "cap-boundary-43-setup-scheduling-chain",
        "cap-boundary-44-loading-bay-conflict",
        "cap-boundary-45-outdoor-rain-contingency",
        "cap-boundary-47-specialist-disagreement",
        "cap-boundary-48-local-vs-global-optimum",
        "cap-boundary-59-conflicting-agents-evidence",
        "cap-boundary-61-large-but-purely-structured",
        "cap-boundary-65-ten-structured-constraints",
    } == {scenario.scenario.scenario_id for scenario in scenarios}


def test_specialist_decision_validation_rejects_invalid_enum_values() -> None:
    with pytest.raises(ValidationError):
        SpecialistDecision.model_validate(
            {
                "specialist_id": "venue",
                "domain": "venue",
                "recommendation": "accept venue",
                "status": "NOT_A_REAL_STATUS",
                "hard_constraints_considered": ["venue"],
                "evidence_references": [],
                "assumptions": ["assume"],
                "unresolved_uncertainties": [],
                "local_score": 1.0,
                "local_rank": 1,
                "recommended_resource_ids": ["venue-a"],
                "reasons_for_rejection": [],
                "dependency_decision_ids": [],
                "notes": ["note"],
            }
        )


def test_conflicting_specialists_do_not_use_majority_vote() -> None:
    scenario = _scenario(
        scenario_id="majority-vote-control",
        request=_request(),
        resources=(
            _venue("venue-a", price="100"),
            _caterer("caterer-a"),
            _activity("activity-a"),
        ),
    )
    candidate = ("venue-a", "caterer-a", "activity-a")
    decisions = (
        _decision("venue", SpecialistDomain.VENUE, ArbitrationOutcome.ACCEPT),
        _decision("catering", SpecialistDomain.CATERING_SAFETY, ArbitrationOutcome.ACCEPT),
        _decision("accessibility", SpecialistDomain.ACCESSIBILITY, ArbitrationOutcome.ACCEPT),
        _decision("scheduling", SpecialistDomain.SCHEDULING_OPERATIONS, ArbitrationOutcome.ACCEPT),
        _decision("budget", SpecialistDomain.BUDGET, ArbitrationOutcome.REJECT),
    )

    trace, selected, _ = v04._coordinate_candidate(scenario, candidate, decisions)

    assert selected == candidate
    assert trace.outcome is ArbitrationOutcome.REJECT
    assert trace.feasibility_outcome is FeasibilityOutcome.NO_FEASIBLE_PLAN
    assert len(trace.accepted_specialist_ids) == 4
    assert len(trace.rejected_specialist_ids) == 1


def test_authoritative_catering_evidence_overrides_marketing_copy() -> None:
    scenario = _scenario(
        scenario_id="catering-authority-control",
        request=_request(allergies=("peanut",)),
        resources=(
            _venue("venue-a", price="100"),
            _caterer("caterer-a"),
            _activity("activity-a"),
        ),
        evidence_documents=(
            _doc(
                document_id="caterer-marketing",
                resource_id="caterer-a",
                document_type=EvidenceDocumentType.ALLERGEN_POLICY,
                text=(
                    "Marketing copy supports allergy-friendly events and says the menu is suitable."
                ),
            ),
            _doc(
                document_id="caterer-policy",
                resource_id="caterer-a",
                document_type=EvidenceDocumentType.ALLERGEN_POLICY,
                text=(
                    "Family Table peanut and tree nut policy: foods containing peanuts "
                    "and tree nuts are prepared in a shared kitchen. The caterer cannot "
                    "guarantee an allergen-free meal."
                ),
            ),
        ),
        requires_evidence=True,
    )
    candidate_resources = scenario.structured_resources

    decision = v04._catering_specialist(scenario, candidate_resources)

    assert decision.status is ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
    assert len(decision.evidence_references) == 2


def test_accessibility_rejection_overrides_marketing_language() -> None:
    scenario = _scenario(
        scenario_id="accessibility-rejection-control",
        request=_request(accessibility_needs=("wheelchair_accessible",)),
        resources=(
            _venue("venue-a", price="100"),
            _caterer("caterer-a"),
            _activity("activity-a"),
        ),
        evidence_documents=(
            _doc(
                document_id="venue-marketing",
                resource_id="venue-a",
                document_type=EvidenceDocumentType.VENUE_POLICY,
                text="The venue says wheelchair access is available.",
            ),
            _doc(
                document_id="venue-policy",
                resource_id="venue-a",
                document_type=EvidenceDocumentType.VENUE_POLICY,
                text=(
                    "The actual event room is not suitable because the path is too close to the "
                    "service stairs."
                ),
            ),
        ),
        requires_evidence=True,
    )

    decision = v04._accessibility_specialist(scenario, scenario.structured_resources)

    assert decision.status is ArbitrationOutcome.REJECT
    assert decision.reasons_for_rejection == ("Accessibility evidence blocks the request.",)


def test_unresolved_safety_uncertainty_triggers_review() -> None:
    scenario = _scenario(
        scenario_id="safety-review-control",
        request=_request(allergies=("peanut",)),
        resources=(
            _venue("venue-a", price="100"),
            _caterer("caterer-a"),
            _activity("activity-a"),
        ),
        evidence_documents=(
            _doc(
                document_id="caterer-marketing",
                resource_id="caterer-a",
                document_type=EvidenceDocumentType.ALLERGEN_POLICY,
                text="Marketing copy supports allergy-friendly events.",
            ),
            _doc(
                document_id="caterer-policy",
                resource_id="caterer-a",
                document_type=EvidenceDocumentType.ALLERGEN_POLICY,
                text=(
                    "Foods containing peanuts and tree nuts are prepared in a shared kitchen. "
                    "The caterer cannot guarantee an allergen-free meal."
                ),
            ),
        ),
        requires_evidence=True,
    )
    decision = v04._catering_specialist(scenario, scenario.structured_resources)
    candidate = ("venue-a", "caterer-a", "activity-a")
    trace, _, _ = v04._coordinate_candidate(
        scenario,
        candidate,
        (
            _decision("venue", SpecialistDomain.VENUE, ArbitrationOutcome.ACCEPT),
            decision,
            _decision("accessibility", SpecialistDomain.ACCESSIBILITY, ArbitrationOutcome.ACCEPT),
            _decision(
                "scheduling",
                SpecialistDomain.SCHEDULING_OPERATIONS,
                ArbitrationOutcome.ACCEPT,
            ),
            _decision("budget", SpecialistDomain.BUDGET, ArbitrationOutcome.ACCEPT),
        ),
    )

    assert decision.status is ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
    assert trace.outcome is ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
    assert trace.feasibility_outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED


def test_global_optimum_prefers_lower_total_cost() -> None:
    scenario = _scenario(
        scenario_id="local-vs-global-optimum-control",
        request=_request(budget=Decimal("500")),
        resources=(
            _venue("venue-a", price="100"),
            _venue("venue-b", price="120"),
            _caterer("caterer-a"),
            _activity("activity-a"),
        ),
        evidence_documents=(
            _doc(
                document_id="venue-a-fee",
                resource_id="venue-a",
                document_type=EvidenceDocumentType.VENUE_POLICY,
                text="Mandatory service fee $100 applies to venue A.",
            ),
        ),
        capability_tags=("global_optimization",),
        requires_evidence=True,
    )

    report = v04.run_v04_multi_agent_experiment((scenario,))
    scenario_result = report.scenarios[0]

    assert scenario_result.baseline.selected_resource_ids[0] == "venue-a"
    assert scenario_result.coordinated.selected_resource_ids[0] == "venue-b"
    assert scenario_result.baseline.global_optimum is False
    assert scenario_result.coordinated.global_optimum is True
    assert report.metrics.global_optimum_scenario_count == 1
    assert report.metrics.baseline.global_optimum_accuracy == 0.0
    assert report.metrics.coordinated.global_optimum_accuracy == 1.0


def test_agreement_control_all_accept_is_accepted() -> None:
    scenario = _scenario(
        scenario_id="agreement-control",
        request=_request(),
        resources=(
            _venue("venue-a", price="100"),
            _caterer("caterer-a"),
            _activity("activity-a"),
        ),
    )
    candidate = ("venue-a", "caterer-a", "activity-a")
    trace, _, _ = v04._coordinate_candidate(
        scenario,
        candidate,
        (
            _decision("venue", SpecialistDomain.VENUE, ArbitrationOutcome.ACCEPT),
            _decision("catering", SpecialistDomain.CATERING_SAFETY, ArbitrationOutcome.ACCEPT),
            _decision("accessibility", SpecialistDomain.ACCESSIBILITY, ArbitrationOutcome.ACCEPT),
            _decision(
                "scheduling",
                SpecialistDomain.SCHEDULING_OPERATIONS,
                ArbitrationOutcome.ACCEPT,
            ),
            _decision("budget", SpecialistDomain.BUDGET, ArbitrationOutcome.ACCEPT),
        ),
    )

    assert trace.outcome is ArbitrationOutcome.ACCEPT
    assert trace.feasibility_outcome is FeasibilityOutcome.FEASIBLE
    assert trace.rejected_specialist_ids == ()


def test_complexity_trap_control_remains_deterministic() -> None:
    scenario = next(
        item
        for item in v04.load_v04_multi_agent_benchmark()
        if item.scenario.scenario_id == "cap-boundary-61-large-but-purely-structured"
    )
    report = v04.run_v04_multi_agent_experiment((scenario,))

    assert scenario.metadata.complexity_trap_flag is True
    assert report.metrics.scenario_count == 1
    assert report.scenarios[0].baseline.failure_stage is None
    assert report.scenarios[0].coordinated.failure_stage is None
    assert report.scenarios[0].baseline.feasibility_outcome is FeasibilityOutcome.FEASIBLE
    assert report.scenarios[0].coordinated.feasibility_outcome is FeasibilityOutcome.FEASIBLE


def test_conflicting_accessibility_evidence_requires_review_instead_of_reject() -> None:
    scenario = next(
        item
        for item in v04.load_v04_multi_agent_benchmark()
        if item.scenario.scenario_id == "cap-boundary-59-conflicting-agents-evidence"
    )

    report = v04.run_v04_multi_agent_experiment((scenario,))
    scenario_result = report.scenarios[0]

    assert scenario_result.expected_feasibility is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    assert (
        scenario_result.coordinated.feasibility_outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    )
    assert scenario_result.coordinated.arbitration is not None
    assert (
        scenario_result.coordinated.arbitration.outcome is ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
    )
    assert scenario_result.coordinated.arbitration.controlling_evidence_ids == (
        "doc-cap59-recommendation-note",
        "doc-cap59-accessibility-analysis",
    )
    assert (
        scenario_result.coordinated.specialist_decisions[0].status
        is ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
    )


def test_eval_v04_cli_defaults_to_timestamped_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixed_timestamp = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
    fixed_output_dir = tmp_path / "multi_agent" / "20260811T120000Z"
    scenario = _scenario(
        scenario_id="cli-control",
        request=_request(),
        resources=(
            _venue("venue-a", price="100"),
            _caterer("caterer-a"),
            _activity("activity-a"),
        ),
    )

    monkeypatch.setattr(eval_v04, "load_v04_multi_agent_benchmark", lambda: (scenario,))
    monkeypatch.setattr(
        eval_v04,
        "default_output_dir",
        lambda timestamp: fixed_output_dir,
    )
    monkeypatch.setattr(
        eval_v04,
        "datetime",
        SimpleNamespace(now=lambda tz: fixed_timestamp),
    )

    exit_code = eval_v04.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Scenario count: 1" in captured.out
    assert "20260811T120000Z" in captured.out
    assert (fixed_output_dir / "v0_4_multi_agent.json").exists()
    assert (fixed_output_dir / "v0_4_multi_agent.md").exists()
