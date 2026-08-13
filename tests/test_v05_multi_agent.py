"""Tests for the PartyPilot v0.5 live multi-agent runtime."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from partypilot.adapters.llm_specialist_agents import build_specialist_agents
from partypilot.adapters.ollama import OllamaConfig, OllamaConnectionError, OllamaTimeoutError
from partypilot.application import multi_agent_runtime as runtime_module
from partypilot.application import v04_multi_agent as v04
from partypilot.application.multi_agent_runtime import (
    load_v05_multi_agent_benchmark,
    run_v05_multi_agent_experiment,
)
from partypilot.cli import eval_v05_llm_multi_agent as eval_v05
from partypilot.cli import smoke_langchain_agents, smoke_langchain_multi_agent, smoke_multi_agent
from partypilot.composition.multi_agent_runtime import build_live_multi_agent_runtime
from partypilot.domain import (
    SPECIALIST_IDENTITIES,
    AccessibilityAttribute,
    Activity,
    ArbitrationOutcome,
    CapabilityBoundaryScenario,
    CapabilityBoundaryScenarioMetadata,
    Caterer,
    CoordinationFailureKind,
    EvidenceDocument,
    EvidenceDocumentMetadata,
    EvidenceDocumentStatus,
    EvidenceDocumentType,
    FeasibilityOutcome,
    PartyRequest,
    ResourceCategory,
    ScenarioCategory,
    SpecialistAdapterVariant,
    SpecialistDecisionEnvelope,
    SpecialistDecisionPayload,
    SpecialistDomain,
    SpecialistFailureKind,
    Venue,
    canonical_specialist_id,
    canonical_specialist_name,
)
from partypilot.domain.evaluation import ComplexityMetadata, DatasetSplit, EvaluationScenario
from partypilot.ports.llm_provider import GenerationRequest, GenerationResponse


def _request() -> PartyRequest:
    return PartyRequest(
        location="Brooklyn, NY",
        event_date=date(2026, 9, 20),
        guest_count=24,
        total_budget=Decimal("1200.00"),
    )


def _venue(resource_id: str) -> Venue:
    return Venue(
        resource_id=resource_id,
        name=resource_id,
        location="Brooklyn, NY",
        price=Decimal("100.00"),
        capacity=40,
        accessibility_attributes=frozenset({AccessibilityAttribute.WHEELCHAIR_ACCESSIBLE}),
    )


def _caterer(resource_id: str) -> Caterer:
    return Caterer(
        resource_id=resource_id,
        name=resource_id,
        location="Brooklyn, NY",
        price=Decimal("50.00"),
    )


def _activity(resource_id: str) -> Activity:
    return Activity(
        resource_id=resource_id,
        name=resource_id,
        location="Brooklyn, NY",
        price=Decimal("25.00"),
        capacity=40,
    )


def _doc(
    *,
    document_id: str,
    resource_id: str,
    document_type: EvidenceDocumentType,
    text: str,
) -> EvidenceDocument:
    return EvidenceDocument(
        metadata=EvidenceDocumentMetadata(
            document_id=document_id,
            resource_id=resource_id,
            document_type=document_type,
            version="1.0",
            effective_date=date(2026, 1, 1),
            status=EvidenceDocumentStatus.CURRENT,
        ),
        text=text,
    )


def _scenario(
    scenario_id: str = "cap-boundary-41-venue-caterer-dependency",
) -> CapabilityBoundaryScenario:
    scenario_from_benchmark: CapabilityBoundaryScenario = next(
        item
        for item in v04.load_v04_multi_agent_benchmark()
        if item.scenario.scenario_id == scenario_id
    )
    if scenario_id != "cap-boundary-41-venue-caterer-dependency":
        return scenario_from_benchmark
    resources = (
        _venue("venue-alpha"),
        _caterer("caterer-alpha"),
        _activity("activity-alpha"),
    )
    base_scenario = EvaluationScenario(
        scenario_id=scenario_id,
        request=_request(),
        expected_feasibility=FeasibilityOutcome.FEASIBLE,
        expected_resource_ids=(
            "venue-alpha",
            "caterer-alpha",
            "activity-alpha",
        ),
        scenario_category=ScenarioCategory.FEASIBLE,
        complexity=ComplexityMetadata(),
        dataset_split=DatasetSplit.DEVELOPMENT,
        labeling_notes=("smoke",),
    )
    metadata = CapabilityBoundaryScenarioMetadata(
        capability_tags=("cross-domain",),
        requires_evidence=False,
        requires_semantic_interpretation=False,
        requires_state_replanning=False,
        cross_domain_dependency_count=2,
        adversarial_flag=False,
        complexity_trap_flag=False,
        milestone_introduced="v0.5-test",
        notes=("smoke",),
    )
    return CapabilityBoundaryScenario(
        scenario=base_scenario,
        metadata=metadata,
        evidence_documents=(
            _doc(
                document_id="doc-venue-alpha",
                resource_id="venue-alpha",
                document_type=EvidenceDocumentType.VENUE_POLICY,
                text="Venue policy allows the proposed event.",
            ),
            _doc(
                document_id="doc-caterer-alpha",
                resource_id="caterer-alpha",
                document_type=EvidenceDocumentType.ALLERGEN_POLICY,
                text="Caterer policy allows the proposed menu.",
            ),
            _doc(
                document_id="doc-activity-alpha",
                resource_id="activity-alpha",
                document_type=EvidenceDocumentType.ACCESSIBILITY_GUIDANCE,
                text="Activity space is accessible.",
            ),
        ),
        structured_resources=resources,
    )


def _benchmark_scenario(scenario_id: str) -> CapabilityBoundaryScenario:
    return next(
        item
        for item in v04.load_v04_multi_agent_benchmark()
        if item.scenario.scenario_id == scenario_id
    )


class RoutingFakeLLMProvider:
    def __init__(
        self,
        *,
        failures: dict[str, Exception] | None = None,
        malformed_outputs: set[str] | None = None,
    ) -> None:
        self.requests: list[tuple[GenerationRequest, float]] = []
        self.failures = failures or {}
        self.malformed_outputs = malformed_outputs or set()

    def generate(self, request: GenerationRequest, *, timeout_seconds: float) -> GenerationResponse:
        self.requests.append((request, timeout_seconds))
        payload = json.loads(request.prompt)
        specialist_id = payload["specialist_id"]
        if specialist_id in self.failures:
            raise self.failures[specialist_id]
        if specialist_id in self.malformed_outputs:
            return GenerationResponse(text="{", structured_output={"unexpected": "shape"})
        return _valid_specialist_response(payload)


class RetryRoutingFakeLLMProvider:
    def __init__(self) -> None:
        self.requests: list[tuple[GenerationRequest, float]] = []
        self._venue_attempts = 0

    def generate(self, request: GenerationRequest, *, timeout_seconds: float) -> GenerationResponse:
        self.requests.append((request, timeout_seconds))
        payload = json.loads(request.prompt)
        if payload["specialist_id"] == "venue":
            self._venue_attempts += 1
            if self._venue_attempts == 1:
                return GenerationResponse(
                    text='{"decision":{"status":"HUMAN_REVIEW_REQUIRED"}}',
                    structured_output={"decision": {"status": "HUMAN_REVIEW_REQUIRED"}},
                )
        return _valid_specialist_response(payload)


class ValidationModeFakeLLMProvider:
    def __init__(self) -> None:
        self.requests: list[tuple[GenerationRequest, float]] = []

    def generate(self, request: GenerationRequest, *, timeout_seconds: float) -> GenerationResponse:
        self.requests.append((request, timeout_seconds))
        payload = json.loads(request.prompt)
        evidence_docs = payload["scoped_evidence_documents"]
        decision = {
            "specialist_id": payload["specialist_id"],
            "domain": payload["domain"],
            "recommendation": f"accept {payload['specialist_id']}",
            "status": ArbitrationOutcome.ACCEPT.value,
            "hard_constraints_considered": ["hard"],
            "evidence_references": (
                [{"evidence_id": evidence_docs[0]["document_id"], "state": "SUPPORTED"}]
                if evidence_docs
                else []
            ),
            "assumptions": ["assume"],
            "unresolved_uncertainties": [],
            "local_score": 1.0,
            "local_rank": 1,
            "recommended_resource_ids": [],
            "reasons_for_rejection": [],
            "dependency_decision_ids": [],
            "notes": ["ok"],
        }
        return GenerationResponse(
            text=json.dumps({"decision": decision}),
            structured_output={"decision": decision},
        )


class StructuredOnlyVarianceFakeLLMProvider:
    def __init__(self) -> None:
        self.requests: list[tuple[GenerationRequest, float]] = []

    def generate(self, request: GenerationRequest, *, timeout_seconds: float) -> GenerationResponse:
        self.requests.append((request, timeout_seconds))
        payload = json.loads(request.prompt)
        specialist_id = payload["specialist_id"]
        evidence_docs = payload["scoped_evidence_documents"]
        if specialist_id == "accessibility":
            decision = {
                "specialist_id": specialist_id,
                "domain": payload["domain"],
                "recommendation": "Accessibility conditions are unclear.",
                "status": ArbitrationOutcome.HUMAN_REVIEW_REQUIRED.value,
                "hard_constraints_considered": ["accessibility"],
                "evidence_references": [],
                "assumptions": ["The room path is not fully confirmed."],
                "unresolved_uncertainties": ["Accessibility needs a human check."],
                "local_score": 0.0,
                "local_rank": 1,
                "recommended_resource_ids": [],
                "reasons_for_rejection": [],
                "dependency_decision_ids": [],
                "notes": ["uncertain"],
            }
        else:
            decision = {
                "specialist_id": specialist_id,
                "domain": payload["domain"],
                "recommendation": f"accept {specialist_id}",
                "status": ArbitrationOutcome.ACCEPT.value,
                "hard_constraints_considered": ["hard"],
                "evidence_references": (
                    [{"evidence_id": evidence_docs[0]["document_id"], "state": "SUPPORTED"}]
                    if evidence_docs
                    else []
                ),
                "assumptions": ["assume"],
                "unresolved_uncertainties": [],
                "local_score": 1.0,
                "local_rank": 1,
                "recommended_resource_ids": [],
                "reasons_for_rejection": [],
                "dependency_decision_ids": [],
                "notes": ["ok"],
            }
        return GenerationResponse(
            text=json.dumps({"decision": decision}),
            structured_output={"decision": decision},
        )


def _valid_specialist_response(payload: dict[str, Any]) -> GenerationResponse:
    specialist_id = payload["specialist_id"]
    candidate_resource_ids = [
        resource["resource_id"] for resource in payload["candidate_resources"]
    ]
    evidence_docs = payload["scoped_evidence_documents"]
    evidence_id = evidence_docs[0]["document_id"] if evidence_docs else None
    decision: dict[str, Any] = {
        "specialist_id": specialist_id,
        "domain": payload["domain"],
        "recommendation": f"accept {specialist_id}",
        "status": ArbitrationOutcome.ACCEPT.value,
        "hard_constraints_considered": ["hard"],
        "evidence_references": (
            [{"evidence_id": evidence_id, "state": "SUPPORTED"}] if evidence_id is not None else []
        ),
        "assumptions": ["assume"],
        "unresolved_uncertainties": [],
        "local_score": 1.0,
        "local_rank": 1,
        "recommended_resource_ids": candidate_resource_ids,
        "reasons_for_rejection": [],
        "dependency_decision_ids": [],
        "notes": ["ok"],
    }
    return GenerationResponse(
        text=json.dumps({"decision": decision}),
        structured_output={"decision": decision},
    )


def test_specialist_examples_validate_against_the_canonical_schema() -> None:
    provider = RoutingFakeLLMProvider()
    agents = build_specialist_agents(provider, timeout_seconds=7.5, model_name="fake-model")

    for agent in agents:
        example = agent._example_payload(agent.domain)
        envelope = SpecialistDecisionEnvelope.model_validate(
            {"decision": example.model_dump(mode="json")}
        )
        assert envelope.decision.specialist_id == canonical_specialist_id(agent.domain)
        assert example.specialist_id == canonical_specialist_id(agent.domain)
        assert envelope.decision.domain is agent.domain
        assert envelope.decision.status.value in {
            "ACCEPT",
            "REJECT",
            "HUMAN_REVIEW_REQUIRED",
            "REPLAN_REQUIRED",
        }


def test_specialist_identities_are_canonical_and_typed() -> None:
    assert tuple(identity.domain for identity in SPECIALIST_IDENTITIES) == (
        SpecialistDomain.VENUE,
        SpecialistDomain.CATERING_SAFETY,
        SpecialistDomain.ACCESSIBILITY,
        SpecialistDomain.SCHEDULING_OPERATIONS,
        SpecialistDomain.BUDGET,
    )
    assert tuple(identity.specialist_id for identity in SPECIALIST_IDENTITIES) == (
        "venue",
        "catering",
        "accessibility",
        "scheduling",
        "budget",
    )
    assert tuple(identity.specialist_name for identity in SPECIALIST_IDENTITIES) == (
        "VenueAgent",
        "CateringSafetyAgent",
        "AccessibilityAgent",
        "SchedulingAgent",
        "BudgetAgent",
    )
    assert canonical_specialist_id(SpecialistDomain.SCHEDULING_OPERATIONS) == "scheduling"
    assert canonical_specialist_name(SpecialistDomain.SCHEDULING_OPERATIONS) == "SchedulingAgent"


def test_scheduling_operations_alias_is_rejected_by_domain_validation() -> None:
    provider = RoutingFakeLLMProvider()
    agent = build_specialist_agents(provider, timeout_seconds=7.5, model_name="fake-model")[3]
    scenario = _scenario("cap-boundary-59-conflicting-agents-evidence")
    candidate_resources = scenario.structured_resources
    planning_state = runtime_module._build_planning_state(scenario, candidate_resources)
    agent_input = runtime_module._specialist_input(
        scenario=scenario,
        planning_state=planning_state,
        candidate_resources=candidate_resources,
        candidate_resource_ids=tuple(resource.resource_id for resource in candidate_resources),
        specialist=agent,
    )

    alias_payload = SpecialistDecisionPayload.model_validate(
        {
            "specialist_id": "scheduling_operations",
            "domain": SpecialistDomain.SCHEDULING_OPERATIONS.value,
            "recommendation": "Timing is fine.",
            "status": ArbitrationOutcome.ACCEPT.value,
            "hard_constraints_considered": ["schedule"],
            "evidence_references": [],
            "assumptions": [],
            "unresolved_uncertainties": [],
            "local_score": 1.0,
            "local_rank": 1,
            "recommended_resource_ids": [],
            "reasons_for_rejection": [],
            "dependency_decision_ids": [],
            "notes": [],
        }
    )

    with pytest.raises(ValueError, match="does not match"):
        agent._build_decision(agent_input, alias_payload)


def test_build_live_multi_agent_runtime_uses_canonical_schema_and_examples() -> None:
    provider = RoutingFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenario = _scenario()

    runtime.plan_scenario(scenario)

    assert provider.requests
    schema = SpecialistDecisionEnvelope.model_json_schema()
    for request, _timeout in provider.requests:
        assert request.structured_output is not None
        assert request.structured_output.json_schema == schema
        system_prompt = request.system_prompt or ""
        assert (
            "Return exactly one JSON object matching the canonical schema below." in system_prompt
        )
        assert "Return only the typed JSON envelope requested by the schema." in system_prompt
        allowed_statuses = (
            "The only legal status values are: ACCEPT, REJECT, "
            "HUMAN_REVIEW_REQUIRED, REPLAN_REQUIRED."
        )
        assert allowed_statuses in system_prompt
        assert "The recommendation field is free-text reasoning, not an enum." in system_prompt
        assert "specialist_id must equal the input specialist_id." in system_prompt
        assert "domain must equal the input domain." in system_prompt
        assert "Canonical specialist identity:" in system_prompt
        assert 'specialist_id MUST be exactly "' in system_prompt
        assert (
            "Never use the specialist_name, the class name, or the domain enum value"
            in system_prompt
        )
        assert "Valid example:" in system_prompt
        assert '"decision"' in system_prompt
        assert '"specialist_id"' in system_prompt
        assert '"domain"' in system_prompt
        assert '"recommendation"' in system_prompt
        assert '"status"' in system_prompt
        assert (
            "Never use synonyms such as accepted, reject, unsafe, accessible, invalidated,"
            in system_prompt
        )


def test_specialist_retry_uses_validation_error_and_canonical_schema() -> None:
    provider = RetryRoutingFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenario = _scenario()

    result = runtime.plan_scenario(scenario)

    venue_outcomes = [
        outcome
        for candidate in result.candidate_results
        for outcome in candidate.specialist_outcomes
        if outcome.trace.specialist_id == "venue"
    ]
    assert venue_outcomes
    assert any(outcome.trace.retry_count == 1 for outcome in venue_outcomes)
    assert all(outcome.trace.validation_succeeded for outcome in venue_outcomes)
    assert all(outcome.failure_kind is None for outcome in venue_outcomes)
    venue_requests = [
        request
        for request, _timeout in provider.requests
        if json.loads(request.prompt)["specialist_id"] == "venue"
    ]
    assert len(venue_requests) >= 2
    retry_system_prompt = venue_requests[1].system_prompt or ""
    assert "Validation errors:" in retry_system_prompt
    assert "Canonical schema:" in retry_system_prompt
    assert "Invalid previous structured output:" in retry_system_prompt
    assert "Return only corrected JSON." in retry_system_prompt


def test_specialist_schema_rejects_invalid_synonyms_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SpecialistDecisionEnvelope.model_validate(
            {
                "decision": {
                    "specialist_id": "venue",
                    "domain": "venue",
                    "recommendation": "Venue is fine.",
                    "status": "unsafe",
                    "hard_constraints_considered": [],
                    "evidence_references": [],
                    "assumptions": [],
                    "unresolved_uncertainties": [],
                    "local_score": 0.5,
                    "local_rank": 1,
                    "recommended_resource_ids": [],
                    "reasons_for_rejection": [],
                    "dependency_decision_ids": [],
                    "notes": [],
                }
            }
        )

    with pytest.raises(ValidationError):
        SpecialistDecisionEnvelope.model_validate(
            {
                "decision": {
                    "specialist_id": "venue",
                    "domain": "venue",
                    "recommendation": "Venue is fine.",
                    "status": "ACCEPT",
                    "hard_constraints_considered": [],
                    "evidence_references": [],
                    "assumptions": [],
                    "unresolved_uncertainties": [],
                    "local_score": 0.5,
                    "local_rank": 1,
                    "recommended_resource_ids": ["venue-alpha"],
                    "reasons_for_rejection": [],
                    "dependency_decision_ids": [],
                    "notes": [],
                    "outside_food_rules": "not allowed",
                }
            }
        )


def test_specialist_accepts_without_citations_when_allowed_evidence_list_is_empty() -> None:
    provider = RoutingFakeLLMProvider()
    agent = build_specialist_agents(provider, timeout_seconds=7.5, model_name="fake-model")[-1]
    scenario = _scenario("cap-boundary-41-venue-caterer-dependency")
    candidate_resources = scenario.structured_resources
    planning_state = runtime_module._build_planning_state(scenario, candidate_resources)
    agent_input = runtime_module._specialist_input(
        scenario=scenario,
        planning_state=planning_state,
        candidate_resources=candidate_resources,
        candidate_resource_ids=tuple(resource.resource_id for resource in candidate_resources),
        specialist=agent,
    )

    assert agent_input.allowed_evidence_document_ids == ()

    payload = SpecialistDecisionPayload.model_validate(
        {
            "specialist_id": "budget",
            "domain": SpecialistDomain.BUDGET.value,
            "recommendation": "Structured cost is within budget.",
            "status": ArbitrationOutcome.ACCEPT.value,
            "hard_constraints_considered": ["budget"],
            "evidence_references": [],
            "assumptions": ["Budget uses structured totals."],
            "unresolved_uncertainties": [],
            "local_score": 1.0,
            "local_rank": 1,
            "recommended_resource_ids": [],
            "reasons_for_rejection": [],
            "dependency_decision_ids": [],
            "notes": ["No documentary evidence is needed."],
        }
    )

    decision = agent._build_decision(agent_input, payload)

    assert decision.status is ArbitrationOutcome.ACCEPT
    assert decision.evidence_references == ()
    assert decision.recommended_resource_ids == ()


def test_specialist_rejects_invented_evidence_ids_and_structured_field_names() -> None:
    provider = RoutingFakeLLMProvider()
    agent = build_specialist_agents(provider, timeout_seconds=7.5, model_name="fake-model")[-1]
    scenario = _scenario("cap-boundary-59-conflicting-agents-evidence")
    candidate_resources = scenario.structured_resources
    planning_state = runtime_module._build_planning_state(scenario, candidate_resources)
    agent_input = runtime_module._specialist_input(
        scenario=scenario,
        planning_state=planning_state,
        candidate_resources=candidate_resources,
        candidate_resource_ids=tuple(resource.resource_id for resource in candidate_resources),
        specialist=agent,
    )

    assert agent_input.allowed_evidence_document_ids == ()

    invented_payload = SpecialistDecisionPayload.model_validate(
        {
            "specialist_id": "budget",
            "domain": SpecialistDomain.BUDGET.value,
            "recommendation": "Budget is fine.",
            "status": ArbitrationOutcome.ACCEPT.value,
            "hard_constraints_considered": ["budget"],
            "evidence_references": [
                {"evidence_id": "doc-accessibility-policy", "state": "SUPPORTED"}
            ],
            "assumptions": [],
            "unresolved_uncertainties": [],
            "local_score": 1.0,
            "local_rank": 1,
            "recommended_resource_ids": [],
            "reasons_for_rejection": [],
            "dependency_decision_ids": [],
            "notes": [],
        }
    )

    structured_field_payload = SpecialistDecisionPayload.model_validate(
        {
            "specialist_id": "budget",
            "domain": SpecialistDomain.BUDGET.value,
            "recommendation": "Budget is fine.",
            "status": ArbitrationOutcome.ACCEPT.value,
            "hard_constraints_considered": ["budget"],
            "evidence_references": [{"evidence_id": "candidate_total_cost", "state": "SUPPORTED"}],
            "assumptions": [],
            "unresolved_uncertainties": [],
            "local_score": 1.0,
            "local_rank": 1,
            "recommended_resource_ids": [],
            "reasons_for_rejection": [],
            "dependency_decision_ids": [],
            "notes": [],
        }
    )

    with pytest.raises(ValueError, match="outside the allowed evidence list"):
        agent._build_decision(agent_input, invented_payload)

    with pytest.raises(ValueError, match="outside the allowed evidence list"):
        agent._build_decision(agent_input, structured_field_payload)


def test_budget_accepts_from_structured_cost_facts_without_evidence() -> None:
    provider = RoutingFakeLLMProvider()
    agent = build_specialist_agents(provider, timeout_seconds=7.5, model_name="fake-model")[-1]
    scenario = _scenario("cap-boundary-59-conflicting-agents-evidence")
    candidate_resources = scenario.structured_resources
    planning_state = runtime_module._build_planning_state(scenario, candidate_resources)
    agent_input = runtime_module._specialist_input(
        scenario=scenario,
        planning_state=planning_state,
        candidate_resources=candidate_resources,
        candidate_resource_ids=tuple(resource.resource_id for resource in candidate_resources),
        specialist=agent,
    )

    assert agent_input.allowed_evidence_document_ids == ()
    assert agent_input.structured_facts
    assert any(fact.startswith("candidate_total_cost=") for fact in agent_input.structured_facts)
    assert any(fact.startswith("budget_ceiling=") for fact in agent_input.structured_facts)

    payload = SpecialistDecisionPayload.model_validate(
        {
            "specialist_id": "budget",
            "domain": SpecialistDomain.BUDGET.value,
            "recommendation": "Structured cost is within budget.",
            "status": ArbitrationOutcome.ACCEPT.value,
            "hard_constraints_considered": ["budget"],
            "evidence_references": [],
            "assumptions": ["Budget uses structured totals."],
            "unresolved_uncertainties": [],
            "local_score": 1.0,
            "local_rank": 1,
            "recommended_resource_ids": [],
            "reasons_for_rejection": [],
            "dependency_decision_ids": [],
            "notes": ["No documentary evidence is needed."],
        }
    )

    decision = agent._build_decision(agent_input, payload)

    assert decision.status is ArbitrationOutcome.ACCEPT
    assert decision.evidence_references == ()
    assert decision.recommended_resource_ids == ()


def test_scheduling_accepts_when_unrelated_catering_conflict_is_out_of_scope() -> None:
    provider = RoutingFakeLLMProvider()
    agent = build_specialist_agents(provider, timeout_seconds=7.5, model_name="fake-model")[3]
    scenario = _scenario("cap-boundary-41-venue-caterer-dependency")
    candidate_resources = scenario.structured_resources
    planning_state = runtime_module._build_planning_state(scenario, candidate_resources)
    agent_input = runtime_module._specialist_input(
        scenario=scenario,
        planning_state=planning_state,
        candidate_resources=candidate_resources,
        candidate_resource_ids=tuple(resource.resource_id for resource in candidate_resources),
        specialist=agent,
    )

    assert agent_input.allowed_evidence_document_ids == (
        "doc-venue-alpha",
        "doc-activity-alpha",
    )

    payload = SpecialistDecisionPayload.model_validate(
        {
            "specialist_id": "scheduling",
            "domain": SpecialistDomain.SCHEDULING_OPERATIONS.value,
            "recommendation": "No scheduling conflict exists.",
            "status": ArbitrationOutcome.ACCEPT.value,
            "hard_constraints_considered": ["schedule"],
            "evidence_references": [],
            "assumptions": ["Catering conflicts are outside scheduling authority."],
            "unresolved_uncertainties": [],
            "local_score": 1.0,
            "local_rank": 1,
            "recommended_resource_ids": [],
            "reasons_for_rejection": [],
            "dependency_decision_ids": [],
            "notes": ["No temporal conflict is present."],
        }
    )

    decision = agent._build_decision(agent_input, payload)

    assert decision.status is ArbitrationOutcome.ACCEPT
    assert decision.evidence_references == ()


def test_deterministic_hard_violation_overrides_llm_acceptance() -> None:
    provider = ValidationModeFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenario = _scenario("cap-boundary-42-venue-activity-dependency")

    result = runtime.plan_scenario(scenario)

    assert result.final_result.feasibility_outcome is FeasibilityOutcome.NO_FEASIBLE_PLAN
    assert result.final_result.failure_stage == "hard_constraints"
    assert any(
        candidate.arbitration_outcome is ArbitrationOutcome.REJECT
        for candidate in result.candidate_results
    )
    assert result.final_result.arbitration is not None
    assert result.final_result.arbitration.controlling_evidence_ids == (
        "doc-cap42-venue-no-prep-room",
    )


def test_evidence_backed_guardrail_preserves_controlling_evidence() -> None:
    provider = ValidationModeFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenario = _benchmark_scenario("cap-boundary-41-venue-caterer-dependency")

    result = runtime.plan_scenario(scenario)

    assert result.final_result.feasibility_outcome is FeasibilityOutcome.NO_FEASIBLE_PLAN
    assert result.final_result.failure_stage == "hard_constraints"
    assert result.final_result.evidence_grounded_arbitration is True
    assert result.final_result.arbitration is not None
    assert result.final_result.arbitration.controlling_evidence_ids == (
        "doc-cap41-approved-caterer-list",
        "doc-cap41-caterer-vendor-rule",
    )


def test_impossible_setup_chain_is_authoritative_even_when_llm_is_uncertain() -> None:
    provider = ValidationModeFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenario = _scenario("cap-boundary-43-setup-scheduling-chain")

    result = runtime.plan_scenario(scenario)

    assert result.final_result.feasibility_outcome is FeasibilityOutcome.NO_FEASIBLE_PLAN
    assert result.final_result.failure_stage == "hard_constraints"
    assert result.final_result.evidence_grounded_arbitration is True
    assert result.final_result.arbitration is not None
    assert result.final_result.arbitration.controlling_evidence_ids == (
        "doc-cap43-venue-access",
        "doc-cap43-setup-window",
        "doc-cap43-caterer-setup",
        "doc-cap43-activity-setup-window",
    )


def test_structured_only_complexity_trap_keeps_deterministic_feasibility() -> None:
    provider = StructuredOnlyVarianceFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenario = _benchmark_scenario("cap-boundary-65-ten-structured-constraints")

    result = runtime.plan_scenario(scenario)

    assert result.final_result.feasibility_outcome is FeasibilityOutcome.FEASIBLE
    assert result.final_result.failure_stage is None
    assert result.final_result.arbitration is not None
    assert result.final_result.arbitration.outcome is ArbitrationOutcome.ACCEPT
    assert result.final_result.arbitration.overridden_specialist_ids == ("accessibility",)
    assert result.final_result.arbitration.controlling_evidence_ids == ()
    assert result.final_result.evidence_grounded_arbitration is True


def test_unresolved_weather_contingency_stays_human_review_required() -> None:
    provider = ValidationModeFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenario = _benchmark_scenario("cap-boundary-45-outdoor-rain-contingency")

    result = runtime.plan_scenario(scenario)

    assert result.final_result.feasibility_outcome is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    assert result.final_result.failure_stage is None
    assert result.final_result.arbitration is not None
    assert result.final_result.arbitration.outcome is ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
    assert result.final_result.arbitration.controlling_evidence_ids == (
        "doc-cap45-outdoor-policy",
        "doc-cap45-rain-contingency",
    )


def test_live_v05_experiment_separates_terminal_mismatches_from_failure_stages() -> None:
    provider = ValidationModeFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenarios = (
        _benchmark_scenario("cap-boundary-41-venue-caterer-dependency"),
        _benchmark_scenario("cap-boundary-45-outdoor-rain-contingency"),
    )

    report = run_v05_multi_agent_experiment(scenarios, runtime=runtime)

    assert report.terminal_outcome_mismatch_scenario_ids == ()
    assert report.diagnostic_failure_stage_scenario_ids == (
        "cap-boundary-41-venue-caterer-dependency",
    )


def test_structured_only_scenario_does_not_fake_evidence_authority_failure() -> None:
    provider = ValidationModeFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenario = _scenario("cap-boundary-65-ten-structured-constraints")

    result = runtime.plan_scenario(scenario)

    assert result.final_result.feasibility_outcome is FeasibilityOutcome.FEASIBLE
    assert result.final_result.failure_stage is None
    assert result.final_result.evidence_grounded_arbitration is True


def test_global_optimizer_still_chooses_the_cheapest_feasible_combination() -> None:
    provider = ValidationModeFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenario = _scenario("cap-boundary-48-local-vs-global-optimum")

    result = runtime.plan_scenario(scenario)

    assert result.final_result.feasibility_outcome is FeasibilityOutcome.FEASIBLE
    assert result.final_result.selected_resource_ids == (
        "venue-brooklyn-loft",
        "caterer-family-table",
    )


def test_live_acceptance_preserves_causally_relevant_evidence_ids() -> None:
    provider = ValidationModeFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenario = _benchmark_scenario("cap-boundary-48-local-vs-global-optimum")

    result = runtime.plan_scenario(scenario)

    assert result.final_result.feasibility_outcome is FeasibilityOutcome.FEASIBLE
    assert result.final_result.evidence_grounded_arbitration is True
    assert result.final_result.arbitration is not None
    assert result.final_result.arbitration.controlling_evidence_ids == (
        "doc-cap48-expensive-venue",
        "doc-cap48-cheap-caterer",
        "doc-cap48-global-cost-note",
    )
    assert "doc-cap48-cheap-venue" not in result.final_result.arbitration.controlling_evidence_ids


def test_build_live_multi_agent_runtime_scopes_specialists_and_structured_outputs() -> None:
    provider = RoutingFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenario = _scenario()

    result = runtime.plan_scenario(scenario)

    assert result.final_result.feasibility_outcome is FeasibilityOutcome.FEASIBLE
    assert len(provider.requests) == len(result.candidate_results) * 5
    assert [specialist.specialist_id for specialist in runtime._specialists] == [
        "venue",
        "catering",
        "accessibility",
        "scheduling",
        "budget",
    ]

    resource_categories = {
        resource.resource_id: resource.category for resource in scenario.structured_resources
    }
    for request, _timeout in provider.requests:
        payload = json.loads(request.prompt)
        specialist_id = payload["specialist_id"]
        candidate_resource_ids = tuple(
            resource["resource_id"] for resource in payload["candidate_resources"]
        )
        selected_resource_ids = tuple(payload["planning_state"]["selected_resource_ids"])
        evidence_ids = tuple(doc["document_id"] for doc in payload["scoped_evidence_documents"])
        assert request.structured_output is not None
        assert request.structured_output.schema_name == "SpecialistDecision"
        assert payload["specialist_id"] == specialist_id
        if specialist_id == "venue":
            expected = tuple(
                resource_id
                for resource_id in selected_resource_ids
                if resource_categories[resource_id] is ResourceCategory.VENUE
            )
            assert candidate_resource_ids == expected
            assert evidence_ids == tuple(
                document.metadata.document_id
                for document in scenario.evidence_documents
                if document.metadata.resource_id in expected
                and document.metadata.document_type.value
                in {"venue_policy", "accessibility_guidance"}
            )
            assert "venueagent" in (request.system_prompt or "").casefold()
            assert "venue" in (request.system_prompt or "").casefold()
        elif specialist_id == "catering":
            expected = tuple(
                resource_id
                for resource_id in selected_resource_ids
                if resource_categories[resource_id]
                in {ResourceCategory.VENUE, ResourceCategory.CATERER}
            )
            assert candidate_resource_ids == expected
            assert evidence_ids == tuple(
                document.metadata.document_id
                for document in scenario.evidence_documents
                if document.metadata.resource_id in expected
                and document.metadata.document_type.value
                in {"venue_policy", "allergen_policy", "outside_food_rules"}
            )
            assert "allergen" in (request.system_prompt or "").casefold()
        elif specialist_id == "accessibility":
            expected = tuple(
                resource_id
                for resource_id in selected_resource_ids
                if resource_categories[resource_id]
                in {ResourceCategory.VENUE, ResourceCategory.ACTIVITY}
            )
            assert candidate_resource_ids == expected
            assert evidence_ids == ("doc-activity-alpha",)
            assert "accessibility requirements" in (request.system_prompt or "").casefold()
        elif specialist_id == "scheduling":
            assert candidate_resource_ids == selected_resource_ids
            assert evidence_ids == ("doc-venue-alpha", "doc-activity-alpha")
            assert "temporal feasibility" in (request.system_prompt or "").casefold()
        elif specialist_id == "budget":
            assert candidate_resource_ids == selected_resource_ids
            assert evidence_ids == ()
            assert "budget ceiling" in (request.system_prompt or "").casefold()
        else:  # pragma: no cover - defensive
            pytest.fail(f"unexpected specialist {specialist_id}")


def test_build_live_multi_agent_runtime_scopes_accessibility_evidence() -> None:
    provider = RoutingFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenario = _scenario("cap-boundary-59-conflicting-agents-evidence")

    runtime.plan_scenario(scenario)

    accessibility_requests = [
        request
        for request, _timeout in provider.requests
        if json.loads(request.prompt)["specialist_id"] == "accessibility"
    ]
    assert accessibility_requests
    for request in accessibility_requests:
        payload = json.loads(request.prompt)
        evidence_ids = tuple(doc["document_id"] for doc in payload["scoped_evidence_documents"])
        assert evidence_ids == (
            "doc-cap59-recommendation-note",
            "doc-cap59-accessibility-analysis",
        )


def test_accepting_validation_response_may_omit_recommended_resources() -> None:
    provider = ValidationModeFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(
        provider,
        timeout_seconds=7.5,
        model_name="fake-model",
    )
    scenario = _scenario()

    result = runtime.plan_scenario(scenario)

    assert result.execution_traces
    assert all(trace.validation_succeeded for trace in result.execution_traces)
    assert all(
        trace.adapter_variant is SpecialistAdapterVariant.NATIVE_OLLAMA
        for trace in result.execution_traces
    )
    assert all(
        outcome.decision is not None and outcome.decision.status is ArbitrationOutcome.ACCEPT
        for candidate in result.candidate_results
        for outcome in candidate.specialist_outcomes
    )
    assert all(
        outcome.decision is not None and not outcome.decision.recommended_resource_ids
        for candidate in result.candidate_results
        for outcome in candidate.specialist_outcomes
    )


def test_live_v05_experiment_reports_live_and_baseline_metrics() -> None:
    provider = RoutingFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(provider, timeout_seconds=7.5, model_name="fake-model")
    scenario = _scenario()

    report = run_v05_multi_agent_experiment((scenario,), runtime=runtime)

    assert report.evaluation_variant == "deterministic_specialists_vs_live_llm_specialists"
    assert report.metrics.scenario_count == 1
    assert report.metrics.runtime.total_specialist_calls == len(provider.requests)
    assert report.metrics.runtime.specialist_success_rate == 1.0
    assert report.metrics.baseline.final_decision_accuracy == 1.0
    assert report.metrics.live.final_decision_accuracy == 1.0
    assert report.metrics.retention_rule_passed is True
    assert report.scenarios[0].live_result.feasibility_outcome is FeasibilityOutcome.FEASIBLE


def test_smoke_multi_agent_cli_reports_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = RoutingFakeLLMProvider()
    scenario = _scenario()
    fake_config = SimpleNamespace(
        model="fake-model",
        timeout_seconds=12.0,
        num_ctx=8192,
        max_retries=2,
    )
    monkeypatch.setattr(smoke_multi_agent, "_ollama_config", lambda **kwargs: fake_config)
    monkeypatch.setattr(smoke_multi_agent, "OllamaAdapter", lambda config, transport: provider)
    monkeypatch.setattr(smoke_multi_agent, "_smoke_scenarios", lambda scenario_ids: (scenario,))

    exit_code = smoke_multi_agent.main(
        ["--scenario-id", "cap-boundary-41-venue-caterer-dependency"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PartyPilot v0.5 Multi-Agent Smoke Test" in captured.out
    assert "Provider I/O timeout: 12.0s" in captured.out
    assert "Ollama context budget: 8192 tokens" in captured.out
    assert "Model: fake-model" in captured.out
    assert "Scenario: cap-boundary-41-venue-caterer-dependency" in captured.out
    assert "Smoke test passed." in captured.out


def test_smoke_multi_agent_cli_reports_provider_failure_with_classification(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = RoutingFakeLLMProvider(
        failures={"venue": OllamaConnectionError("Could not connect to Ollama")}
    )
    scenario = _scenario()
    fake_config = SimpleNamespace(
        model="fake-model",
        timeout_seconds=12.0,
        num_ctx=8192,
        max_retries=2,
    )
    monkeypatch.setattr(smoke_multi_agent, "_ollama_config", lambda **kwargs: fake_config)
    monkeypatch.setattr(smoke_multi_agent, "OllamaAdapter", lambda config, transport: provider)
    monkeypatch.setattr(smoke_multi_agent, "_smoke_scenarios", lambda scenario_ids: (scenario,))

    exit_code = smoke_multi_agent.main(
        ["--scenario-id", "cap-boundary-41-venue-caterer-dependency"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Provider I/O timeout: 12.0s" in captured.out
    assert "Ollama context budget: 8192 tokens" in captured.out
    assert (
        "FAILED | PROVIDER_CONNECTION_ERROR | OllamaConnectionError: Could not connect to Ollama"
        in captured.out
    )


def test_smoke_langchain_multi_agent_cli_reports_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_outcome = SimpleNamespace(
        trace=SimpleNamespace(
            specialist_name="VenueAgent",
            adapter_variant=SimpleNamespace(value="langchain_chatollama"),
            evidence_document_ids=(),
            validation_succeeded=True,
            retry_count=0,
            latency_ms=1.2,
        ),
        decision=SimpleNamespace(
            status=SimpleNamespace(value="ACCEPT"),
            recommended_resource_ids=(),
            evidence_references=(),
        ),
        failure_kind=None,
    )
    fake_candidate = SimpleNamespace(candidate_resource_ids=(), specialist_outcomes=(fake_outcome,))
    fake_result = SimpleNamespace(
        final_result=SimpleNamespace(
            selected_resource_ids=(),
            feasibility_outcome=SimpleNamespace(value="FEASIBLE"),
        ),
        candidate_results=(fake_candidate,),
    )
    fake_config = SimpleNamespace(
        model="fake-model",
        timeout_seconds=12.0,
        num_ctx=8192,
        max_retries=2,
    )

    class FakeRuntime:
        def plan_scenario(self, scenario: Any) -> Any:
            return fake_result

    monkeypatch.setattr(smoke_langchain_multi_agent, "_ollama_config", lambda **kwargs: fake_config)
    monkeypatch.setattr(
        smoke_langchain_multi_agent,
        "build_live_multi_agent_runtime",
        lambda **kwargs: FakeRuntime(),
    )
    monkeypatch.setattr(
        smoke_langchain_multi_agent,
        "_smoke_scenarios",
        lambda scenario_ids: (_scenario(),),
    )
    monkeypatch.setattr(
        smoke_langchain_multi_agent,
        "_selected_candidate",
        lambda result: fake_candidate,
    )

    exit_code = smoke_langchain_multi_agent.main(
        ["--scenario-id", "cap-boundary-41-venue-caterer-dependency"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PartyPilot v0.6b LangChain Multi-Agent Smoke Test" in captured.out
    assert "Adapter kind: langchain" in captured.out
    assert "Provider I/O timeout: 12.0s" in captured.out
    assert "Ollama context budget: 8192 tokens" in captured.out
    assert "Scenario: cap-boundary-41-venue-caterer-dependency" in captured.out
    assert "adapter=langchain_chatollama" in captured.out
    assert "Smoke test passed." in captured.out


def test_smoke_langchain_agents_cli_reports_tool_usage(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_outcome = SimpleNamespace(
        trace=SimpleNamespace(
            specialist_name="VenueAgent",
            adapter_variant=SimpleNamespace(value="langchain_agent"),
            evidence_document_ids=("doc-venue-alpha",),
            validation_succeeded=True,
            retry_count=0,
            latency_ms=1.2,
            tool_call_count=2,
            tool_call_traces=(
                SimpleNamespace(tool_name="inspect_selected_venue", success=True, error_kind=None),
                SimpleNamespace(
                    tool_name="get_allowed_venue_evidence", success=True, error_kind=None
                ),
            ),
            agent_execution_limit_hit=False,
            failure_reason=None,
        ),
        decision=SimpleNamespace(
            status=SimpleNamespace(value="ACCEPT"),
            recommended_resource_ids=(),
            evidence_references=(),
        ),
        failure_kind=None,
    )
    fake_candidate = SimpleNamespace(candidate_resource_ids=(), specialist_outcomes=(fake_outcome,))
    fake_result = SimpleNamespace(
        final_result=SimpleNamespace(
            selected_resource_ids=(),
            feasibility_outcome=SimpleNamespace(value="FEASIBLE"),
        ),
        candidate_results=(fake_candidate,),
    )
    fake_config = SimpleNamespace(
        model="fake-model",
        timeout_seconds=12.0,
        num_ctx=8192,
        max_retries=2,
    )

    class FakeRuntime:
        def plan_scenario(self, scenario: Any) -> Any:
            return fake_result

    monkeypatch.setattr(smoke_langchain_agents, "_ollama_config", lambda **kwargs: fake_config)
    monkeypatch.setattr(
        smoke_langchain_agents,
        "build_live_multi_agent_runtime",
        lambda **kwargs: FakeRuntime(),
    )
    monkeypatch.setattr(
        smoke_langchain_agents,
        "_smoke_scenarios",
        lambda scenario_ids: (_scenario(),),
    )
    monkeypatch.setattr(
        smoke_langchain_agents,
        "_selected_candidate",
        lambda result: fake_candidate,
    )

    exit_code = smoke_langchain_agents.main(
        ["--scenario-id", "cap-boundary-41-venue-caterer-dependency"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PartyPilot v0.6c LangChain Agent Smoke Test" in captured.out
    assert "Adapter kind: langchain_agent" in captured.out
    assert "Provider I/O timeout: 12.0s" in captured.out
    assert "Ollama context budget: 8192 tokens" in captured.out
    assert "Agent execution bound: 8" in captured.out
    assert "Tools invoked: yes" in captured.out
    assert "tool_calls=2" in captured.out
    assert "agent_limit_hit=False" in captured.out
    assert "Smoke test passed." in captured.out


def test_smoke_langchain_agents_cli_diagnostic_reports_tool_request_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_config = OllamaConfig(
        base_url="http://localhost:11434",
        model="fake-model",
        timeout_seconds=12.0,
        num_ctx=8192,
        max_retries=0,
    )
    fake_trace = SimpleNamespace(
        validation_succeeded=True,
        agent_execution_limit_hit=False,
        tool_call_count=1,
        tool_call_traces=(
            SimpleNamespace(
                tool_name="inspect_selected_resource_accessibility",
                request_summary="resource_id=venue-alpha",
                success=True,
                error_kind=None,
            ),
        ),
        failure_reason=None,
    )
    fake_outcome = SimpleNamespace(
        trace=fake_trace,
        raw_structured_output={"decision": {"status": "ACCEPT"}},
        decision=SimpleNamespace(
            status=SimpleNamespace(value="ACCEPT"),
            evidence_references=(),
        ),
    )
    fake_specialist = SimpleNamespace(
        specialist_id="accessibility",
        run=lambda agent_input: fake_outcome,
    )

    monkeypatch.setattr(
        smoke_langchain_agents, "_smoke_scenarios", lambda scenario_ids: (_scenario(),)
    )
    monkeypatch.setattr(
        smoke_langchain_agents,
        "build_specialist_agents",
        lambda **kwargs: (fake_specialist,),
    )
    smoke_module = cast(Any, smoke_langchain_agents)
    monkeypatch.setattr(
        smoke_module.runtime_module,
        "_build_planning_state",
        lambda scenario, candidate_resources: SimpleNamespace(),
    )
    monkeypatch.setattr(
        smoke_module.runtime_module,
        "_specialist_input",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        smoke_module.v04,
        "_candidate_combinations",
        lambda scenario: iter([("venue-alpha", "caterer-alpha", "activity-alpha")]),
    )

    exit_code = smoke_langchain_agents._run_tool_necessity_diagnostic(
        config=fake_config,
        scenario_id="cap-boundary-59-conflicting-agents-evidence",
        specialist_id="accessibility",
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Tool-Necessity Diagnostic" in captured.out
    assert "Diagnostic specialist_id: accessibility" in captured.out
    assert "Tool calls: 1" in captured.out
    assert "request=resource_id=venue-alpha" in captured.out
    assert "Diagnostic passed." in captured.out


def test_runtime_classifies_provider_timeout() -> None:
    provider = RoutingFakeLLMProvider(
        failures={"venue": OllamaTimeoutError("Ollama request timed out")}
    )
    runtime = build_live_multi_agent_runtime(provider, timeout_seconds=7.5, model_name="fake-model")
    scenario = _scenario()

    result = runtime.plan_scenario(scenario)

    failed_outcomes = [
        outcome
        for candidate in result.candidate_results
        for outcome in candidate.specialist_outcomes
        if outcome.failure_kind is not None
    ]
    assert failed_outcomes
    assert all(
        outcome.failure_kind is SpecialistFailureKind.PROVIDER_TIMEOUT
        for outcome in failed_outcomes
    )
    assert all(
        outcome.trace.failure_error_type == "OllamaTimeoutError" for outcome in failed_outcomes
    )
    assert all(outcome.decision is None for outcome in failed_outcomes)


def test_runtime_classifies_structured_output_validation_failure() -> None:
    provider = RoutingFakeLLMProvider(malformed_outputs={"venue"})
    runtime = build_live_multi_agent_runtime(provider, timeout_seconds=7.5, model_name="fake-model")
    scenario = _scenario()

    result = runtime.plan_scenario(scenario)

    failed_outcomes = [
        outcome
        for candidate in result.candidate_results
        for outcome in candidate.specialist_outcomes
        if outcome.failure_kind is not None
    ]
    assert failed_outcomes
    assert all(
        outcome.failure_kind is SpecialistFailureKind.STRUCTURED_OUTPUT_VALIDATION_ERROR
        for outcome in failed_outcomes
    )
    assert all(outcome.trace.failure_error_type == "ValidationError" for outcome in failed_outcomes)
    assert all(outcome.decision is None for outcome in failed_outcomes)


def test_runtime_completes_with_one_failed_specialist_and_four_successful_specialists() -> None:
    provider = RoutingFakeLLMProvider(
        failures={"venue": OllamaConnectionError("Could not connect to Ollama")}
    )
    runtime = build_live_multi_agent_runtime(provider, timeout_seconds=7.5, model_name="fake-model")
    scenario = _scenario()

    result = runtime.plan_scenario(scenario)

    assert result.execution_traces
    assert len(result.execution_traces) == len(result.candidate_results) * 5
    assert all(
        trace.adapter_variant is SpecialistAdapterVariant.NATIVE_OLLAMA
        for trace in result.execution_traces
    )
    assert any(
        outcome.failure_kind is SpecialistFailureKind.PROVIDER_CONNECTION_ERROR
        for candidate in result.candidate_results
        for outcome in candidate.specialist_outcomes
    )
    assert any(
        outcome.decision is not None
        for candidate in result.candidate_results
        for outcome in candidate.specialist_outcomes
    )
    assert result.final_result.arbitration is not None
    assert result.final_result.arbitration.outcome is ArbitrationOutcome.HUMAN_REVIEW_REQUIRED


def test_runtime_classifies_coordinator_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = RoutingFakeLLMProvider()
    runtime = build_live_multi_agent_runtime(provider, timeout_seconds=7.5, model_name="fake-model")
    scenario = _scenario()

    def _raise_coordinate_candidate(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        raise RuntimeError("coordinator boom")

    monkeypatch.setattr(v04, "_coordinate_candidate", _raise_coordinate_candidate)

    result = runtime.plan_scenario(scenario)

    assert result.candidate_results
    assert all(
        candidate.coordinated_result.failure_kind is CoordinationFailureKind.COORDINATOR_ERROR
        for candidate in result.candidate_results
    )
    assert all(
        candidate.coordinated_result.failure_stage == "coordinator_error"
        for candidate in result.candidate_results
    )


def test_smoke_multi_agent_requires_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("PARTYPILOT_OLLAMA_MODEL", raising=False)
    monkeypatch.setattr(
        smoke_multi_agent,
        "OllamaAdapter",
        lambda *args, **kwargs: pytest.fail("provider should not be constructed"),
    )

    exit_code = smoke_multi_agent.main(
        ["--scenario-id", "cap-boundary-41-venue-caterer-dependency"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PARTYPILOT_OLLAMA_MODEL is required" in captured.err


def test_eval_v05_cli_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = RoutingFakeLLMProvider()
    scenario = _scenario()
    fake_config = SimpleNamespace(
        model="fake-model",
        timeout_seconds=12.0,
        num_ctx=8192,
        max_retries=2,
    )
    monkeypatch.setattr(eval_v05, "_ollama_config", lambda **kwargs: fake_config)
    monkeypatch.setattr(eval_v05, "OllamaAdapter", lambda config, transport: provider)
    monkeypatch.setattr(
        eval_v05,
        "load_v05_multi_agent_benchmark",
        lambda: (scenario,),
    )

    exit_code = eval_v05.main(
        [
            "--output-dir",
            str(tmp_path),
            "--scenario-id",
            "cap-boundary-41-venue-caterer-dependency",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PartyPilot v0.5 Live Multi-Agent Runtime Experiment" in captured.out
    assert "Scenario count: 1" in captured.out
    assert "Retention rule passed:" in captured.out
    assert "Terminal outcome mismatches:" in captured.out
    assert "Diagnostic failure-stage cases:" in captured.out
    assert (tmp_path / "v0_5_llm_multi_agent.json").exists()
    assert (tmp_path / "v0_5_llm_multi_agent.md").exists()


def test_load_v05_multi_agent_benchmark_reuses_the_bounded_development_subset() -> None:
    scenarios = load_v05_multi_agent_benchmark()

    assert len(scenarios) == 10
    assert {scenario.scenario.scenario_id for scenario in scenarios} == {
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
    }
