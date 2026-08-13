from __future__ import annotations

import inspect
import json
from datetime import date
from decimal import Decimal
from typing import Any, cast

import httpx
import pytest
from langchain_ollama import ChatOllama

import partypilot.adapters.langchain_specialist_agents as langchain_specialist_agents
from partypilot.adapters.langchain_specialist_agents import (
    LangChainAccessibilityAgent,
    LangChainBudgetAgent,
    LangChainCateringSafetyAgent,
    LangChainSchedulingAgent,
    LangChainVenueAgent,
    SchedulingOperationsAgent,
    build_langchain_specialist_agents,
)
from partypilot.adapters.llm_specialist_agents import VenueAgent, build_specialist_prompt_payload
from partypilot.adapters.ollama import OllamaConfig, OllamaTimeoutError
from partypilot.domain import (
    SPECIALIST_IDENTITIES,
    AccessibilityAttribute,
    ArbitrationOutcome,
    PartyRequest,
    PlanningState,
    SpecialistAdapterVariant,
    SpecialistDecisionEnvelope,
    SpecialistDomain,
    SpecialistFailureKind,
    canonical_specialist_id,
)
from partypilot.domain.multi_agent import SpecialistAgentInput
from partypilot.domain.resources import Venue
from partypilot.ports.specialist_agent import SpecialistAgent


def _request() -> PartyRequest:
    return PartyRequest(
        location="Brooklyn, NY",
        event_date=date(2026, 9, 20),
        guest_count=24,
        total_budget=Decimal("1200.00"),
    )


def _planning_state() -> PlanningState:
    return PlanningState(revision_number=1, request=_request())


def _resource() -> Venue:
    return Venue(
        resource_id="venue-alpha",
        name="venue-alpha",
        location="Brooklyn, NY",
        price=Decimal("100.00"),
        capacity=40,
        accessibility_attributes=frozenset({AccessibilityAttribute.WHEELCHAIR_ACCESSIBLE}),
    )


def _ollama_config() -> OllamaConfig:
    return OllamaConfig.model_validate(
        {
            "base_url": "http://ollama.test:11434",
            "model": "qwen-test",
            "timeout_seconds": 5.0,
            "max_retries": 0,
        }
    )


def _agent_input(domain: SpecialistDomain) -> SpecialistAgentInput:
    return SpecialistAgentInput(
        run_id="run-1",
        specialist_id=canonical_specialist_id(domain),
        specialist_name=f"{domain.value}-agent",
        domain=domain,
        planning_state=_planning_state(),
        candidate_resources=(_resource(),),
        allowed_evidence_document_ids=(),
        scoped_evidence_documents=(),
        structured_facts=("candidate total cost is within budget",),
    )


class FakeStructuredRunnable:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[tuple[object, object | None]] = []

    def invoke(self, input: object, config: object | None = None) -> object:
        self.calls.append((input, config))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeChatModel:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[tuple[type[object] | dict[str, Any], bool, dict[str, Any]]] = []
        self.runnable = FakeStructuredRunnable(outcome)

    def with_structured_output(
        self,
        schema: type[object] | dict[str, Any],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> FakeStructuredRunnable:
        self.calls.append((schema, include_raw, kwargs))
        return self.runnable


class FakeChatOllamaConstructor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeChatModel:
        self.calls.append(kwargs)
        return FakeChatModel(_valid_envelope(SpecialistDomain.VENUE))


def _valid_envelope(domain: SpecialistDomain) -> dict[str, Any]:
    return {
        "decision": {
            "specialist_id": canonical_specialist_id(domain),
            "domain": domain.value,
            "recommendation": "accept the candidate",
            "status": ArbitrationOutcome.ACCEPT.value,
            "hard_constraints_considered": ["budget"],
            "evidence_references": [],
            "assumptions": [],
            "unresolved_uncertainties": [],
            "local_score": 0.9,
            "local_rank": 1,
            "recommended_resource_ids": [],
            "reasons_for_rejection": [],
            "dependency_decision_ids": [],
            "notes": ["ok"],
        }
    }


class FakeNativeTimeoutProvider:
    def generate(self, request: Any, *, timeout_seconds: float) -> Any:  # pragma: no cover
        raise OllamaTimeoutError("native timeout")


def test_canonical_specialist_identity_table_matches_typed_domain_model() -> None:
    assert [
        (identity.domain, identity.specialist_id, identity.specialist_name)
        for identity in SPECIALIST_IDENTITIES
    ] == [
        (SpecialistDomain.VENUE, "venue", "VenueAgent"),
        (SpecialistDomain.CATERING_SAFETY, "catering", "CateringSafetyAgent"),
        (SpecialistDomain.ACCESSIBILITY, "accessibility", "AccessibilityAgent"),
        (SpecialistDomain.SCHEDULING_OPERATIONS, "scheduling", "SchedulingAgent"),
        (SpecialistDomain.BUDGET, "budget", "BudgetAgent"),
    ]


class DummyNativeProvider:
    def generate(self, request: Any, *, timeout_seconds: float) -> Any:  # pragma: no cover
        raise AssertionError("prompt-only tests should not call the provider")


def _invalid_specialist_id_envelope() -> dict[str, object]:
    payload = _valid_envelope(SpecialistDomain.SCHEDULING_OPERATIONS)
    decision = dict(payload["decision"])
    payload["decision"] = {
        **decision,
        "specialist_id": "scheduling_operations",
        "domain": SpecialistDomain.SCHEDULING_OPERATIONS.value,
    }
    return payload


def _invented_evidence_envelope() -> dict[str, object]:
    payload = _valid_envelope(SpecialistDomain.VENUE)
    decision = dict(payload["decision"])
    payload["decision"] = {
        **decision,
        "evidence_references": [{"evidence_id": "invented-evidence", "state": "SUPPORTED"}],
    }
    return payload


def _bad_schema_envelope() -> dict[str, object]:
    payload = _valid_envelope(SpecialistDomain.VENUE)
    decision = dict(payload["decision"])
    payload["decision"] = {
        **decision,
        "status": "NOT_A_REAL_STATUS",
    }
    return payload


def test_langchain_adapter_satisfies_specialist_agent_port() -> None:
    agent = LangChainVenueAgent(chat_model=FakeChatModel(_valid_envelope(SpecialistDomain.VENUE)))

    assert isinstance(agent, SpecialistAgent)


def test_chatollama_constructor_and_structured_output_api_are_v1_compatible() -> None:
    model = ChatOllama(
        model="qwen-test",
        base_url="http://localhost:11434",
        temperature=0,
    )
    structured = model.with_structured_output(
        SpecialistDecisionEnvelope,
        include_raw=True,
    )
    signature = inspect.signature(ChatOllama.with_structured_output)

    assert "include_raw" in signature.parameters
    assert "method" in signature.parameters
    assert signature.parameters["method"].default == "json_schema"
    assert hasattr(structured, "invoke")


def test_lazy_chatollama_construction_propagates_timeout_into_sync_and_async_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_constructor = FakeChatOllamaConstructor()
    monkeypatch.setattr(langchain_specialist_agents, "_ChatOllama", fake_constructor)

    agent = LangChainVenueAgent(
        ollama_config=_ollama_config(),
        timeout_seconds=12.5,
    )

    chat_model = agent._resolve_chat_model()

    assert isinstance(chat_model, FakeChatModel)
    assert fake_constructor.calls == [
        {
            "model": "qwen-test",
            "base_url": "http://ollama.test:11434",
            "num_ctx": 8192,
            "temperature": 0,
            "sync_client_kwargs": {"timeout": 12.5},
            "async_client_kwargs": {"timeout": 12.5},
        }
    ]


def test_langchain_builder_returns_five_specialists() -> None:
    agents = build_langchain_specialist_agents(
        chat_model_factory=lambda _config: FakeChatModel(_valid_envelope(SpecialistDomain.VENUE)),
        ollama_config=_ollama_config(),
    )

    assert [agent.specialist_id for agent in agents] == [
        "venue",
        "catering",
        "accessibility",
        "scheduling",
        "budget",
    ]
    assert [agent.domain for agent in agents] == [
        SpecialistDomain.VENUE,
        SpecialistDomain.CATERING_SAFETY,
        SpecialistDomain.ACCESSIBILITY,
        SpecialistDomain.SCHEDULING_OPERATIONS,
        SpecialistDomain.BUDGET,
    ]


def test_all_langchain_specialists_satisfy_specialist_agent_port() -> None:
    agents = build_langchain_specialist_agents(
        chat_model_factory=lambda _config: FakeChatModel(_valid_envelope(SpecialistDomain.VENUE)),
        ollama_config=_ollama_config(),
    )

    assert all(isinstance(agent, SpecialistAgent) for agent in agents)


@pytest.mark.parametrize(
    ("cls", "domain"),
    [
        (LangChainVenueAgent, SpecialistDomain.VENUE),
        (LangChainCateringSafetyAgent, SpecialistDomain.CATERING_SAFETY),
        (LangChainAccessibilityAgent, SpecialistDomain.ACCESSIBILITY),
        (LangChainSchedulingAgent, SpecialistDomain.SCHEDULING_OPERATIONS),
        (LangChainBudgetAgent, SpecialistDomain.BUDGET),
    ],
)
def test_each_specialist_prompt_uses_canonical_specialist_id(
    cls: type[Any],
    domain: SpecialistDomain,
) -> None:
    agent = cls(chat_model=FakeChatModel(_valid_envelope(domain)))
    system_prompt = agent._system_prompt(_agent_input(domain), None)

    assert (
        f'Canonical specialist identity: domain="{domain.value}", '
        f'specialist_name="{agent.specialist_name}", '
        f'specialist_id="{canonical_specialist_id(domain)}".'
    ) in system_prompt
    assert f'specialist_id MUST be exactly "{canonical_specialist_id(domain)}".' in system_prompt
    assert (
        f"Never use the specialist_name, the class name, or the domain enum value "
        f'"{domain.value}" as specialist_id.'
    ) in system_prompt


@pytest.mark.parametrize(
    ("cls", "domain"),
    [
        (LangChainVenueAgent, SpecialistDomain.VENUE),
        (LangChainCateringSafetyAgent, SpecialistDomain.CATERING_SAFETY),
        (LangChainAccessibilityAgent, SpecialistDomain.ACCESSIBILITY),
        (SchedulingOperationsAgent, SpecialistDomain.SCHEDULING_OPERATIONS),
        (LangChainBudgetAgent, SpecialistDomain.BUDGET),
    ],
)
def test_each_specialist_example_uses_canonical_specialist_id(
    cls: type[Any],
    domain: SpecialistDomain,
) -> None:
    agent = cls(chat_model=FakeChatModel(_valid_envelope(domain)))
    example = agent._example_payload(domain)

    assert example.specialist_id == canonical_specialist_id(domain)
    assert example.model_dump(mode="json")["specialist_id"] == canonical_specialist_id(domain)
    assert example.model_dump(mode="json")["domain"] == domain.value


@pytest.mark.parametrize(
    ("cls", "domain"),
    [
        (LangChainVenueAgent, SpecialistDomain.VENUE),
        (LangChainCateringSafetyAgent, SpecialistDomain.CATERING_SAFETY),
        (LangChainAccessibilityAgent, SpecialistDomain.ACCESSIBILITY),
        (LangChainSchedulingAgent, SpecialistDomain.SCHEDULING_OPERATIONS),
        (LangChainBudgetAgent, SpecialistDomain.BUDGET),
    ],
)
def test_each_specialist_example_validates_against_canonical_envelope(
    cls: type[Any],
    domain: SpecialistDomain,
) -> None:
    agent = cls(chat_model=FakeChatModel(_valid_envelope(domain)))
    example = agent._example_payload(domain)

    assert example.specialist_id == canonical_specialist_id(domain)
    validated = SpecialistDecisionEnvelope.model_validate(
        {"decision": example.model_dump(mode="json")}
    )
    assert validated.decision.specialist_id == canonical_specialist_id(domain)
    assert validated.decision.domain is domain


def test_native_and_langchain_prompt_payloads_match_for_identical_inputs() -> None:
    domain = SpecialistDomain.VENUE
    agent_input = _agent_input(domain)
    native_agent = VenueAgent(DummyNativeProvider())
    langchain_agent = LangChainVenueAgent(
        chat_model=FakeChatModel(_valid_envelope(SpecialistDomain.VENUE))
    )

    native_payload = json.loads(native_agent._prompt(agent_input, None))
    langchain_payload = json.loads(langchain_agent._prompt(agent_input, None))

    assert build_specialist_prompt_payload(agent_input) == native_payload
    assert native_payload == langchain_payload
    assert native_payload["specialist_identity"] == {
        "domain": domain.value,
        "specialist_name": "VenueAgent",
        "specialist_id": "venue",
    }


def test_langchain_adapter_maps_valid_pydantic_response_to_party_pilot_types() -> None:
    agent = LangChainVenueAgent(chat_model=FakeChatModel(_valid_envelope(SpecialistDomain.VENUE)))
    outcome = agent.run(_agent_input(SpecialistDomain.VENUE))
    chat_model = agent._chat_model
    assert chat_model is not None

    assert outcome.decision is not None
    assert outcome.decision.specialist_id == "venue"
    assert outcome.trace.validation_succeeded is True
    assert outcome.trace.adapter_variant is SpecialistAdapterVariant.LANGCHAIN_CHATOLLAMA
    assert outcome.raw_structured_output == _valid_envelope(SpecialistDomain.VENUE)
    assert not hasattr(outcome.raw_structured_output, "additional_kwargs")
    assert chat_model.calls[0][0] is SpecialistDecisionEnvelope
    assert chat_model.calls[0][1] is True
    assert chat_model.runnable.calls[0][0] == [
        ("system", agent._system_prompt(_agent_input(SpecialistDomain.VENUE), None)),
        ("user", agent._prompt(_agent_input(SpecialistDomain.VENUE), None)),
    ]


def test_langchain_adapter_rejects_wrong_specialist_identity() -> None:
    agent = SchedulingOperationsAgent(chat_model=FakeChatModel(_invalid_specialist_id_envelope()))

    outcome = agent.run(_agent_input(SpecialistDomain.SCHEDULING_OPERATIONS))

    assert outcome.decision is None
    assert outcome.failure_kind is SpecialistFailureKind.SPECIALIST_DOMAIN_VALIDATION_ERROR
    assert "scheduling_operations" in (outcome.failure_reason or "")


def test_langchain_adapter_rejects_invented_evidence() -> None:
    agent = LangChainVenueAgent(chat_model=FakeChatModel(_invented_evidence_envelope()))

    outcome = agent.run(_agent_input(SpecialistDomain.VENUE))

    assert outcome.decision is None
    assert outcome.failure_kind is SpecialistFailureKind.SPECIALIST_DOMAIN_VALIDATION_ERROR
    assert "outside the allowed evidence list" in (outcome.failure_reason or "")


def test_langchain_adapter_maps_structured_output_validation_error() -> None:
    agent = LangChainVenueAgent(chat_model=FakeChatModel(_bad_schema_envelope()))

    outcome = agent.run(_agent_input(SpecialistDomain.VENUE))

    assert outcome.decision is None
    assert outcome.failure_kind is SpecialistFailureKind.STRUCTURED_OUTPUT_VALIDATION_ERROR
    assert outcome.trace.validation_succeeded is False


def test_langchain_adapter_maps_provider_timeout() -> None:
    agent = LangChainVenueAgent(chat_model=FakeChatModel(TimeoutError("timed out")))

    outcome = agent.run(_agent_input(SpecialistDomain.VENUE))

    assert outcome.decision is None
    assert outcome.failure_kind is SpecialistFailureKind.PROVIDER_TIMEOUT
    assert "timed out" in (outcome.failure_reason or "")


def test_langchain_adapter_maps_provider_connection_error() -> None:
    agent = LangChainVenueAgent(chat_model=FakeChatModel(ConnectionError("offline")))

    outcome = agent.run(_agent_input(SpecialistDomain.VENUE))

    assert outcome.decision is None
    assert outcome.failure_kind is SpecialistFailureKind.PROVIDER_CONNECTION_ERROR
    assert "offline" in (outcome.failure_reason or "")


def test_native_and_langchain_timeout_traces_are_comparable() -> None:
    native_agent = VenueAgent(FakeNativeTimeoutProvider())
    langchain_agent = LangChainVenueAgent(chat_model=FakeChatModel(httpx.ReadTimeout("timed out")))

    native_outcome = native_agent.run(_agent_input(SpecialistDomain.VENUE))
    langchain_outcome = langchain_agent.run(_agent_input(SpecialistDomain.VENUE))

    assert native_outcome.decision is None
    assert native_outcome.failure_kind is SpecialistFailureKind.PROVIDER_TIMEOUT
    assert native_outcome.trace.adapter_variant is SpecialistAdapterVariant.NATIVE_OLLAMA
    assert native_outcome.trace.validation_succeeded is False
    assert native_outcome.trace.retry_count == 0

    assert langchain_outcome.decision is None
    assert langchain_outcome.failure_kind is SpecialistFailureKind.PROVIDER_TIMEOUT
    assert langchain_outcome.trace.adapter_variant is SpecialistAdapterVariant.LANGCHAIN_CHATOLLAMA
    assert langchain_outcome.trace.validation_succeeded is False
    assert langchain_outcome.trace.retry_count == 0


def test_langchain_adapter_maps_httpx_read_timeout_without_retry() -> None:
    agent = cast(
        Any, LangChainVenueAgent(chat_model=FakeChatModel(httpx.ReadTimeout("request timed out")))
    )
    agent._max_retries = 2

    outcome = agent.run(_agent_input(SpecialistDomain.VENUE))

    assert outcome.decision is None
    assert outcome.failure_kind is SpecialistFailureKind.PROVIDER_TIMEOUT
    assert outcome.trace.validation_succeeded is False
    assert outcome.trace.retry_count == 0
    assert "timed out" in (outcome.failure_reason or "")


def test_langchain_adapter_maps_context_capacity_errors_to_provider_response_errors() -> None:
    agent = LangChainVenueAgent(chat_model=FakeChatModel(ValueError("exceed_context_size_error")))

    outcome = agent.run(_agent_input(SpecialistDomain.VENUE))

    assert outcome.decision is None
    assert outcome.failure_kind is SpecialistFailureKind.PROVIDER_RESPONSE_ERROR
    assert outcome.trace.retry_count == 0
    assert "exceed_context_size_error" in (outcome.failure_reason or "")
