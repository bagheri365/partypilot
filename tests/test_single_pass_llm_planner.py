from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from partypilot.adapters.ollama import OllamaConnectionError
from partypilot.application.single_pass_llm_planner import (
    LLMPlanFailureCategory,
    SinglePassLLMPlanner,
    SinglePassPlannerProviderError,
)
from partypilot.domain.party_request import PartyRequest
from partypilot.ports.llm_provider import (
    FailingFakeLLMProvider,
    FakeLLMProvider,
    GenerationResponse,
)


def request(**updates: Any) -> PartyRequest:
    values: dict[str, Any] = dict(
        location="Boston", event_date=date(2026, 9, 1), guest_count=10, total_budget=Decimal("500")
    )
    values.update(updates)
    return PartyRequest(**values)


def resource(*, price: str = "100", location: str = "Boston", capacity: int = 20) -> dict[str, Any]:
    return {
        "resource_id": "llm-venue",
        "name": "Imagined Venue",
        "location": location,
        "price": price,
        "capacity": capacity,
        "availability": [],
        "age_restrictions": None,
        "accessibility_attributes": [],
        "category": "venue",
    }


def test_single_pass_records_ungrounded_resources_without_repair() -> None:
    provider = FakeLLMProvider(
        [
            GenerationResponse(
                text="",
                structured_output={
                    "resources": [resource()],
                    "claimed_total_cost": "100",
                    "assumptions": [],
                },
            )
        ]
    )
    result = SinglePassLLMPlanner(provider).plan(request())
    assert result.plan is not None
    assert result.validation is not None and result.validation.feasible
    assert result.failure_categories == (LLMPlanFailureCategory.HALLUCINATED_RESOURCES,)
    assert len(provider.requests) == 1


def test_single_pass_records_constraint_and_arithmetic_failures() -> None:
    provider = FakeLLMProvider(
        [
            GenerationResponse(
                text="",
                structured_output={
                    "resources": [resource(price="600", location="Cambridge", capacity=5)],
                    "claimed_total_cost": "42",
                    "assumptions": [],
                },
            )
        ]
    )
    result = SinglePassLLMPlanner(provider).plan(request())
    assert LLMPlanFailureCategory.HALLUCINATED_RESOURCES in result.failure_categories
    assert LLMPlanFailureCategory.ARITHMETIC_MISTAKE in result.failure_categories
    assert LLMPlanFailureCategory.CONSTRAINT_VIOLATIONS in result.failure_categories


def test_single_pass_records_unsupported_assumptions() -> None:
    provider = FakeLLMProvider(
        [
            GenerationResponse(
                text="",
                structured_output={
                    "resources": [],
                    "claimed_total_cost": "0",
                    "assumptions": ["Outdoor space is available"],
                },
            )
        ]
    )
    result = SinglePassLLMPlanner(provider).plan(request())
    assert result.failure_categories == (LLMPlanFailureCategory.UNSUPPORTED_ASSUMPTIONS,)


def test_schema_invalid_structured_output_is_recorded() -> None:
    provider = FakeLLMProvider(
        [GenerationResponse(text="not json", structured_output={"wrong": "shape"})]
    )
    result = SinglePassLLMPlanner(provider).plan(request())
    assert result.plan is None
    assert result.failure_categories == (LLMPlanFailureCategory.SCHEMA_INVALID,)


def test_malformed_json_is_recorded() -> None:
    provider = FakeLLMProvider([GenerationResponse(text="not json", structured_output=None)])
    result = SinglePassLLMPlanner(provider).plan(request())
    assert result.plan is None
    assert result.failure_categories == (LLMPlanFailureCategory.MALFORMED_JSON,)


def test_request_with_unsupported_safety_constraint_fails_validation() -> None:
    provider = FakeLLMProvider(
        [
            GenerationResponse(
                text="",
                structured_output={"resources": [], "claimed_total_cost": "0", "assumptions": []},
            )
        ]
    )
    result = SinglePassLLMPlanner(provider).plan(request(allergies=("peanuts",)))
    assert result.validation is not None and not result.validation.feasible
    assert LLMPlanFailureCategory.CONSTRAINT_VIOLATIONS in result.failure_categories


def test_provider_error_is_translated() -> None:
    planner = SinglePassLLMPlanner(FailingFakeLLMProvider(OllamaConnectionError("offline")))
    with pytest.raises(SinglePassPlannerProviderError, match="OllamaConnectionError"):
        planner.plan(request())


def test_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError):
        SinglePassLLMPlanner(FakeLLMProvider([]), timeout_seconds=0)
