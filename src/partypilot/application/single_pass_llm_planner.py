"""Single-pass, ungrounded LLM baseline planner."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, ValidationError

from partypilot.application.budget_validation import CostComponent, calculate_total_cost
from partypilot.application.candidate_filtering import CandidateRequirements
from partypilot.application.constraint_engine import (
    ConstraintEngineInput,
    ConstraintEngineResult,
    validate_constraints,
)
from partypilot.domain.constraints import Constraint, ConstraintOperator, ConstraintType
from partypilot.domain.party_plan import PartyPlan
from partypilot.domain.party_request import PartyRequest
from partypilot.domain.resources import AccessibilityAttribute
from partypilot.domain.temporal import TimeWindow
from partypilot.ports.llm_provider import (
    GenerationRequest,
    LLMProvider,
    StructuredOutputExpectation,
    UsageMetadata,
)


class LLMPlanFailureCategory(StrEnum):
    HALLUCINATED_RESOURCES = "hallucinated_resources"
    UNSUPPORTED_ASSUMPTIONS = "unsupported_assumptions"
    CONSTRAINT_VIOLATIONS = "constraint_violations"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_INVALID = "schema_invalid"
    ARITHMETIC_MISTAKE = "arithmetic_mistake"
    PROVIDER_ERROR = "provider_error"


SINGLE_PASS_PROMPT_VERSION = "single-pass-v1"


class SinglePassPlannerError(Exception):
    """Base typed error for the single-pass planner."""


class SinglePassPlannerProviderError(SinglePassPlannerError):
    """Provider failure translated at the application boundary."""


class SinglePassLLMResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    plan: PartyPlan | None
    validation: ConstraintEngineResult | None
    failure_categories: tuple[LLMPlanFailureCategory, ...]
    errors: tuple[str, ...] = ()
    usage: UsageMetadata | None = None

    @property
    def feasible(self) -> bool:
        return (
            self.plan is not None
            and self.validation is not None
            and self.validation.feasible
            and not self.failure_categories
        )


class SinglePassLLMPlanner:
    """Generate exactly once, then validate deterministically without repair."""

    def __init__(self, provider: LLMProvider, *, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._provider = provider
        self._timeout_seconds = timeout_seconds

    def plan(self, request: PartyRequest) -> SinglePassLLMResult:
        generation_request = GenerationRequest(
            system_prompt=(
                "Return one PartyPlan as structured JSON only. "
                "Do not claim that resource facts are verified."
            ),
            prompt=json.dumps(request.model_dump(mode="json"), sort_keys=True),
            structured_output=StructuredOutputExpectation(
                schema_name="PartyPlan",
                json_schema=PartyPlan.model_json_schema(),
            ),
        )
        try:
            response = self._provider.generate(
                generation_request, timeout_seconds=self._timeout_seconds
            )
        except Exception as exc:  # provider implementations are translated here
            raise SinglePassPlannerProviderError(
                f"LLM provider generation failed: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            if response.structured_output is None:
                return SinglePassLLMResult(
                    plan=None,
                    validation=None,
                    failure_categories=(LLMPlanFailureCategory.MALFORMED_JSON,),
                    errors=("Provider response could not be parsed as JSON.",),
                    usage=response.usage,
                )
            if not isinstance(response.structured_output, dict):
                return SinglePassLLMResult(
                    plan=None,
                    validation=None,
                    failure_categories=(LLMPlanFailureCategory.SCHEMA_INVALID,),
                    errors=("Provider returned structured output with an unexpected shape.",),
                    usage=response.usage,
                )
            plan = PartyPlan.model_validate(response.structured_output)
        except (ValidationError, ValueError, TypeError) as exc:
            return SinglePassLLMResult(
                plan=None,
                validation=None,
                failure_categories=(LLMPlanFailureCategory.SCHEMA_INVALID,),
                errors=(str(exc),),
                usage=response.usage,
            )

        components = tuple(
            CostComponent(component_id=r.resource_id, description=r.name, amount=r.price)
            for r in plan.resources
        )
        actual_total = calculate_total_cost(components)
        requirements, hard_constraints = _validation_inputs(request)
        validation = validate_constraints(
            ConstraintEngineInput(
                hard_constraints=hard_constraints,
                selected_resources=plan.resources,
                candidate_requirements=requirements,
                budget=request.total_budget,
                cost_components=components,
            )
        )

        failures: list[LLMPlanFailureCategory] = []
        errors: list[str] = []
        if plan.resources:
            failures.append(LLMPlanFailureCategory.HALLUCINATED_RESOURCES)
            errors.append(
                "Resource claims are ungrounded because this baseline has no "
                "retrieval or resource tools."
            )
        if plan.assumptions:
            failures.append(LLMPlanFailureCategory.UNSUPPORTED_ASSUMPTIONS)
            errors.append("The plan contains unsupported assumptions.")
        if plan.claimed_total_cost != actual_total:
            failures.append(LLMPlanFailureCategory.ARITHMETIC_MISTAKE)
            errors.append(
                f"Claimed total {plan.claimed_total_cost} does not equal calculated "
                f"total {actual_total}."
            )
        if not validation.feasible:
            failures.append(LLMPlanFailureCategory.CONSTRAINT_VIOLATIONS)
            errors.append("Deterministic constraint validation failed.")

        return SinglePassLLMResult(
            plan=plan,
            validation=validation,
            failure_categories=tuple(failures),
            errors=tuple(errors),
            usage=response.usage,
        )


def _validation_inputs(
    request: PartyRequest,
) -> tuple[CandidateRequirements, tuple[Constraint, ...]]:
    accessibility: set[AccessibilityAttribute] = set()
    unsupported_needs = False
    for need in request.accessibility_needs:
        try:
            accessibility.add(AccessibilityAttribute(need.strip().casefold().replace(" ", "_")))
        except ValueError:
            unsupported_needs = True

    event_window = None
    if request.event_time is not None:
        start = datetime.combine(request.event_date, request.event_time)
        event_window = TimeWindow(start=start, end=start + timedelta(hours=2))

    requirements = CandidateRequirements(
        location=request.location,
        guest_count=request.guest_count,
        child_age=request.child_age,
        child_age_range=request.child_age_range,
        availability=event_window,
        accessibility=frozenset(accessibility),
    )
    constraints = [
        Constraint(
            identifier="request-location",
            key="location",
            operator=ConstraintOperator.EQ,
            value=request.location,
            constraint_type=ConstraintType.HARD,
            description="Resources must match location.",
        ),
        Constraint(
            identifier="request-capacity",
            key="guest_count",
            operator=ConstraintOperator.GTE,
            value=request.guest_count,
            constraint_type=ConstraintType.HARD,
            description="Resources must support guest count.",
        ),
        Constraint(
            identifier="request-budget",
            key="total_budget",
            operator=ConstraintOperator.LTE,
            value=request.total_budget,
            constraint_type=ConstraintType.HARD,
            description="Plan must fit budget.",
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
                description="Resources must support child age.",
            )
        )
    if event_window is not None:
        constraints.append(
            Constraint(
                identifier="request-availability",
                key="availability",
                operator=ConstraintOperator.EQ,
                value="event_window",
                constraint_type=ConstraintType.HARD,
                description="Resources must be available.",
            )
        )
    if accessibility:
        constraints.append(
            Constraint(
                identifier="request-accessibility",
                key="accessibility",
                operator=ConstraintOperator.CONTAINS,
                value=tuple(x.value for x in sorted(accessibility, key=str)),
                constraint_type=ConstraintType.HARD,
                description="Resources must satisfy accessibility needs.",
            )
        )
    for key, present in (
        ("allergies", bool(request.allergies)),
        ("dietary_restrictions", bool(request.dietary_restrictions)),
        ("other_constraints", bool(request.other_constraints)),
        ("unsupported_accessibility", unsupported_needs),
    ):
        if present:
            constraints.append(
                Constraint(
                    identifier=f"request-{key}",
                    key=key,
                    operator=ConstraintOperator.EQ,
                    value=True,
                    constraint_type=ConstraintType.HARD,
                    description=f"Request includes {key}.",
                )
            )
    return requirements, tuple(constraints)
