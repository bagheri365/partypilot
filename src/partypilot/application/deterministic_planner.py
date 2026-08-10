"""Transparent deterministic baseline planner for PartyPilot."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from itertools import product

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application.budget_validation import CostComponent, validate_budget
from partypilot.application.candidate_filtering import CandidateRequirements, filter_candidates
from partypilot.application.constraint_engine import ConstraintEngineInput, validate_constraints
from partypilot.domain.constraints import Constraint, ConstraintOperator, ConstraintType
from partypilot.domain.party_request import PartyRequest
from partypilot.domain.resources import AccessibilityAttribute, Resource, ResourceCategory
from partypilot.domain.temporal import Duration, TimeWindow
from partypilot.ports.resource_store import ResourceSearchCriteria, ResourceStore


class PreferenceWeights(BaseModel):
    """Simple documented baseline ranking weights.

    The baseline currently optimizes only total cost. Lower cost is better, so
    ``cost`` is multiplied by the negative total cost to produce a score.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cost: Decimal = Field(default=Decimal("1"), gt=0)


class PlannerConfig(BaseModel):
    """Deterministic planner configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_duration: Duration = Field(default_factory=lambda: Duration.hours(2))
    preference_weights: PreferenceWeights = Field(default_factory=PreferenceWeights)
    max_candidates: int = Field(default=10, gt=0)


class PlanCandidate(BaseModel):
    """A validated deterministic resource combination."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resources: tuple[Resource, ...]
    total_cost: Decimal = Field(ge=0)
    score: Decimal

    @property
    def resource_ids(self) -> tuple[str, ...]:
        """Return resource IDs in deterministic category order."""
        return tuple(resource.resource_id for resource in self.resources)


class PlannerResult(BaseModel):
    """Result of deterministic baseline planning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: tuple[PlanCandidate, ...]
    unresolved_request_constraints: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        """Whether at least one feasible candidate was found."""
        return bool(self.candidates)


class DeterministicPlanner:
    """Non-LLM baseline planner using structured resources and deterministic rules."""

    def __init__(self, resource_store: ResourceStore, config: PlannerConfig | None = None) -> None:
        self._resource_store = resource_store
        self._config = config or PlannerConfig()

    def plan(self, request: PartyRequest) -> PlannerResult:
        """Return the best feasible combinations for a validated party request."""
        event_window = self._event_window(request)
        accessibility, unresolved_accessibility = self._parse_accessibility(request)
        unresolved = list(unresolved_accessibility)
        if request.allergies:
            unresolved.append("allergies")
        if request.dietary_restrictions:
            unresolved.append("dietary_restrictions")
        if request.other_constraints:
            unresolved.append("other_constraints")

        # These requirements cannot yet be validated from structured resource data.
        if unresolved:
            return PlannerResult(candidates=(), unresolved_request_constraints=tuple(unresolved))

        by_category: dict[ResourceCategory, tuple[Resource, ...]] = {}
        for category in (
            ResourceCategory.VENUE,
            ResourceCategory.CATERER,
            ResourceCategory.ACTIVITY,
        ):
            resources = self._resource_store.search(
                ResourceSearchCriteria(
                    location=request.location,
                    minimum_capacity=request.guest_count,
                    category=category,
                )
            )
            resources = tuple(
                resource for resource in resources if self._available_on_date(resource, request)
            )
            requirements = CandidateRequirements(
                location=request.location,
                guest_count=request.guest_count,
                child_age=request.child_age,
                child_age_range=request.child_age_range,
                availability=event_window,
                accessibility=accessibility
                if category is not ResourceCategory.CATERER
                else frozenset(),
            )
            by_category[category] = filter_candidates(resources, requirements).eligible

        if any(not by_category[category] for category in by_category):
            return PlannerResult(candidates=())

        hard_constraints = self._hard_constraints(request, event_window, accessibility)
        feasible: list[PlanCandidate] = []
        for combination in product(
            by_category[ResourceCategory.VENUE],
            by_category[ResourceCategory.CATERER],
            by_category[ResourceCategory.ACTIVITY],
        ):
            resources = tuple(combination)
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

            requirements = CandidateRequirements(
                location=request.location,
                guest_count=request.guest_count,
                child_age=request.child_age,
                child_age_range=request.child_age_range,
                availability=event_window,
            )
            validation = validate_constraints(
                ConstraintEngineInput(
                    hard_constraints=hard_constraints,
                    selected_resources=resources,
                    candidate_requirements=requirements,
                    budget=request.total_budget,
                    cost_components=components,
                )
            )
            if not validation.feasible:
                continue

            score = -(budget.total_cost * self._config.preference_weights.cost)
            feasible.append(
                PlanCandidate(resources=resources, total_cost=budget.total_cost, score=score)
            )

        feasible.sort(key=lambda item: (-item.score, item.resource_ids))
        return PlannerResult(candidates=tuple(feasible[: self._config.max_candidates]))

    def _event_window(self, request: PartyRequest) -> TimeWindow | None:
        if request.event_time is None:
            return None
        start = datetime.combine(request.event_date, request.event_time)
        return TimeWindow(start=start, end=start + self._config.event_duration.value)

    @staticmethod
    def _available_on_date(resource: Resource, request: PartyRequest) -> bool:
        return any(
            window.start.date() <= request.event_date <= window.end.date()
            for window in resource.availability
        )

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
    def _hard_constraints(
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
                    value="event_window",
                    constraint_type=ConstraintType.HARD,
                    description="Resources must be available during the event window.",
                )
            )
        if accessibility:
            constraints.append(
                Constraint(
                    identifier="request-accessibility",
                    key="accessibility",
                    operator=ConstraintOperator.CONTAINS,
                    value=tuple(attribute.value for attribute in sorted(accessibility, key=str)),
                    constraint_type=ConstraintType.HARD,
                    description="Resources must satisfy structured accessibility needs.",
                )
            )
        return tuple(constraints)
