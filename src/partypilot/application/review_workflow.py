"""Framework-independent human-review workflow models for PartyPilot."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

NonEmptyString = str


class HumanReviewAction(StrEnum):
    """Supported safe actions for a suspended human review."""

    APPROVE_CURRENT_PLAN = "approve_current_plan"
    REJECT_CURRENT_PLAN = "reject_current_plan"
    REQUEST_REPLAN = "request_replan"


class HumanReviewRequest(BaseModel):
    """JSON-serializable request payload presented to a reviewer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: NonEmptyString = Field(min_length=1)
    scenario_id: NonEmptyString = Field(min_length=1)
    planning_revision: int = Field(ge=0)
    review_reason: NonEmptyString = Field(min_length=1)
    selected_resource_ids: tuple[NonEmptyString, ...] = ()
    controlling_evidence_ids: tuple[NonEmptyString, ...] = ()
    unresolved_issues: tuple[NonEmptyString, ...] = ()
    targeted_domains: tuple[NonEmptyString, ...] = ()
    safe_actions: tuple[HumanReviewAction, ...] = Field(
        default_factory=lambda: (
            HumanReviewAction.APPROVE_CURRENT_PLAN,
            HumanReviewAction.REJECT_CURRENT_PLAN,
            HumanReviewAction.REQUEST_REPLAN,
        )
    )
    notes: tuple[NonEmptyString, ...] = ()

    @field_validator(
        "execution_id",
        "scenario_id",
        "review_reason",
        "selected_resource_ids",
        "controlling_evidence_ids",
        "unresolved_issues",
        "targeted_domains",
        "notes",
    )
    @classmethod
    def validate_unique_text_items(cls, value: str | tuple[str, ...]) -> str | tuple[str, ...]:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError("value cannot be blank")
            return normalized
        normalized_items = tuple(item.strip() for item in value)
        if any(not item for item in normalized_items):
            raise ValueError("values cannot be blank")
        if len(set(normalized_items)) != len(normalized_items):
            raise ValueError("values must be unique")
        return normalized_items


class HumanReviewResponse(BaseModel):
    """Validated human response used to resume a suspended execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: NonEmptyString = Field(min_length=1)
    planning_revision: int = Field(ge=0)
    action: HumanReviewAction
    candidate_resource_ids: tuple[NonEmptyString, ...] = ()
    notes: tuple[NonEmptyString, ...] = ()

    @field_validator("execution_id", "candidate_resource_ids", "notes")
    @classmethod
    def validate_unique_text_items(cls, value: str | tuple[str, ...]) -> str | tuple[str, ...]:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError("value cannot be blank")
            return normalized
        normalized_items = tuple(item.strip() for item in value)
        if any(not item for item in normalized_items):
            raise ValueError("values cannot be blank")
        if len(set(normalized_items)) != len(normalized_items):
            raise ValueError("values must be unique")
        return normalized_items


__all__ = [
    "HumanReviewAction",
    "HumanReviewRequest",
    "HumanReviewResponse",
]
