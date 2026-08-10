"""Shared helpers for baseline metric classification and aggregation."""

from __future__ import annotations

from enum import StrEnum

from partypilot.application.single_pass_llm_planner import (
    LLMPlanFailureCategory,
    SinglePassLLMResult,
)
from partypilot.domain.feasibility import FeasibilityOutcome


class BaselineFailureLabel(StrEnum):
    """Typed labels for per-scenario baseline failures."""

    PROVIDER_FAILURE = "provider_failure"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_INVALID = "schema_invalid"
    HARD_CONSTRAINT_VIOLATION = "hard_constraint_violation"
    FEASIBILITY_MISCLASSIFICATION = "feasibility_misclassification"
    ARITHMETIC_ERROR = "arithmetic_error"
    HALLUCINATED_RESOURCE = "hallucinated_resource"
    UNSUPPORTED_ASSUMPTION = "unsupported_assumption"
    VALID = "valid"


def classify_single_pass_failure_labels(
    result: SinglePassLLMResult,
    expected_outcome: FeasibilityOutcome,
) -> tuple[BaselineFailureLabel, ...]:
    """Derive a deterministic failure taxonomy for one LLM scenario."""
    labels: list[BaselineFailureLabel] = []

    if LLMPlanFailureCategory.PROVIDER_ERROR in result.failure_categories:
        labels.append(BaselineFailureLabel.PROVIDER_FAILURE)

    if result.plan is None:
        if _looks_like_malformed_json(result.errors):
            labels.append(BaselineFailureLabel.MALFORMED_JSON)
        else:
            labels.append(BaselineFailureLabel.SCHEMA_INVALID)
    else:
        if LLMPlanFailureCategory.HALLUCINATED_RESOURCES in result.failure_categories:
            labels.append(BaselineFailureLabel.HALLUCINATED_RESOURCE)
        if LLMPlanFailureCategory.UNSUPPORTED_ASSUMPTIONS in result.failure_categories:
            labels.append(BaselineFailureLabel.UNSUPPORTED_ASSUMPTION)
        if LLMPlanFailureCategory.ARITHMETIC_MISTAKE in result.failure_categories:
            labels.append(BaselineFailureLabel.ARITHMETIC_ERROR)
        if result.validation is not None and not result.validation.feasible:
            labels.append(BaselineFailureLabel.HARD_CONSTRAINT_VIOLATION)

    if result.plan is None or _predicted_outcome(result) is not expected_outcome:
        labels.append(BaselineFailureLabel.FEASIBILITY_MISCLASSIFICATION)

    if not labels:
        labels.append(BaselineFailureLabel.VALID)

    return tuple(dict.fromkeys(labels))


def _looks_like_malformed_json(errors: tuple[str, ...]) -> bool:
    if not errors:
        return False
    text = " ".join(errors).casefold()
    return (
        "invalid json" in text
        or "json object" in text
        or "missing text output" in text
        or "did not return a json object" in text
    )


def _predicted_outcome(result: SinglePassLLMResult) -> FeasibilityOutcome:
    if result.validation is not None and result.validation.feasible and result.plan is not None:
        return FeasibilityOutcome.FEASIBLE
    if result.validation is not None and result.validation.unresolved_constraint_ids:
        return FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
    return FeasibilityOutcome.NO_FEASIBLE_PLAN
