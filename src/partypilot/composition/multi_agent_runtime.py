"""Live multi-agent composition for PartyPilot v0.5."""

from __future__ import annotations

from partypilot.adapters.llm_specialist_agents import (
    AccessibilityAgent,
    BudgetAgent,
    CateringSafetyAgent,
    LLMBaseSpecialistAgent,
    SchedulingAgent,
    VenueAgent,
)
from partypilot.application.multi_agent_runtime import (
    LIVE_ARCHITECTURE,
    MultiAgentPlanningRuntime,
)
from partypilot.ports.llm_provider import LLMProvider


def build_live_multi_agent_runtime(
    provider: LLMProvider,
    *,
    timeout_seconds: float = 30.0,
    model_name: str | None = None,
    max_workers: int | None = None,
) -> MultiAgentPlanningRuntime:
    """Construct the live multi-agent runtime with five LLM-backed specialists."""

    specialists: tuple[LLMBaseSpecialistAgent, ...] = (
        VenueAgent(provider, timeout_seconds=timeout_seconds, model_name=model_name),
        CateringSafetyAgent(provider, timeout_seconds=timeout_seconds, model_name=model_name),
        AccessibilityAgent(provider, timeout_seconds=timeout_seconds, model_name=model_name),
        SchedulingAgent(provider, timeout_seconds=timeout_seconds, model_name=model_name),
        BudgetAgent(provider, timeout_seconds=timeout_seconds, model_name=model_name),
    )
    return MultiAgentPlanningRuntime(
        specialists,
        model_name=model_name or LIVE_ARCHITECTURE,
        max_workers=max_workers,
    )
