"""Live multi-agent composition for PartyPilot v0.5."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from partypilot.adapters.langchain_agent_specialist_agents import (
    build_langchain_agent_specialist_agents,
)
from partypilot.adapters.langchain_specialist_agents import (
    build_langchain_specialist_agents,
)
from partypilot.adapters.llm_specialist_agents import (
    AccessibilityAgent,
    BudgetAgent,
    CateringSafetyAgent,
    LLMBaseSpecialistAgent,
    SchedulingAgent,
    VenueAgent,
)
from partypilot.adapters.ollama import OllamaConfig
from partypilot.application.multi_agent_runtime import (
    LIVE_ARCHITECTURE,
    MultiAgentPlanningRuntime,
)
from partypilot.ports.llm_provider import LLMProvider


class SpecialistAdapterKind(StrEnum):
    """Selectable specialist-adapter families for composition."""

    NATIVE = "native"
    LANGCHAIN = "langchain"
    LANGCHAIN_AGENT = "langchain_agent"


def build_specialist_agents(
    provider: LLMProvider | None = None,
    *,
    adapter_kind: SpecialistAdapterKind = SpecialistAdapterKind.NATIVE,
    timeout_seconds: float = 30.0,
    model_name: str | None = None,
    ollama_config: OllamaConfig | None = None,
    chat_model_factory: Callable[[OllamaConfig], Any] | None = None,
) -> tuple[LLMBaseSpecialistAgent, ...]:
    """Construct the five specialists for the selected adapter family."""

    if adapter_kind is SpecialistAdapterKind.NATIVE:
        if provider is None:
            raise ValueError("provider is required for the native specialist adapter")
        return (
            VenueAgent(provider, timeout_seconds=timeout_seconds, model_name=model_name),
            CateringSafetyAgent(provider, timeout_seconds=timeout_seconds, model_name=model_name),
            AccessibilityAgent(provider, timeout_seconds=timeout_seconds, model_name=model_name),
            SchedulingAgent(provider, timeout_seconds=timeout_seconds, model_name=model_name),
            BudgetAgent(provider, timeout_seconds=timeout_seconds, model_name=model_name),
        )

    resolved_config = ollama_config or OllamaConfig.from_env()
    if adapter_kind is SpecialistAdapterKind.LANGCHAIN_AGENT:
        return build_langchain_agent_specialist_agents(
            timeout_seconds=timeout_seconds,
            model_name=model_name or resolved_config.model,
            chat_model_factory=chat_model_factory,
            ollama_config=resolved_config,
        )
    return build_langchain_specialist_agents(
        timeout_seconds=timeout_seconds,
        model_name=model_name or resolved_config.model,
        chat_model_factory=chat_model_factory,
        ollama_config=resolved_config,
    )


def build_live_multi_agent_runtime(
    provider: LLMProvider | None = None,
    *,
    timeout_seconds: float = 30.0,
    model_name: str | None = None,
    max_workers: int | None = None,
    adapter_kind: SpecialistAdapterKind = SpecialistAdapterKind.NATIVE,
    ollama_config: OllamaConfig | None = None,
    chat_model_factory: Callable[[OllamaConfig], Any] | None = None,
) -> MultiAgentPlanningRuntime:
    """Construct the live multi-agent runtime with five LLM-backed specialists."""

    specialists = build_specialist_agents(
        provider,
        adapter_kind=adapter_kind,
        timeout_seconds=timeout_seconds,
        model_name=model_name,
        ollama_config=ollama_config,
        chat_model_factory=chat_model_factory,
    )
    return MultiAgentPlanningRuntime(
        specialists,
        model_name=model_name or LIVE_ARCHITECTURE,
        max_workers=max_workers,
    )
