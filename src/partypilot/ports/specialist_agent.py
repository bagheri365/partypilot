"""Provider-neutral specialist-agent port for PartyPilot v0.5."""

from __future__ import annotations

from typing import Protocol

from partypilot.domain.coordination import SpecialistDomain
from partypilot.domain.multi_agent import (
    SpecialistAgentInput,
    SpecialistExecutionOutcome,
)


class SpecialistAgent(Protocol):
    """Port for a single typed specialist agent."""

    specialist_id: str
    specialist_name: str
    domain: SpecialistDomain

    def run(self, agent_input: SpecialistAgentInput) -> SpecialistExecutionOutcome:
        """Execute the specialist on a scoped input and return a typed outcome."""
        ...
