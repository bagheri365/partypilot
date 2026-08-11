"""Reusable deterministic test doubles for PartyPilot."""

from partypilot.testing.failure_injection import (
    InjectedProviderUnavailableError,
    InjectedToolError,
    RaisingEvidenceRetriever,
    ScriptedLLMProvider,
    StaticEvidenceRetriever,
    empty_retriever,
    malformed_structured_output_provider,
    repeated_invalid_output_provider,
    stale_evidence,
    timeout_provider,
    tool_exception_retriever,
    unavailable_provider,
    wrong_vendor_evidence,
)

__all__ = [
    "InjectedProviderUnavailableError",
    "InjectedToolError",
    "RaisingEvidenceRetriever",
    "ScriptedLLMProvider",
    "StaticEvidenceRetriever",
    "empty_retriever",
    "malformed_structured_output_provider",
    "repeated_invalid_output_provider",
    "stale_evidence",
    "timeout_provider",
    "tool_exception_retriever",
    "unavailable_provider",
    "wrong_vendor_evidence",
]
