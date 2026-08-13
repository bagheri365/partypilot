"""LangChain-backed specialist agents for PartyPilot v0.6a."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from pydantic import BaseModel, ValidationError

from partypilot.adapters.llm_specialist_agents import (
    SPECIALIST_DECISION_SCHEMA_TEXT,
    LLMBaseSpecialistAgent,
)
from partypilot.adapters.ollama import OllamaConfig
from partypilot.domain.coordination import SpecialistDomain
from partypilot.domain.multi_agent import (
    SpecialistAdapterVariant,
    SpecialistAgentInput,
    SpecialistDecisionEnvelope,
    SpecialistExecutionOutcome,
    SpecialistFailureKind,
    canonical_specialist_id,
    canonical_specialist_name,
)

_ChatOllama: Any | None = None
try:  # pragma: no cover - exercised indirectly when LangChain is installed
    from langchain_ollama import ChatOllama as _ImportedChatOllama
except ModuleNotFoundError:  # pragma: no cover - offline test environment
    pass
else:  # pragma: no cover - exercised indirectly when LangChain is installed
    _ChatOllama = _ImportedChatOllama


class _UnusedProvider:
    """Sentinel provider required by the shared native base class."""

    def generate(self, request: Any, *, timeout_seconds: float) -> Any:  # pragma: no cover
        raise RuntimeError("LangChain specialist adapter does not use the native provider")


@dataclass(frozen=True, slots=True)
class LangChainStructuredResult:
    """Normalized result returned from a LangChain structured-output runnable."""

    raw_text: str | None
    parsed: object | None
    parsing_error: Exception | None


class _StructuredOutputRunnable(Protocol):
    def invoke(self, input: Sequence[tuple[str, str]], config: Any | None = None) -> Any: ...


class _ChatModel(Protocol):
    def with_structured_output(
        self,
        schema: type[BaseModel] | dict[str, Any],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> _StructuredOutputRunnable: ...


def _load_chat_ollama() -> Any:
    if _ChatOllama is None:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("langchain-ollama is required for the LangChain specialist adapter")
    return cast(Any, _ChatOllama)


def _is_timeout_error(error: Exception) -> bool:
    name = type(error).__name__.casefold()
    message = str(error).casefold()
    return "timeout" in name or "timed out" in message


def _is_connection_error(error: Exception) -> bool:
    name = type(error).__name__.casefold()
    message = str(error).casefold()
    return any(
        fragment in name or fragment in message
        for fragment in ("connection", "connect", "unreachable", "refused")
    )


def _is_structured_output_error(error: Exception) -> bool:
    name = type(error).__name__.casefold()
    message = str(error).casefold()
    return any(
        fragment in name or fragment in message
        for fragment in ("structured", "outputparser", "parse", "validation", "json")
    )


def _is_provider_response_error(error: Exception) -> bool:
    message = str(error).casefold()
    return any(
        fragment in message
        for fragment in (
            "exceed_context_size_error",
            "context size",
            "context window",
            "context length",
            "prompt too long",
            "input is too long",
        )
    )


def _normalize_json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    return value


class LangChainBaseSpecialistAgent(LLMBaseSpecialistAgent):
    """Shared LangChain-backed specialist adapter."""

    adapter_variant: SpecialistAdapterVariant = SpecialistAdapterVariant.LANGCHAIN_CHATOLLAMA

    def __init__(
        self,
        *,
        specialist_id: str,
        specialist_name: str,
        domain: SpecialistDomain,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        model_name: str | None = None,
        chat_model: Any | None = None,
        chat_model_factory: Callable[[OllamaConfig], Any] | None = None,
        ollama_config: OllamaConfig | None = None,
    ) -> None:
        super().__init__(
            _UnusedProvider(),
            specialist_id=specialist_id,
            specialist_name=specialist_name,
            domain=domain,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            model_name=model_name,
        )
        self._chat_model = chat_model
        self._chat_model_factory = chat_model_factory
        self._ollama_config = ollama_config

    def run(self, agent_input: SpecialistAgentInput) -> SpecialistExecutionOutcome:
        started_at = datetime.now(UTC)
        retry_count = 0
        raw_text: str | None = None
        raw_structured_output: object | None = None
        failure_kind: SpecialistFailureKind | None = None
        failure_error_type: str | None = None
        failure_reason: str | None = None
        validation_feedback: str | None = None

        for attempt in range(self._max_retries + 1):
            system_prompt = self._system_prompt(agent_input, validation_feedback)
            prompt = self._prompt(agent_input, validation_feedback)
            try:
                structured_result = self._invoke_structured_output(system_prompt, prompt)
                langchain_result = self._normalize_result(structured_result)
                raw_text = langchain_result.raw_text
                raw_structured_output = _normalize_json_value(langchain_result.parsed)
                if langchain_result.parsing_error is not None:
                    parsing_error_name = type(langchain_result.parsing_error).__name__
                    failure_kind = SpecialistFailureKind.STRUCTURED_OUTPUT_VALIDATION_ERROR
                    failure_error_type = parsing_error_name
                    failure_reason = self._structured_validation_feedback(
                        error_text=f"{parsing_error_name}: {langchain_result.parsing_error}",
                        raw_text=raw_text,
                        raw_structured_output=raw_structured_output,
                    )
                    if attempt < self._max_retries:
                        retry_count += 1
                        validation_feedback = failure_reason
                        continue
                    break
                if langchain_result.parsed is None:
                    failure_kind = SpecialistFailureKind.STRUCTURED_OUTPUT_VALIDATION_ERROR
                    failure_error_type = "ValueError"
                    failure_reason = self._structured_validation_feedback(
                        error_text="structured output parser returned no parsed value",
                        raw_text=raw_text,
                        raw_structured_output=raw_structured_output,
                    )
                    if attempt < self._max_retries:
                        retry_count += 1
                        validation_feedback = failure_reason
                        continue
                    break

                envelope = self._validate_envelope(langchain_result.parsed)
                decision = self._build_decision(agent_input, envelope.decision)
            except TimeoutError as exc:
                failure_kind = SpecialistFailureKind.PROVIDER_TIMEOUT
                failure_error_type = type(exc).__name__
                failure_reason = f"{type(exc).__name__}: {exc}"
                break
            except ConnectionError as exc:
                failure_kind = SpecialistFailureKind.PROVIDER_CONNECTION_ERROR
                failure_error_type = type(exc).__name__
                failure_reason = f"{type(exc).__name__}: {exc}"
                break
            except ValidationError as exc:
                failure_kind = SpecialistFailureKind.STRUCTURED_OUTPUT_VALIDATION_ERROR
                failure_error_type = type(exc).__name__
                failure_reason = self._structured_validation_feedback(
                    error_text=str(exc),
                    raw_text=raw_text,
                    raw_structured_output=raw_structured_output,
                )
                if attempt < self._max_retries:
                    retry_count += 1
                    validation_feedback = failure_reason
                    continue
                break
            except ValueError as exc:
                failure_error_type = type(exc).__name__
                if _is_provider_response_error(exc):
                    failure_kind = SpecialistFailureKind.PROVIDER_RESPONSE_ERROR
                    failure_reason = f"{type(exc).__name__}: {exc}"
                else:
                    failure_kind = SpecialistFailureKind.SPECIALIST_DOMAIN_VALIDATION_ERROR
                    failure_reason = str(exc)
                break
            except Exception as exc:  # pragma: no cover - defensive safeguard
                failure_kind = self._classify_unexpected_exception(exc)
                failure_error_type = type(exc).__name__
                failure_reason = f"{type(exc).__name__}: {exc}"
                break
            else:
                completed_at = datetime.now(UTC)
                trace = self._build_trace(
                    agent_input=agent_input,
                    started_at=started_at,
                    completed_at=completed_at,
                    retry_count=retry_count,
                    decision=decision,
                    validation_succeeded=True,
                    failure_kind=None,
                    failure_error_type=None,
                    failure_reason=None,
                    response=None,
                )
                return SpecialistExecutionOutcome(
                    decision=decision,
                    trace=trace,
                    raw_text=raw_text,
                    raw_structured_output=raw_structured_output,
                )

        completed_at = datetime.now(UTC)
        trace = self._build_trace(
            agent_input=agent_input,
            started_at=started_at,
            completed_at=completed_at,
            retry_count=retry_count,
            decision=None,
            validation_succeeded=False,
            failure_kind=failure_kind,
            failure_error_type=failure_error_type,
            failure_reason=failure_reason,
            response=None,
        )
        return SpecialistExecutionOutcome(
            trace=trace,
            failure_kind=failure_kind,
            failure_error_type=failure_error_type,
            failure_reason=failure_reason,
            raw_text=raw_text,
            raw_structured_output=raw_structured_output,
        )

    def _normalize_result(self, result: object) -> LangChainStructuredResult:
        if isinstance(result, dict) and {"raw", "parsed", "parsing_error"} <= set(result):
            raw_message = result.get("raw")
            raw_text = cast(str | None, getattr(raw_message, "content", None))
            parsed = result.get("parsed")
            parsing_error = result.get("parsing_error")
            if raw_text is None and parsed is not None:
                raw_text = json.dumps(_normalize_json_value(parsed), sort_keys=True)
            if isinstance(parsing_error, Exception):
                return LangChainStructuredResult(raw_text, parsed, parsing_error)
            if parsing_error is None:
                return LangChainStructuredResult(raw_text, parsed, None)
            return LangChainStructuredResult(raw_text, parsed, Exception(str(parsing_error)))

        if isinstance(result, BaseModel):
            return LangChainStructuredResult(
                raw_text=json.dumps(result.model_dump(mode="json"), sort_keys=True),
                parsed=result,
                parsing_error=None,
            )

        if isinstance(result, dict):
            return LangChainStructuredResult(
                raw_text=json.dumps(_normalize_json_value(result), sort_keys=True),
                parsed=result,
                parsing_error=None,
            )

        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Exception):
            return LangChainStructuredResult(None, result[0], result[1])

        return LangChainStructuredResult(raw_text=str(result), parsed=result, parsing_error=None)

    def _validate_envelope(self, parsed: object) -> SpecialistDecisionEnvelope:
        envelope = (
            parsed
            if isinstance(parsed, SpecialistDecisionEnvelope)
            else SpecialistDecisionEnvelope.model_validate(parsed)
        )
        return envelope

    def _structured_validation_feedback(
        self,
        *,
        error_text: str,
        raw_text: str | None,
        raw_structured_output: object | None,
    ) -> str:
        lines = [
            "Validation errors:",
            error_text,
            "Canonical schema:",
            SPECIALIST_DECISION_SCHEMA_TEXT,
        ]
        if raw_text is not None:
            lines.extend(["Invalid previous text response:", raw_text])
        if raw_structured_output is not None:
            lines.extend(
                [
                    "Invalid previous structured output:",
                    json.dumps(raw_structured_output, sort_keys=True, ensure_ascii=False),
                ]
            )
        lines.append("Return only corrected JSON.")
        return "\n".join(lines)

    def _classify_unexpected_exception(self, exc: Exception) -> SpecialistFailureKind:
        if _is_timeout_error(exc):
            return SpecialistFailureKind.PROVIDER_TIMEOUT
        if _is_connection_error(exc):
            return SpecialistFailureKind.PROVIDER_CONNECTION_ERROR
        if _is_provider_response_error(exc):
            return SpecialistFailureKind.PROVIDER_RESPONSE_ERROR
        if _is_structured_output_error(exc):
            return SpecialistFailureKind.PROVIDER_RESPONSE_ERROR
        return SpecialistFailureKind.SPECIALIST_EXECUTION_ERROR

    def _resolve_chat_model(self) -> _ChatModel:
        if self._chat_model is not None:
            return cast(_ChatModel, self._chat_model)
        if self._chat_model_factory is not None:
            if self._ollama_config is None:
                raise ValueError("ollama_config is required when chat_model_factory is supplied")
            self._chat_model = self._chat_model_factory(self._ollama_config)
            return cast(_ChatModel, self._chat_model)

        chat_ollama = _load_chat_ollama()
        if self._ollama_config is None:
            raise ValueError("ollama_config is required when ChatOllama is constructed lazily")
        self._chat_model = chat_ollama(
            model=self._ollama_config.model,
            base_url=self._ollama_config.base_url,
            num_ctx=self._ollama_config.num_ctx,
            temperature=0,
            sync_client_kwargs={"timeout": self._timeout_seconds},
            async_client_kwargs={"timeout": self._timeout_seconds},
        )
        return cast(_ChatModel, self._chat_model)

    def _invoke_structured_output(self, system_prompt: str, prompt: str) -> Any:
        chat_model = self._resolve_chat_model()
        structured_model = chat_model.with_structured_output(
            SpecialistDecisionEnvelope,
            include_raw=True,
        )
        if not hasattr(structured_model, "invoke"):
            raise TypeError("structured output runnable does not provide invoke()")
        return structured_model.invoke([("system", system_prompt), ("user", prompt)])


class LangChainVenueAgent(LangChainBaseSpecialistAgent):
    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        model_name: str | None = None,
        chat_model: Any | None = None,
        chat_model_factory: Callable[[OllamaConfig], Any] | None = None,
        ollama_config: OllamaConfig | None = None,
    ) -> None:
        super().__init__(
            specialist_id=canonical_specialist_id(SpecialistDomain.VENUE),
            specialist_name=canonical_specialist_name(SpecialistDomain.VENUE),
            domain=SpecialistDomain.VENUE,
            timeout_seconds=timeout_seconds,
            model_name=model_name,
            chat_model=chat_model,
            chat_model_factory=chat_model_factory,
            ollama_config=ollama_config,
        )

    def _domain_prompt_lines(self, agent_input: SpecialistAgentInput) -> tuple[str, ...]:
        return (
            "",
            "Authority: venue eligibility, venue policy compatibility, venue-specific resource",
            "restrictions, venue-side vendor or activity compatibility, and venue facts that are",
            "directly documented in the scoped evidence.",
            "Do not decide catering safety, total budget, or non-venue scheduling feasibility.",
            "Do not issue a global accessibility conclusion beyond the venue facts that are "
            "documented.",
            "If no venue-specific violation or unresolved venue uncertainty exists, return ACCEPT.",
            "If the venue itself is not clearly compatible, recommend REJECT or "
            "HUMAN_REVIEW_REQUIRED.",
            "If another domain looks problematic but the venue is valid, return the venue "
            "decision only.",
        )


class LangChainCateringSafetyAgent(LangChainBaseSpecialistAgent):
    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        model_name: str | None = None,
        chat_model: Any | None = None,
        chat_model_factory: Callable[[OllamaConfig], Any] | None = None,
        ollama_config: OllamaConfig | None = None,
    ) -> None:
        super().__init__(
            specialist_id=canonical_specialist_id(SpecialistDomain.CATERING_SAFETY),
            specialist_name=canonical_specialist_name(SpecialistDomain.CATERING_SAFETY),
            domain=SpecialistDomain.CATERING_SAFETY,
            timeout_seconds=timeout_seconds,
            model_name=model_name,
            chat_model=chat_model,
            chat_model_factory=chat_model_factory,
            ollama_config=ollama_config,
        )

    def _domain_prompt_lines(self, agent_input: SpecialistAgentInput) -> tuple[str, ...]:
        return (
            "",
            "Authority: allergen and dietary safety, caterer policy, venue-caterer compatibility,",
            "and food-handling or vendor restrictions.",
            "Do not decide general venue eligibility, budget, or non-catering accessibility "
            "issues.",
            "Do not treat a friendly tone as proof of safe food handling.",
            "If no catering/allergen/vendor issue exists, return ACCEPT.",
            "If unrelated evidence appears, ignore it and stay within catering safety.",
        )


class LangChainAccessibilityAgent(LangChainBaseSpecialistAgent):
    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        model_name: str | None = None,
        chat_model: Any | None = None,
        chat_model_factory: Callable[[OllamaConfig], Any] | None = None,
        ollama_config: OllamaConfig | None = None,
    ) -> None:
        super().__init__(
            specialist_id=canonical_specialist_id(SpecialistDomain.ACCESSIBILITY),
            specialist_name=canonical_specialist_name(SpecialistDomain.ACCESSIBILITY),
            domain=SpecialistDomain.ACCESSIBILITY,
            timeout_seconds=timeout_seconds,
            model_name=model_name,
            chat_model=chat_model,
            chat_model_factory=chat_model_factory,
            ollama_config=ollama_config,
        )

    def _domain_prompt_lines(self, agent_input: SpecialistAgentInput) -> tuple[str, ...]:
        return (
            "",
            "Authority: accessibility requirements, physical access, accommodation evidence, and",
            "conflicting accessibility evidence.",
            "Focus on step-free access, wheelchair access, accessible restrooms, accessible paths,",
            "and room-specific accessibility constraints.",
            "Do not infer catering, budget, or scheduling failures from unrelated evidence.",
            "If no accessibility issue exists from the structured facts or allowed evidence, "
            "return ACCEPT.",
            "If another domain looks problematic but accessibility evidence is valid, return your",
            "own accessibility decision rather than escalating globally.",
        )


class LangChainSchedulingAgent(LangChainBaseSpecialistAgent):
    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        model_name: str | None = None,
        chat_model: Any | None = None,
        chat_model_factory: Callable[[OllamaConfig], Any] | None = None,
        ollama_config: OllamaConfig | None = None,
    ) -> None:
        super().__init__(
            specialist_id=canonical_specialist_id(SpecialistDomain.SCHEDULING_OPERATIONS),
            specialist_name=canonical_specialist_name(SpecialistDomain.SCHEDULING_OPERATIONS),
            domain=SpecialistDomain.SCHEDULING_OPERATIONS,
            timeout_seconds=timeout_seconds,
            model_name=model_name,
            chat_model=chat_model,
            chat_model_factory=chat_model_factory,
            ollama_config=ollama_config,
        )

    def _domain_prompt_lines(self, agent_input: SpecialistAgentInput) -> tuple[str, ...]:
        return (
            "",
            "Authority: temporal feasibility, setup and teardown windows, loading or resource",
            "timing conflicts, and operational dependencies.",
            "Use venue setup windows, caterer setup windows, and activity setup requirements "
            "when they are provided in structured facts or allowed evidence.",
            "If timing or dependency order makes the plan impossible, reject it.",
            "Do not reject solely because accessibility evidence conflicts, catering evidence is",
            "incomplete, or venue policy is ambiguous outside its scheduling implications.",
            "If no scheduling or operational conflict exists, return ACCEPT.",
            "If unrelated evidence appears, ignore it and stay within scheduling operations.",
        )


class LangChainBudgetAgent(LangChainBaseSpecialistAgent):
    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        model_name: str | None = None,
        chat_model: _ChatModel | None = None,
        chat_model_factory: Callable[[OllamaConfig], _ChatModel] | None = None,
        ollama_config: OllamaConfig | None = None,
    ) -> None:
        super().__init__(
            specialist_id=canonical_specialist_id(SpecialistDomain.BUDGET),
            specialist_name=canonical_specialist_name(SpecialistDomain.BUDGET),
            domain=SpecialistDomain.BUDGET,
            timeout_seconds=timeout_seconds,
            model_name=model_name,
            chat_model=chat_model,
            chat_model_factory=chat_model_factory,
            ollama_config=ollama_config,
        )

    def _domain_prompt_lines(self, agent_input: SpecialistAgentInput) -> tuple[str, ...]:
        return (
            "",
            "Authority: costs, fees, budget ceilings, cost dependencies, and local or global cost",
            "implications.",
            "Use the provided candidate total cost and budget ceiling from structured facts; "
            "do not invent extra fees.",
            "Do not escalate merely because unrelated domain evidence is uncertain.",
            "If the structured cost facts are within budget and no budget dependency is "
            "violated, return ACCEPT.",
            "Budget decisions do not require documentary evidence when structured facts are "
            "sufficient.",
            "If the candidate is over budget, reject it; if the cost is uncertain, ask for review.",
        )


def build_langchain_specialist_agents(
    *,
    timeout_seconds: float = 30.0,
    model_name: str | None = None,
    chat_model_factory: Callable[[OllamaConfig], Any] | None = None,
    ollama_config: OllamaConfig | None = None,
) -> tuple[LangChainBaseSpecialistAgent, ...]:
    """Construct the five LangChain-backed specialist agents."""

    return (
        LangChainVenueAgent(
            timeout_seconds=timeout_seconds,
            model_name=model_name,
            chat_model_factory=chat_model_factory,
            ollama_config=ollama_config,
        ),
        LangChainCateringSafetyAgent(
            timeout_seconds=timeout_seconds,
            model_name=model_name,
            chat_model_factory=chat_model_factory,
            ollama_config=ollama_config,
        ),
        LangChainAccessibilityAgent(
            timeout_seconds=timeout_seconds,
            model_name=model_name,
            chat_model_factory=chat_model_factory,
            ollama_config=ollama_config,
        ),
        LangChainSchedulingAgent(
            timeout_seconds=timeout_seconds,
            model_name=model_name,
            chat_model_factory=chat_model_factory,
            ollama_config=ollama_config,
        ),
        LangChainBudgetAgent(
            timeout_seconds=timeout_seconds,
            model_name=model_name,
            chat_model_factory=chat_model_factory,
            ollama_config=ollama_config,
        ),
    )


SchedulingOperationsAgent = LangChainSchedulingAgent
