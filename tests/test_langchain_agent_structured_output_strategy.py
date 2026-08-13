from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel


class TinySchema(BaseModel):
    answer: str


@tool
def ping() -> str:
    """Return pong."""

    return "pong"


class FakeBoundRunnable:
    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self.calls: list[tuple[object, object | None]] = []

    def invoke(self, messages: object, config: object | None = None) -> AIMessage:
        self.calls.append((messages, config))
        if self.mode == "provider":
            return AIMessage(content='{"answer":"ok"}')
        if self.mode == "tool":
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "TinySchema",
                        "args": {"answer": "ok"},
                        "id": "structured-response",
                        "type": "tool_call",
                    }
                ],
            )
        raise RuntimeError(f"unknown mode: {self.mode}")


def _patch_chat_ollama(monkeypatch: Any, *, bound: FakeBoundRunnable) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_bind(self: ChatOllama, **kwargs: Any) -> FakeBoundRunnable:
        captured["bind"] = kwargs
        return bound

    def fake_bind_tools(
        self: ChatOllama,
        tools: list[Any],
        *,
        tool_choice: str | dict[str, Any] | bool | None = None,
        **kwargs: Any,
    ) -> FakeBoundRunnable:
        captured["bind_tools"] = {
            "tool_names": [getattr(tool, "name", None) for tool in tools],
            "tool_choice": tool_choice,
            **kwargs,
        }
        return bound

    monkeypatch.setattr(ChatOllama, "bind", fake_bind, raising=True)
    monkeypatch.setattr(ChatOllama, "bind_tools", fake_bind_tools, raising=True)
    return captured


def test_auto_response_format_uses_tool_strategy_for_chatollama_without_profile(
    monkeypatch: Any,
) -> None:
    bound = FakeBoundRunnable(mode="tool")
    captured = _patch_chat_ollama(monkeypatch, bound=bound)

    agent = create_agent(
        model=ChatOllama(model="qwen-test", base_url="http://localhost:11434", temperature=0),
        tools=[ping],
        response_format=TinySchema,
        name="test",
    )
    result = agent.invoke({"messages": [HumanMessage(content="hi")]})

    assert result["structured_response"] == TinySchema(answer="ok")
    assert "bind_tools" in captured
    assert captured["bind_tools"]["tool_names"] == ["ping", "TinySchema"]
    assert captured["bind_tools"]["tool_choice"] == "any"
    assert "response_format" not in captured["bind_tools"]


def test_explicit_tool_strategy_binds_synthetic_schema_tool(
    monkeypatch: Any,
) -> None:
    bound = FakeBoundRunnable(mode="tool")
    captured = _patch_chat_ollama(monkeypatch, bound=bound)

    agent = create_agent(
        model=ChatOllama(model="qwen-test", base_url="http://localhost:11434", temperature=0),
        tools=[ping],
        response_format=ToolStrategy(TinySchema, handle_errors=False),
        name="test",
    )
    result = agent.invoke({"messages": [HumanMessage(content="hi")]})

    assert result["structured_response"] == TinySchema(answer="ok")
    assert "bind_tools" in captured
    assert captured["bind_tools"]["tool_names"] == ["ping", "TinySchema"]
    assert captured["bind_tools"]["tool_choice"] == "any"
    assert "response_format" not in captured["bind_tools"]


def test_explicit_provider_strategy_binds_response_format_without_schema_tool(
    monkeypatch: Any,
) -> None:
    bound = FakeBoundRunnable(mode="provider")
    captured = _patch_chat_ollama(monkeypatch, bound=bound)

    agent = create_agent(
        model=ChatOllama(model="qwen-test", base_url="http://localhost:11434", temperature=0),
        tools=[ping],
        response_format=ProviderStrategy(TinySchema),
        name="test",
    )
    result = agent.invoke({"messages": [HumanMessage(content="hi")]})

    assert result["structured_response"] == TinySchema(answer="ok")
    assert "bind_tools" in captured
    assert captured["bind_tools"]["tool_names"] == ["ping"]
    assert "tool_choice" in captured["bind_tools"]
    assert "response_format" in captured["bind_tools"]
    assert captured["bind_tools"]["response_format"]["type"] == "json_schema"
