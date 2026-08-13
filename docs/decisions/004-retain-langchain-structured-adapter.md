# ADR 004: Retain LangChain Structured Adapter

## Status

Accepted

## Context

The specialist layer needs a stable adapter boundary that standardizes model invocation without changing PartyPilot semantics.

## Decision

Retain the LangChain structured `ChatOllama` adapter.

## Evidence

The canonical v0.6 controlled evaluation retained the structured adapter path while comparing it with the tool-using path:

- `evals/results/v0_6/langchain/20260813T042641Z/v0_6_langchain_controlled_evaluation.json`
- `evals/results/v0_6/langchain/20260813T042641Z/v0_6_langchain_controlled_evaluation.md`

The structured adapter remained the useful standardization layer for PartyPilot's specialist output contracts.

## Consequences

LangChain stays in adapter/composition infrastructure as the default structured specialist path, while higher-level agentic paths remain separate experimental variants.
