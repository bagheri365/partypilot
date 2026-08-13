# ADR 003: Live LLM Specialists Experimental

## Status

Accepted experimentally

## Context

PartyPilot needs to test whether independently invoked live specialists behave like real agents rather than simulated components.

## Decision

Use live LLM specialists as a real multi-agent execution path, but keep the result experimentally constrained by provider reliability and timeout behavior.

## Evidence

The canonical v0.5 live multi-agent run shows real specialist execution with independent provider attempts and isolated failures:

- `evals/results/v0_5/llm_multi_agent/20260812T060624Z/v0_5_llm_multi_agent.json`
- `evals/results/v0_5/llm_multi_agent/20260812T060624Z/v0_5_llm_multi_agent.md`

The measured run retained final decision accuracy while exposing substantial live-provider latency and timeout sensitivity.

## Consequences

Live specialists are meaningful as an experimental architecture milestone, but provider reliability remains a first-order constraint on their practical use.
