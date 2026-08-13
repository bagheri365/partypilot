# ADR 005: Reject `create_agent` as Default

## Status

Rejected as default

## Context

Tool-using specialist agents were explored in v0.6 as a possible next step beyond structured model output.

## Decision

Do not promote `create_agent` to the default specialist strategy.

## Evidence

The canonical v0.6 evaluation recorded zero tool calls for the `langchain_agent` variant and no decision-quality advantage over the structured adapter path:

- `evals/results/v0_6/langchain/20260813T042641Z/v0_6_langchain_controlled_evaluation.md`
- `evals/results/v0_6/langchain/langchain_agent/20260813T042641Z/20260813T042641Z-1-3/run.json`

## Consequences

`create_agent` remains an experimental comparison path. PartyPilot should not add tools or prompt tuning just to manufacture tool use.
