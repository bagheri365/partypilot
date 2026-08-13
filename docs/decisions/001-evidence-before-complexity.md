# ADR 001: Evidence Before Complexity

## Status

Accepted

## Context

PartyPilot is a research system, not a framework demo. The project needs a disciplined way to decide when additional orchestration or agentic machinery is warranted.

## Decision

Introduce complexity only after a simpler baseline has produced a measurable failure that the new capability is meant to address.

## Evidence

The version history follows this pattern:

- `docs/experiments/001_baselines.md`
- `evals/results/v0_2/v0_2_evidence_grounded_evaluation.md`
- `evals/results/v0_3/replanning/20260811T045039Z/v0_3_replanning.md`
- `evals/results/v0_4/multi_agent/20260811T060908Z/v0_4_multi_agent.md`

Each step adds one capability and evaluates it against a frozen benchmark.

## Consequences

Future architecture changes should justify themselves with measured outcomes before they are retained as defaults.
