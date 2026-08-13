# ADR 002: Deterministic Coordinator

## Status

Accepted

## Context

Multi-agent coordination introduces arbitration, disagreement handling, and global-constraint resolution. Those concerns benefit from explicit authority and testability.

## Decision

Keep the coordinator deterministic.

## Evidence

The v0.4 multi-agent experiment showed that deterministic coordination can reach strong measured outcomes on the benchmark while preserving explicit arbitration behavior:

- `evals/results/v0_4/multi_agent/20260811T060908Z/v0_4_multi_agent.md`

The coordinator remains the place where global constraints, evidence conflicts, and arbitration decisions are resolved.

## Consequences

Specialists can remain focused on their own domains while the coordinator stays auditable, reproducible, and testable.
