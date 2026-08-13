# ADR 006: Retain LangGraph Experimentally

## Status

Retained experimentally

## Context

LangGraph adds explicit routing, targeted replanning, and checkpointed review semantics that are useful for controlled orchestration experiments.

## Decision

Retain LangGraph as an experimental orchestration backend, not as the default claim of equivalent work under the benchmark.

## Evidence

The canonical v0.7 artifacts support two conclusions:

- LangGraph adds useful orchestration capabilities, including routing, replanning, and human-review checkpoints.
- The post-hoc accounting audit classified the comparison as `VALID_BUT_DIFFERENT_WORK_POLICY`, not a pure backend-efficiency comparison.

References:

- `evals/results/v0_7/langgraph/20260813T164446Z/20260813T164446Z/v0_7_controlled_orchestration_evaluation.md`
- `evals/results/v0_7/langgraph/20260813T164446Z/20260813T164446Z/accounting_audit.md`

## Consequences

LangGraph remains available for orchestration research and review-state experiments, but the canonical evidence does not justify treating it as a pure drop-in efficiency improvement over the imperative orchestration backend.
