# Experiment 001 — Baselines

## Question

How does PartyPilot's transparent deterministic planner compare with the single-pass LLM planner on objective correctness and reliability metrics in the v0.1 benchmark?

## Hypothesis

The deterministic baseline is expected to provide stronger hard-constraint validity and fewer unsupported claims because it selects only from structured resources and validates plans deterministically. The single-pass LLM baseline may produce structurally useful plans, but without retrieval or resource tools it is expected to be more vulnerable to unsupported resource claims and invalid assumptions.

This hypothesis is predeclared here as an architectural expectation, not as a measured conclusion.

## Compared variants

### A. Deterministic baseline

The deterministic planner queries the structured resource store, filters hard constraints, constructs candidate combinations, calculates costs, validates candidates, and ranks feasible candidates using documented deterministic weights.

### B. Single-pass LLM baseline

The single-pass baseline sends one `PartyRequest` to the configured LLM provider and requests a typed `PartyPlan`. It has no structured resource tools or retrieval. Every parsed plan is subsequently checked by the deterministic constraint engine, and failures are recorded rather than automatically repaired.

## Metrics

The objective comparison uses:

- feasibility accuracy
- hard-constraint validity
- structured-output validity
- unsupported-claim rate where measurable
- mean latency
- token usage where available

Subjective plan quality is intentionally excluded from the objective comparison.

## Predeclared decision criteria

A baseline may be preferred for the next architecture only if the measured evidence supports the choice. The decision order is:

1. Hard-constraint validity is a safety/correctness gate; a material regression is not accepted for gains in other metrics.
2. Feasibility accuracy should improve without reducing hard-constraint validity.
3. Unsupported-claim rate should be minimized and must not be hidden by successful structured parsing.
4. Structured-output validity must be high enough for deterministic downstream validation to operate reliably.
5. Latency and token usage are secondary efficiency measures after correctness requirements are met.
6. If the variants have not both been run under a comparable configuration, no winner is declared.

No threshold is retroactively tuned against the frozen-test split.

## Measured results

### Deterministic baseline

A measured run over the 24-scenario v0.1 dataset produced:

| Metric | Result |
|---|---:|
| Scenarios | 24 |
| Feasibility accuracy | 0.875 |
| Hard-constraint validity | 1.000 |
| No-feasible-plan accuracy | 0.923 |
| Mean latency | 0.059 ms |

The latency value is specific to the local sandbox run and is not a production performance claim.

### Single-pass LLM baseline

No comparable live-provider benchmark result is recorded yet. The implementation and fake-provider tests exercise parsing, validation, failure categorization, and metric calculation, but fake-provider outputs are not reported as model benchmark results.

Accordingly, structured-output validity, unsupported-claim rate, latency, and token-usage results for a real single-pass model remain unmeasured in this experiment report.

## Failure analysis

The deterministic run made three feasibility errors out of 24 scenarios:

- `dev-accessible-07`: expected `FEASIBLE`, predicted `NO_FEASIBLE_PLAN`.
- `frozen-accessibility-15`: expected `FEASIBLE`, predicted `NO_FEASIBLE_PLAN`.
- `frozen-resource-conflict-17`: expected `NO_FEASIBLE_PLAN`, predicted `FEASIBLE`.

Despite those feasibility-label errors, the measured hard-constraint validity metric was 1.000 for emitted deterministic candidates. This suggests the next investigation should focus on coverage/representation of accessibility and resource-conflict semantics rather than relaxing hard-constraint validation.

For the single-pass LLM variant, unit tests demonstrate expected failure categories such as invalid structured output, unsupported assumptions, hallucinated resources, arithmetic mistakes, and constraint violations. These are implementation-path tests, not frequency estimates for a real model.

## Conclusion

The deterministic baseline has measured v0.1 results and currently provides a useful correctness reference point. A baseline winner **cannot be declared** because the single-pass LLM variant has not yet been benchmarked with a real, explicitly configured provider/model under comparable conditions.

The current evidence therefore supports retaining the deterministic planner as the reference baseline while treating the single-pass LLM implementation as an unevaluated comparison variant, not as a demonstrated improvement or regression.

## Next architectural question

Can grounding the language-model planner in PartyPilot's structured resource store and evidence/provenance layer reduce unsupported claims while preserving deterministic hard-constraint validity and improving feasibility accuracy?

## Release tag status

`v0.1-baselines` is **not tagged** by this change. The tag should only be created after the baseline implementations, tests, and comparable evaluations are complete, including a real single-pass LLM benchmark run with traceable experiment configuration.
