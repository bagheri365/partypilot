# PartyPilot v0.1 Baseline Experiment

- Experiment ID: baseline-v0.1-development-20260810T040158Z
- Dataset split: development
- Timestamp: 2026-08-10T04:01:58.552402+00:00
- Commit SHA: d39e8a94551c1008276f3080bc8f0514ad39067d
- Working tree dirty: False
- Git metadata error: none
- Model provider: ollama
- Model name: qwen3:4b-instruct-2507-q4_K_M
- Prompt version: single-pass-v1

# Baseline Comparison

Objective metrics only; subjective plan quality is not included in this report.

| Variant | Feasibility accuracy | Hard-constraint validity | Structured-output validity | Unsupported-claim rate | Mean latency (ms) | Median latency (ms) | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic baseline | 0.900 | 1.000 | 1.000 | n/a | 0.110 | 0.080 | n/a |
| Single-pass LLM baseline | 0.000 | 0.000 | 0.000 | n/a | 9322.746 | 10766.600 | 5633 |
