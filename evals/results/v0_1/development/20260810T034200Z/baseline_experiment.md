# PartyPilot v0.1 Baseline Experiment

- Experiment ID: baseline-v0.1-development-20260810T034200Z
- Dataset split: development
- Timestamp: 2026-08-10T03:42:00.365657+00:00
- Commit SHA: d6d2b011c90e0947c2fad7f4c1203ae0ec768422
- Working tree dirty: False
- Git metadata error: none
- Model provider: ollama
- Model name: qwen3:4b-instruct-2507-q4_K_M
- Prompt version: single-pass-v1

# Baseline Comparison

Objective metrics only; subjective plan quality is not included in this report.

| Variant | Feasibility accuracy | Hard-constraint validity | Structured-output validity | Unsupported-claim rate | Mean latency (ms) | Median latency (ms) | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic baseline | 0.900 | 1.000 | 1.000 | n/a | 0.100 | 0.077 | n/a |
| Single-pass LLM baseline | 0.000 | 0.000 | 0.000 | n/a | 9003.298 | 10176.554 | 5543 |
