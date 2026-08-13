# PartyPilot v0.7d Controlled Orchestration Evaluation

- Benchmark: `v0.5 live multi-agent benchmark`
- Benchmark version: `1.0`
- Scenario count: `10`
- Run order blocks: `(('imperative', 'langgraph'), ('langgraph', 'imperative'), ('imperative', 'langgraph'))`
- Retention rule passed: `True`

## Provenance

- Experiment start Git SHA: `719270dbbff6845cf69abd68b2decdcf45aa6554`
- Experiment start working tree dirty: `False`
- Experiment start git metadata error: `n/a`
- Artifact Git SHA: `719270dbbff6845cf69abd68b2decdcf45aa6554`
- Artifact working tree dirty: `False`
- Artifact git metadata error: `n/a`
- Canonical start guard enforced: `True`
- Exploratory mode: `False`

## Reproducibility

- Git SHA: `719270dbbff6845cf69abd68b2decdcf45aa6554`
- Working tree dirty: `False`
- Timestamp: `2026-08-13T16:44:46.779459+00:00`
- Python: `3.12.13`
- Model: `n/a`

## Backend Summaries

### imperative

- Runs: `3`
- Final decision accuracy: `mean=0.700, range=0.700..0.700`
- Evidence-grounded arbitration: `mean=0.750, range=0.750..0.750`
- Hard-constraint validity: `mean=0.600, range=0.600..0.600`
- Global-optimum accuracy: `mean=1.000, range=1.000..1.000`
- Human-review calibration: `mean=1.000, range=1.000..1.000`
- Specialist success rate: `mean=0.180, range=0.100..0.220`
- Mean scenario wall-clock latency (ms): `mean=52461.449, range=49584.938..54052.771`
- Top-level specialist invocations: `150`
- Successful top-level specialist invocations: `27`
- Total specialist invocations: `150`
- Successful specialist invocations: `27`
- Provider attempts: `150`
- Retry count: `0`
- Specialist timeout outcomes: `123`
- Structured-output failures: `0`
- Specialist-domain validation failures: `0`
- Graph executions: `n/a`
- Targeted specialist rerun count: `0`
- Human-review route count: `n/a`
- Interrupt count: `n/a`
- Resume count: `n/a`
- Disposition: `BASELINE`

### langgraph

- Runs: `3`
- Final decision accuracy: `mean=0.700, range=0.700..0.700`
- Evidence-grounded arbitration: `mean=0.625, range=0.625..0.625`
- Hard-constraint validity: `mean=0.600, range=0.600..0.600`
- Global-optimum accuracy: `mean=1.000, range=1.000..1.000`
- Human-review calibration: `mean=1.000, range=1.000..1.000`
- Specialist success rate: `mean=0.200, range=0.167..0.233`
- Mean scenario wall-clock latency (ms): `mean=39715.193, range=38953.974..41110.655`
- Top-level specialist invocations: `90`
- Successful top-level specialist invocations: `18`
- Total specialist invocations: `90`
- Successful specialist invocations: `18`
- Provider attempts: `90`
- Retry count: `0`
- Specialist timeout outcomes: `72`
- Structured-output failures: `0`
- Specialist-domain validation failures: `0`
- Graph executions: `30`
- Targeted specialist rerun count: `0`
- Human-review route count: `18`
- Interrupt count: `18`
- Resume count: `0`
- Disposition: `RETAIN_EXPERIMENTALLY`

#### Node Execution Counts

- `accessibility`: `18`
- `budget`: `18`
- `catering`: `18`
- `coordinator`: `18`
- `finalize`: `30`
- `preflight`: `30`
- `scheduling`: `18`
- `venue`: `18`

#### Route Counts

- `end`: `30`
- `fan_out`: `18`
- `finalize`: `12`
- `human_review`: `18`

## Orchestration Sub-Benchmark

# PartyPilot v0.7d Orchestration/Replan Sub-Benchmark

- Benchmark: `deterministic orchestration/replan fixture`
- Scenario count: `4`
- Targeted domain selection correct: `4`
- Untouched specialists preserved: `4`
- Stale specialist outcome replaced: `4`
- PlanningState revision progression: `4`
- Coordinator rerun count: `2`
- Graph termination count: `4`
- Loop-bound handling count: `1`
- Passed: `True`

## Route Counts

- `finalize`: `2`
- `human_review`: `1`
- `replan`: `1`

Deterministic offline fixture used to keep targeted replanning and loop-bound handling under regression coverage.

## Human Review Sub-Benchmark

# PartyPilot v0.7d Human-Review Sub-Benchmark

- Benchmark: `deterministic human-review fixture`
- Interrupt emitted: `4`
- Checkpoint created: `4`
- Execution ID retained: `4`
- Resume from same execution: `4`
- Stale response rejected: `2`
- Invalid response rejected: `2`
- Valid approval routed: `1`
- Valid rejection routed: `1`
- Valid replan routed: `1`
- Deterministic hard constraints preserved after resume: `4`
- Passed: `True`

Deterministic offline fixture used to validate checkpointed interrupt/resume semantics without manual interaction.