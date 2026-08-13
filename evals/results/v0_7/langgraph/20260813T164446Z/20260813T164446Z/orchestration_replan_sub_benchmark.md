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