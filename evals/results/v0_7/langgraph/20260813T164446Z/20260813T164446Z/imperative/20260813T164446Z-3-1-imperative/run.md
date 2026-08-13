# PartyPilot v0.7d Controlled Orchestration Run

- Run ID: `20260813T164446Z-3-1-imperative`
- Backend: `imperative`
- Repetition: `3`
- Order block: `3`
- Order position: `1`
- Scenario count: `10`
- Model: `qwen3:4b-instruct-2507-q4_K_M`
- Specialist adapter: `langchain_chatollama`
- Orchestration backend: `imperative`
- Provider I/O timeout: `30.0s`
- Ollama context budget: `8192`
- Structured-output strategy: `with_structured_output`

## Provenance

- Experiment start Git SHA: `719270dbbff6845cf69abd68b2decdcf45aa6554`
- Experiment start working tree dirty: `False`
- Experiment start git metadata error: `n/a`
- Artifact Git SHA: `719270dbbff6845cf69abd68b2decdcf45aa6554`
- Artifact working tree dirty: `False`
- Artifact git metadata error: `n/a`
- Canonical start guard enforced: `True`
- Exploratory mode: `False`

## Metrics

- Final decision accuracy: `0.700`
- Evidence-grounded arbitration: `0.750`
- Hard-constraint validity: `0.600`
- Global-optimum accuracy: `1.000`
- Human-review calibration: `1.000`
- Specialist success rate: `0.220`
- Mean scenario wall-clock latency (ms): `53746.637`
- Top-level specialist invocations: `50`
- Total specialist invocations: `50`
- Provider attempts: `50`
- Retry count: `0`
- Graph executions: `n/a`
- Coordinator node executions: `n/a`
- Finalize executions: `n/a`
- Human-review route count: `n/a`
- Interrupt count: `n/a`
- Resume count: `n/a`

## Scenarios

### cap-boundary-41-venue-caterer-dependency
Venue and caterer dependency

- Expected feasibility: `NO_FEASIBLE_PLAN`
- Live feasibility: `NO_FEASIBLE_PLAN`
- Wall-clock latency: `30057.547` ms
- Selected resources: `venue-boston-studio, caterer-boston-buffet`
- Graph trace events: `0`

### cap-boundary-42-venue-activity-dependency
Venue and activity dependency

- Expected feasibility: `NO_FEASIBLE_PLAN`
- Live feasibility: `NO_FEASIBLE_PLAN`
- Wall-clock latency: `30043.466` ms
- Selected resources: `venue-brooklyn-loft, activity-craft-party`
- Graph trace events: `0`

### cap-boundary-43-setup-scheduling-chain
Setup scheduling chain

- Expected feasibility: `NO_FEASIBLE_PLAN`
- Live feasibility: `NO_FEASIBLE_PLAN`
- Wall-clock latency: `34996.721` ms
- Selected resources: `venue-boston-evening-hall, caterer-boston-buffet, activity-craft-party`
- Graph trace events: `0`

### cap-boundary-44-loading-bay-conflict
Loading-bay conflict

- Expected feasibility: `NO_FEASIBLE_PLAN`
- Live feasibility: `NO_FEASIBLE_PLAN`
- Wall-clock latency: `31933.283` ms
- Selected resources: `venue-brooklyn-terrace, caterer-brooklyn-hosted`
- Graph trace events: `0`

### cap-boundary-45-outdoor-rain-contingency
Outdoor rain contingency

- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Live feasibility: `HUMAN_REVIEW_REQUIRED`
- Wall-clock latency: `40246.598` ms
- Selected resources: `venue-cambridge-garden`
- Graph trace events: `0`

### cap-boundary-47-specialist-disagreement
Specialist disagreement

- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Live feasibility: `HUMAN_REVIEW_REQUIRED`
- Wall-clock latency: `30037.170` ms
- Selected resources: `venue-boston-garden`
- Graph trace events: `0`

### cap-boundary-48-local-vs-global-optimum
Local versus global optimum

- Expected feasibility: `FEASIBLE`
- Live feasibility: `HUMAN_REVIEW_REQUIRED`
- Wall-clock latency: `68720.996` ms
- Selected resources: `venue-brooklyn-loft, caterer-family-table`
- Graph trace events: `0`

### cap-boundary-59-conflicting-agents-evidence
Conflicting agents and evidence

- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Live feasibility: `HUMAN_REVIEW_REQUIRED`
- Wall-clock latency: `35123.648` ms
- Selected resources: `venue-boston-garden`
- Graph trace events: `0`

### cap-boundary-61-large-but-purely-structured
Large but purely structured

- Expected feasibility: `FEASIBLE`
- Live feasibility: `HUMAN_REVIEW_REQUIRED`
- Wall-clock latency: `193188.836` ms
- Selected resources: `venue-boston-hall-c, caterer-boston-a, activity-boston-structured`
- Graph trace events: `0`

### cap-boundary-65-ten-structured-constraints
Ten structured constraints

- Expected feasibility: `FEASIBLE`
- Live feasibility: `HUMAN_REVIEW_REQUIRED`
- Wall-clock latency: `43118.109` ms
- Selected resources: `venue-boston-hall-d, caterer-boston-c, activity-boston-c`
- Graph trace events: `0`
