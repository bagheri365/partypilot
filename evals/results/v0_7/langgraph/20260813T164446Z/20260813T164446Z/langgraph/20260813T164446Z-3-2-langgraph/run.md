# PartyPilot v0.7d Controlled Orchestration Run

- Run ID: `20260813T164446Z-3-2-langgraph`
- Backend: `langgraph`
- Repetition: `3`
- Order block: `3`
- Order position: `2`
- Scenario count: `10`
- Model: `qwen3:4b-instruct-2507-q4_K_M`
- Specialist adapter: `langchain_chatollama`
- Orchestration backend: `langgraph`
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
- Evidence-grounded arbitration: `0.625`
- Hard-constraint validity: `0.600`
- Global-optimum accuracy: `1.000`
- Human-review calibration: `1.000`
- Specialist success rate: `0.200`
- Mean scenario wall-clock latency (ms): `39080.950`
- Top-level specialist invocations: `30`
- Total specialist invocations: `30`
- Provider attempts: `30`
- Retry count: `0`
- Graph executions: `10`
- Coordinator node executions: `6`
- Finalize executions: `10`
- Human-review route count: `6`
- Interrupt count: `6`
- Resume count: `0`

## Scenarios

### cap-boundary-41-venue-caterer-dependency
Venue and caterer dependency

- Expected feasibility: `NO_FEASIBLE_PLAN`
- Live feasibility: `NO_FEASIBLE_PLAN`
- Wall-clock latency: `10.765` ms
- Selected resources: `venue-boston-studio, caterer-boston-buffet`
- Graph trace events: `4`

### cap-boundary-42-venue-activity-dependency
Venue and activity dependency

- Expected feasibility: `NO_FEASIBLE_PLAN`
- Live feasibility: `NO_FEASIBLE_PLAN`
- Wall-clock latency: `2.201` ms
- Selected resources: `venue-brooklyn-loft, activity-craft-party`
- Graph trace events: `4`

### cap-boundary-43-setup-scheduling-chain
Setup scheduling chain

- Expected feasibility: `NO_FEASIBLE_PLAN`
- Live feasibility: `NO_FEASIBLE_PLAN`
- Wall-clock latency: `1.271` ms
- Selected resources: `venue-boston-evening-hall, caterer-boston-buffet, activity-craft-party`
- Graph trace events: `4`

### cap-boundary-44-loading-bay-conflict
Loading-bay conflict

- Expected feasibility: `NO_FEASIBLE_PLAN`
- Live feasibility: `NO_FEASIBLE_PLAN`
- Wall-clock latency: `1.479` ms
- Selected resources: `venue-brooklyn-terrace, caterer-brooklyn-hosted`
- Graph trace events: `4`

### cap-boundary-45-outdoor-rain-contingency
Outdoor rain contingency

- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Live feasibility: `HUMAN_REVIEW_REQUIRED`
- Wall-clock latency: `30074.670` ms
- Selected resources: `venue-cambridge-garden`
- Graph trace events: `18`

### cap-boundary-47-specialist-disagreement
Specialist disagreement

- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Live feasibility: `HUMAN_REVIEW_REQUIRED`
- Wall-clock latency: `30052.865` ms
- Selected resources: `venue-boston-garden`
- Graph trace events: `18`

### cap-boundary-48-local-vs-global-optimum
Local versus global optimum

- Expected feasibility: `FEASIBLE`
- Live feasibility: `HUMAN_REVIEW_REQUIRED`
- Wall-clock latency: `72692.388` ms
- Selected resources: `venue-brooklyn-loft, caterer-family-table`
- Graph trace events: `18`

### cap-boundary-59-conflicting-agents-evidence
Conflicting agents and evidence

- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Live feasibility: `HUMAN_REVIEW_REQUIRED`
- Wall-clock latency: `30078.128` ms
- Selected resources: `venue-boston-garden`
- Graph trace events: `18`

### cap-boundary-61-large-but-purely-structured
Large but purely structured

- Expected feasibility: `FEASIBLE`
- Live feasibility: `HUMAN_REVIEW_REQUIRED`
- Wall-clock latency: `197661.468` ms
- Selected resources: `venue-boston-hall-c, caterer-boston-a, activity-boston-structured`
- Graph trace events: `18`

### cap-boundary-65-ten-structured-constraints
Ten structured constraints

- Expected feasibility: `FEASIBLE`
- Live feasibility: `HUMAN_REVIEW_REQUIRED`
- Wall-clock latency: `30234.269` ms
- Selected resources: `venue-boston-hall-d, caterer-boston-c, activity-boston-c`
- Graph trace events: `18`
