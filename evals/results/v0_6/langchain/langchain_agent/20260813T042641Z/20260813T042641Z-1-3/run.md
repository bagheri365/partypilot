# PartyPilot v0.6d Controlled LangChain Run

- Run ID: `20260813T042641Z-1-3`
- Variant: `langchain_agent`
- Repetition: `1`
- Order block: `1`
- Order position: `3`
- Scenario count: `10`
- Model: `qwen3:4b-instruct-2507-q4_K_M`
- Provider I/O timeout: `30.0s`
- Ollama context budget: `8192`
- Structured-output strategy: `ProviderStrategy(SpecialistDecisionEnvelope)`

## Provenance

- Experiment start Git SHA: `8c3cde626733eb200aaaff9a9e0a009ba56d40e1`
- Experiment start working tree dirty: `False`
- Experiment start git metadata error: `n/a`
- Artifact Git SHA: `8c3cde626733eb200aaaff9a9e0a009ba56d40e1`
- Artifact working tree dirty: `False`
- Artifact git metadata error: `n/a`
- Canonical start guard enforced: `True`
- Exploratory mode: `False`

## Environment

- Python: `3.12.13`
- LangChain: `1.3.15`
- langchain-core: `1.5.4`
- langchain-ollama: `1.1.0`
- LangGraph: `1.2.11`

# PartyPilot v0.5 Live Multi-Agent Runtime Experiment

Benchmark name: `v0.5 live multi-agent benchmark`
Benchmark version: `1.0`
Evaluation variant: `langchain_agent_vs_deterministic`
Scenario count: **10**

## Reproducibility Metadata

- Git SHA: `8c3cde626733eb200aaaff9a9e0a009ba56d40e1`
- Working tree dirty: `False`
- Dataset version: `1.0`
- Baseline architecture: `v0.4_deterministic_specialist_coordination`
- Live architecture: `langchain_agent`
- Model name: `n/a`

## Metric Definitions

- `final_decision_accuracy`: Fraction of scenarios whose terminal feasibility outcome matches the benchmark label.
- `hard_constraint_validity`: Fraction of scenarios where the chosen result respects deterministic hard constraints.
- `cross_domain_compatibility_accuracy`: Fraction of scenarios where cross-resource dependencies are handled correctly.
- `evidence_grounded_arbitration_accuracy`: Fraction of evidence-relevant scenarios where arbitration uses authoritative evidence.
- `global_optimum_accuracy`: Fraction of global-optimization scenarios where the lowest-cost viable option is chosen.
- `human_review_calibration`: Fraction of HUMAN_REVIEW_REQUIRED scenarios routed to human review.
- `specialist_call_count`: Total number of specialist recommendations produced by the architecture.
- `coordination_overhead_count`: Total number of explicit coordination/dependency checks performed by the coordinator.
- `mean_latency_ms`: Mean wall-clock latency per scenario for the architecture.

## Aggregate Metrics

| Metric | Baseline | Live |
|---|---:|---:|
| Final decision accuracy | 1.000 | 0.700 |
| Hard-constraint validity | 0.600 | 0.600 |
| Cross-domain compatibility | 0.700 | 0.700 |
| Evidence-grounded arbitration | 1.000 | 0.625 |
| Global optimum accuracy | 1.000 | 1.000 |
| Human review calibration | 1.000 | 1.000 |
| Specialist calls | 50 | 10 |
| Coordination overhead | 80 | 80 |

## Runtime Metrics

- Total specialist calls: `80`
- Mean specialists invoked per scenario: `8.000`
- Specialist success rate: `0.200`
- Structured output validation failure rate: `0.000`
- Retry rate: `0.000`
- Specialist disagreement rate: `0.000`
- Coordinator override count: `5`
- Mean latency (ms): `49183.029`

- Retention rule passed: `False`

## Scenario Diagnostics

- Terminal outcome mismatches: `cap-boundary-48-local-vs-global-optimum, cap-boundary-61-large-but-purely-structured, cap-boundary-65-ten-structured-constraints`
- Diagnostic failure-stage cases: `cap-boundary-41-venue-caterer-dependency, cap-boundary-42-venue-activity-dependency, cap-boundary-43-setup-scheduling-chain, cap-boundary-44-loading-bay-conflict, cap-boundary-48-local-vs-global-optimum, cap-boundary-61-large-but-purely-structured, cap-boundary-65-ten-structured-constraints`

## Per-Scenario Results

### cap-boundary-41-venue-caterer-dependency
Venue and caterer dependency

- Capability tags: `cross_domain, dependency_chain, replanning`
- Expected feasibility: `NO_FEASIBLE_PLAN`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `NO_FEASIBLE_PLAN` | 1.000 | 1.000 | 1.000 | 0.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `NO_FEASIBLE_PLAN` | 1.000 | 1.000 | 1.000 | 0.000 | 1 | 5 | hard_constraints |

#### Runtime Trace

- Wall-clock latency: `30256.250` ms
- Selected resources: `venue-boston-studio, caterer-boston-buffet`
- Execution traces: `5`
- Arbitration outcome: `REJECT`
- Controlling evidence: `doc-cap41-approved-caterer-list, doc-cap41-caterer-vendor-rule`
- Accepted specialists: `budget`
- Rejected specialists: `none`

### cap-boundary-42-venue-activity-dependency
Venue and activity dependency

- Capability tags: `cross_domain, activity_dependency, replanning`
- Expected feasibility: `NO_FEASIBLE_PLAN`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 1.000 | 0.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 1.000 | 0.000 | 1 | 5 | hard_constraints |

#### Runtime Trace

- Wall-clock latency: `30037.594` ms
- Selected resources: `venue-brooklyn-loft, activity-craft-party`
- Execution traces: `5`
- Arbitration outcome: `REJECT`
- Controlling evidence: `doc-cap42-venue-no-prep-room`
- Accepted specialists: `catering`
- Rejected specialists: `none`

### cap-boundary-43-setup-scheduling-chain
Setup scheduling chain

- Capability tags: `temporal_dependency, setup_chain, replanning`
- Expected feasibility: `NO_FEASIBLE_PLAN`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 1.000 | 0.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 1.000 | 0.000 | 1 | 5 | hard_constraints |

#### Runtime Trace

- Wall-clock latency: `35694.758` ms
- Selected resources: `venue-boston-evening-hall, caterer-boston-buffet, activity-craft-party`
- Execution traces: `5`
- Arbitration outcome: `REJECT`
- Controlling evidence: `doc-cap43-venue-access, doc-cap43-setup-window, doc-cap43-caterer-setup, doc-cap43-activity-setup-window`
- Accepted specialists: `venue`
- Rejected specialists: `none`

### cap-boundary-44-loading-bay-conflict
Loading-bay conflict

- Capability tags: `cross_domain, loading_bay, logistics_conflict`
- Expected feasibility: `NO_FEASIBLE_PLAN`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 1.000 | 0.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 1.000 | 0.000 | 1 | 5 | hard_constraints |

#### Runtime Trace

- Wall-clock latency: `30044.321` ms
- Selected resources: `venue-brooklyn-terrace, caterer-brooklyn-hosted`
- Execution traces: `5`
- Arbitration outcome: `REJECT`
- Controlling evidence: `doc-cap44-loading-bay-window, doc-cap44-caterer-delivery-window`
- Accepted specialists: `none`
- Rejected specialists: `none`

### cap-boundary-45-outdoor-rain-contingency
Outdoor rain contingency

- Capability tags: `temporal_dependency, contingency, replanning`
- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 0.000 | 1.000 | 0.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 0.000 | 1.000 | 0.000 | 1 | 5 | none |

#### Runtime Trace

- Wall-clock latency: `31895.927` ms
- Selected resources: `venue-cambridge-garden`
- Execution traces: `5`
- Arbitration outcome: `HUMAN_REVIEW_REQUIRED`
- Controlling evidence: `doc-cap45-outdoor-policy, doc-cap45-rain-contingency`
- Accepted specialists: `none`
- Rejected specialists: `none`

### cap-boundary-47-specialist-disagreement
Specialist disagreement

- Capability tags: `arbitration, cross_domain, conflict`
- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 1.000 | 0.000 | 1.000 | 1 | 5 | none |

#### Runtime Trace

- Wall-clock latency: `30039.078` ms
- Selected resources: `venue-boston-garden`
- Execution traces: `5`
- Arbitration outcome: `HUMAN_REVIEW_REQUIRED`
- Controlling evidence: `none`
- Accepted specialists: `venue`
- Rejected specialists: `none`

### cap-boundary-48-local-vs-global-optimum
Local versus global optimum

- Capability tags: `global_optimization, cross_domain, budget_tradeoff`
- Expected feasibility: `FEASIBLE`
- Requires evidence: `True`
- Requires global optimum: `True`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 10 | none |
| `v0.5_live_llm_specialist_agents` | `HUMAN_REVIEW_REQUIRED` | 0.000 | 1.000 | 0.000 | 1.000 | 1 | 10 | evidence_authority |

#### Runtime Trace

- Wall-clock latency: `60102.203` ms
- Selected resources: `venue-brooklyn-loft, caterer-family-table`
- Execution traces: `10`
- Arbitration outcome: `HUMAN_REVIEW_REQUIRED`
- Controlling evidence: `none`
- Accepted specialists: `budget`
- Rejected specialists: `none`

### cap-boundary-59-conflicting-agents-evidence
Conflicting agents and evidence

- Capability tags: `arbitration, cross_domain, conflict`
- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 1.000 | 0.000 | 1.000 | 1 | 5 | none |

#### Runtime Trace

- Wall-clock latency: `30045.245` ms
- Selected resources: `venue-boston-garden`
- Execution traces: `5`
- Arbitration outcome: `HUMAN_REVIEW_REQUIRED`
- Controlling evidence: `none`
- Accepted specialists: `venue`
- Rejected specialists: `none`

### cap-boundary-61-large-but-purely-structured
Large but purely structured

- Capability tags: `structured_only, large_candidate_set, optimization`
- Expected feasibility: `FEASIBLE`
- Requires evidence: `False`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 30 | none |
| `v0.5_live_llm_specialist_agents` | `HUMAN_REVIEW_REQUIRED` | 0.000 | 1.000 | 1.000 | 1.000 | 1 | 30 | outcome |

#### Runtime Trace

- Wall-clock latency: `183676.887` ms
- Selected resources: `venue-boston-hall-c, caterer-boston-a, activity-boston-structured`
- Execution traces: `30`
- Arbitration outcome: `HUMAN_REVIEW_REQUIRED`
- Controlling evidence: `none`
- Accepted specialists: `none`
- Rejected specialists: `none`

### cap-boundary-65-ten-structured-constraints
Ten structured constraints

- Capability tags: `structured_constraints, many_constraints, optimization`
- Expected feasibility: `FEASIBLE`
- Requires evidence: `False`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `HUMAN_REVIEW_REQUIRED` | 0.000 | 1.000 | 1.000 | 1.000 | 1 | 5 | outcome |

#### Runtime Trace

- Wall-clock latency: `30038.030` ms
- Selected resources: `venue-boston-hall-d, caterer-boston-c, activity-boston-c`
- Execution traces: `5`
- Arbitration outcome: `HUMAN_REVIEW_REQUIRED`
- Controlling evidence: `none`
- Accepted specialists: `none`
- Rejected specialists: `none`

## Notes

- This experiment is fully offline and deterministic except for the live specialist model calls.
- The coordinator remains deterministic; specialists are the only live LLM-backed components.
- The benchmark is intentionally bounded and reused from the v0.4 development subset.