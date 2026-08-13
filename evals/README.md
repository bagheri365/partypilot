# PartyPilot Evaluation Guide

This directory contains the canonical evidence for PartyPilot's architecture experiments.
The evaluation artifacts are meant to be read as research records, not as raw machine output that needs reverse-engineering.

## Evaluation Philosophy

PartyPilot follows a repeatable research loop:

`baseline -> measured failure -> capability -> controlled evaluation -> disposition`

The point is to add one capability at a time, measure it against a frozen benchmark, and then decide whether the capability should be:

- `RETAIN`: adopted as part of the default retained architecture
- `RETAIN_EXPERIMENTALLY`: kept available for comparison, but not promoted as the default
- `REJECT` / `REJECT_AS_DEFAULT`: kept as a reference variant or discarded as the default path because the evidence did not justify promotion

The project principle is the same one used in the README: evidence before complexity.

## Canonical Versus Exploratory Runs

PartyPilot distinguishes exploratory development runs from canonical evaluation runs.

- Exploratory runs are used while implementing or debugging a capability.
- Canonical runs happen after implementation freeze.
- Canonical runs start from a clean working tree.
- The experiment-start Git SHA is captured and recorded in the artifact provenance.
- Generated artifacts must not rewrite or obscure experiment-start provenance.
- Canonical runs avoid selective reruns that would bias the work-count or decision metrics.

If a run begins from a dirty tree, it is not a canonical evidence record.

## Benchmark Lineage

The benchmark family evolves in small steps:

- v0.1 established a deterministic planning baseline and a reproducible reference point.
- v0.2 added evidence grounding with retrieval and live constraint extraction.
- v0.3 added planning state and dependency-aware targeted replanning.
- v0.4 added deterministic multi-agent coordination.
- v0.5 added live LLM specialist execution.
- v0.6 added LangChain specialist adapters and rejected `create_agent` as the default structured path.
- v0.7 added a LangGraph orchestration backend, while the imperative orchestration backend remained the compatibility/default comparison path; the result was later audited as a different work policy rather than a pure framework-efficiency comparison.

This directory stores the canonical records for those experiments and their supporting sub-benchmarks.

## Metric Definitions

The benchmark files use a shared vocabulary. The most important terms are:

- `final decision accuracy`: fraction of scenarios whose terminal feasibility outcome matches the benchmark label
- `evidence-grounded arbitration`: fraction of evidence-relevant scenarios where the controlling evidence and arbitration outcome align with the authoritative evidence
- `hard-constraint validity`: fraction of scenarios where the chosen result respects deterministic hard constraints
- `global-optimum accuracy`: fraction of globally optimizable scenarios where the chosen combination is the lowest-total-cost viable option
- `specialist success rate`: fraction of specialist invocations that completed successfully
- `specialist timeout outcome rate`: fraction of specialist invocations that ended in provider timeout
- `provider attempts`: total provider/model requests issued by the architecture
- `tool calls`: total tool invocations produced inside an agent execution, when the architecture uses tools
- `graph executions`: total LangGraph scenario executions
- `human-review interrupts`: total LangGraph interruptions that suspend execution for review
- `human-review resumes`: total LangGraph resumptions from a suspended execution

Two important accounting distinctions:

- `top-level specialist invocations` count the specialist runs the architecture intentionally launched.
- `provider/model attempts` count the underlying provider requests made by those specialist runs.

Those counts usually track together, but they are not conceptually identical. A benchmark may track retries, provider failures, or tool use separately from the top-level invocation count.

## Repetition Methodology

When a live benchmark is repeated, PartyPilot uses balanced run order blocks so that backend comparisons are less sensitive to local provider drift, warm caches, or time-of-day effects.

Repeated live runs matter because local LLM/provider behavior can vary substantially across runs even when the code and benchmark inputs are frozen.

That is why the canonical evidence records:

- run order blocks
- repetition index
- scenario count
- per-run provenance
- per-run metrics

The goal is to compare architectures under the same benchmark, not to accidentally compare one backend's unlucky run with another backend's lucky run.

## Provenance

Canonical artifacts always distinguish:

- experiment-start Git state
- artifact-time Git state

The experiment-start state is the state at which the canonical run began.
The artifact-time state is the state when the report was written.

For canonical runs, PartyPilot enforces a clean-tree expectation and records whether that expectation was met.
That makes it possible to tell the difference between a frozen evidence record and a later regenerated summary.

## Known Experimental Caveats

Several caveats should be kept in mind when reading the evaluation artifacts:

- local provider timeout rates can be high and can dominate live specialist success
- the v0.6 `create_agent` experiment produced zero tool calls in the canonical evaluation and was not promoted to the default
- the v0.7 LangGraph comparison was classified as `VALID_BUT_DIFFERENT_WORK_POLICY`, so it should not be described as a pure orchestration-backend efficiency comparison or a 40% efficiency gain

## Canonical Artifact Index

The table below points to the canonical artifact directories for each major benchmark version.

| Version | Canonical artifact directory |
|---|---|
| v0.1 | [evals/results/v0_1/development/20260810T034200Z](results/v0_1/development/20260810T034200Z/) |
| v0.2 | [evals/results/v0_2](results/v0_2/) |
| v0.3 | [evals/results/v0_3/replanning/20260811T045039Z](results/v0_3/replanning/20260811T045039Z/) |
| v0.4 | [evals/results/v0_4/multi_agent/20260811T060908Z](results/v0_4/multi_agent/20260811T060908Z/) |
| v0.5 | [evals/results/v0_5/llm_multi_agent/20260812T060624Z](results/v0_5/llm_multi_agent/20260812T060624Z/) |
| v0.6 | [evals/results/v0_6/langchain/20260813T042641Z](results/v0_6/langchain/20260813T042641Z/) |
| v0.7 | [evals/results/v0_7/langgraph/20260813T164446Z/20260813T164446Z](results/v0_7/langgraph/20260813T164446Z/20260813T164446Z/) |

The v0.7 directory contains:

- the controlled orchestration evaluation summary
- the orchestration/replan sub-benchmark
- the human-review sub-benchmark
- the post-hoc accounting audit
