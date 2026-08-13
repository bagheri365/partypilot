# PartyPilot

PartyPilot v0.2 is a reproducible evidence-grounded event-planning benchmark and release milestone.
It includes:

- typed domain models and application services
- deterministic structured candidate filtering and hard-constraint validation
- plain `BM25EvidenceRetriever`
- a live `LLMConstraintExtractor` backed by Ollama
- deterministic request-specific interpretation of evidence-backed constraints
- evidence-state resolution plus provenance and citation validation
- benchmark dataset loading and evaluation infrastructure
- automated quality checks for formatting, linting, typing, and tests

PartyPilot v0.2 intentionally does not yet include:

- conditional query rewriting in the retained runtime
- semantic retrieval in the retained runtime
- RRF in the retained runtime
- reranking in the retained runtime
- LangGraph
- specialist multi-agent orchestration
- MCP or A2A integrations
- production deployment infrastructure

## v0.6a Adapter Foundation

PartyPilot v0.6a begins the specialist model-execution adapter layer for the v0.5 multi-agent runtime.

- The PartyPilot `SpecialistAgent` port remains the architectural boundary.
- Native Ollama specialists remain available as the baseline comparator.
- LangChain support is introduced only inside adapter/composition code.
- The first LangChain path uses `langchain_ollama.ChatOllama` with structured output against PartyPilot's typed `SpecialistDecisionEnvelope`.
- `create_agent` is intentionally deferred until tools are introduced.

## Requirements

- Python 3.12+

## Run v0.2

Start from the repository root.

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the project with development dependencies.

```bash
python -m pip install -e '.[dev]'
```

3. Run the quality gate.

```bash
make check
```

4. If you want to run the live v0.2 evaluation or the Ollama smoke tests, make sure Ollama is available and `PARTYPILOT_OLLAMA_MODEL` is set.

5. Run the canonical v0.2 evaluation.

```bash
make eval-v02
```

By default, `make eval-v02` evaluates the `development` split and writes JSON and Markdown artifacts under `evals/results/v0_2/development/<timestamp>/`.

To evaluate a different split, pass `SPLIT`, for example:

```bash
make eval-v02 SPLIT=frozen_test
```

## Benchmark Surfaces

PartyPilot now keeps two separate benchmark surfaces:

1. Canonical v0.2 release benchmark

- The current 10-scenario `development` split.
- Used for reproducible release metrics and retained runtime comparisons.
- This is the benchmark behind the canonical `make eval-v02` release flow.

2. Expanded capability-boundary benchmark

- A forward-looking scenario suite in `data/evaluation/capability_boundary_scenarios.json`.
- Intentionally includes cases beyond the current retained v0.2 architecture.
- It is an inventory and comparison surface, not part of the canonical release metric set.
- It is frozen with a benchmark version and deterministic checksum in `data/evaluation/capability_boundary_manifest.json`.
- Future architecture milestones must not silently edit these scenarios; any benchmark change must be explicit, versioned, and justified.
- Inspect it with `make capability-boundary-inventory`.

3. Future architecture comparison

- The same expanded scenarios are intended for future comparisons across deterministic, single LLM, evidence-grounded, decomposed, multi-agent, and adaptive architectures.
- Those comparisons are not implemented in the retained v0.2 runtime.
- The current v0.2 architecture should not be treated as solving the expanded suite.
- A simpler architecture matching or outperforming a multi-agent architecture on these scenarios is a valid and important result.

## Quality Gate

Use the Makefile for local checks:

```bash
make format       # format Python files with Ruff
make lint         # run Ruff linting
make typecheck    # run strict mypy checks
make test         # run pytest with coverage
make eval-v02     # run the canonical v0.2 release evaluation
make check        # verify formatting, linting, typing, and tests
```

`make check` is the complete, non-mutating quality gate. It stops with a non-zero exit status if Ruff formatting, Ruff linting, strict type checking, or tests fail.

## Measured v0.2 Results

### Canonical v0.2 Development Release

The canonical v0.2 development-split release metrics are:

- scenario count: `10`
- feasibility accuracy: `0.900`
- hard-constraint validity: `1.000`
- grounded-decision accuracy: `1.000`
- source-attribution accuracy: `1.000`
- derived-constraint accuracy: `1.000`
- unsupported-claim rate: `0.000`
- wrong-source/version rate: `0.000`
- no-feasible-plan accuracy: `0.800`
- mean latency: approximately `40.5` seconds in the measured local Ollama run

These are the canonical v0.2 release metrics for the current `development` split.

### Research / Comparison Result

An earlier broader live comparison over 24 scenarios produced the following research metrics:

- feasibility accuracy: `0.875`
- hard-constraint validity: `1.000`
- grounded-decision accuracy: `1.000`
- source-attribution accuracy: `1.000`
- derived-constraint accuracy: `1.000`
- unsupported-claim rate: `0.000`
- wrong-source/version rate: `0.000`
- no-feasible-plan accuracy: `0.769`

That comparison was useful for deciding between runtime variants, but it is not the canonical v0.2 release score.

## Behavioral Notes

- Structured data determines candidate eligibility.
- Evidence contributes contextual policy constraints.
- `SUPPORTED` means the evidence supports a fact, not necessarily that the user's requirement is satisfied.
- Request-specific compatibility is evaluated deterministically.
- Provenance and source/version validation remain explicit.
- The live constraint extractor uses Ollama.
- Multi-agent coordination and LangGraph are not implemented yet.

## Current Limitations

- Live local-model evaluation is slow.
- Token and cost metrics are not yet collected.
- No-feasible-plan accuracy remains below perfect.
- The evaluation corpus is intentionally small.
- Semantic, RRF, and rewrite experiments are research comparisons, not retained runtime dependencies.

## v0.3 Experiment

PartyPilot v0.3 is currently exploring explicit planning state, dependency tracking, and targeted replanning.

- The goal is to test whether stateful decomposition solves dynamic planning failures before multi-agent coordination is justified.
- Multi-agent orchestration, LangGraph, specialist agents, and coordinator behavior are not implemented.
- The retained v0.2 runtime remains unchanged; v0.3 experiments run beside it as deterministic research surfaces.
- The replanning benchmark compares full replanning against dependency-aware targeted replanning on offline fixtures.
- Run it with `make eval-v03-replanning`.
