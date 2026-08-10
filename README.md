# PartyPilot

PartyPilot v0.1 is a typed Python event-planning benchmark and baseline-evaluation surface. It now includes:

- typed domain models and application services
- a deterministic feasibility planner
- a single-pass LLM baseline planner
- an Ollama adapter behind a provider-neutral port
- benchmark dataset loading and evaluation infrastructure
- deterministic validation of plans and constraints
- automated quality checks for formatting, linting, typing, and tests

PartyPilot v0.1 intentionally does not yet include:

- RAG or evidence retrieval
- LangGraph
- specialist multi-agent orchestration
- MCP or A2A integrations
- production deployment infrastructure

## Requirements

- Python 3.12+

## Run v0.1

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

4. If you want to run the live single-pass LLM baseline, make sure Ollama is available and `PARTYPILOT_OLLAMA_MODEL` is set.

5. Run the canonical baseline evaluation.

```bash
make eval-baseline
```

By default, `make eval-baseline` evaluates the `development` split and writes JSON and Markdown artifacts under `evals/results/v0_1/development/<timestamp>/`.

To evaluate a different split, pass `SPLIT`, for example:

```bash
make eval-baseline SPLIT=frozen_test
```

## Quality Gate

Use the Makefile for local checks:

```bash
make format       # format Python files with Ruff
make lint         # run Ruff linting
make typecheck    # run strict mypy checks
make test         # run pytest with coverage
make eval-baseline # run the canonical baseline evaluation
make check        # verify formatting, linting, typing, and tests
```

`make check` is the complete, non-mutating quality gate. It stops with a non-zero exit status if Ruff formatting, Ruff linting, strict type checking, or tests fail.
