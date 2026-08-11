.PHONY: format format-check lint typecheck test check eval-baseline eval-v02 compare-v0-2 capability-boundary-inventory smoke-ollama smoke-constraint-extractor

SPLIT ?= development
EVAL_BASELINE_ARGS ?=
EVAL_V02_ARGS ?=
COMPARISON_ARGS ?=
CAPABILITY_BOUNDARY_INVENTORY_ARGS ?=

format:
	./.venv/bin/ruff format .

format-check:
	./.venv/bin/ruff format --check .

lint:
	./.venv/bin/ruff check .

typecheck:
	./.venv/bin/mypy src tests

test:
	./.venv/bin/pytest --cov=partypilot --cov-report=term-missing

eval-baseline:
	./.venv/bin/python -m partypilot.cli.eval_baseline --split $(SPLIT) $(EVAL_BASELINE_ARGS)

eval-v02:
	./.venv/bin/python -m partypilot.cli.eval_v02 --split $(SPLIT) $(EVAL_V02_ARGS)

compare-v0-2:
	./.venv/bin/python -m evals.run_v0_2_comparison $(COMPARISON_ARGS)

capability-boundary-inventory:
	./.venv/bin/python -m partypilot.cli.capability_boundary_inventory $(CAPABILITY_BOUNDARY_INVENTORY_ARGS)

smoke-ollama:
	./.venv/bin/python -m partypilot.cli.smoke_ollama $(SMOKE_OLLAMA_ARGS)

smoke-constraint-extractor:
	./.venv/bin/python -m partypilot.cli.smoke_constraint_extractor $(SMOKE_CONSTRAINT_EXTRACTOR_ARGS)

check: format-check lint typecheck test
