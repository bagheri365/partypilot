.PHONY: format format-check lint typecheck test check eval-baseline smoke-ollama

SPLIT ?= development
EVAL_BASELINE_ARGS ?=

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

smoke-ollama:
	./.venv/bin/python -m partypilot.cli.smoke_ollama $(SMOKE_OLLAMA_ARGS)

check: format-check lint typecheck test
