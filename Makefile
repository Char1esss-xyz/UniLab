.PHONY: sync setup format type check check-tests test test-cov test-slow test-all

sync:
	uv sync

setup: sync

format:
	uv run ruff format
	uv run ruff check --fix

type:
	uv run mypy src/unilab
	uv run pyright

check: format type check-tests

check-tests:
	uv run ruff check tests --select F401,F821,F811,F841 --output-format concise

test:
	uv run pytest -m "not slow"

test-cov:
	uv run pytest -m "not slow" --cov=src/unilab --cov-report=term-missing

test-slow:
	uv run pytest -m "slow" -v

test-all: check test-cov test-slow
