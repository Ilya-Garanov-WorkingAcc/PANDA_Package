.DEFAULT_GOAL := help
.PHONY: help run push lint deps install test test-env lint-check typecheck check build wheel-check docs clean

help:  ## list the commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

run:  ## the panda command line: make run args="shapes --pore slit"
	uv run panda $(args)
push:
	git add .
	@if [ -z "$(msg)" ]; then \
		read -p "Enter commit message: " user_msg; \
		git commit -m "$$user_msg"; \
	else \
		git commit -m "$(msg)"; \
	fi
	git push
lint:  ## black + ruff with automatic fixes
	uv run black . -l 120
	uv run ruff check --fix
deps:
	sed -i 's/>/~/g' pyproject.toml

install:  ## development environment (.venv): the package with all extras + dev tools
	uv sync --all-extras

test:  ## run the tests
	uv run pytest

test-env:  ## tests in a separate clean environment: the package with [md, plot] and the test group only
	UV_PROJECT_ENVIRONMENT=.venv-test uv sync --no-default-groups --group test --extra md --extra plot
	UV_PROJECT_ENVIRONMENT=.venv-test uv run --no-sync pytest

lint-check:  ## black + ruff without changing files (as in CI)
	uv run black . -l 120 --check
	uv run ruff check

typecheck:  ## mypy (strict)
	uv run mypy

check: lint-check typecheck test  ## everything CI checks

build: clean  ## build sdist and wheel into dist/
	uv build

wheel-check: build  ## install the built wheel into a clean environment, import it and run panda
	uv run --isolated --no-project --with dist/*.whl python -c "import panda_md; print('panda_md', panda_md.__version__)"
	uv run --isolated --no-project --with dist/*.whl panda shapes --pore slit > /dev/null

docs:  ## documentation (mkdocs) into site/; strict: a warning is an error
	uv run --group docs mkdocs build --strict

clean:  ## remove build artefacts and caches
	rm -rf dist build site .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
