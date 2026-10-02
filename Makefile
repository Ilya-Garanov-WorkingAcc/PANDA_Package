.DEFAULT_GOAL := help
.PHONY: help run push lint deps install test test-env lint-check typecheck check build wheel-check clean

help:  ## список команд
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

run:
	uv run -m src.main
push:
	git add .
	@if [ -z "$(msg)" ]; then \
		read -p "Enter commit message: " user_msg; \
		git commit -m "$$user_msg"; \
	else \
		git commit -m "$(msg)"; \
	fi
	git push
lint:  ## black + ruff с автоисправлением
	uv run black . -l 120
	uv run ruff check --fix
deps:
	sed -i 's/>/~/g' pyproject.toml

install:  ## окружение разработки (.venv): пакет + группы test и lint
	uv sync

test:  ## тесты
	uv run pytest

test-env:  ## тесты в отдельном чистом окружении: только пакет и группа test
	UV_PROJECT_ENVIRONMENT=.venv-test uv sync --no-default-groups --group test
	UV_PROJECT_ENVIRONMENT=.venv-test uv run --no-sync pytest

lint-check:  ## black + ruff без изменения файлов (как в CI)
	uv run black . -l 120 --check
	uv run ruff check

typecheck:  ## mypy (strict)
	uv run mypy

check: lint-check typecheck test  ## всё, что проверяет CI

build: clean  ## собрать sdist и wheel в dist/
	uv build

wheel-check: build  ## установить собранный wheel в чистое окружение и импортировать
	uv run --isolated --no-project --with dist/*.whl python -c "import panda_md; print('panda_md', panda_md.__version__)"

clean:  ## удалить артефакты сборки и кэши
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
