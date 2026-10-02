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
lint:
	black . -l 120
	ruff check --fix
deps:
	sed -i 's/>/~/g' pyproject.toml
