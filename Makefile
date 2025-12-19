.PHONY: help install-uv install-deps setup

PYTHON ?= python3

help:
	@echo "Targets disponibles :"
	@echo "  make install-uv     -> Installe uv si absent"
	@echo "  make install-deps   -> Installe les dependances (system)"
	@echo "  make setup          -> install-uv + install-deps"

install-uv:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo ">> uv non detecte, installation..."; \
		pip install -U uv; \
	else \
		echo ">> uv deja installe : $$(uv --version)"; \
	fi

install-deps:
	@echo ">> Installation des dependances depuis pyproject.toml (system)"
	uv pip install --system -r pyproject.toml

setup: install-uv install-deps
	@echo ">> Setup termine avec succes"
