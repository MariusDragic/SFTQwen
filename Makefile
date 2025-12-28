.PHONY: help install-uv sync venv train eval generate annotate setup

UV ?= uv
PYTHON ?= python3
MAIN ?= main.py

help:
	@echo "Available targets:"
	@echo "  make install-uv    -> Install uv if missing"
	@echo "  make sync          -> Install deps with uv sync"
	@echo "  make venv          -> Show how to activate the uv virtualenv"
	@echo "  make train         -> Run training"
	@echo "  make eval          -> Run evaluation"
	@echo "  make generate      -> Run generation / inference"
	@echo "  make annotate      -> Run annotation pipeline"
	@echo "  make setup         -> install-uv + sync"

install-uv:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo ">> uv not detected, installing..."; \
		pip install -U uv; \
	else \
		echo ">> uv already installed: $$(uv --version)"; \
	fi

sync:
	@echo ">> Installing dependencies with uv sync"
	$(UV) sync

venv:
	@echo ">> To activate the uv virtual environment manually:"
	@echo "   source .venv/bin/activate  (if local)"
	@echo "   or: source /venv/main/bin/activate (container / cloud)"

train:
	@echo ">> Running training"
	$(UV) run python $(MAIN) --mode train

eval:
	@echo ">> Running evaluation"
	$(UV) run python $(MAIN) --mode eval

generate:
	@echo ">> Running generation"
	$(UV) run python $(MAIN) --mode generate

annotate:
	@echo ">> Running annotation"
	$(UV) run python $(MAIN) --mode annotate

setup: install-uv sync
	@echo ">> Setup completed successfully"
