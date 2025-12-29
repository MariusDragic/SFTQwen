.PHONY: help install-uv sync venv train eval generate annotate config setup pull

help:
	@echo "Available targets:"
	@echo "  make install-uv    -> Install uv if missing"
	@echo "  make sync          -> Install deps with uv sync"
	@echo "  make venv          -> Show how to activate the uv virtualenv"
	@echo "  make pull          -> Clone or update the SFTQwen repository"
	@echo "  make train         -> Run training"
	@echo "  make eval          -> Run evaluation"
	@echo "  make generate      -> Run generation / inference"
	@echo "  make annotate      -> Run annotation pipeline"
	@echo "  make config        -> Print full configuration"
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
	uv sync

venv:
	@echo ">> To activate the uv virtual environment manually:"
	@echo "   source .venv/bin/activate        (local)"
	@echo "   source /venv/main/bin/activate  (container / cloud)"

clone:
	@if [ ! -d "SFTQwen" ]; then \
		echo ">> Cloning SFTQwen repository"; \
		git clone https://github.com/MariusDragic/SFTQwen.git; \
	else \
		echo ">> Updating SFTQwen repository"; \
		cd SFTQwen && git pull; \
	fi

pull:
	@echo ">> Pulling latest changes for SFTQwen"
	git pull

train:
	@echo ">> Running training"
	uv run python main.py --mode train

eval:
	@echo ">> Running evaluation"
	uv run python main.py --mode eval

generate:
	@echo ">> Running generation"
	uv run python main.py --mode generate

annotate:
	@echo ">> Running annotation"
	uv run python main.py --mode annotate

config:
	@echo ">> Printing configuration"
	uv run python main.py --mode config

setup: install-uv sync
	@echo ">> Setup completed successfully"
