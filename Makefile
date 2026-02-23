.PHONY: help install test lint demo pipeline clean

CONFIG ?= configs/experiment1_dev.yaml

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (uv sync --extra dev)
	uv sync --extra dev

test: ## Run pytest
	uv run pytest tests/ -v

lint: ## Run ruff linter
	uv run ruff check src/ scripts/ tests/

demo: ## Run walkthrough notebook on fixture data (CPU)
	uv run jupyter execute notebooks/walkthrough.ipynb

pipeline: ## Run full 5-step pipeline (GPU required for steps 1-2)
	uv run python scripts/01_compute_steering_vector.py --config $(CONFIG)
	uv run python scripts/02_generate_responses.py --config $(CONFIG)
	uv run python scripts/03_embed_responses.py --config $(CONFIG)
	uv run python scripts/04_compute_metrics.py --config $(CONFIG)
	uv run python scripts/05_visualize.py --config $(CONFIG)

examples: ## Regenerate example outputs from fixture data
	uv run python scripts/generate_examples.py

clean: ## Remove generated outputs
	rm -rf outputs/
