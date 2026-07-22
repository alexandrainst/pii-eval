# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: docs help demo

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Set the PATH env var used by cargo and uv
export PATH := ${HOME}/.local/bin:${HOME}/.cargo/bin:$(PATH)

# Set the shell to bash, enabling the use of `source` statements
SHELL := /bin/bash

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing the 'pii-eval' project..."
	@$(MAKE) --quiet install-uv
	@$(MAKE) --quiet install-venv
	@echo "Installed the 'pii-eval' project! You can now activate your virtual environment with 'source .venv/bin/activate'."
	@echo "Note that this is a 'uv' project. Use 'uv add <package>' to install new dependencies and 'uv remove <package>' to remove them."

install-uv:
	@if [ "$(shell which uv)" = "" ]; then \
		if [ "$(shell which rustup)" = "" ]; then \
			curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
			echo "Installed Rust."; \
		fi; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "Installed uv."; \
    else \
		echo "Updating uv..."; \
		uv self update || true; \
	fi

install-venv:
	@uv sync

DEMO_HOST ?= 127.0.0.1
DEMO_PORT ?= 8000

demo: ## Spin up the PII detection web demo (override with DEMO_HOST / DEMO_PORT)
	@echo "Starting pii-eval demo at http://$(DEMO_HOST):$(DEMO_PORT)/demo/ ..."
	@( sleep 2; (xdg-open "http://$(DEMO_HOST):$(DEMO_PORT)/demo/" || open "http://$(DEMO_HOST):$(DEMO_PORT)/demo/") >/dev/null 2>&1 & )
	@uv run pii-eval serve --host $(DEMO_HOST) --port $(DEMO_PORT)
