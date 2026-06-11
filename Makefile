# Kokoro TTS MCP Server - Makefile (TCP Mode)
# Provides convenience targets for common operations with MCP-over-TCP transport.

.PHONY: help build up down rebuild test clean logs check container

# Configuration variables (can be overridden via command line)
CONTAINER ?= kokoro-tts-mcp
IMAGE     ?= kokoro_docker-kokoro-tts-mcp
VOICE     ?= af_bella
ifndef TEXT
	TEXT      ?= Hello, this is a test of the Kokoro TTS system.
endif
CLONE_AUDIO ?=  # Optional path to reference WAV for voice cloning test (requires kokoclone)

# TCP connection options (server defaults: localhost:8765)
TEST_HOST ?= localhost
TEST_PORT ?= 8765

# Default target shows help
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Kokoro TTS MCP Server - Available Commands"
	@echo "==========================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Docker Compose Operations

up: ## Start the MCP server container (detached)
	docker compose up -d
	@echo ""
	@echo "Container started. Status:"
	docker ps --filter name=$(CONTAINER) --format "table {{.Names}}\t{{.Status}}"

down: ## Stop and remove the container
	docker compose down
	@echo "Container stopped."

build: ## Build the Docker image
	docker compose build

rebuild: ## Rebuild with no cache (use after code changes)
	docker compose build --no-cache
	docker compose up -d

logs: ## Show container logs
	docker logs $(CONTAINER) -f

# Health Check Operations

check: ## Verify the container is running and healthy
	@echo "Checking container status..."
	@if docker ps --filter name=$(CONTAINER) --format "{{.Status}}" | grep -q "healthy"; then \
		echo "✓ Container is running and healthy"; \
	else \
		echo "✗ Container is not healthy or not running"; \
		exit 1; \
	fi

container: ## Show detailed container info
	docker inspect $(CONTAINER) --format '{{.Name}}: {{.State.Status}} (since {{.State.StartedAt}})'

# Testing - MCP-over-TCP mode

test: check ## Run MCP server test (list voices + synthesize with auto-play)
	@echo ""
	@echo "Running Kokoro TTS TCP MCP test..."
ifdef CLONE_AUDIO
	@echo "(Including voice cloning test with $(CLONE_AUDIO))"
	python3 mcp_test_client.py \
		--host $(TEST_HOST) \
		--port $(TEST_PORT) \
		--voice $(VOICE) \
		--text "$(TEXT)" \
		--test synthesize clone \
		--clone-audio "$(CLONE_AUDIO)"
else
	python3 mcp_test_client.py \
		--host $(TEST_HOST) \
		--port $(TEST_PORT) \
		--voice $(VOICE) \
		--text "$(TEXT)" \
		--test all
endif

test-voices: check ## Only list available voices
	@echo ""
	@echo "Listing available MCP tools..."
	python3 mcp_test_client.py \
		--host $(TEST_HOST) \
		--port $(TEST_PORT) \
		--voice $(VOICE) \
		--test voices

# Voice Cloning Test - requires reference audio file (WAV format, 10-60 seconds recommended)
CLONE_VOICE ?= cloned_$(shell date +%Y%m%d_%H%M%S)_$(shell head /dev/urandom | tr -dc a-z0-9 | head -c8)

test-clone: check ## Test voice cloning + synthesis (requires CLONE_AUDIO variable)
	@echo ""
	@echo "Running voice cloning test..."
ifdef CLONE_AUDIO
	@echo "(Using reference audio: $(CLONE_AUDIO))"
	python3 mcp_test_client.py \
		--host $(TEST_HOST) \
		--port $(TEST_PORT) \
		--keep-audio \
		--voice $(VOICE) \
		--text "$(TEXT)" \
		--test clone \
		--clone-audio "$(CLONE_AUDIO)"
else
	@echo "ERROR: CLONE_AUDIO must be specified"
	@echo "Usage: make test-clone CLONE_AUDIO=/path/to/reference.wav"
	@exit 1
endif

# Cleanup

clean: down ## Stop container and remove volumes (WARNING: deletes model cache!)
	docker compose down -v
	@echo "Container stopped and volumes removed."
