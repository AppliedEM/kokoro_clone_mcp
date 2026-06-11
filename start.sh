#!/bin/bash
# Start the Kokoro TTS MCP Server
# 
# Usage:
#   ./start.sh              # Run with docker-compose (stdio mode)
#   ./start.sh http         # Run HTTP server on port 8765
#   ./start.sh dev          # Run in development mode (local python, no docker)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "${1:-}" in
    http)
        echo "Starting Kokoro TTS MCP Server (HTTP mode on port 8765)..."
        docker compose up --build
        ;;
    dev)
        echo "Starting Kokoro TTS MCP Server (development mode - local Python)..."
        
        # Activate virtual environment if exists
        if [ -d "$SCRIPT_DIR/venv" ]; then
            source "$SCRIPT_DIR/venv/bin/activate"
        fi
        
        # Ensure dependencies are installed
        pip install -q mcp 2>/dev/null || true
        
        # Run the MCP server directly (requires kokoro scripts in ../kokoro)
        python "$SCRIPT_DIR/mcp_server/server.py" --device cpu --quality fp16
        ;;
    *)
        echo "Starting Kokoro TTS MCP Server (docker-compose, stdio mode)..."
        docker compose up --build
        ;;
esac
