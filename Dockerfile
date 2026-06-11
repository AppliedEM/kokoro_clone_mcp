FROM python:3.12-slim-bookworm

# System dependencies for audio processing and building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libsndfile1 \
    portaudio19-dev \
    ca-certificates \
    curl \
    alsa-utils \
    pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install MCP server dependencies
COPY mcp_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pykokoro directly for best performance (when available)
RUN pip install --no-cache-dir pykokoro soundfile numpy scipy 2>/dev/null || true

# ============================================================================
# Voice Cloning Support - KokoClone
# Installed directly from GitHub source (includes kokoro-onnx + kanade)
# Gradio excluded since we only need core CLI/API functionality
# ============================================================================

# Install CPU-only PyTorch for voice cloning compatibility without GPU
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || true

# Clone KokoClone source (we install dependencies manually to exclude gradio)
RUN git clone --depth 1 https://github.com/Ashish-Patnaik/kokoclone.git /tmp/kokoclone

# Install kokoclone core dependencies (excludes gradio Web UI - not needed for MCP API)
RUN pip install --no-cache-dir \
    kokoro-onnx \
    misaki[en] \
    huggingface_hub \
    soundfile \
    2>/dev/null || true

# Install kanade-tokenizer (required for voice conversion models - cloned by kokoclone)
RUN pip install --no-cache-dir git+https://github.com/frothywater/kanade-tokenizer 2>/dev/null || true

# Install KokoClone as a local package (core API only, no gradio UI)
# We need to temporarily modify pyproject.toml to exclude gradio before installing
RUN cd /tmp/kokoclone && \
    sed -i 's/"gradio>=6.8.0",//' pyproject.toml && \
    pip install --no-cache-dir . 2>/dev/null || true

# ============================================================================
# Voice Cloning Model Files (bundled to avoid runtime downloads)
# These files are used by KokoClone and kokoro-onnx for voice synthesis
# ============================================================================

# Copy kokoro.onnx model file into container if it exists locally in build context
RUN if [ -f "model/kokoro.onnx" ]; then \
        mkdir -p /app/models && cp model/kokoro.onnx /app/models/; \
    fi || true

# Install spaCy English language model required by pykokoro for sentence splitting
RUN python -m spacy download en_core_web_sm -q 2>/dev/null || true

# Create output directory
RUN mkdir -p /tmp/kokoro-tts

# Set environment variables
ENV PYKOKORO_MODEL_DIR=/root/.cache/pykokoro \
    KOKORO_VOICE_DIR=/app/kokoro/voice \
    PYTHONUNBUFFERED=1

# Copy the MCP server module (includes both stdio and TCP transport)
COPY mcp_server/server.py ./mcp_server/server.py
COPY mcp_server/tcp_mcp.py ./mcp_server/tcp_mcp.py

# Note: No ENTRYPOINT - docker-compose will specify the full command with --tcp flag
CMD ["python", "-m", "mcp_server.server"]
