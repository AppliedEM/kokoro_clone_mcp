# Kokoro TTS MCP Server

Dockerized Model Context Protocol (MCP) server for Kokoro Text-to-Speech synthesis with voice cloning capabilities. Supports both stdio and TCP transport modes.

## Features

- **Text-to-Speech**: Synthesize text into high-quality speech using 54+ voices
- **Voice Cloning**: Clone voices from reference audio files with in-memory caching
- **Local Kanade Model**: Pre-loaded kanade model (~458MB) avoids HuggingFace downloads on container restart
- **Bundled kokoro.onnx**: Voice cloning ONNX inference model included in Docker image (~89MB), no runtime download needed
- **Audio Playback**: Play generated audio through system speakers (Linux/macOS/Windows)
- **Auto-Cleanup**: Automatic file deletion after playback to manage disk space
- **In-Memory Cache**: Voice cloning state cached between calls for performance
- **TCP Transport Mode**: Persistent server accepting multiple concurrent clients via TCP socket

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd kokoro_docker
```

### 2. Download Kanade Model (One-Time Setup)

For voice cloning without HuggingFace downloads:

```bash
# Download kanade model weights (~458MB)
curl -L https://huggingface.co/frothywater/kanade-12.5hz/resolve/main/model.safetensors \
  -o models/kanade.safetensors

# Download kanade config (3KB)
curl -L https://huggingface.co/frothywater/kanade-12.5hz/raw/main/config.yaml \
  -o models/kanade-config.yaml
```

**Note:** The kokoro.onnx voice cloning model (~89MB) is now bundled in the Docker image by default, eliminating the need for runtime downloads. You can optionally replace it with a custom version if needed.

### 3. Start the Server

**TCP Transport Mode (Recommended for MCP-over-TCP):**
```bash
docker compose up -d
# Server starts on port 8765 inside container, mapped to host:8765
```

**Stdio Transport Mode (For Claude Desktop, Cursor, etc.):**
```bash
# Connect via stdio using docker exec
docker exec -i kokoro-tts-mcp python3 /app/mcp_server/server.py
```

### 4. Verify Connection

```bash
# Check container status and health
make check

# Run MCP-over-TCP tests (list tools + synthesize)
make test

# Test voice cloning with sample audio
make test-clone CLONE_AUDIO=./samples/voice_sample.wav
```

---

## MCP Transport Modes

### TCP Mode (Default)

The server runs as a persistent process accepting JSON-RPC 2.0 connections over TCP:

| Property | Value |
|----------|-------|
| Port | `8765` (container), mapped to host |
| Protocol | JSON-RPC 2.0 with newline-delimited frames |
| Client Library | See `mcp_test_client.py` for reference implementation |

**Connection Flow:**
1. Client connects to TCP port 8765
2. Server sends initial handshake: `{"jsonrpc":"2.0","id":null,"method":"server/initialized",...}`
3. Client sends tool calls: `{"jsonrpc":"2.0","id":<timestamp>,"method":"tools/call",...}\n`
4. Server responds with JSON-RPC response

**Test Connection:**
```bash
python3 mcp_test_client.py --test voices    # List available tools
python3 mcp_test_client.py                  # Run all tests (voices + synthesize)
```

### Stdio Mode

For MCP clients that require stdin/stdout transport:
```bash
docker exec -i kokoro-tts-mcp python3 /app/mcp_server/server.py --voice af_bella
```

---

## MCP Tools

### 1. `tts_synthesize`

Synthesize text to speech using built-in Kokoro voices.

**Parameters:**
| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `input` | string | Yes | Text to synthesize into speech | - |
| `voice` | string | No | Voice name (e.g., 'af_bella', 'am_michael') | af_bella |
| `quality` | string | No | Model quality: fp32, fp16, q8, q4 | fp16 |
| `device` | string | No | Compute device: cpu, cuda, auto | cpu |
| `auto_play` | boolean | No | Auto-play audio after synthesis | false |
| `delete_after_playback` | boolean | No | Delete generated file after playback (only effective with auto_play) | true |
| `max_sentences` | integer | No | Maximum sentences per API call | 20 |
| `max_chunks` | integer | No | Maximum text chunks for splitting | 100 |

**Example:**
```json
{
  "name": "tts_synthesize",
  "arguments": {
    "input": "Hello world!",
    "voice": "af_bella",
    "auto_play": true,
    "delete_after_playback": false
  }
}
```

### 2. `tts_clone`

Synthesize text using voice cloning from a reference audio file. Uses in-memory caching to avoid reloading the cloned model on every call.

**Parameters:**
| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `input` | string | Yes | Text to synthesize using the cloned voice | - |
| `ref_audio` | string | Yes | Path to reference audio file (WAV/MP3) for voice cloning | - |
| `ref_text` | string | No | Transcript of the reference audio (optional, helps accuracy) | - |
| `voice` | string | No | Default voice name | af_bella |
| `quality` | string | No | Model quality: fp32, fp16, q8, q4 | fp16 |
| `device` | string | No | Compute device | cpu |
| `auto_play` | boolean | No | Auto-play audio after synthesis | false |
| `delete_after_playback` | boolean | No | Delete generated file after playback (only effective with auto_play) | true |
| `max_sentences` | integer | No | Maximum sentences per API call | 20 |
| `max_chunks` | integer | No | Maximum text chunks for splitting | 100 |

**Example:**
```json
{
  "name": "tts_clone",
  "arguments": {
    "input": "This is a cloned voice message.",
    "ref_audio": "/tmp/kokoro-tts/voice_sample.wav",
    "auto_play": true,
    "delete_after_playback": false
  }
}
```

### 3. `list_voices`

List all available built-in Kokoro TTS voices with their language and gender classification.

**Example:**
```json
{
  "name": "list_voices",
  "arguments": {}
}
```

### 4. `play_audio` (New!)

Play a WAV audio file through system speakers. Supports local files within the container. Automatically deletes files after playback by default.

**Parameters:**
| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `file_path` | string | Yes | Path to WAV audio file to play | - |
| `delete_after_play` | boolean | No | Auto-delete the file after successful playback | true |

**Example:**
```json
{
  "name": "play_audio",
  "arguments": {
    "file_path": "/tmp/kokoro-tts/output.wav",
    "delete_after_play": true
  }
}
```

---

## Voice Cloning with In-Memory Caching

The server maintains cloned voice state in memory between calls for optimal performance:

**First Call:**
- Loads kanade model from **local files** (~1-2 seconds) or downloads from HuggingFace if not available (~30-60 seconds)
- Extracts reference audio features from provided WAV file
- Stores cloning configuration in memory

**Subsequent Calls (Same ref_audio):**
- Reuses cached voice state immediately
- No model reload needed
- Significantly faster synthesis

**Example Workflow:**
```json
// 1. Load reference voice into memory
{"name": "tts_clone", "arguments": {"input": "", "ref_audio": "/path/to/sample.wav"}}

// 2. Use cloned voice for multiple messages (cached!)
{"name": "tts_synthesize", "arguments": {"input": "First message using cloned voice"}}
{"name": "tts_synthesize", "arguments": {"input": "Second message - instant!"}}
```

---

## Docker Configuration

### docker-compose.yml (Default)

```yaml
services:
  kokoro-tts-mcp:
    build: .
    container_name: kokoro-tts-mcp
    restart: unless-stopped
    volumes:
      - kokoro-models:/root/.cache/pykokoro     # Model cache persistence
      - ./models/hf-cache:/root/.cache/huggingface  # HF model cache (kanade)
      - ./models/kanade.safetensors:/app/models/kanade.safetensors:ro  # Local kanade weights
      - ./models/kanade-config.yaml:/app/models/kanade-config.yaml:ro   # Local kanade config
      - ./model/kokoro.onnx:/app/models/kokoro.onnx:ro  # Bundled kokoro ONNX model (~89MB)
      - ../kokoro:/app/kokoro-original:ro        # Kokoro source (if available)
      - ./output:/tmp/kokoro-tts                 # Audio output directory
    ports:
      - "8765:8765"                              # MCP-over-TCP port
    environment:
      PYKOKORO_MODEL_DIR: /root/.cache/pykokoro
      KOKORO_VOICE_DIR: /app/kokoro/voice
      PYTHONUNBUFFERED: "1"
      HF_HOME: /root/.cache/huggingface
      KOKORO_LOCAL_MODEL_DIR: /app/models        # Local kanade model path
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OUTPUT_DIR` | Where to save output WAV files | `/tmp/kokoro-tts` |
| `VOICE` | Default voice name | `af_bella` |
| `DEVICE` | Compute device (cpu/cuda/auto) | `cpu` |
| `QUALITY` | Model quality preset | `fp16` |
| `KOKORO_LOCAL_MODEL_DIR` | Path to local kanade model files | `/app/models` |

---

## Testing & Utilities

### Makefile Commands

```bash
make help              # Show all available commands
make up                # Start the MCP server container (detached)
make down              # Stop and remove the container  
make rebuild           # Rebuild Docker image and restart
make test              # Run MCP-over-TCP tests (list tools + synthesize)
make check             # Verify container is running and healthy
make logs              # Show container logs in real-time

# Voice cloning test
make test-clone CLONE_AUDIO=./samples/voice_sample.wav
```

### Test Client

A Python test client (`mcp_test_client.py`) is included for manual verification:

```bash
# Run all tests (list voices + synthesize)
python3 mcp_test_client.py

# Use specific voice and custom text
python3 mcp_test_client.py --voice af_nicole --text "Hello from Kokoro!"

# Only test voice listing
python3 mcp_test_client.py --test voices

# Test with voice cloning (requires reference WAV file)
python3 mcp_test_client.py --test clone --clone-audio ./my_voice.wav
```

---

## File Management

### Output Directory
- Default: `/tmp/kokoro-tts/` (configurable in Docker)
- Format: `{type}_{timestamp}.wav` (e.g., `direct_abc123.wav`)
- Audio specs: 24kHz, mono, 16-bit WAV

### Auto-Cleanup Strategy
When `auto_play` or `delete_after_play` is enabled:
1. Generate audio file
2. Get file metadata (size for response)
3. Play through speakers
4. **Delete** the temporary file automatically

This ensures no disk space accumulation from repeated synthesis operations.

### File Persistence Control with `delete_after_playback`

For more granular control over generated audio files:

```json
// Auto-play and delete immediately (default behavior)
{
  "name": "tts_synthesize",
  "arguments": {
    "input": "Hear this then delete it",
    "auto_play": true,
    "delete_after_playback": true
  }
}

// Auto-play but KEEP the file for review/archival
{
  "name": "tts_synthesize",
  "arguments": {
    "input": "Generate audio to keep",
    "auto_play": true,
    "delete_after_playback": false
  }
}
```

**When NOT to use auto_cleanup:**
- ❌ Need to preserve audio for later review (use `auto_play: true, delete_after_playback: false`)
- ❌ Archiving conversation history
- ❌ Debugging playback issues

### CLI Test Client with --keep-audio Flag

The test client includes a `--keep-audio` flag that sets `delete_after_playback=false`:

```bash
# Test synthesis and keep the generated file
python3 mcp_test_client.py --keep-audio --text "Hello world"

# Voice cloning test, preserve output
python3 mcp_test_client.py --test clone --clone-audio ./samples/voice_sample.wav --keep-audio
```

---

## Troubleshooting

### Container Keeps Restarting
The MCP server uses a retry loop to handle stdio transport disconnections. If you see rapid restarts:
- Check the healthcheck configuration in docker-compose.yml (default start_period is 120s)
- Monitor logs with `make logs` or `docker logs kokoro-tts-mcp -f`

### Missing spaCy Language Model Error
If TTS synthesis fails with "en_core_web_sm not found":
```bash
# The model is pre-installed in the Docker image. If missing:
docker exec kokoro-tts-mcp python3 -m spacy download en_core_web_sm
```

### Voice Cloning Not Working (kokoclone)
Voice cloning requires the `kokoclone` library which is included by default in this setup. To verify:

1. **Check if kanade model is loaded locally:**
   ```bash
   docker exec kokoro-tts-mcp ls -lh /app/models/kanade.safetensors
   ```

2. **Test voice cloning with sample audio:**
   ```bash
   make test-clone CLONE_AUDIO=./samples/voice_sample.wav
   ```

3. **If local model missing, download it (see "Quick Start" section above)**

### Audio Not Playing on Linux
```bash
# Check if paplay or aplay is available
which paplay aplay

# If missing, install ALSA utilities
sudo apt-get install alsa-utils
```

### MCP-over-TCP Connection Issues
```bash
# Verify port is exposed
docker ps --filter name=kokoro-tts-mcp -q | xargs docker inspect --format '{{.NetworkSettings.Ports}}'

# Test TCP connectivity from host
nc -zv localhost 8765 || echo "Port not accessible"

# Check server logs for connection errors
docker logs kokoro-tts-mcp --tail 50
```

---

## API Response Format

All tools return MCP-compliant responses:

```json
{
  "result": {
    "content": [
      {"type": "text", "text": "Audio generated successfully.\nFile: /tmp/kokoro-tts/output.wav\nSize: 147 KB | Sample rate: 24kHz | Format: WAV"}
    ],
    "isError": false
  }
}
```

---

## License

MIT License - See LICENSE file for details.

## Credits

- **Kokoro**: High-quality text-to-speech model by [Kokoro project](https://github.com/hassan-hq/kokoro)
- **pykokoro**: Python bindings for Kokoro TTS
- **KokoClone**: Voice cloning library using Kanade tokenizer
- **Kanade**: Speech tokenizer model by [frothywater](https://huggingface.co/frothywater/kanade-12.5hz)
- **MCP SDK**: Model Context Protocol implementation
