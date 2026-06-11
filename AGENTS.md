# AGENTS.md - MCP Server Interface Guide

This document provides essential information for AI agents interfacing with the Kokoro TTS MCP server via Model Context Protocol (MCP).

---

## Transport Modes

The server supports two transport modes:

### 1. TCP Mode (Recommended)
- **Port**: `8765` (default, configurable via docker-compose.yml)
- **Protocol**: JSON-RPC 2.0 with newline-delimited frames
- **Client Library**: See `mcp_test_client.py` for reference implementation
- **Advantage**: Persistent server accepting multiple concurrent clients

### 2. Stdio Mode
- For MCP clients requiring stdin/stdout transport (e.g., Claude Desktop, Cursor)
- Connect via: `docker exec -i kokoro-tts-mcp python3 /app/mcp_server/server.py`

---

## Tool Endpoints

### Available Tools (4)

| Tool Name | Description | Required Args | Optional Args |
|-----------|-------------|---------------|---------------|
| `tts_synthesize` | Text-to-speech with built-in voices | `input` (text) | `voice`, `quality`, `device`, `auto_play`, `delete_after_playback`, `max_sentences`, `max_chunks` |
| `tts_clone` | Voice cloning + synthesis | `input`, `ref_audio` | `ref_text`, `voice`, `quality`, `device`, `auto_play`, `delete_after_playback`, `max_sentences`, `max_chunks` |
| `list_voices` | List available voices | None | None |
| `play_audio` | Play WAV file through speakers | `file_path` | `delete_after_play` |

---

## Efficiency Best Practices

### 1. Voice Cloning: Use In-Memory Caching

**Critical optimization**: The server caches cloned voice state in memory between calls with the same `ref_audio`.

**Inefficient pattern (AVOID):**
```json
// Calling every time - forces reload on first call, then cached
{"name": "tts_clone", "arguments": {"input": "Hello", "ref_audio": "/path/to/sample.wav"}}
{"name": "tts_clone", "arguments": {"input": "World", "ref_audio": "/path/to/sample.wav"}}  // Cached now
```

**Efficient pattern (RECOMMENDED):**
```json
// First: Load voice into memory once
{"name": "tts_clone", "arguments": {"input": "", "ref_audio": "/path/to/sample.wav"}}

// Subsequent calls: Reuse cached state - use tts_synthesize with ref_audio hint
{"name": "tts_synthesize", "arguments": {"input": "First message using cloned voice"}}
{"name": "tts_synthesize", "arguments": {"input": "Second message - instant!"}}
```

**Performance difference:**
- First clone call: ~1-2 seconds (kanade model from local files) or 30-60s (HF download fallback)
- Cached calls: ~2-8 seconds (only synthesis, no model reload)

### 2. Avoid Unnecessary File I/O with auto_play

When you only need to hear the result and don't need file persistence:

```json
// Generates → Plays → Deletes (no disk residue)
{
  "name": "tts_synthesize",
  "arguments": {
    "input": "Quick verification message",
    "voice": "af_bella",
    "auto_play": true,
    "delete_after_playback": true
  }
}
```

**Control File Persistence with delete_after_playback:**

When `auto_play` is enabled, you can choose whether to keep or delete the generated audio file:

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

**When to use auto_play:**
- ✅ Quick voice verification tests
- ✅ Interactive feedback loops
- ✅ Limited disk space environments

**When NOT to use auto_play:**
- ❌ Need to preserve audio for later review (use `auto_play: true, delete_after_playback: false` instead)
- ❌ Archiving conversation history
- ❌ Debugging playback issues

### 3. Batch Voice List Queries

Only call `list_voices` once and cache results locally. The voice list rarely changes between sessions.

```json
// Call once, then use cached data
{"name": "list_voices", "arguments": {}}
```

### 4. Quality vs Performance Trade-offs

| Setting | Speed | Size | Quality | Recommended Use |
|---------|-------|------|---------|-----------------|
| `fp32` | Slowest | Largest | Highest | Production archives |
| `fp16` (default) | Fast | Medium | High | Balanced default |
| `q8` | Faster | Smaller | Good | Real-time/low-latency |
| `q4` | Fastest | Smallest | Acceptable | Quick previews |

**Recommendation**: Use `fp16` for general use, downgrade to `q8` or `q4` only if latency is critical.

---

## MCP-over-TCP Protocol Details

### Connection Handshake
When a client connects via TCP, the server immediately sends initialization information:

```json
{
  "jsonrpc": "2.0",
  "id": null,
  "method": "server/initialized",
  "result": {
    "version": "2024-11-05",
    "serverName": "kokoro-tts-mcp",
    "toolCount": 4,
    "tools": [
      {"name": "tts_synthesize", "description": "..."},
      {"name": "tts_clone", "description": "..."},
      {"name": "list_voices", "description": "..."},
      {"name": "play_audio", "description": "..."}
    ]
  }
}
```

### Request Format
All tool calls follow JSON-RPC 2.0 format with newline delimiters:

```json
{"jsonrpc":"2.0","id":<timestamp>,"method":"tools/call","params":{"name":"tts_synthesize","arguments":{...}}}
```

**Note**: Use `tools/call` or `call_tool` as the method name (both are supported).

### Response Format
Responses include both standard MCP fields and JSON-RPC metadata:

```json
{
  "jsonrpc": "2.0",
  "id": <request_id>,
  "result": {
    "isError": false,
    "content": [
      {"type": "text", "text": "Audio generated successfully.\nFile: /tmp/kokoro-tts/abc123.wav"}
    ]
  }
}
```

---

## Response Parsing Guide

### Standard Success Response Structure

```json
{
  "result": {
    "content": [
      {"type": "text", "text": "Audio generated successfully.\nFile: /path/to/file.wav\nSize: 147 KB | Sample rate: 24kHz | Format: WAV"}
    ],
    "isError": false
  }
}
```

### Key Extraction Pattern (Pseudocode)

```python
def parse_response(response):
    if not isinstance(response, dict):
        return None
        
    result = response.get("result")  
    if not isinstance(result, dict):
        return None
    
    # Direct CallToolResult in "result" field - no wrapping key needed
    if "content" in result:  # This IS the CallToolResult!
        content_list = result["content"]
        error_flag = result.get("isError", False)
        
        text_parts = []
        for item in (content_list or []):
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
                
        return {
            "success": not error_flag,
            "content": "\n".join(text_parts),
            "raw_result": result
        }
    
    return None  # Unexpected response structure
```

### File Path Extraction

File paths appear in the first text content item:

```python
import re
text = "Audio generated successfully.\nFile: /tmp/kokoro-tts/direct_abc123.wav\nSize: ..."
match = re.search(r'File:\s*(.+\.wav)', text)
if match:
    file_path = match.group(1).strip()
```

### Error Response Pattern

```json
{
  "result": {
    "content": [
      {"type": "text", "text": "Error: [Errno 2] No such file or directory: '/path/to/file.wav'"}
    ],
    "isError": true
  }
}
```

---

## Resource Management Tips

### Disk Space Cleanup

The server does NOT automatically clean up files generated without `auto_play`. For long-running sessions:

**Recommended cleanup interval**: Every 50-100 synthesis calls or periodically via cron.

**Manual cleanup pattern:**
```json
// Use play_audio with delete_after_play=true for automatic cleanup
{"name": "play_audio", "arguments": {"file_path": "/tmp/kokoro-tts/old_file.wav"}}
```

### Memory Considerations (Voice Cloning)

- **Kanade model**: ~458MB loaded into memory once per session (from local files)
- **KokoClone engine**: Additional ~200MB for voice cloning
- **Each cached voice**: Adds ~5MB additional memory overhead

**Best practice**: Only cache 1-2 reference voices simultaneously. If switching between many different reference files, the server will swap models in/out of memory automatically.

---

## Audio File Specifications

All generated audio follows these standards:

| Property | Value |
|----------|-------|
| Format | WAV (RIFF) |
| Sample Rate | 24,000 Hz |
| Bit Depth | 16-bit PCM |
| Channels | Mono |
| Encoding | Signed Little-Endian |

**Approximate file sizes:**
- 1 second of speech: ~5-7 KB
- 5 seconds of speech: ~25-35 KB  
- Typical short phrase (2-4 words): ~30-50 KB

---

## Error Handling Recommendations

### Common Errors and Recovery

| Error | Likely Cause | Recommended Action |
|-------|--------------|-------------------|
| `File not found` | Invalid file_path for play_audio | Verify path exists in container; mount volumes correctly |
| `KokoClone not available` | Missing kokoclone package | Check Docker image includes kokoclone dependency |
| `TTS generation failed: 404` | Wrong HF repo for cloning | Ensure hf_repo is set to `PatnaikAshish/kokoclone` |
| Timeout on clone call | First-time model download | With local kanade model, this should be ~1-2 seconds. Retry with longer timeout (30s) if using HF fallback |

### Retry Strategy

```python
# Suggested retry pattern for unreliable networks
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def safe_call(tool, args):
    for attempt in range(MAX_RETRIES):
        try:
            response = send_request(tool, args)
            result = parse_response(response)
            if result and result["success"]:
                return result
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None
```

---

## MCP Connection Configuration

### TCP Transport Setup (Default)

For connecting via TCP socket (persistent server):

**Docker configuration:**
```yaml
ports:
  - "8765:8765"
```

**With arguments:**
- `--host`: Bind address (default: 0.0.0.0)
- `--port`: Port to listen on (default: 8765)

### Stdio Transport Setup

For connecting via stdio (recommended for containerized deployments):

**Docker exec method:**
```bash
docker exec -i kokoro-tts-mcp python3 /app/mcp_server/server.py --voice af_bella
```

**With arguments:**
- `--voice`: Default voice name
- `--port`: Optional port for future HTTP extension (default: 8765)

### Protocol Version

The server uses MCP protocol version **2024-11-05**. Ensure your MCP client supports this version.

---

## Quick Reference Card

```
# One-liner synthesis with auto-play (keeps file)
{"name":"tts_synthesize","arguments":{"input":"Hello","voice":"af_bella","auto_play":true,"delete_after_playback":false}}

# One-liner voice cloning + play  
{"name":"tts_clone","arguments":{"input":"Cloned message","ref_audio":"/samples/voice.wav","auto_play":true,"delete_after_playback":false}}

# Manual playback of existing file
{"name":"play_audio","arguments":{"file_path":"/tmp/kokoro-tts/output.wav"}}

# List available voices (cache this!)
{"name":"list_voices","arguments":{}}

# TCP connection test
python3 mcp_test_client.py --test all

# Test with keep-audio flag (preserves generated files)
python3 mcp_test_client.py --keep-audio --text "Hello world"

# Voice cloning test with sample audio
python3 mcp_test_client.py --test clone --clone-audio ./samples/voice_sample.wav --keep-audio
```
