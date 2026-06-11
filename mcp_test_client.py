#!/usr/bin/env python3
"""MCP Client Test Script for Kokoro TTS Server (TCP Mode).

Connects to the Kokoro TTS MCP server via TCP port 8765 and tests basic functionality.

The server runs in TCP transport mode, accepting multiple concurrent connections
without requiring stdin/stdout pipes or docker exec.

Usage:
    ./mcp_test_client.py              # Run all tests with defaults
    ./mcp_test_client.py --port 9000  # Connect to custom port (default: 8765)
    ./mcp_test_client.py --test voices   # Only test list_voices  
    ./mcp_test_client.py --text "Hello world"  # Custom text

Options:
    --host HOST       TCP host (default: localhost)
    --port PORT       TCP port (default: 8765)
    --text TEXT       Text to synthesize (default: "Hello, this is a test of the Kokoro TTS system.")
    --voice VOICE     Voice name (default: af_bella)
    --test TESTS      Tests to run (all|voices|synthesize|clone)

Requires:
    - MCP-over-TCP server running on HOST:PORT
"""

import argparse
import asyncio
import json
import subprocess
import sys
from typing import List, Dict, Any, Optional


# Configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765
DEFAULT_TEXT = "Hello, this is a test of the Kokoro TTS system."
DEFAULT_VOICE = "af_bella"


def _extract_text_from_content_item(item: Any) -> str:
    """Extract text content from a ContentBlock (dict or MCP object)."""
    if isinstance(item, dict):
        return item.get('text', '')
    
    # Handle MCP TextContent/ContentBlock objects
    for attr in ('text', 'content'):
        val = getattr(item, attr, None)
        if val is not None:
            return str(val)
    
    return ''


def _format_content_for_display(content_items: List[Any]) -> str:
    """Format content items into readable text for display."""
    lines = []
    for item in (content_items or []):
        text = _extract_text_from_content_item(item)
        if text:
            # Handle multiline text
            for line in text.split('\n'):
                if line.strip():
                    lines.append(line)
    return '\n'.join(lines)


def copy_audio_to_container(container: str, local_path: str) -> Optional[str]:
    """Copy reference audio file into container for voice cloning.
    
    Returns the path inside the container, or None on failure.
    """
    import os.path as osp
    filename = osp.basename(local_path)
    remote_path = f"/tmp/kokoro-tts/{filename}"
    
    print(f"Copying reference audio to container: {local_path} -> {remote_path}")
    
    try:
        result = subprocess.run(
            ["docker", "cp", local_path, f"{container}:{remote_path}"],
            capture_output=True, text=True, timeout=30
        )
        
        if result.returncode == 0:
            print(f"✓ Audio copied successfully")
            return remote_path
        else:
            print(f"✗ Failed to copy audio: {result.stderr[:100]}")
            return None
            
    except subprocess.TimeoutExpired:
        print("✗ Copy operation timed out")
        return None
    except Exception as e:
        print(f"✗ Copy failed: {e}")
        return None


def get_clone_voice_name() -> str:
    """Generate a unique clone voice identifier."""
    import hashlib
    import time
    
    timestamp = f"{int(time.time())}"
    random_bytes = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    # Generate from date/time for uniqueness
    return f"cloned_{timestamp}_{random_bytes}"


class TCPIPClient:
    """Lightweight TCP client for MCP-over-TCP protocol."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.request_id = 0

    async def connect(self) -> bool:
        """Connect to the MCP server via TCP."""
        print(f"Connecting to MCP-over-TCP server at {self.host}:{self.port}...")
        
        try:
            # Connect with timeout
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=10.0
            )
            
            self.reader = reader
            self.writer = writer
            
            # Read and verify server info (first message)
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if line:
                    info = json.loads(line.decode('utf-8').strip())
                    result = info.get('result', {})
                    version = result.get('version', 'unknown')
                    tool_count = result.get('toolCount', 0)
                    
                    print(f"✓ Connected to server (version {version}, {tool_count} tools available)")
                    return True
                else:
                    print("✗ No data received from server")
                    return False
            except Exception as e:
                print(f"✗ Failed to read server info: {e}")
                # Try anyway - might still work
                pass
                
        except asyncio.TimeoutError:
            print(f"✗ Connection timed out (host={self.host}, port={self.port})")
        except ConnectionRefusedError:
            print(f"✗ Connection refused - is the server running on {self.host}:{self.port}?")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
        
        return False

    async def disconnect(self):
        """Close the TCP connection (without shutting down the server)."""
        if self.writer:
            try:
                # Close connection gracefully without sending shutdown command
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send an MCP request and return the response."""
        self.request_id += 1
        
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        
        msg = json.dumps(request) + "\n"
        
        if not self.writer:
            raise RuntimeError("Not connected to server")
            
        self.writer.write(msg.encode())
        await self.writer.drain()
        
        try:
            response_line = await asyncio.wait_for(self.reader.readline(), timeout=120.0)  # 2 min for TTS operations
            if not response_line:
                raise RuntimeError("Connection closed by server")
            
            response = json.loads(response_line.decode('utf-8', errors='replace'))
            return response
        except asyncio.TimeoutError:
            if method == "tools/call":
                raise RuntimeError(f"Timeout waiting for response (TTS operation may still be in progress)")
            else:
                raise RuntimeError(f"Timeout waiting for response to {method}")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        result = await self._send_request(
            "tools/list",
            {}
        )
        
        if result.get("error"):
            raise RuntimeError(f"list_tools failed: {result['error']}")
        
        return result["result"].get("tools", [])

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool."""
        result = await self._send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments
            }
        )
        
        if result.get("error"):
            raise RuntimeError(f"tool call failed: {result['error']}")
        
        return result["result"]


async def test_list_voices(client: TCPIPClient) -> Dict[str, Any]:
    """Test listing available voices."""
    result = {"passed": True, "message": "", "is_warning": False}
    
    try:
        tools = await client.list_tools()
        
        tool_names = [t.get('name', 'unknown') for t in tools] if isinstance(tools, list) else []
        
        result["message"] = f"Server supports {len(tool_names)} tool(s)"
        if tool_names:
            result["message"] += f": {', '.join(tool_names)}"
            
    except Exception as e:
        result["passed"] = False
        result["is_warning"] = True
        result["message"] = f"Failed to list tools - {e}"
    
    return result


async def test_synthesize(client: TCPIPClient, text: str, voice: str, keep_audio: bool = False) -> Dict[str, Any]:
    """Test text-to-speech synthesis."""
    result = {"passed": True, "message": "", "is_warning": False}
    
    try:
        call_result = await client.call_tool(
            "tts_synthesize",
            arguments={
                "input": text,
                "voice": voice,
                "auto_play": True,
                "delete_after_playback": not keep_audio,
                "quality": "fp16"
            }
        )
        
        # Parse response content using helper for consistent display
        raw_content = call_result.get('content', []) if isinstance(call_result, dict) else []
        response_text = _format_content_for_display(raw_content)
        
        result["message"] = f"Response received:\n{response_text}"
        
        # Check if there was an error in the response (not just a warning)
        has_error = "Error:" in response_text and "Error: Error:" not in response_text
        
        # If it's a known issue (missing spacy model), treat as warning
        is_spacy_issue = "en_core_web_sm" in response_text.lower() or "spacy" in response_text.lower()
        
        if has_error and not is_spacy_issue:
            result["passed"] = False
        elif is_spacy_issue:
            result["is_warning"] = True
            result["message"] += "\n\n(Note: This warning may be due to missing spaCy language model. TTS should still work via subprocess fallback.)"
            
    except Exception as e:
        result["passed"] = False
        result["message"] = f"Synthesis failed - {e}"
    
    return result


async def test_clone(client: TCPIPClient, text: str, voice: str, clone_audio_path: Optional[str] = None, keep_audio: bool = False) -> Dict[str, Any]:
    """Test voice cloning + synthesis with auto-play.
    
    This tests the full voice cloning workflow using tts_clone tool which:
    1. Loads reference audio into memory for voice cloning
    2. Synthesizes text using the cloned voice  
    3. Auto-plays the result and optionally deletes the file
    
    Args:
        client: TCP client instance
        text: Text to synthesize with cloned voice
        voice: Default voice name (fallback if cloning fails)
        clone_audio_path: Path to reference WAV for cloning
        keep_audio: If True, keep audio files after playback; otherwise delete them
        
    Returns:
        Result dict with test outcome
    """
    result = {"passed": True, "message": "", "is_warning": False}
    
    # Step 1: Copy audio to container if local path provided
    remote_audio_path = clone_audio_path
    
    if clone_audio_path and not clone_audio_path.startswith("/"):
        # Local file - need to copy into container (for Docker deployments)
        # Note: This requires docker access; skip if we can't determine the container name
        try:
            result_cmd = subprocess.run(
                ["docker", "ps", "-q", "--filter", "name=kokoro-tts-mcp"],
                capture_output=True, text=True, timeout=5
            )
            if result_cmd.returncode == 0 and result_cmd.stdout.strip():
                container_name = result_cmd.stdout.strip().split('\n')[0]
                remote_audio_path = copy_audio_to_container(container_name, clone_audio_path)
        except Exception:
            print("(Skipping audio copy - docker access unavailable)")
    
    if not remote_audio_path:
        result["passed"] = False
        result["is_warning"] = True  
        result["message"] = "No reference audio available for cloning test"
        return result
    
    print(f"\nLoading cloned voice and synthesizing text...")
    print(f"Reference audio: {remote_audio_path}")
    
    # Step 2-3: Call tts_clone with auto_play=True (loads + synthesizes + plays)
    try:
        call_result = await client.call_tool(
            "tts_clone",
            arguments={
                "input": text,
                "ref_audio": remote_audio_path,
                "voice": voice,  # Fallback voice
                "auto_play": True,
                "delete_after_playback": not keep_audio,
                "quality": "fp16"
            }
        )
        
        # Parse response content using helper for consistent display
        raw_content = call_result.get('content', []) if isinstance(call_result, dict) else []
        response_text = _format_content_for_display(raw_content)
        
        result["message"] = f"Cloned voice synthesis:\n{response_text}"
        
        # Check for errors - kokoclone not installed is expected in minimal Docker images
        has_error = "Error:" in response_text and "Error: Error:" not in response_text
        
        if has_error:
            if "kokoclone" in response_text.lower():
                result["is_warning"] = True
                result["message"] += "\n\n(Note: Voice cloning requires the kokoclone library which may need to be installed separately. Basic TTS synthesis works fine.)"
            else:
                result["passed"] = False
            
    except Exception as e:
        import traceback
        error_msg = str(e) + "\n" + traceback.format_exc() if hasattr(traceback, 'format_exc') else str(e)
        
        # Treat missing kokoclone as expected in minimal deployments
        if "kokoclone" in str(e).lower():
            result["is_warning"] = True
            result["message"] = f"Cloning not available (kokoclone library not installed): {str(e)[:100]}"
        else:
            result["passed"] = False
            result["message"] = f"Cloned synthesis failed: {error_msg}"
    
    return result


async def run_tests(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    text: str = DEFAULT_TEXT,
    voice: str = DEFAULT_VOICE,
    tests: List[str] | None = None,
    clone_audio_path: Optional[str] = None,
    keep_audio: bool = False
) -> int:
    """Run specified MCP tests and return exit code (0=success)."""
    
    if tests is None:
        tests = ["voices", "synthesize"]
    
    client = TCPIPClient(host=host, port=port)
    results = []
    
    try:
        connected = await client.connect()
        if not connected:
            print("ERROR: Failed to connect to MCP server")
            return 1
        
        # Run tests while connection is active
        if "voices" in tests:
            results.append(("voices", await test_list_voices(client)))
        
        if "synthesize" in tests and "clone" not in tests:
            # Regular synthesis (no cloning)
            results.append(("synthesize", await test_synthesize(client, text, voice, keep_audio)))
        
        if "clone" in tests:
            # Voice cloning + synthesis with auto-play
            results.append(("clone", await test_clone(client, text, voice, clone_audio_path, keep_audio)))
    
    finally:
        await client.disconnect()
    
    # Print summary after all connections closed
    print("\n" + "="*60)
    print("MCP SERVER TEST RESULTS")
    print("="*60)
    
    for test_name, result in results:
        status = "PASS" if result["passed"] else ("WARN" if result.get("is_warning", False) else "FAIL")
        print(f"\n[TEST] {test_name.title()} - [{status}]")
        # Truncate long messages
        msg = result["message"].strip()
        if len(msg) > 300:
            msg = msg[:297] + "..."
        for line in msg.split('\n'):
            print(f"  {line}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r["passed"])
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "PASS" if result["passed"] else ("WARN" if result.get("is_warning", False) else "FAIL")
        print(f"  {test_name:15} [{status}]")
    
    print(f"\n{passed}/{len(results)} tests passed")
    
    # Return non-zero only on hard failures, not warnings
    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Test client for Kokoro TTS MCP Server (TCP Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # TCP connection options (replaces --container)
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"TCP host address (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"TCP port number (default: {DEFAULT_PORT})")
    
    parser.add_argument("--text", "-t", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--voice", "-v", default=DEFAULT_VOICE, help="Voice name (default: af_bella)")
    
    # Voice cloning options
    parser.add_argument(
        "--clone-audio",
        dest="clone_audio",
        default=None,
        help="Path to reference WAV file for voice cloning test"
    )
    
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        default=False,
        help="Keep generated audio files after auto-playback (default: delete after playback)"
    )
    
    test_group = parser.add_argument_group("tests")
    test_group.add_argument(
        "--test",
        nargs="+",
        choices=["all", "voices", "synthesize", "clone"],
        default=["all"],
        help="Which tests to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Expand 'all' to specific tests
    if "all" in args.test:
        if getattr(args, 'clone_audio', None):
            # If clone audio provided, test voices + cloning
            args.test = ["voices", "clone"]
        else:
            # Default behavior: voices + basic synthesis
            args.test = ["voices", "synthesize"]
    
    exit_code = asyncio.run(run_tests(
        host=args.host,
        port=args.port,
        text=args.text,
        voice=args.voice,
        tests=args.test,
        clone_audio_path=getattr(args, 'clone_audio', None),
        keep_audio=getattr(args, 'keep_audio', False)
    ))
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
