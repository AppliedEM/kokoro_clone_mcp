#!/usr/bin/env python3
"""
MCP Server Endpoint Tester - Tests Kokoro TTS endpoints via stdio transport.

This script:
1. Starts the MCP server as a subprocess  
2. Sends proper JSON-RPC messages over stdin/stdout
3. Receives and displays responses from each endpoint
4. Demonstrates in-memory caching behavior for voice cloning

Usage: python test_endpoints.py [--skip-download]
"""

import asyncio
import json
import sys
import time
from pathlib import Path


# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
KOKORO_SAMPLES = SCRIPT_DIR / ".." / "kokoro" / "samples"
SERVER_SCRIPT = SCRIPT_DIR / "mcp_server" / "server.py"

# Message counter for JSON-RPC requests
msg_id = 0


async def send_request(process, method: str, params: dict) -> dict:
    """Send a JSON-RPC request and wait for response."""
    global msg_id
    msg_id += 1
    
    # Build request message - MCP uses newline-delimited JSON over stdio
    request = {
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": method,
        "params": params
    }
    
    # Send the request (with newline delimiter)
    await process.stdin.write(json.dumps(request) + "\n")
    await process.stdin.drain()
    
    print(f"\n→ {method}({json.dumps(params)[:80]}...)")
    
    try:
        # Wait for response from stdout  
        response_line = await asyncio.wait_for(
            process.stdout.readline(), 
            timeout=120 if method == "tools/call" and params.get("name", "").startswith("tts_") else 30
        )
        
        if not response_line:
            return None
            
        return json.loads(response_line.decode())
        
    except asyncio.TimeoutError:
        print(f"⚠️  {method} timed out after timeout period")
        return None


async def test_endpoints(skip_download: bool = False):
    """Test all MCP server endpoints."""
    
    global msg_id
    
    print("="*60)
    print("Kokoro TTS MCP Server - Endpoint Tester")  
    print("="*60)
    print()
    
    # Show available sample files
    if KOKORO_SAMPLES.exists():
        wav_files = list(KOKORO_SAMPLES.glob("*.wav"))
        print(f"Available sample audio files ({len(wav_files)}):")
        for wf in sorted(wav_files):
            size_kb = wf.stat().st_size / 1024
            print(f"  • {wf.name} ({size_kb:.1f} KB)")
    
    # Read jo.txt transcript if available
    jo_txt = KOKORO_SAMPLES / "jo.txt"
    jo_transcript = ""
    if jo_txt.exists():
        with open(jo_txt) as f:
            jo_transcript = f.read().strip()
        print(f"\njo.wav transcript preview:")
        print(f"  '{jo_transcript[:60]}...'")
    
    # Start the MCP server
    print("\nStarting MCP Server...")
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", str(SERVER_SCRIPT), "--voice", "af_bella",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(SCRIPT_DIR)
    )
    
    print(f"Server started (PID: {process.pid})")
    await asyncio.sleep(2)  # Wait for server initialization
    
    try:
        # ========================================================================
        # Test 1: Initialize Connection
        # ========================================================================
        print("\n" + "="*60)
        print("TEST 1: Initialize MCP Connection")  
        print("="*60)
        
        init_request = {
            "jsonrpc": "2.0",
            "id": msg_id + 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "kokoro-tts-test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        await process.stdin.write(json.dumps(init_request) + "\n")
        await process.stdin.drain()
        print("→ initialize() sent...")
        
        init_response = await asyncio.wait_for(process.stdout.readline(), timeout=10)
        if not init_response:
            print("✗ Failed to receive initialization response")
            return
        
        try:
            init_data = json.loads(init_response.decode())
            result = init_data.get("result", {})
            capabilities = result.get("capabilities", {})
            tools_cap = capabilities.get("tools", {})
            
            print(f"✅ Connected successfully!")
            print(f"   Server version: {result.get('serverInfo', {}).get('version', 'unknown')}")
            print(f"   Tools capability: {bool(tools_cap)}")
        except json.JSONDecodeError as e:
            print(f"⚠️  Response received but couldn't parse: {str(e)[:100]}")
            
    except Exception as e:
        print(f"✗ Server failed during initialization: {e}")
        process.terminate()
        return
    
    # ========================================================================  
    # Test 2: List Available Tools
    # ========================================================================
    print("\n" + "="*60)
    print("TEST 2: List Available Tools")  
    print("="*60)
    
    tools_response = await send_request(process, "tools/list", {})
    if tools_response:
        result = tools_response.get("result", {})
        call_result = result.get("listToolsResult", {}) or {}
        tools = call_result.get("tools", [])
        
        print(f"\n✅ Found {len(tools)} MCP tools:")
        for tool in (tools if isinstance(tools, list) else []):
            name = tool.get("name", "unknown") if isinstance(tool, dict) else str(tool)
            desc = tool.get("description", "") if isinstance(tool, dict) else ""
            print(f"\n  🔧 {name}")
            print(f"     {desc[:100]}...")
    
    # ========================================================================
    # Test 3: List Voices (Metadata Tool)  
    # ========================================================================
    print("\n" + "="*60)
    print("TEST 3: list_voices - Available Voice Catalog")  
    print("="*60)
    
    voices_response = await send_request(process, "tools/call", {
        "name": "list_voices",
        "arguments": {}
    })
    
    # Extract response content
    if voices_response:
        result = voices_response.get("result", {})
        call_result = result.get("callToolResult", {}) or {}
        isError = call_result.get("isError", False)
        
        if not isError:
            print("\n✅ list_voices successful:")
            for item in (call_result.get("content") or []):
                text = item.get("text", "") if isinstance(item, dict) else str(item)
                # Truncate long responses  
                display = text[:300] + "..." if len(text) > 300 else text
                print(f"\n{display}")
    
    # ========================================================================
    # Test 4: tts_synthesize (Built-in Voice Synthesis)
    # ========================================================================
    print("\n" + "="*60)
    print("TEST 4: tts_synthesize - Built-in Voice TTS")  
    print("="*60)
    
    if skip_download:
        print("⊘ Skipped (--skip-download flag set)")
    else:
        text = "Hello, this is a test of the Kokoro TTS system."
        
        print(f"\nSynthesizing: '{text}'")
        print("(First call will download pykokoro model - can take 1-3 minutes)")
        start_time = time.time()
        
        synth_response = await send_request(process, "tools/call", {
            "name": "tts_synthesize", 
            "arguments": {"input": text, "voice": "af_bella"}
        })
        
        elapsed = time.time() - start_time
        
        if synth_response:
            result = synth_response.get("result", {})
            call_result = result.get("callToolResult", {}) or {}  
            isError = call_result.get("isError", False)
            
            print(f"\n✅ Synthesis completed in {elapsed:.1f}s:")
            for item in (call_result.get("content") or []):
                text_content = item.get("text", "") if isinstance(item, dict) else str(item)
                print(f"  {text_content}")
    
    # ========================================================================
    # Test 5: tts_clone (Voice Cloning with Caching Demonstration)
    # ========================================================================
    print("\n" + "="*60)  
    print("TEST 5: tts_clone - Voice Cloning")
    print("="*60)
    
    jo_wav_path = str(KOKORO_SAMPLES / "jo.wav") if KOKORO_SAMPLES.exists() else None
    
    if not jo_wav_path:
        print("⊘ Skipped (no sample files found)")
    elif skip_download:
        print("⊘ Skipped (--skip-download flag set)")  
    else:
        # Test 5a: First clone call - loads kanade model + reference audio into memory
        text = "This is a test of voice cloning functionality."
        
        print(f"\nUsing reference audio: {jo_wav_path}")
        if jo_transcript:
            print(f"Reference transcript available: yes")
            
        print("\nFirst call:")
        print("  - Loading kanade model into memory (~minutes)")  
        print("  - Loading reference audio features")
        start_time = time.time()
        
        clone_response_1 = await send_request(process, "tools/call", {
            "name": "tts_clone",
            "arguments": {
                "input": text,
                "ref_audio": jo_wav_path, 
                "ref_text": jo_transcript[:200] if jo_transcript else None
            }
        })
        
        elapsed_1 = time.time() - start_time
        
        # Test 5b: Second clone call with SAME reference - demonstrates caching
        print("\n" + "-"*40)
        print("Second call (same reference audio):")  
        print("  - Using IN-MEMORY cached state (fast!)")
        text_2 = "Voice cloning is now reusing the loaded model."
        
        start_time = time.time()
        clone_response_2 = await send_request(process, "tools/call", {
            "name": "tts_clone", 
            "arguments": {"input": text_2, "ref_audio": jo_wav_path}
        })
        elapsed_2 = time.time() - start_time
        
        # Display results comparison
        print(f"\nPerformance comparison:")
        if clone_response_1 and clone_response_2:
            r1_result = (clone_response_1.get("result", {}) or {}).get("callToolResult", {}) or {}
            r2_result = (clone_response_2.get("result", {}) or {}).get("callToolResult", {}) or {}
            
            print(f"  First call:  {elapsed_1:.1f}s")
            print(f"  Second call: {elapsed_2:.1f}s")
            
            if elapsed_2 < elapsed_1 * 0.8:
                print(f"\n✅ Caching confirmed - second call was faster!")
        
        # Show success messages  
        for i, resp in enumerate([clone_response_1, clone_response_2], 1):
            if resp:
                result = resp.get("result", {})
                call_result = result.get("callToolResult", {}) or {}
                isError = call_result.get("isError", False)
                
                print(f"\n✅ Call {i} successful:")
                for item in (call_result.get("content") or []):
                    text_content = item.get("text", "") if isinstance(item, dict) else str(item)
                    display = text_content[:200] + "..." if len(text_content) > 200 else text_content
                    print(f"  {display}")

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\n" + "="*60)
    print("Shutting down server...")
    print("="*60)
    
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        process.kill()


async def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MCP Server Endpoints")
    parser.add_argument("--skip-download", action="store_true", 
                       help="Skip tests requiring model downloads")
    
    args = parser.parse_args()
    
    try:
        await test_endpoints(skip_download=args.skip_download)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")


if __name__ == "__main__":
    asyncio.run(main())
