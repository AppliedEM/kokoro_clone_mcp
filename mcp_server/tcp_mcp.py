"""
MCP-over-TCP Server Implementation for Kokoro TTS

A lightweight JSON-RPC 2.0 transport layer over TCP sockets that exposes
the same tool interface as the standard MCP SDK but without requiring stdin/stdout pipes.

This allows the server to run persistently in Docker and accept multiple concurrent clients.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("kokoro-mcp-server")


class TcpMcpServer:
    """TCP-based MCP server that handles JSON-RPC 2.0 over TCP sockets."""

    PROTOCOL_VERSION = "2024-11-05"
    
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self._server_task: Optional[asyncio.Task] = None
        self._running = False
        self._clients = set()
        
        # Tool handlers (populated by server.py)
        self.tools = []
        self.tool_handlers = {}
        
    def register_tools(self, tools):
        """Register MCP tool definitions and their handlers."""
        self.tools = tools
        
    def register_tool_handler(self, name, handler):
        """Register a tool call handler."""
        self.tool_handlers[name] = handler
        
    async def start(self):
        """Start the TCP server listener."""
        logger.info(f"Starting MCP-over-TCP server on {self.host}:{self.port}")
        
        self._server_task = asyncio.create_task(self._accept_connections())
        self._running = True
        
        print(f"\n{'='*60}")
        print("MCP-over-TCP Server Ready")
        print(f"Address: {self.host}:{self.port}")
        tools_names = ', '.join(t.name if hasattr(t, 'name') else t.get('name', '') for t in self.tools) if self.tools else 'none'
        print(f"Tools: {tools_names}")
        print('='*60 + '\n')

    async def stop(self):
        """Stop the TCP server."""
        logger.info("Shutting down MCP-over-TCP server...")
        self._running = False
        
        for reader, writer in list(self._clients):
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
        
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass

    async def _accept_connections(self):
        """Accept incoming TCP connections."""
        server = None
        try:
            server = await asyncio.start_server(
                self._handle_client, 
                self.host, 
                self.port
            )
            
            addr = server.sockets[0].getsockname()
            logger.info(f"Listening on {addr}")
            
            while self._running:
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            pass
        finally:
            if server:
                server.close()
                await server.wait_closed()

    async def _handle_client(self, reader, writer):
        """Handle a single client connection."""
        client_addr = writer.get_extra_info('peername')
        logger.info(f"Client connected from {client_addr}")
        
        self._clients.add((reader, writer))
        
        try:
            # Send server info to client (first message)
            server_info = json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "method": "server/initialized",
                "result": {
                    "version": self.PROTOCOL_VERSION,
                    "serverName": "kokoro-tts-mcp",
                    "toolCount": len(self.tools),
                    "tools": [{"name": t.name if hasattr(t, "name") else t.get("name", ""), "description": getattr(t, "description", "") or (t.get("description", ""))}
                             for t in self.tools] if self.tools else []
                }
            }) + "\n"
            
            writer.write(server_info.encode())
            await writer.drain()
            
            while self._running:
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=30)
                    if not line:
                        break
                        
                    request_text = line.decode('utf-8').strip()
                    if not request_text:
                        continue
                        
                    try:
                        request = json.loads(request_text)
                    except json.JSONDecodeError as e:
                        error_response = self._make_error(None, -32700, f"Parse error: {e}")
                        writer.write(error_response.encode() + b"\n")
                        await writer.drain()
                        continue
                    
                    response = await self._process_request(request)
                    
                    if response is not None:
                        writer.write(response.encode() + b"\n")
                        await writer.drain()
                        
                except asyncio.TimeoutError:
                    try:
                        ping = json.dumps({
                            "jsonrpc": "2.0", 
                            "id": None,
                            "method": "server/ping"
                        }) + "\n"
                        writer.write(ping.encode())
                        await writer.drain()
                    except Exception:
                        break
                        
        except asyncio.CancelledError:
            pass
        finally:
            self._clients.discard((reader, writer))
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.info(f"Client disconnected: {client_addr}")

    async def _process_request(self, request):
        """Process a JSON-RPC 2.0 request."""
        method = request.get("method", "")
        params = request.get("params", {})
        msg_id = request.get("id")
        
        try:
            if method == "tools/list":
                # Convert Tool objects to dictionaries for JSON serialization
                tools_list = []
                for t in self.tools:
                    tool_dict = {
                        "name": t.name if hasattr(t, 'name') else (t.get('name', '') if isinstance(t, dict) else ''),
                        "description": getattr(t, 'description', '') or (t.get('description', '') if isinstance(t, dict) else '')
                    }
                    tools_list.append(tool_dict)
                
                result = {
                    "toolCount": len(self.tools),
                    "tools": tools_list
                }
                return self._make_response(msg_id, result)
                
            elif method in ("tools/call", "call_tool"):
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                
                handler = self.tool_handlers.get(tool_name)
                if not handler:
                    return self._make_error(msg_id, -32601, f"Unknown tool: {tool_name}")
                
                # Call the handler (may be sync or async)
                result = None
                
                if callable(handler):
                    try:
                        import inspect
                        if asyncio.iscoroutinefunction(handler):
                            result = await handler(arguments)
                        else:
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(None, lambda: handler(arguments))
                    except Exception as e:
                        return self._make_error(msg_id, -32000, f"Tool execution error: {e}")
                
                # Extract response data - handle both dict and MCP object formats
                if isinstance(result, dict) and "isError" in result:
                    is_error = result.get("isError", False)
                    content_raw = result.get("content", [])
                elif hasattr(result, 'model_dump'):  # Pydantic v2 model (MCP 1.27+)
                    # Use model_dump() for proper serialization
                    dump = result.model_dump() if callable(getattr(result, 'model_dump')) else {}
                    is_error = dump.get("isError", False)
                    content_raw = list(dump.get("content", [])) if isinstance(dump.get("content"), (list, tuple)) else []
                elif hasattr(result, 'to_dict'):  # Pydantic v1 model (older MCP)
                    is_error = getattr(result, 'isError', False)
                    content_raw = list(getattr(result, 'content', [])) if hasattr(result, 'content') else []
                elif isinstance(result, dict):  # Generic dict response
                    is_error = result.get("isError", False)
                    content_raw = result.get("content", [])
                else:
                    logger.debug(f"Unexpected result type: {type(result).__name__}")
                    is_error = False
                    content_raw = [{"type": "text", "text": str(result)}] if result else []
                
                # Convert ContentBlock objects to dictionaries for JSON serialization
                content = []
                logger.debug(f"Serializing {len(content_raw or [])} content items")
                for idx, item in enumerate(content_raw or []):
                    if isinstance(item, dict):
                        content.append(item)
                    elif hasattr(item, 'type') and hasattr(item, 'text'):  # TextContent object with attributes
                        try:
                            item_type = getattr(item, 'type', 'text')
                            item_text = str(getattr(item, 'text', ''))
                            logger.debug(f"    Extracted type='{item_type}', text length={len(item_text)}")
                            logger.debug(f"    Extracted type='{item_type}', text length={len(item_text)}")
                            logger.debug(f"    Extracted type='{item_type}', text length={len(item_text)}")
                            content.append({
                                "type": item_type,
                                "text": item_text
                            })
                        except Exception as e:
                            logger.warning(f"Failed to serialize ContentBlock: {e}")
                            # Fallback - try to_dict if available
                            if hasattr(item, 'to_dict'):
                                content.append(item.to_dict())
                            else:
                                content.append({"type": "text", "text": str(item)})
                    else:
                        # Try to_dict as fallback for MCP objects
                        if hasattr(item, 'to_dict') and callable(getattr(item, 'to_dict')):
                            try:
                                content.append(item.to_dict())
                                continue
                            except Exception:
                                pass
                        # Fallback - convert to string
                        content.append({"type": "text", "text": str(item)})
                
                return self._make_response(msg_id, {
                    "isError": is_error,
                    "content": content
                })
                
            elif method in ("server/shutdown", "shutdown"):
                logger.info("Received shutdown request from client")
                asyncio.create_task(self.stop())
                return self._make_response(msg_id, {"status": "shutting down"})
                
            else:
                return self._make_error(msg_id, -32601, f"Unknown method: {method}")
                
        except Exception as e:
            logger.error(f"Request processing error: {e}", exc_info=True)
            return self._make_error(msg_id, -32603, str(e))

    def _make_response(self, msg_id, result):
        """Create a JSON-RPC 2.0 response."""
        return json.dumps({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result
        })

    def _make_error(self, msg_id, code: int, message: str):
        """Create a JSON-RPC 2.0 error response."""
        return json.dumps({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": code,
                "message": message
            }
        })


class TcpMcpClient:
    """Simple TCP client for connecting to MCP-over-TCP servers."""

    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None
        
    async def connect(self):
        """Connect to the MCP server."""
        logger.info(f"Connecting to {self.host}:{self.port}...")
        
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port),
            timeout=10.0
        )
        
        self.reader = reader
        self.writer = writer
        
        try:
            server_info_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if server_info_line:
                server_info = json.loads(server_info_line.decode('utf-8').strip())
                logger.info(f"Server connected: {server_info.get('result', {}).get('version', 'unknown')}")
        except Exception as e:
            logger.warning(f"Could not read server info: {e}")
            
    async def call_tool(self, tool_name, arguments):
        """Call a tool on the MCP server."""
        request = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments}
        }
        
        self.writer.write((json.dumps(request) + "\n").encode())
        await self.writer.drain()
        
        try:
            response_line = await asyncio.wait_for(self.reader.readline(), timeout=300.0)
            if not response_line:
                return {"error": "No response received"}
                
            response = json.loads(response_line.decode('utf-8').strip())
            
            if "result" in response:
                return response["result"]
            elif "error" in response:
                return f"Error: {response['error']['message']}"
                
        except asyncio.TimeoutError:
            return {"error": "Request timed out"}
        
    async def list_tools(self):
        """List available tools."""
        request = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "tools/list"
        }
        
        self.writer.write((json.dumps(request) + "\n").encode())
        await self.writer.drain()
        
        try:
            response_line = await asyncio.wait_for(self.reader.readline(), timeout=5.0)
            if response_line:
                response = json.loads(response_line.decode('utf-8').strip())
                return response.get("result", {})
        except asyncio.TimeoutError:
            pass
        
    async def close(self):
        """Close the connection."""
        if self.writer:
            try:
                request = {
                    "jsonrpc": "2.0",
                    "id": int(time.time() * 1000),
                    "method": "server/shutdown"
                }
                self.writer.write((json.dumps(request) + "\n").encode())
                await asyncio.wait_for(self.writer.drain(), timeout=2.0)
            except Exception:
                pass
            
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass


async def main_tcp(host="0.0.0.0", port=8765):
    """Main entry point for TCP mode."""
    import sys
    
    logger = logging.getLogger("kokoro-mcp-server")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    sys.path.insert(0, str(Path(__file__).parent))
    import server as mcp_server_module
    
    tcp_server = TcpMcpServer(host=host, port=port)
    
    tools = mcp_server_module.get_tool_definitions()
    tcp_server.register_tools(tools)
    
    logger.info(f"Starting TCP mode with {len(tools)} tools")
    
    try:
        await tcp_server.start()
        
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await tcp_server.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kokoro TTS MCP Server (TCP Mode)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--voice", default=None, help="Default voice name")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Compute device")
    
    args = parser.parse_args()
    
    asyncio.run(main_tcp(host=args.host, port=args.port))
