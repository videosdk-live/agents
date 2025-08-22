import json
import asyncio
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlparse

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession, stdio_client
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters
from mcp.types import JSONRPCMessage
from mcp.client.streamable_http import streamablehttp_client

from videosdk.agents.utils import RawFunctionTool, ToolError, create_generic_mcp_adapter, FunctionTool


class MCPConnectionManager:
    """Manages MCP client connections and lifecycle"""
    
    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout_seconds = timeout_seconds
        self.session: Optional[ClientSession] = None
        self.context_stack = AsyncExitStack()
        self.is_connected = False
        
    async def establish_connection(self, stream_provider) -> ClientSession:
        """Establish connection using provided stream source"""
        if self.is_connected:
            return self.session
            
        try:
            streams = await self.context_stack.enter_async_context(stream_provider)
            rx_stream, tx_stream = streams[0], streams[1]
            
            timeout_delta = timedelta(seconds=self.timeout_seconds) if self.timeout_seconds else None
            
            self.session = await self.context_stack.enter_async_context(
                ClientSession(rx_stream, tx_stream, read_timeout_seconds=timeout_delta)
            )
            
            await self.session.initialize()
            self.is_connected = True
            return self.session
            
        except Exception:
            await self.cleanup()
            raise
    
    async def cleanup(self):
        """Clean up connection resources"""
        try:
            await self.context_stack.aclose()
        finally:
            self.session = None
            self.is_connected = False


class MCPToolRegistry:
    """Registry for managing MCP tools with caching"""
    
    def __init__(self):
        self.cached_tools: Optional[List[FunctionTool]] = None
        self.needs_refresh = True
        
    def mark_stale(self):
        """Mark the tool cache as needing refresh"""
        self.needs_refresh = True
        
    def has_valid_cache(self) -> bool:
        """Check if cached tools are still valid"""
        return not self.needs_refresh and self.cached_tools is not None
        
    def update_cache(self, tools: List[FunctionTool]):
        """Update the tool cache"""
        self.cached_tools = tools
        self.needs_refresh = False
        
    def get_cached_tools(self) -> List[FunctionTool]:
        """Get cached tools if available"""
        if not self.has_valid_cache():
            raise RuntimeError("No valid tool cache available")
        return self.cached_tools


class MCPToolExecutor:
    """Handles execution of MCP tools with proper error handling"""
    
    def __init__(self, connection_manager: MCPConnectionManager):
        self.connection_manager = connection_manager
        
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute an MCP tool and process the response"""
        if not self.connection_manager.is_connected or not self.connection_manager.session:
            raise ToolError(f"Cannot execute tool '{tool_name}': MCP connection not established")
            
        try:
            execution_result = await self.connection_manager.session.call_tool(tool_name, parameters)
            return self._process_tool_result(tool_name, execution_result)
            
        except Exception as e:
            if isinstance(e, ToolError):
                raise
            raise ToolError(f"Tool execution failed for '{tool_name}': {str(e)}")
    
    def _process_tool_result(self, tool_name: str, result) -> Any:
        """Process and format tool execution result"""
        if result.isError:
            error_details = " | ".join(str(item) for item in result.content)
            raise ToolError(f"Tool '{tool_name}' reported error: {error_details}")
            
        if not result.content:
            return {"output": None, "tool": tool_name}
            
        if len(result.content) == 1:
            return self._format_single_content(result.content[0])
            
        return self._format_multiple_content(result.content)
    
    def _format_single_content(self, content) -> Dict[str, Any]:
        """Format single content item"""
        if hasattr(content, 'type') and content.type == 'text' and hasattr(content, 'text'):
            return {"output": content.text, "type": "text"}
            
        try:
            if hasattr(content, 'model_dump_json'):
                parsed_json = json.loads(content.model_dump_json())
                return parsed_json if isinstance(parsed_json, dict) else {"output": parsed_json}
        except (json.JSONDecodeError, AttributeError):
            pass
            
        return {"output": str(content), "type": "raw"}
    
    def _format_multiple_content(self, content_items) -> Dict[str, Any]:
        """Format multiple content items"""
        try:
            formatted_items = []
            for item in content_items:
                if hasattr(item, 'type') and item.type == 'text' and hasattr(item, 'text'):
                    formatted_items.append({"content": item.text, "type": "text"})
                else:
                    try:
                        formatted_items.append(item.model_dump())
                    except AttributeError:
                        formatted_items.append({"content": str(item), "type": "raw"})
                        
            return {"output": formatted_items, "type": "multi_content"}
            
        except Exception:
            return {"output": [str(item) for item in content_items], "type": "raw_list"}


class MCPServiceProvider(ABC):
    """Abstract base for MCP service providers"""
    
    def __init__(self, connection_timeout: float = 5.0):
        self.connection_mgr = MCPConnectionManager(connection_timeout)
        self.tool_registry = MCPToolRegistry()
        self.tool_executor = MCPToolExecutor(self.connection_mgr)
        
    @property
    def is_ready(self) -> bool:
        """Check if the service provider is ready"""
        return self.connection_mgr.is_connected
        
    async def connect(self):
        """Establish connection to MCP service"""
        await self.connection_mgr.establish_connection(self.get_stream_provider())
        
    async def get_available_tools(self) -> List[FunctionTool]:
        """Retrieve available tools from MCP service"""
        if not self.is_ready:
            raise RuntimeError("MCP service provider not connected")
            
        if self.tool_registry.has_valid_cache():
            return self.tool_registry.get_cached_tools()
            
        mcp_tools_list = await self.connection_mgr.session.list_tools()
        framework_tools = []
        
        for mcp_tool in mcp_tools_list.tools:
            tool_executor = partial(self.tool_executor.execute_tool, mcp_tool.name)
            
            adapted_tool = create_generic_mcp_adapter(
                tool_name=mcp_tool.name,
                tool_description=mcp_tool.description,
                input_schema=mcp_tool.inputSchema,
                client_call_function=tool_executor
            )
            
            framework_tools.append(adapted_tool)
            
        self.tool_registry.update_cache(framework_tools)
        return framework_tools
    
    def invalidate_tool_cache(self):
        """Force refresh of tool cache on next request"""
        self.tool_registry.mark_stale()
        
    async def disconnect(self):
        """Disconnect from MCP service"""
        await self.connection_mgr.cleanup()
        self.tool_registry.mark_stale()
        
    @abstractmethod
    def get_stream_provider(self):
        """Get the stream provider for this service type"""
        pass


class HTTPTransportDetector:
    """Utility class to detect HTTP transport type"""
    
    @staticmethod
    def detect_transport_mode(url: str) -> str:
        """Detect transport mode based on URL characteristics"""
        parsed = urlparse(url.lower())
        path_segments = parsed.path.strip('/').split('/')
        
        if path_segments and path_segments[-1] == 'mcp':
            return 'streamable_http'
        elif path_segments and path_segments[-1] == 'sse':
            return 'sse'
        else:
            return 'sse'
    
    @staticmethod
    def is_streamable_http(url: str) -> bool:
        """Check if URL should use streamable HTTP transport"""
        return HTTPTransportDetector.detect_transport_mode(url) == 'streamable_http'


class MCPServerHTTP(MCPServiceProvider):
    """HTTP/Web-based MCP service provider with automatic transport detection"""
    
    def __init__(
        self,
        endpoint_url: str,
        request_headers: Optional[Dict[str, Any]] = None,
        connection_timeout: float = 10.0,
        stream_read_timeout: float = 300.0,
        session_timeout: float = 5.0,
    ):
        super().__init__(session_timeout)
        self.endpoint_url = endpoint_url
        self.request_headers = request_headers or {}
        self.connection_timeout = connection_timeout
        self.stream_read_timeout = stream_read_timeout
        
        self.transport_mode = HTTPTransportDetector.detect_transport_mode(endpoint_url)
        
    def get_stream_provider(self):
        """Get appropriate stream provider based on detected transport"""
        timeout_delta = timedelta(seconds=self.connection_timeout)
        
        if self.transport_mode == 'streamable_http':
            return streamablehttp_client(
                url=self.endpoint_url,
                headers=self.request_headers,
                timeout=timeout_delta,
            )
        else: 
            return sse_client(
                url=self.endpoint_url,
                headers=self.request_headers,
                timeout=self.connection_timeout,
            )
    
    def __repr__(self) -> str:
        return f"MCPServerHTTP(url={self.endpoint_url}, transport={self.transport_mode})"


class MCPServerStdio(MCPServiceProvider):
    """Process-based MCP service provider for local applications"""
    
    def __init__(
        self,
        executable_path: str,
        process_arguments: List[str],
        environment_vars: Optional[Dict[str, str]] = None,
        working_directory: Optional[str | Path] = None,
        session_timeout: float = 5.0,
    ):
        super().__init__(session_timeout)
        self.executable_path = executable_path
        self.process_arguments = process_arguments
        self.environment_vars = environment_vars
        self.working_directory = Path(working_directory) if working_directory and not isinstance(working_directory, Path) else working_directory
        
    def get_stream_provider(self):
        """Get stdio stream provider for process communication"""
        server_params = StdioServerParameters(
            command=self.executable_path,
            args=self.process_arguments,
            env=self.environment_vars,
            cwd=self.working_directory
        )
        return stdio_client(server_params)
    
    def __repr__(self) -> str:
        return (f"MCPServerStdio(executable={self.executable_path}, "
                f"args={self.process_arguments}, cwd={self.working_directory})")