import json
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import logging
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

try:
    from mcp import ClientSession, stdio_client
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters
    from mcp.types import JSONRPCMessage
except ImportError as e:
    raise ImportError(
        "The 'mcp' package is required to run the MCP server integration but is not installed.\n"
        "To fix this, install the optional dependency: pip install 'livekit-agents[mcp]'"
    ) from e


from .utils import RawFunctionTool, ToolError, create_generic_mcp_adapter, FunctionTool

logger = logging.getLogger(__name__)

# Type alias for MCP tools 
MCPTool = RawFunctionTool


class MCPServer(ABC):
    """Base class for MCP server connections"""
    
    def __init__(self, *, client_session_timeout_seconds: float) -> None:
        self._client: Optional[ClientSession] = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._read_timeout = client_session_timeout_seconds

        self._cache_dirty = True
        self._tools_cache: Optional[List[FunctionTool]] = None

    @property
    def initialized(self) -> bool:
        """Check if the server is initialized"""
        return self._client is not None

    def invalidate_cache(self) -> None:
        """Invalidate the tools cache"""
        self._cache_dirty = True
        self._tools_cache = None

    async def initialize(self) -> None:
        """Initialize the MCP server connection"""
        try:
            receive_stream, send_stream = await self._exit_stack.enter_async_context(
                self.client_streams()
            )
            self._client = await self._exit_stack.enter_async_context(
                ClientSession(
                    receive_stream,
                    send_stream,
                    read_timeout_seconds=timedelta(seconds=self._read_timeout)
                    if self._read_timeout
                    else None,
                )
            )
            await self._client.initialize()
        except Exception:
            await self.aclose()
            raise

    async def list_tools(self) -> List[FunctionTool]:
        """Get the list of tools available on the MCP server as framework FunctionTools"""
        if self._client is None:
            raise RuntimeError("MCPServer isn't initialized")

        if not self._cache_dirty and self._tools_cache is not None:
            return self._tools_cache

        # Get tools from MCP server
        mcp_tools_response = await self._client.list_tools()
        
        # Convert MCP tools to framework tools
        framework_tools = []
        for tool in mcp_tools_response.tools:
            # Create a closure for calling this specific tool
            tool_caller = partial(self._call_mcp_tool, tool.name)
            
            # Create a framework tool adapter
            adapter = create_generic_mcp_adapter(
                tool_name=tool.name,
                tool_description=tool.description,
                input_schema=tool.inputSchema,
                client_call_function=tool_caller
            )
            
            framework_tools.append(adapter)

        # Cache the tools
        self._tools_cache = framework_tools
        self._cache_dirty = False
        return framework_tools

    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Internal method to call an MCP tool and handle the response"""
        if self._client is None:
            raise ToolError("MCP server not initialized")
            
        try:
            # Log the call for debugging
            logger.info(f"Calling MCP tool '{tool_name}' with arguments: {arguments}")
            
            # Call the tool
            result = await self._client.call_tool(tool_name, arguments)
            
            # Handle error response
            if result.isError:
                error_str = "\n".join(str(part) for part in result.content)
                raise ToolError(f"Error in MCP tool '{tool_name}': {error_str}")
                
            # Process successful response
            if not result.content:
                return {"result": None}
                
            if len(result.content) == 1:
                content = result.content[0]
                # Handle text content - wrap it in dictionary for API compatibility
                if hasattr(content, 'type') and content.type == 'text' and hasattr(content, 'text'):
                    return {"result": content.text}
                
                # Try to parse JSON if possible
                try:
                    if hasattr(content, 'model_dump_json'):
                        json_str = content.model_dump_json()
                        json_obj = json.loads(json_str)
                        return json_obj if isinstance(json_obj, dict) else {"result": json_obj}
                except:
                    # Fallback - return as string
                    return {"result": str(content)}
            else:
                # Multiple content items - combine as array in result field
                try:
                    items = [
                        (item.text if hasattr(item, 'type') and item.type == 'text' and hasattr(item, 'text') 
                         else item.model_dump()) 
                        for item in result.content
                    ]
                    return {"result": items}
                except:
                    # Fallback if model_dump fails
                    return {"result": [str(item) for item in result.content]}
                
        except Exception as e:
            if not isinstance(e, ToolError):
                raise ToolError(f"Error calling MCP tool '{tool_name}': {str(e)}")
            raise

    async def aclose(self) -> None:
        """Close the MCP server connection"""
        try:
            await self._exit_stack.aclose()
        finally:
            self._client = None
            self._tools_cache = None

    @abstractmethod
    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]: 
        """Get client streams for communication with the MCP server"""
        pass


class MCPServerHTTP(MCPServer):
    """HTTP-based MCP server connection"""

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, Any]] = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
        client_session_timeout_seconds: float = 5,
    ) -> None:
        super().__init__(client_session_timeout_seconds=client_session_timeout_seconds)
        self.url = url
        self.headers = headers
        self._timeout = timeout
        self._sse_read_timeout = sse_read_timeout

    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        return sse_client(
            url=self.url,
            headers=self.headers,
            timeout=self._timeout,
            sse_read_timeout=self._sse_read_timeout,
        )

    def __repr__(self) -> str:
        return f"MCPServerHTTP(url={self.url})"


class MCPServerStdio(MCPServer):
    """Stdio-based MCP server connection (for local process communication)"""
    
    def __init__(
        self,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str | Path] = None,
        client_session_timeout_seconds: float = 5,
    ) -> None:
        super().__init__(client_session_timeout_seconds=client_session_timeout_seconds)
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd

    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        return stdio_client(
            StdioServerParameters(command=self.command, args=self.args, env=self.env, cwd=self.cwd)
        )

    def __repr__(self) -> str:
        return f"MCPServerStdio(command={self.command}, args={self.args}, cwd={self.cwd})"