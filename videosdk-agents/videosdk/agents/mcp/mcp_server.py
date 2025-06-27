import json
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession, stdio_client
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters
from mcp.types import JSONRPCMessage

from mcp.client.streamable_http import streamablehttp_client


from ..utils import RawFunctionTool, ToolError, create_generic_mcp_adapter, FunctionTool

# MCPTool alias remains, as it's a type hint
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

    async def initialize(self) -> None:
        """Initialize the MCP server connection"""
        try:
            streams = await self._exit_stack.enter_async_context(
                self.client_streams()
            )
            receive_stream, send_stream = streams[0], streams[1]
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

        mcp_tools_response = await self._client.list_tools()
        
        framework_tools = []
        for tool in mcp_tools_response.tools:
            tool_caller = partial(self._call_mcp_tool, tool.name)
            
            adapter = create_generic_mcp_adapter(
                tool_name=tool.name,
                tool_description=tool.description,
                input_schema=tool.inputSchema,
                client_call_function=tool_caller
            )
            
            framework_tools.append(adapter)

        self._tools_cache = framework_tools
        self._cache_dirty = False
        return framework_tools

    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Internal method to call an MCP tool and handle the response"""
        if self._client is None:
            raise ToolError("MCP server not initialized")
            
        try:
            result = await self._client.call_tool(tool_name, arguments)
            
            if result.isError:
                error_str = "\n".join(str(part) for part in result.content)
                raise ToolError(f"Error in MCP tool '{tool_name}': {error_str}")
                
            if not result.content:
                return {"result": None}
                
            if len(result.content) == 1:
                content = result.content[0]
                
                if hasattr(content, 'type') and content.type == 'text' and hasattr(content, 'text'):
                    return {"result": content.text}
                
                try:
                    if hasattr(content, 'model_dump_json'):
                        json_str = content.model_dump_json()
                        json_obj = json.loads(json_str)
                        return json_obj if isinstance(json_obj, dict) else {"result": json_obj}
                except Exception:
                    return {"result": str(content)}
            else:
                try:
                    items = [
                        (item.text if hasattr(item, 'type') and item.type == 'text' and hasattr(item, 'text') 
                         else item.model_dump()) 
                        for item in result.content
                    ]
                    return {"result": items}
                except Exception:
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
            ...,
        ]
    ]: 
        """Get client streams for communication with the MCP server"""
        pass


class MCPServerHTTP(MCPServer):
    """
    HTTP-based MCP server to detect transport type based on URL path.
    
    - URLs ending with 'mcp' use streamable HTTP transport
    - URLs ending with 'sse' use Server-Sent Events (SSE) transport
    - For other URLs, defaults to SSE transport for backward compatibility
    
    Note: SSE transport is being deprecated in favor of streamable HTTP transport.
    See: https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206
    """

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
        self._use_streamable_http = self._should_use_streamable_http(url)

    def _should_use_streamable_http(self, url: str) -> bool:
        """
        Determine transport type based on URL path.
        
        Returns True for streamable HTTP if URL ends with 'mcp',
        False for SSE if URL ends with 'sse' or for backward compatibility.
        """
        url_lower = url.lower().rstrip('/')
        return url_lower.endswith('mcp')

    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
            ...,
        ]
    ]:
        if self._use_streamable_http:
            return streamablehttp_client(
                url=self.url,
                headers=self.headers,
                timeout=timedelta(seconds=self._timeout),
            )
        else:
            return sse_client(  # type: ignore[no-any-return]
                url=self.url,
                headers=self.headers,
                timeout=self._timeout,
            )

    def __repr__(self) -> str:
        transport_type = "streamable_http" if self._use_streamable_http else "sse"
        return f"MCPServerHTTP(url={self.url}, transport={transport_type})"


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
            ...,
        ]
    ]:
        return stdio_client(  # type: ignore[no-any-return]
            StdioServerParameters(command=self.command, args=self.args, env=self.env, cwd=self.cwd)
        )

    def __repr__(self) -> str:
        return f"MCPServerStdio(command={self.command}, args={self.args}, cwd={self.cwd})"