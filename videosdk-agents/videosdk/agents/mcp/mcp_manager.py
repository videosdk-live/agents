import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from videosdk.agents.mcp.mcp_server import MCPServiceProvider, MCPServerStdio, MCPServerHTTP
from videosdk.agents.utils import FunctionTool, ToolError


class MCPToolManager:
    """
    Manages MCP service providers and their tools for agents.
    """

    def __init__(self) -> None:
        """
        Initialize the MCPToolManager.
        """
        self._active_providers: List[MCPServiceProvider] = []
        self._registered_tools: List[FunctionTool] = []

    @property
    def providers(self) -> List[MCPServiceProvider]:
        """
        Get all active MCP service providers.

        Returns:
            List[MCPServiceProvider]: A copy of the list of active providers.
        """
        return self._active_providers.copy()

    @property
    def tools(self) -> List[FunctionTool]:
        """
        Get all registered tools from all providers.

        Returns:
            List[FunctionTool]: A copy of the list of all available tools.
        """
        return self._registered_tools.copy()

    async def add_mcp_server(self, provider: MCPServiceProvider) -> None:
        """
        Add a new MCP service provider and initialize it.
        Args:
            provider (MCPServiceProvider): The MCP service provider to add.

        Raises:
            Exception: If provider initialization fails (connection, tool discovery, etc.).
        """
        if not provider.is_ready:
            try:
                await provider.connect()

                tools = await provider.get_available_tools()

                self._registered_tools.extend(tools)
                if provider not in self._active_providers:
                    self._active_providers.append(provider)
            except Exception as e:
                await provider.disconnect()
                raise

    async def remove_provider(self, provider: MCPServiceProvider) -> None:
        """
        Remove a service provider and its tools.
        Args:
            provider (MCPServiceProvider): The MCP service provider to remove.
        """
        if provider in self._active_providers:
            await provider.disconnect()
            self._active_providers.remove(provider)

            self._registered_tools.clear()
            for active_provider in self._active_providers:
                if active_provider.is_ready:
                    tools = await active_provider.get_available_tools()
                    self._registered_tools.extend(tools)

    async def refresh_tools(self) -> None:
        """
        Refresh tools from all active providers.
        """
        self._registered_tools.clear()
        for provider in self._active_providers:
            if provider.is_ready:
                provider.invalidate_tool_cache()
                tools = await provider.get_available_tools()
                self._registered_tools.extend(tools)

    async def cleanup(self) -> None:
        """
        Close all MCP service providers and clear the tools list.
        """
        for provider in self._active_providers:
            await provider.disconnect()
        self._active_providers.clear()
        self._registered_tools.clear()

    def get_tool_by_name(self, tool_name: str) -> Optional[FunctionTool]:
        """
        Get a specific tool by name from the registry.

        Args:
            tool_name (str): The name of the tool to find.

        Returns:
            Optional[FunctionTool]: The found tool or None if not found.
        """
        for tool in self._registered_tools:
            tool_info = getattr(tool, "_tool_info", None)
            if tool_info and hasattr(tool_info, 'name') and tool_info.name == tool_name:
                return tool
            elif hasattr(tool, 'name') and tool.name == tool_name:
                return tool
        return None

    def get_tools_by_provider(self, provider: MCPServiceProvider) -> List[FunctionTool]:
        """
        Get all tools from a specific provider.

        Args:
            provider (MCPServiceProvider): The provider whose tools to retrieve.

        Returns:
            List[FunctionTool]: List of tools from the specified provider, or empty list if provider not found.
        """
        if provider not in self._active_providers:
            return []

        return [tool for tool in self._registered_tools if hasattr(tool, "_tool_info")]

    async def add_mcp_server_http(
        self,
        url: str,
        headers: Optional[Dict[str, Any]] = None,
        connection_timeout: float = 10.0,
        stream_read_timeout: float = 300.0,
        session_timeout: float = 5.0
    ) -> MCPServerHTTP:
        """
        Convenience method to create and add an HTTP MCP server.

        Args:
            url (str): The endpoint URL for the MCP server.
            headers (Optional[Dict[str, Any]], optional): Optional request headers for authentication or configuration.
            connection_timeout (float, optional): Connection timeout in seconds. Defaults to 10.0.
            stream_read_timeout (float, optional): Stream read timeout in seconds. Defaults to 300.0.
            session_timeout (float, optional): Session timeout in seconds. Defaults to 5.0.

        Returns:
            MCPServerHTTP: The created and configured MCPServerHTTP instance.
        """
        provider = MCPServerHTTP(
            endpoint_url=url,
            request_headers=headers,
            connection_timeout=connection_timeout,
            stream_read_timeout=stream_read_timeout,
            session_timeout=session_timeout
        )
        await self.add_mcp_server(provider)
        return provider

    async def add_mcp_server_stdio(
        self,
        executable_path: str,
        process_arguments: List[str],
        environment_vars: Optional[Dict[str, str]] = None,
        working_directory: Optional[str | Path] = None,
        session_timeout: float = 5.0
    ) -> MCPServerStdio:
        """
        Convenience method to create and add a stdio MCP server.

        Args:
            executable_path (str): Path to the executable MCP server.
            process_arguments (List[str]): Command line arguments to pass to the executable.
            environment_vars (Optional[Dict[str, str]], optional): Optional environment variables for the process.
            working_directory (Optional[str | Path], optional): Working directory for the process. Can be string or Path.
            session_timeout (float, optional): Session timeout in seconds. Defaults to 5.0.

        Returns:
            MCPServerStdio: The created and configured MCPServerStdio instance.
        """
        provider = MCPServerStdio(
            executable_path=executable_path,
            process_arguments=process_arguments,
            environment_vars=environment_vars,
            working_directory=working_directory,
            session_timeout=session_timeout
        )
        await self.add_mcp_server(provider)
        return provider

    async def __aenter__(self):
        """
        Async context manager entry point.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit point.
        """
        await self.cleanup()
