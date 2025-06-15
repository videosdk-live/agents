import asyncio
from typing import Any, Dict, List, Optional

# from videosdk.agents.agent import Agent 
from videosdk.agents.mcp.mcp_server import MCPServer
from videosdk.agents.utils import FunctionTool, ToolError


class MCPToolManager:
    """
    Manages MCP servers and their tools for agents.
    
    This class provides a simple way to register MCP servers and their tools
    with agents. It handles server initialization, tool registration, and cleanup.
    """

    def __init__(self) -> None:
        """Initialize the MCPToolManager."""
        self.servers: List[MCPServer] = []
        self.tools: List[FunctionTool] = []

    async def add_mcp_server(self, server: MCPServer) -> None:
        """
        Add a new MCP server and initialize it.
        
        Args:
            server: The MCP server to add
        
        Raises:
            Exception: If server initialization fails
        """
        if not server.initialized:
            try:
                await server.initialize()
                
                tools = await server.list_tools()
                
                self.tools.extend(tools)
                if server not in self.servers:
                    self.servers.append(server)
            except Exception as e:
                await server.aclose()
                raise


    async def cleanup(self) -> None:
        """Close all MCP servers and clear the tools list."""
        for server in self.servers:
            await server.aclose()
        self.servers = []
        self.tools = []