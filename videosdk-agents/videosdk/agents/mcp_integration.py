import asyncio
import logging
from typing import Any, Dict, List, Optional

from videosdk.agents.agent import Agent
from videosdk.agents.mcp_server import MCPServer
from videosdk.agents.utils import FunctionTool, ToolError

logger = logging.getLogger(__name__)


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
                # Initialize the server
                await server.initialize()
                
                # Get all tools from the server
                tools = await server.list_tools()
                logger.info(f"Added MCP server with {len(tools)} tools")
                
                # Log tool information for debugging
                for tool in tools:
                    try:
                        # Basic validation check and log information
                        from inspect import signature
                        sig = signature(tool)
                        logger.debug(f"Loaded MCP tool: {tool.__name__} with parameters: {sig}")
                    except Exception as e:
                        logger.warning(f"Unable to introspect tool {tool.__name__}: {e}")
                
                # Store the tools and server
                self.tools.extend(tools)
                self.servers.append(server)
            except Exception as e:
                # Clean up if initialization fails
                logger.error(f"Failed to initialize MCP server: {e}")
                await server.aclose()
                raise

    async def register_mcp_tools(self, agent: Agent) -> None:
        """
        Register MCP tools with an agent.

        Args:
            agent: The agent to register tools with
        
        Raises:
            ValueError: If the agent doesn't have a register_tools method
        """
        if not hasattr(agent, "register_tools"):
            raise ValueError("Agent does not have a register_tools method")

        # Register all tools with the agent
        agent._tools.extend(self.tools)
        agent.register_tools()
        logger.info(f"Registered {len(self.tools)} MCP tools with agent by extending agent._tools and calling agent.register_tools()")

    async def cleanup(self) -> None:
        """Close all MCP servers and clear the tools list."""
        for server in self.servers:
            await server.aclose()
        self.servers = []
        self.tools = []