from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Literal
import inspect
from .event_emitter import EventEmitter
from .llm.chat_context import ChatContext
from .utils import FunctionTool, is_function_tool
from .a2a.protocol import A2AProtocol
from .a2a.card import AgentCard
import uuid
from .llm.chat_context import ChatContext, ChatRole
from .mcp.mcp_manager import MCPToolManager
from .mcp.mcp_server import MCPServiceProvider
import logging
logger = logging.getLogger(__name__)

class Agent(EventEmitter[Literal["agent_started"]], ABC):
    """
    Abstract base class for creating custom agents.
    Inherits from EventEmitter to handle agent events and state updates.
    """
    def __init__(self, instructions: str, tools: List[FunctionTool] = None, agent_id: str = None, mcp_servers: List[MCPServiceProvider] = None):
        super().__init__()
        self._tools = tools
        self._llm = None
        self._stt = None
        self._tts = None
        self.chat_context = ChatContext.empty()
        self.instructions = instructions
        self._tools = tools if tools else []
        self._mcp_servers = mcp_servers if mcp_servers else []
        self._mcp_initialized = False
        self._register_class_tools()
        self.register_tools()
        self.a2a = A2AProtocol(self)
        self._agent_card = None 
        self.id = agent_id or str(uuid.uuid4())
        self.mcp_manager = MCPToolManager()

    def _register_class_tools(self) -> None:
        """Internal Method: Register all function tools defined in the class"""
        for name, attr in inspect.getmembers(self):
            if is_function_tool(attr):
                self._tools.append(attr)

    @property
    def instructions(self) -> str:
        """Get the instructions for the agent"""
        return self._instructions

    @instructions.setter
    def instructions(self, value: str) -> None:
        """Set the instructions for the agent"""
        self._instructions = value
        self.chat_context.add_message(
            role=ChatRole.SYSTEM,
            content=value
        )

    @property
    def tools(self) -> List[FunctionTool]:
        """Get the tools for the agent"""
        return self._tools
    
    def register_tools(self) -> None:
        """Internal Method: Register external function tools for the agent"""
        for tool in self._tools:
            if not is_function_tool(tool):
                raise ValueError(f"Tool {tool.__name__ if hasattr(tool, '__name__') else tool} is not a valid FunctionTool")
    
    def update_tools(self, tools: List[FunctionTool]) -> None:
        """Update the tools for the agent"""
        self._tools.extend(tools)
        self._register_class_tools()
        self.register_tools()
    
    async def hangup(self) -> None:
        """Hang up the agent"""
        await self.session.close()
    
    async def initialize_mcp(self) -> None:
        """Internal Method: Initialize the agent, including any MCP server if provided."""
        if self._mcp_servers and not self._mcp_initialized:
            for server in self._mcp_servers:
                await self.add_server(server)
            self._mcp_initialized = True
    
    async def add_server(self, mcp_server: MCPServiceProvider) -> None:
        """Internal Method: Initialize the MCP server and register the tools"""
        await self.mcp_manager.add_mcp_server(mcp_server)
        self._tools.extend(self.mcp_manager.tools)
    
    @abstractmethod
    async def on_enter(self) -> None:
        """Called when session starts, to be implemented in your custom agent implementation."""
        pass

    def on_speech_in(self, data: dict) -> None:
        """Called when user speech is detected, to be implemented in your custom agent implementation."""
        pass

    def on_speech_out(self, data: dict) -> None:
        """Called when agent speech is generated, to be implemented in your custom agent implementation."""
        pass

    async def register_a2a(self, card: AgentCard) -> None:
        """ Register the agent for A2A communication"""
        self._agent_card = card
        await self.a2a.register(card)

    async def unregister_a2a(self) -> None:
        """Unregister the agent from A2A communication"""
        await self.a2a.unregister()
        self._agent_card = None

    async def cleanup(self) -> None:
        """Internal Method: Cleanup agent resources"""
        logger.info("Cleaning up agent resources")        
        if self.mcp_manager:
            try:
                await self.mcp_manager.cleanup()
                logger.info("MCP manager cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up MCP manager: {e}")
            self.mcp_manager = None

        self._tools = []
        self._mcp_servers = []
        self.chat_context = None
        self._agent_card = None        
        if hasattr(self, 'session'):
            self.session = None        
        logger.info("Agent cleanup completed")
    
    @abstractmethod
    async def on_exit(self) -> None:
        """Called when session ends, to be implemented in your custom agent implementation."""
        pass
