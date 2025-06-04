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
from .mcp.mcp_server import MCPServer


AgentEventTypes = Literal[
    "instructions_updated",
    "tools_updated",
]

class Agent(EventEmitter[AgentEventTypes], ABC):
    """
    Abstract base class for creating custom agents.
    Inherits from EventEmitter to handle agent events and state updates.
    """
    def __init__(self, instructions: str, tools: List[FunctionTool] = [],agent_id: str = None, mcp_servers: List[MCPServer] = None):
        super().__init__()
        self._tools = tools
        self._llm = None
        self._stt = None
        self._tts = None
        self.chat_context = ChatContext.empty()
        self.instructions = instructions
        self._tools = list(tools)
        self._mcp_servers = mcp_servers if mcp_servers else []
        self._mcp_initialized = False
        self._register_class_tools()
        self.register_tools()
        self.a2a = A2AProtocol(self)
        self._agent_card = None 
        self.id = agent_id or str(uuid.uuid4())
        self.mcp_manager = MCPToolManager()

    def _register_class_tools(self) -> None:
        """Register all function tools defined in the class"""
        for name, attr in inspect.getmembers(self):
            if is_function_tool(attr):
                self._tools.append(attr)

    @property
    def instructions(self) -> str:
        return self._instructions

    @instructions.setter
    def instructions(self, value: str) -> None:
        self._instructions = value
        self.chat_context.add_message(
            role=ChatRole.SYSTEM,
            content=value
        )
        # global_event_emitter.emit("instructions_updated", {"instructions": value})

    @property
    def tools(self) -> List[FunctionTool]:
        return self._tools

    def register_tools(self) -> None:
        """Register external function tools for the agent"""
        for tool in self._tools:
            if not is_function_tool(tool):
                raise ValueError(f"Tool {tool.__name__ if hasattr(tool, '__name__') else tool} is not a valid FunctionTool")
        # global_event_emitter.emit("tools_updated", {"tools": self._tools})
    
    async def initialize_mcp(self) -> None:
        """Initialize the agent, including any MCP server if provided."""
        if self._mcp_servers and not self._mcp_initialized:
            for server in self._mcp_servers:
                await self.add_server(server)
            self._mcp_initialized = True
    
    async def add_server(self, mcp_server: MCPServer) -> None:
        """Initialize the MCP server and register the tools"""
        await self.mcp_manager.add_mcp_server(mcp_server)
        self._tools.extend(self.mcp_manager.tools)
        # self.register_tools()
    
    @abstractmethod
    async def on_enter(self) -> None:
        """Called when session starts"""
        pass

    async def register_a2a(self, card: AgentCard) -> None:
        """Register the agent for A2A communication"""
        self._agent_card = card
        await self.a2a.register(card)

    async def unregister_a2a(self) -> None:
        """Unregister the agent from A2A communication"""
        await self.a2a.unregister()
        self._agent_card = None

    @abstractmethod
    async def on_exit(self) -> None:
        """Called when session ends"""
        pass

    async def process_stt_output(self, text: str) -> str:
        """
        Process STT output before it goes to LLM.
        Override this method to add custom processing.
        
        Args:
            text: The text from STT
            
        Returns:
            Processed text
        """
        return text

    async def process_llm_output(self, text: str) -> str:
        """
        Process LLM output before it goes to TTS.
        Override this method to add custom processing.
        
        Args:
            text: The text from LLM
            
        Returns:
            Processed text
        """
        return text
