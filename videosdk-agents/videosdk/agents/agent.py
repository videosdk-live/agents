from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Literal, TYPE_CHECKING
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
from .telemetry.videosdk_telemetry import VideoSDKTelemetry
from opentelemetry.trace import StatusCode


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
        self.telemetry: VideoSDKTelemetry | None = None
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
        if self.telemetry:
            self.telemetry.trace_auto_complete("Agent Set Instructions", {
                "instructions.length": len(value),
                "agent.type": type(self).__name__
            })
        
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
        
        if self.telemetry:
            self.telemetry.trace_auto_complete("Agent Register Tools", {
                "tools.count": len(self._tools),
                "tools.names": [tool.__name__ for tool in self._tools],
                "agent.type": type(self).__name__
            })
        # global_event_emitter.emit("tools_updated", {"tools": self._tools})
    
    async def initialize_mcp(self) -> None:
        """Initialize the agent, including any MCP server if provided."""
        mcp_span = None
        if self.telemetry:
            mcp_span = self.telemetry.trace("Agent MCP Setup", {
                "agent.type": type(self).__name__,
                "mcp.server_count": len(self._mcp_servers),
                "mcp.already_initialized": self._mcp_initialized
            })
        
        try:
            if self._mcp_servers and not self._mcp_initialized:
                for server in self._mcp_servers:
                    await self.add_server(server)
                self._mcp_initialized = True
                if self.telemetry:
                    self.telemetry.add_span_attribute(mcp_span, "mcp.initialization_completed", True)
            else:
                if self.telemetry:
                    self.telemetry.add_span_attribute(mcp_span, "mcp.initialization_skipped", True)
                    
            if self.telemetry:
                self.telemetry.complete_span(mcp_span, StatusCode.OK, "MCP initialization completed")
        except Exception as e:
            if self.telemetry:
                self.telemetry.complete_span(mcp_span, StatusCode.ERROR, f"MCP initialization failed: {str(e)}")
            raise
    
    async def add_server(self, mcp_server: MCPServer) -> None:
        """Initialize the MCP server and register the tools"""
        if self.telemetry:
            with self.telemetry.span_context("Agent Add MCP Server", {
                "agent.type": type(self).__name__,
                "server.name": mcp_server.name if hasattr(mcp_server, 'name') else str(mcp_server)
            }):
                await self.mcp_manager.add_mcp_server(mcp_server)
                self._tools.extend(self.mcp_manager.tools)
                
                if self.telemetry:
                    self.telemetry.add_span_attribute(None, "tools.added", [tool.__name__ for tool in self.mcp_manager.tools])
                    self.telemetry.add_span_attribute(None, "tools.total", len(self._tools))
        else:
            await self.mcp_manager.add_mcp_server(mcp_server)
            self._tools.extend(self.mcp_manager.tools)
        # self.register_tools()
    
    @abstractmethod
    async def on_enter(self) -> None:
        """Called when session starts"""
        pass

    async def register_a2a(self, card: AgentCard) -> None:
        """Register the agent for A2A communication"""
        if self.telemetry:
            with self.telemetry.span_context("Agent A2A Registration", {
                "agent.type": type(self).__name__,
                "agent.id": self.id,
                "card.info": str(card.to_dict())
            }):
                self._agent_card = card
                await self.a2a.register(card)
        else:
            self._agent_card = card
            await self.a2a.register(card)

    async def unregister_a2a(self) -> None:
        """Unregister the agent from A2A communication"""
        if self.telemetry:
            self.telemetry.trace_auto_complete("Agent A2A Unregistration", {
                "agent.type": type(self).__name__,
                "agent.id": self.id
            })
        
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
        if self.telemetry:
            with self.telemetry.span_context("Agent STT Processing", {
                "agent.type": type(self).__name__,
                "input.text_length": len(text),
                "agent.id": self.id
            }) as span:
                processed_text = text
                if span and self.telemetry:
                    self.telemetry.add_span_attribute(span, "output.text_length", len(processed_text))
                return processed_text
        else:
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
        if self.telemetry:
            with self.telemetry.span_context("Agent LLM Processing", {
                "agent.type": type(self).__name__,
                "input.text_length": len(text),
                "agent.id": self.id
            }) as span:
                processed_text = text
                if span and self.telemetry:
                    self.telemetry.add_span_attribute(span, "output.text_length", len(processed_text))
                return processed_text
        else:
            return text
