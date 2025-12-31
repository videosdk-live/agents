from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional
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
from .background_audio import BackgroundAudioHandlerConfig
import logging
import os
import av
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .knowledge_base.base import KnowledgeBase
    
logger = logging.getLogger(__name__)

class ToolList(list):
    """
    Custom list class that supports addition and subtraction for Tool management.
    """
    def __add__(self, other):
        if isinstance(other, list):
            return ToolList(super().__add__(other))
        return ToolList(super().__add__([other]))

    def __sub__(self, other):
        if not isinstance(other, list):
            other = [other]
        return ToolList([item for item in self if item not in other])

if 'AgentSession' not in globals():
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from .agent_session import AgentSession

class Agent(EventEmitter[Literal["agent_started"]], ABC):
    """
    Abstract base class for creating custom agents.
    Inherits from EventEmitter to handle agent events and state updates.
    """
    def __init__(self, instructions: str, tools: List[FunctionTool] = None, agent_id: str = None, mcp_servers: List[MCPServiceProvider] = None, inherit_context: bool = False, knowledge_base: KnowledgeBase | None = None):
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
        self.session: "AgentSession"
        self._thinking_background_config: Optional[BackgroundAudioHandlerConfig] = None
        self.knowledge_base = knowledge_base 
        self.inherit_context = inherit_context

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
    def tools(self) -> ToolList[FunctionTool]:
        """Get the tools for the agent"""
        return ToolList(self._tools)
    
    def register_tools(self) -> None:
        """Internal Method: Register external function tools for the agent"""
        for tool in self._tools:
            if not is_function_tool(tool):
                raise ValueError(f"Tool {tool.__name__ if hasattr(tool, '__name__') else tool} is not a valid FunctionTool")

    def update_tools(self, tools: List[FunctionTool]) -> None:
        """Update the tools for the agent"""
        self._tools = tools
        self._register_class_tools()
        self.register_tools()
    
    async def hangup(self) -> None:
        """Hang up the agent"""
        await self.session.hangup("manual_hangup")
    
    def set_thinking_audio(self, file: str = None, volume: float = 0.3):
        """Set the thinking background for the agent"""
        if file is None:
            file = os.path.join(os.path.dirname(__file__), 'resources', 'agent_keyboard.wav')
        self._thinking_background_config = BackgroundAudioHandlerConfig(file_path=file,volume=volume,looping=True,enabled=True)

    async def play_background_audio(self, file: str = None, volume: float = 1.0, looping: bool = False, override_thinking: bool = True) -> None:
        """Play background audio on demand"""
        if file is None:
            file = os.path.join(os.path.dirname(__file__), 'resources', 'classical.wav')
        
        config = BackgroundAudioHandlerConfig(file_path=file,volume=volume,looping=looping,enabled=True,mode='mixing') 
        
        await self.session.play_background_audio(config, override_thinking)

    async def stop_background_audio(self) -> None:
        """Stop background audio on demand"""
        await self.session.stop_background_audio()
    
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

    def capture_frames(self, num_of_frames: int = 1) -> list[av.VideoFrame]:
        """Capture the latest video frames from the pipeline (max 5)."""
        if num_of_frames > 5:
            raise ValueError("num_of_frames cannot exceed 5")

        pipeline = getattr(getattr(self, 'session', None), 'pipeline', None)
        if not pipeline:
            logger.warning("Pipeline not available")
            return []

        return pipeline.get_latest_frames(num_of_frames)