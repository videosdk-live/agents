from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Literal, Optional
import inspect
from .event_bus import global_event_emitter, EventTypes
from .event_emitter import EventEmitter
from .llm.chat_context import ChatContext
from .utils import FunctionTool, is_function_tool
from .llm.llm import LLMResponse
from .llm.chat_context import ChatContext, ChatRole
from .stt.stt import STTResponse

AgentEventTypes = Literal[
    "instructions_updated",
    "tools_updated",
]

class Agent(EventEmitter[AgentEventTypes], ABC):
    """
    Abstract base class for creating custom agents.
    Inherits from EventEmitter to handle agent events and state updates.
    """
    def __init__(self, instructions: str, tools: List[FunctionTool] = []):
        super().__init__()
        self._tools = tools
        self._llm = None
        self._stt = None
        self._tts = None
        self.chat_context = ChatContext.empty()
        self.instructions = instructions
        self._register_class_tools()
        self.register_tools()

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
        global_event_emitter.emit("instructions_updated", {"instructions": value})

    @property
    def tools(self) -> List[FunctionTool]:
        return self._tools

    def register_tools(self) -> None:
        """Register external function tools for the agent"""
        for tool in self._tools:
            if not is_function_tool(tool):
                raise ValueError(f"Tool {tool.__name__ if hasattr(tool, '__name__') else tool} is not a valid FunctionTool")
        global_event_emitter.emit("tools_updated", {"tools": self._tools})
    @abstractmethod
    async def on_enter(self) -> None:
        """Called when session starts"""
        pass
    
    @abstractmethod
    async def on_exit(self) -> None:
        """Called when session ends"""
        pass
    
    async def process_with_llm(
        self,
        context: ChatContext,
        tools: Optional[List[FunctionTool]] = None,
    ) -> AsyncIterator[LLMResponse]:
        """
        Process a chat context with the LLM
        
        Args:
            context: The chat context to process
            tools: Optional list of function tools to use
            
        Returns:
            AsyncIterator yielding LLMResponse objects
        """
        if self._llm is None:
            raise RuntimeError("No LLM instance available. Make sure the agent is properly initialized with a session.")
        tools = tools or self._tools
        
        async for response in self._llm.chat(
            context=context,
            tools=tools,
        ):
            yield response    

    async def process_with_stt(
        self,
        audio_frames: AsyncIterator[bytes],
        language: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterator[STTResponse]:
        """
        Process audio frames with the STT engine
        
        Args:
            audio_frames: Iterator of audio frames to process
            language: Optional language code for recognition
            **kwargs: Additional provider-specific arguments
            
        Returns:
            AsyncIterator yielding STTResponse objects
        """
        if self._stt is None:
            raise RuntimeError("No STT instance available. Make sure the agent is properly initialized with a session.")
        
        async for response in self._stt.process_audio(
            audio_frames=audio_frames,
            language=language,
            **kwargs
        ):
            yield response

    async def process_with_tts(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Process text with the TTS engine
        
        Args:
            text: Text to convert to speech (either string or async iterator of strings)
            voice_id: Optional voice identifier
            **kwargs: Additional provider-specific arguments
        """
        if self._tts is None:
            raise RuntimeError("No TTS instance available. Make sure the agent is properly initialized with a session.")
        
        await self._tts.synthesize(
            text=text,
            voice_id=voice_id,
            **kwargs
        )