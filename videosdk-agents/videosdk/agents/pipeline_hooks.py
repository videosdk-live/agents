from __future__ import annotations

from typing import Callable, Awaitable, AsyncIterator, Any, Literal, TYPE_CHECKING
import asyncio
import logging

if TYPE_CHECKING:
    import av

logger = logging.getLogger(__name__)


class PipelineHooks:
    """
    Manages pipeline hooks/middleware for intercepting and processing data at different stages.
    
    Supported hooks:
    - speech_in: Process raw incoming user audio (async iterator)
    - speech_out: Process outgoing agent audio after TTS (async iterator)
    - stt: Process user transcript after STT, before LLM
    - llm: Control LLM invocation (can bypass with direct response)
    - agent_response: Process agent response after LLM, before TTS
    - vision_frame: Process video frames when vision is enabled (async iterator)
    - user_turn_start: Called when user turn starts
    - user_turn_end: Called when user turn ends
    - agent_turn_start: Called when agent processing starts
    - agent_turn_end: Called when agent finishes speaking
    - content_generated: Called when LLM content is generated (receives dict with "text" key)
    """
    
    def __init__(self) -> None:
        # Vision hooks (async iterator support)
        self._vision_frame_hooks: list[Callable[[AsyncIterator[Any]], AsyncIterator[Any]]] = []
        
        # Stream processing hooks
        self._stt_stream_hook: Callable[[AsyncIterator[bytes]], AsyncIterator[Any]] | None = None
        self._tts_stream_hook: Callable[[AsyncIterator[str]], AsyncIterator[bytes]] | None = None
        
        # Gate hook (can yield to bypass LLM, or return None to continue to LLM)
        self._llm_hook: Callable[[str], AsyncIterator[str] | Awaitable[None]] | None = None
        
        self._agent_response_hooks: list[Callable[[str], Awaitable[AsyncIterator[str]] | str]] = []

        # Lifecycle hooks (side effects only)
        self._user_turn_start_hooks: list[Callable[[str], Awaitable[None]]] = []
        self._user_turn_end_hooks: list[Callable[[], Awaitable[None]]] = []
        self._agent_turn_start_hooks: list[Callable[[], Awaitable[None]]] = []
        self._agent_turn_end_hooks: list[Callable[[], Awaitable[None]]] = []
        self._content_generated_hooks: list[Callable[[dict], Awaitable[None]]] = []
    
    def on(
        self, 
        event: Literal["stt", "tts", "llm", "agent_response", "vision_frame", "user_turn_start", "user_turn_end", "agent_turn_start", "agent_turn_end", "content_generated"]
    ) -> Callable:
        """
        Decorator to register a hook for a specific event.
        
        Examples:
            @pipeline.on("stt")
            async def stt_stream_hook(audio_stream):
                '''Stream STT hook (audio -> events)'''
                async for event in run_stt(audio_stream):
                    yield event

            @pipeline.on("tts")
            async def tts_stream_hook(text_stream):
                '''Stream TTS hook (text -> audio)'''
                async for audio_frame in run_tts(text_stream):
                    yield audio_frame
            
            @pipeline.on("vision_frame")
            async def process_frames(frame_stream):
                '''Apply filters to video frames'''
                async for frame in frame_stream:
                    # Process av.VideoFrame
                    filtered_frame = apply_filter(frame)
                    yield filtered_frame
            
            @pipeline.on("user_turn_start")
            async def on_user_turn_start(transcript: str) -> None:
                '''Log when user starts speaking'''
                print(f"User said: {transcript}")
            
            @pipeline.on("user_turn_end")
            async def on_user_turn_end() -> None:
                '''Log when user turn ends'''
                print("User turn ended")
            
            @pipeline.on("agent_turn_start")
            async def on_agent_turn_start() -> None:
                '''Log when agent starts processing'''
                print("Agent processing started")
            
            @pipeline.on("agent_turn_end")
            async def on_agent_turn_end() -> None:
                '''Log when agent finishes speaking'''
                print("Agent finished speaking")
            
            @pipeline.on("content_generated")
            async def on_content_generated(data: dict) -> None:
                '''Handle generated content from LLM'''
                text = data.get("text", "")
                print(f"Generated: {text}")
            
            @pipeline.on("llm")
            async def custom_processing(transcript: str):
                '''Bypass LLM with streaming response or don't yield for normal flow'''
                if "hours" in transcript.lower():
                    # Yield to bypass LLM and stream response directly to TTS
                    for word in "We're open 24/7".split():
                        yield word + " "
                # If no yields, the generator will be empty and LLM will be used
        """
        def decorator(func: Callable) -> Callable:
            if event == "stt":
                if self._stt_stream_hook:
                    logger.warning("[on][PipelineHooks]STT stream hook already registered, overwriting")
                self._stt_stream_hook = func
                logger.info("[on][PipelineHooks]Registered STT stream hook")
            elif event == "tts":
                if self._tts_stream_hook:
                    logger.warning("[on][PipelineHooks]TTS stream hook already registered, overwriting")
                self._tts_stream_hook = func
                logger.info("[on][PipelineHooks]Registered TTS stream hook")
            elif event == "llm":
                if self._llm_hook is not None:
                    logger.warning("[on][PipelineHooks]llm hook already registered, overwriting")
                self._llm_hook = func
                logger.info("[on][PipelineHooks]Registered llm hook")
            elif event == "agent_response":
                self._agent_response_hooks.append(func)     
            elif event == "vision_frame":
                self._vision_frame_hooks.append(func)
            elif event == "user_turn_start":
                self._user_turn_start_hooks.append(func)
            elif event == "user_turn_end":
                self._user_turn_end_hooks.append(func)
            elif event == "agent_turn_start":
                self._agent_turn_start_hooks.append(func)
            elif event == "agent_turn_end":
                self._agent_turn_end_hooks.append(func)
            elif event == "content_generated":
                self._content_generated_hooks.append(func)
            else:
                raise ValueError(f"[on][PipelineHooks]Unknown event: {event}")
            
            logger.info(f"[on][PipelineHooks]Registered hook for event: {event}")
            return func
        
        return decorator
    
    async def process_vision_frame(self, frames: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """
        Process video frames through registered vision_frame hooks.
        
        Args:
            frames: Async iterator of av.VideoFrame objects
            
        Yields:
            Processed video frames
        """
        if not self._vision_frame_hooks:
            async for frame in frames:
                yield frame
            return
        
        # Process through hooks
        current_stream = frames
        for hook in self._vision_frame_hooks:
            try:
                current_stream = hook(current_stream)
            except Exception as e:
                logger.error(f"[process_vision_frame] Error in vision_frame hook: {e}", exc_info=True)
        
        async for frame in current_stream:
            yield frame
    
    async def trigger_user_turn_start(self, transcript: str) -> None:
        """
        Trigger all user_turn_start hooks.
        
        Args:
            transcript: User transcript
        """
        for hook in self._user_turn_start_hooks:
            try:
                await hook(transcript)
            except Exception as e:
                logger.error(f"[trigger_user_turn_start] Error in user_turn_start hook: {e}", exc_info=True)
    
    async def trigger_user_turn_end(self) -> None:
        """
        Trigger all user_turn_end hooks.
        """
        for hook in self._user_turn_end_hooks:
            try:
                await hook()
            except Exception as e:
                logger.error(f"[trigger_user_turn_end] Error in user_turn_end hook: {e}", exc_info=True)
    
    async def trigger_agent_turn_start(self) -> None:
        """
        Trigger all agent_turn_start hooks.
        """
        for hook in self._agent_turn_start_hooks:
            try:
                await hook()
            except Exception as e:
                logger.error(f"[trigger_agent_turn_start] Error in agent_turn_start hook: {e}", exc_info=True)
    
    async def trigger_agent_turn_end(self) -> None:
        """
        Trigger all agent_turn_end hooks.
        """
        for hook in self._agent_turn_end_hooks:
            try:
                await hook()
            except Exception as e:
                logger.error(f"[trigger_agent_turn_end] Error in agent_turn_end hook: {e}", exc_info=True)
    
    def has_vision_frame_hooks(self) -> bool:
        """Check if any vision_frame hooks are registered."""
        return len(self._vision_frame_hooks) > 0
    
    def has_llm_hook(self) -> bool:
        """Check if a llm hook is registered."""
        return self._llm_hook is not None
    
    def has_agent_response_hooks(self) -> bool:
        """Check if any agent response hooks are registered."""
        return len(self._agent_response_hooks) > 0

    def has_user_turn_start_hooks(self) -> bool:
        """Check if any user_turn_start hooks are registered."""
        return len(self._user_turn_start_hooks) > 0
    
    def has_user_turn_end_hooks(self) -> bool:
        """Check if any user_turn_end hooks are registered."""
        return len(self._user_turn_end_hooks) > 0
    
    def has_agent_turn_start_hooks(self) -> bool:
        """Check if any agent_turn_start hooks are registered."""
        return len(self._agent_turn_start_hooks) > 0
    
    def has_agent_turn_end_hooks(self) -> bool:
        """Check if any agent_turn_end hooks are registered."""
        return len(self._agent_turn_end_hooks) > 0
    
    def has_content_generated_hooks(self) -> bool:
        """Check if any content_generated hooks are registered."""
        return len(self._content_generated_hooks) > 0
    
    async def trigger_content_generated(self, data: dict) -> None:
        """
        Trigger all content_generated hooks.
        
        Args:
            data: Dictionary containing "text" key with generated content
        """
        for hook in self._content_generated_hooks:
            try:
                await hook(data)
            except Exception as e:
                logger.error(f"[trigger_content_generated] Error in content_generated hook: {e}", exc_info=True)
    
    async def process_llm_gate(self, transcript: str) -> AsyncIterator[str] | None:
        """
        Process turn through llm gate hook.
        
        Args:
            transcript: User transcript
            
        Returns:
            AsyncIterator[str] if hook wants to bypass LLM (yields response chunks)
            None if hook wants to continue to LLM normally (empty generator or no hook)
        """
        if not self._llm_hook:
            return None
        
        try:
            result = self._llm_hook(transcript)
            
            # Check if result is an async generator (has __anext__)
            if hasattr(result, '__anext__'):
                # Peek at the first value to see if generator is empty
                try:
                    first_value = await result.__anext__()
                    
                    # Generator has at least one value - create a new generator that yields first value + rest
                    async def full_generator():
                        yield first_value
                        async for chunk in result:
                            yield chunk
                    
                    return full_generator()
                    
                except StopAsyncIteration:
                    # Generator is empty - continue to LLM
                    return None
                    
            elif hasattr(result, '__await__'):
                # It's a coroutine - await it
                awaited_result = await result
                if awaited_result is None:
                    return None
                elif hasattr(awaited_result, '__anext__'):
                    return awaited_result
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            logger.error(f"[process_llm_gate] Error in llm hook: {e}", exc_info=True)
            return None
    
    def has_stt_stream_hook(self) -> bool:
        """Check if STT stream hook is registered."""
        return self._stt_stream_hook is not None

    def has_tts_stream_hook(self) -> bool:
        """Check if TTS stream hook is registered."""
        return self._tts_stream_hook is not None

    async def process_stt_stream(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[Any]:
        """
        Process audio through STT stream hook.
        
        Args:
            audio_stream: Async iterator of audio bytes
            
        Yields:
            Speech events
        """
        if not self._stt_stream_hook:
            return
        
        try:
            result = self._stt_stream_hook(audio_stream)
            async for event in result:
                yield event
        except Exception as e:
            logger.error(f"[process_stt_stream] Error in STT stream hook: {e}", exc_info=True)

    async def process_tts_stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """
        Process text through TTS stream hook.
        
        Args:
            text_stream: Async iterator of text
            
        Yields:
            Audio frames
        """
        if not self._tts_stream_hook:
            return
            
        try:
            result = self._tts_stream_hook(text_stream)
            async for frame in result:
                yield frame
        except Exception as e:
            logger.error(f"[process_tts_stream] Error in TTS stream hook: {e}", exc_info=True)

    def clear_all_hooks(self) -> None:
        """Clear all registered hooks."""
        self._vision_frame_hooks.clear()
        self._agent_response_hooks.clear()
        self._stt_stream_hook = None
        self._tts_stream_hook = None
        self._llm_hook = None
        self._user_turn_start_hooks.clear()
        self._user_turn_end_hooks.clear()
        self._agent_turn_start_hooks.clear()
        self._agent_turn_end_hooks.clear()
        self._content_generated_hooks.clear()
        logger.info("[clear_all_hooks] Cleared all pipeline hooks")
