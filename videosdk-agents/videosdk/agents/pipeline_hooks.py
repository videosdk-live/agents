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
    """
    
    def __init__(self) -> None:
        # Audio stream hooks (async iterator support)
        self._speech_in_hooks: list[Callable[[AsyncIterator[bytes]], AsyncIterator[bytes]]] = []
        self._speech_out_hooks: list[Callable[[AsyncIterator[bytes]], AsyncIterator[bytes]]] = []
        
        # Vision hooks (async iterator support)
        self._vision_frame_hooks: list[Callable[[AsyncIterator[Any]], AsyncIterator[Any]]] = []
        
        # Text transformation hooks
        self._stt_hooks: list[Callable[[str], Awaitable[str]]] = []
        self._agent_response_hooks: list[Callable[[str], Awaitable[AsyncIterator[str]] | str]] = []
        
        # Gate hook (can yield to bypass LLM, or return None to continue to LLM)
        self._llm_hook: Callable[[str], AsyncIterator[str] | Awaitable[None]] | None = None
        
        # Lifecycle hooks (side effects only)
        self._user_turn_start_hooks: list[Callable[[str], Awaitable[None]]] = []
        self._user_turn_end_hooks: list[Callable[[], Awaitable[None]]] = []
        self._agent_turn_start_hooks: list[Callable[[], Awaitable[None]]] = []
        self._agent_turn_end_hooks: list[Callable[[], Awaitable[None]]] = []
    
    def on(
        self, 
        event: Literal["speech_in", "speech_out", "stt", "llm", "agent_response", "vision_frame", "user_turn_start", "user_turn_end", "agent_turn_start", "agent_turn_end"]
    ) -> Callable:
        """
        Decorator to register a hook for a specific event.
        
        Examples:
            @pipeline.on("speech_in")
            async def process_audio(audio_stream):
                '''Apply noise reduction to incoming audio'''
                async for audio_chunk in audio_stream:
                    # Process audio_chunk (bytes)
                    processed = apply_noise_reduction(audio_chunk)
                    yield processed
            
            @pipeline.on("stt")
            async def clean_transcript(transcript: str) -> str:
                '''Remove filler words from transcript'''
                return transcript.replace("um", "").replace("uh", "")
            
            @pipeline.on("agent_response")
            async def process_response(response: str):
                '''Stream modified response to TTS'''
                for word in response.split():
                    yield word.replace("API", "A P I") + " "
            
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
            if event == "speech_in":
                self._speech_in_hooks.append(func)
            elif event == "speech_out":
                self._speech_out_hooks.append(func)
            elif event == "stt":
                self._stt_hooks.append(func)
            elif event == "llm":
                if self._llm_hook is not None:
                    logger.warning("llm hook already registered, overwriting")
                self._llm_hook = func
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
            else:
                raise ValueError(f"Unknown event: {event}")
            
            logger.info(f"Registered hook for event: {event}")
            return func
        
        return decorator
    
    async def process_speech_in(self, audio: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """
        Process incoming audio through registered speech_in hooks.
        
        Args:
            audio: Async iterator of audio bytes
            
        Yields:
            Processed audio bytes
        """
        if not self._speech_in_hooks:
            async for chunk in audio:
                yield chunk
            return
        
        # Process through hooks
        current_stream = audio
        for hook in self._speech_in_hooks:
            try:
                current_stream = hook(current_stream)
            except Exception as e:
                logger.error(f"Error in speech_in hook: {e}", exc_info=True)
        
        async for chunk in current_stream:
            yield chunk
    
    async def process_speech_out(self, audio: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """
        Process outgoing audio through registered speech_out hooks.
        
        Args:
            audio: Async iterator of audio bytes
            
        Yields:
            Processed audio bytes
        """
        if not self._speech_out_hooks:
            async for chunk in audio:
                yield chunk
            return
        
        # Process through hooks
        current_stream = audio
        for hook in self._speech_out_hooks:
            try:
                current_stream = hook(current_stream)
            except Exception as e:
                logger.error(f"Error in speech_out hook: {e}", exc_info=True)
        
        async for chunk in current_stream:
            yield chunk
    
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
                logger.error(f"Error in vision_frame hook: {e}", exc_info=True)
        
        async for frame in current_stream:
            yield frame
    
    async def process_stt(self, transcript: str) -> str:
        """
        Process user transcript through all registered stt hooks.
        
        Args:
            transcript: Original transcript from STT
            
        Returns:
            Processed transcript to send to LLM
        """
        processed = transcript
        
        for hook in self._stt_hooks:
            try:
                processed = await hook(processed)
            except Exception as e:
                logger.error(f"Error in stt hook: {e}", exc_info=True)
        
        return processed
    
    async def process_agent_response(self, response: str | AsyncIterator[str]) -> AsyncIterator[str]:
        """
        Process agent response through all registered hooks.
        
        Args:
            response: Original response from LLM (string or async iterator)
            
        Yields:
            Processed response chunks to send to TTS
        """
        # If no hooks, pass through
        if not self._agent_response_hooks:
            if isinstance(response, str):
                yield response
            else:
                async for chunk in response:
                    yield chunk
            return
        
        # Collect full response if it's a stream
        if isinstance(response, str):
            full_response = response
        else:
            parts = []
            async for chunk in response:
                parts.append(chunk)
            full_response = "".join(parts)
        
        # Process through hooks
        for hook in self._agent_response_hooks:
            try:
                result = hook(full_response)
                
                # Check if result is an async generator or coroutine
                if hasattr(result, '__anext__'):
                    # It's an async generator - stream it
                    async for chunk in result:
                        yield chunk
                elif hasattr(result, '__await__'):
                    # It's a coroutine - await it
                    processed = await result
                    yield processed
                else:
                    # It's a regular value
                    yield result
                return  # Only process through first hook for now
                    
            except Exception as e:
                logger.error(f"Error in agent_response hook: {e}", exc_info=True)
                # On error, yield original response
                yield full_response
                return
    
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
                logger.error(f"Error in user_turn_start hook: {e}", exc_info=True)
    
    async def trigger_user_turn_end(self) -> None:
        """
        Trigger all user_turn_end hooks.
        """
        for hook in self._user_turn_end_hooks:
            try:
                await hook()
            except Exception as e:
                logger.error(f"Error in user_turn_end hook: {e}", exc_info=True)
    
    async def trigger_agent_turn_start(self) -> None:
        """
        Trigger all agent_turn_start hooks.
        """
        for hook in self._agent_turn_start_hooks:
            try:
                await hook()
            except Exception as e:
                logger.error(f"Error in agent_turn_start hook: {e}", exc_info=True)
    
    async def trigger_agent_turn_end(self) -> None:
        """
        Trigger all agent_turn_end hooks.
        """
        for hook in self._agent_turn_end_hooks:
            try:
                await hook()
            except Exception as e:
                logger.error(f"Error in agent_turn_end hook: {e}", exc_info=True)
    
    def has_speech_in_hooks(self) -> bool:
        """Check if any speech_in hooks are registered."""
        return len(self._speech_in_hooks) > 0
    
    def has_speech_out_hooks(self) -> bool:
        """Check if any speech_out hooks are registered."""
        return len(self._speech_out_hooks) > 0
    
    def has_vision_frame_hooks(self) -> bool:
        """Check if any vision_frame hooks are registered."""
        return len(self._vision_frame_hooks) > 0
    
    def has_stt_hooks(self) -> bool:
        """Check if any stt hooks are registered."""
        return len(self._stt_hooks) > 0
    
    def has_agent_response_hooks(self) -> bool:
        """Check if any agent response hooks are registered."""
        return len(self._agent_response_hooks) > 0
    
    def has_llm_hook(self) -> bool:
        """Check if a llm hook is registered."""
        return self._llm_hook is not None
    
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
            logger.error(f"Error in llm hook: {e}", exc_info=True)
            return None
    
    def clear_all_hooks(self) -> None:
        """Clear all registered hooks."""
        self._speech_in_hooks.clear()
        self._speech_out_hooks.clear()
        self._vision_frame_hooks.clear()
        self._stt_hooks.clear()
        self._agent_response_hooks.clear()
        self._llm_hook = None
        self._user_turn_start_hooks.clear()
        self._user_turn_end_hooks.clear()
        self._agent_turn_start_hooks.clear()
        self._agent_turn_end_hooks.clear()
        logger.info("Cleared all pipeline hooks")
