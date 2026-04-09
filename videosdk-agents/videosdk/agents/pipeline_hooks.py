from __future__ import annotations

from typing import Callable, Awaitable, AsyncIterator, Any, Literal, TYPE_CHECKING
import asyncio
import inspect
import logging

if TYPE_CHECKING:
    import av

logger = logging.getLogger(__name__)


class PipelineHooks:
    """
    Manages pipeline hooks/middleware for intercepting and processing data at different stages.

    Supported hooks:
    - stt: STT processing (async iterator: audio -> events)
    - tts: TTS processing (async iterator: text -> audio)
    - llm: Two patterns:
           Streaming (async generator): receives text chunks from LLM, yields modified chunks to TTS.
             Zero added latency. Use for real-time text modification.
           Observation (async function): receives dict with full text after generation.
             Use for logging, analytics, memory storage. Cannot modify TTS output.
    - vision_frame: Process video frames when vision is enabled (async iterator)
    - user_turn_start: Called when user turn starts
    - user_turn_end: Called when user turn ends
    - agent_turn_start: Called when agent processing starts
    - agent_turn_end: Called when agent finishes speaking
    """

    def __init__(self) -> None:
        # Vision hooks (async iterator support)
        self._vision_frame_hooks: list[Callable[[AsyncIterator[Any]], AsyncIterator[Any]]] = []

        # Stream processing hooks
        self._stt_stream_hook: Callable[[AsyncIterator[bytes]], AsyncIterator[Any]] | None = None
        self._tts_stream_hook: Callable[[AsyncIterator[str]], AsyncIterator[bytes]] | None = None
        self._llm_stream_hook: Callable[[AsyncIterator[str]], AsyncIterator[str]] | None = None

        # LLM observation hooks - fire after full response is collected
        self._llm_hooks: list[Callable[[dict], Awaitable[str | AsyncIterator[str] | None]]] = []
        self._user_turn_start_hooks: list[Callable[[str], Awaitable[None]]] = []
        self._user_turn_end_hooks: list[Callable[[], Awaitable[None]]] = []
        self._agent_turn_start_hooks: list[Callable[[], Awaitable[None]]] = []
        self._agent_turn_end_hooks: list[Callable[[], Awaitable[None]]] = []
    
    def on(
        self,
        event: Literal["stt", "tts", "llm", "vision_frame", "user_turn_start", "user_turn_end", "agent_turn_start", "agent_turn_end"]
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
                    filtered_frame = apply_filter(frame)
                    yield filtered_frame

            @pipeline.on("user_turn_start")
            async def on_user_turn_start(transcript: str) -> None:
                '''Log when user starts speaking'''
                print(f"User said: {transcript}")

            @pipeline.on("llm")
            async def strip_markdown(text_stream):
                '''Streaming hook: modify LLM text before TTS (zero latency)'''
                async for chunk in text_stream:
                    yield chunk.replace("**", "")

            @pipeline.on("llm")
            async def on_llm(data: dict):
                '''Observation hook: fires after full response is collected'''
                text = data.get("text", "")
                print(f"Generated: {text}")
        """
        def decorator(func: Callable) -> Callable:
            if event == "stt":
                if self._stt_stream_hook:
                    logger.warning("STT stream hook already registered, overwriting")
                self._stt_stream_hook = func
                logger.info("Registered STT stream hook")
            elif event == "tts":
                if self._tts_stream_hook:
                    logger.warning("TTS stream hook already registered, overwriting")
                self._tts_stream_hook = func
                logger.info("Registered TTS stream hook")
            elif event == "llm":
                if inspect.isasyncgenfunction(func):
                    if self._llm_stream_hook:
                        logger.warning("LLM stream hook already registered, overwriting")
                    self._llm_stream_hook = func
                    logger.info("Registered LLM stream hook")
                else:
                    self._llm_hooks.append(func)
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
    
    def has_vision_frame_hooks(self) -> bool:
        """Check if any vision_frame hooks are registered."""
        return len(self._vision_frame_hooks) > 0
    
    def has_llm_hooks(self) -> bool:
        """Check if any llm observation hooks are registered."""
        return len(self._llm_hooks) > 0

    def has_llm_stream_hook(self) -> bool:
        """Check if LLM stream hook is registered."""
        return self._llm_stream_hook is not None

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

    async def trigger_llm(self, data: dict) -> str | None:
        """
        Trigger all llm hooks. Hooks are chained — each receives the (possibly modified) text.

        If a hook yields/returns a string, it replaces the response text for subsequent
        hooks and for TTS. If it returns None, the text is kept as-is.

        Args:
            data: Dictionary containing "text" key with generated content

        Returns:
            Modified text if any hook modified it, None otherwise.
        """
        current_text = data.get("text", "")
        modified = False

        for hook in self._llm_hooks:
            try:
                result = hook({"text": current_text})

                if hasattr(result, '__anext__'):
                    # Async generator — collect yielded chunks
                    chunks = []
                    async for chunk in result:
                        if chunk is not None:
                            chunks.append(chunk)
                    if chunks:
                        current_text = "".join(chunks)
                        modified = True

                elif hasattr(result, '__await__'):
                    # Coroutine — await it
                    awaited = await result
                    if isinstance(awaited, str):
                        current_text = awaited
                        modified = True

            except Exception as e:
                logger.error(f"Error in llm hook: {e}", exc_info=True)

        return current_text if modified else None
    
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
            logger.error(f"Error in STT stream hook: {e}", exc_info=True)

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
            logger.error(f"Error in TTS stream hook: {e}", exc_info=True)

    async def process_llm_stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """
        Process LLM text chunks through the registered stream hook.
        Passthrough when no hook is registered.

        Args:
            text_stream: Async iterator of text chunks from the LLM

        Yields:
            Modified text chunks for TTS consumption
        """
        if not self._llm_stream_hook:
            async for chunk in text_stream:
                yield chunk
            return

        try:
            result = self._llm_stream_hook(text_stream)
            async for chunk in result:
                yield chunk
        except Exception as e:
            logger.error(f"Error in LLM stream hook: {e}", exc_info=True)

    def clear_all_hooks(self) -> None:
        """Clear all registered hooks."""
        self._vision_frame_hooks.clear()
        self._stt_stream_hook = None
        self._tts_stream_hook = None
        self._llm_stream_hook = None
        self._llm_hooks.clear()
        self._user_turn_start_hooks.clear()
        self._user_turn_end_hooks.clear()
        self._agent_turn_start_hooks.clear()
        self._agent_turn_end_hooks.clear()
        logger.info("Cleared all pipeline hooks")
