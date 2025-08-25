from __future__ import annotations

from abc import ABC
from typing import Awaitable, Callable, Literal, AsyncIterator, Any
import time
import json
import asyncio
from .event_emitter import EventEmitter
from .stt.stt import STT, STTResponse
from .llm.llm import LLM
from .llm.chat_context import ChatRole
from .utils import is_function_tool, get_tool_info, graceful_cancel
from .tts.tts import TTS
from .stt.stt import SpeechEventType
from .agent import Agent
from .event_bus import global_event_emitter
from .vad import VAD, VADResponse, VADEventType
from .eou import EOU
from .metrics import cascading_metrics_collector
from .denoise import Denoise
import logging

logger = logging.getLogger(__name__)

class ConversationFlow(EventEmitter[Literal["transcription"]], ABC):
    """
    Manages the conversation flow by listening to transcription events.
    """

    def __init__(self, agent: Agent, stt: STT | None = None, llm: LLM | None = None, tts: TTS | None = None, vad: VAD | None = None, turn_detector: EOU | None = None, denoise: Denoise | None = None) -> None:
        """Initialize conversation flow with event emitter capabilities"""
        super().__init__()
        self.transcription_callback: Callable[[
            STTResponse], Awaitable[None]] | None = None
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.turn_detector = turn_detector
        self.agent = agent
        self.denoise = denoise
        self._stt_started = False

        self.stt_lock = asyncio.Lock()
        self.llm_lock = asyncio.Lock()
        self.tts_lock = asyncio.Lock()

        self.user_speech_callback: Callable[[], None] | None = None
        if self.stt:
            self.stt.on_stt_transcript(self.on_stt_transcript)
        if self.vad:
            self.vad.on_vad_event(self.on_vad_event)

        self._current_tts_task: asyncio.Task | None = None
        self._current_llm_task: asyncio.Task | None = None
        self._partial_response = ""
        self._is_interrupted = False

    async def start(self) -> None:
        global_event_emitter.on("speech_started", self.on_speech_started_stt)
        global_event_emitter.on("speech_stopped", self.on_speech_stopped_stt)

        if self.agent and self.agent.instructions:
            cascading_metrics_collector.set_system_instructions(
                self.agent.instructions)

    def on_transcription(self, callback: Callable[[str], None]) -> None:
        """
        Set the callback for transcription events.

        Args:
            callback: Function to call when transcription occurs, takes transcribed text as argument
        """
        self.on("transcription_event", lambda data: callback(data["text"]))

    async def send_audio_delta(self, audio_data: bytes) -> None:
        """
        Send audio delta to the STT
        """
        asyncio.create_task(self._process_audio_delta(audio_data))

    async def _process_audio_delta(self, audio_data: bytes) -> None:
        """Background processing of audio delta"""
        try:
            if self.denoise:
                audio_data = await self.denoise.denoise(audio_data)
            if self.stt:
                async with self.stt_lock:
                    await self.stt.process_audio(audio_data)
            if self.vad:
                await self.vad.process_audio(audio_data)
        except Exception as e:
            self.emit("error", f"Audio processing failed: {str(e)}")

    async def on_vad_event(self, vad_response: VADResponse) -> None:
        if vad_response.event_type == VADEventType.START_OF_SPEECH:
            await self.on_speech_started()
        elif vad_response.event_type == VADEventType.END_OF_SPEECH:
            self.on_speech_stopped()

    async def on_stt_transcript(self, stt_response: STTResponse) -> None:
        """Handle STT transcript events"""
        if stt_response.event_type == SpeechEventType.FINAL:
            user_text = stt_response.data.text
            await self._process_final_transcript(user_text)

    async def _process_final_transcript(self, user_text: str) -> None:
        """Process final transcript with EOU detection and response generation"""

        # Fallback: If VAD is missing, this can start the turn. Otherwise, the collector handles it.
        if not cascading_metrics_collector.data.current_turn:
            cascading_metrics_collector.on_user_speech_start()

        cascading_metrics_collector.set_user_transcript(user_text)
        cascading_metrics_collector.on_stt_complete()

        # Fallback: If VAD is present but hasn't called on_user_speech_end yet,
        if self.vad and cascading_metrics_collector.data.is_user_speaking:
            cascading_metrics_collector.on_user_speech_end()
        elif not self.vad:
            cascading_metrics_collector.on_user_speech_end()

        self.agent.chat_context.add_message(
            role=ChatRole.USER,
            content=user_text
        )

        await self.on_turn_start(user_text)

        if self.turn_detector:
            cascading_metrics_collector.on_eou_start()
            eou_detected = self.turn_detector.detect_end_of_utterance(
                self.agent.chat_context)
            cascading_metrics_collector.on_eou_complete()

            if eou_detected:
                asyncio.create_task(
                    self._generate_and_synthesize_response(user_text))
            else:
                cascading_metrics_collector.complete_current_turn()
        else:
            asyncio.create_task(
                self._generate_and_synthesize_response(user_text))

        await self.on_turn_end()

    async def _generate_and_synthesize_response(self, user_text: str) -> None:
        """Generate agent response"""
        self._is_interrupted = False

        full_response = ""
        self._partial_response = ""

        try:
            llm_stream = self.run(user_text)

            q = asyncio.Queue(maxsize=50)

            async def collector():
                response_parts = []
                try:
                    async for chunk in llm_stream:
                        if self._is_interrupted:
                            logger.info("LLM collection interrupted")
                            await q.put(None)
                            return "".join(response_parts)

                        self._partial_response = "".join(response_parts)
                        await q.put(chunk)
                        response_parts.append(chunk)

                    await q.put(None)
                    return "".join(response_parts)
                except asyncio.CancelledError:
                    logger.info("LLM collection cancelled")
                    await q.put(None)
                    return "".join(response_parts)

            async def tts_consumer():
                async def tts_stream_gen():
                    while True:
                        if self._is_interrupted:
                            break

                        chunk = await q.get()
                        if chunk is None:
                            break
                        yield chunk

                if self.tts:
                    try:
                        await self._synthesize_with_tts(tts_stream_gen())
                    except asyncio.CancelledError:
                        pass

            collector_task = asyncio.create_task(collector())
            tts_task = asyncio.create_task(tts_consumer())

            self._current_llm_task = collector_task
            self._current_tts_task = tts_task

            await asyncio.gather(collector_task, tts_task, return_exceptions=True)

            if not collector_task.cancelled() and not self._is_interrupted:
                full_response = collector_task.result()
            else:
                full_response = self._partial_response

            if full_response and not self._is_interrupted:
                cascading_metrics_collector.set_agent_response(full_response)
                self.agent.chat_context.add_message(
                    role=ChatRole.ASSISTANT,
                    content=full_response
                )

        finally:
            self._current_tts_task = None
            self._current_llm_task = None
            cascading_metrics_collector.complete_current_turn()

    async def process_with_llm(self) -> AsyncIterator[str]:
        """
        Process the current chat context with LLM and yield response chunks.
        This method can be called by user implementations to get LLM responses.
        """
        async with self.llm_lock:
            if not self.llm:
                return

            cascading_metrics_collector.on_llm_start()
            first_chunk_received = False

            async for llm_chunk_resp in self.llm.chat(
                self.agent.chat_context,
                tools=self.agent._tools
            ):
                if self._is_interrupted:
                    logger.info("LLM processing interrupted")
                    break

                if not first_chunk_received:
                    first_chunk_received = True
                    cascading_metrics_collector.on_llm_complete()

                if llm_chunk_resp.metadata and "function_call" in llm_chunk_resp.metadata:
                    func_call = llm_chunk_resp.metadata["function_call"]

                    cascading_metrics_collector.add_function_tool_call(
                        func_call["name"])

                    self.agent.chat_context.add_function_call(
                        name=func_call["name"],
                        arguments=json.dumps(func_call["arguments"]),
                        call_id=func_call.get(
                            "call_id", f"call_{int(time.time())}")
                    )

                    try:
                        tool = next(
                            (t for t in self.agent.tools if is_function_tool(
                                t) and get_tool_info(t).name == func_call["name"]),
                            None
                        )
                    except Exception as e:
                        logger.error(f"Error while selecting tool: {e}")
                        continue

                    if tool:
                        try:
                            result = await tool(**func_call["arguments"])
                            self.agent.chat_context.add_function_output(
                                name=func_call["name"],
                                output=json.dumps(result),
                                call_id=func_call.get(
                                    "call_id", f"call_{int(time.time())}")
                            )

                            async for new_resp in self.llm.chat(self.agent.chat_context):
                                if self._is_interrupted:
                                    break
                                if new_resp.content:
                                    yield new_resp.content
                        except Exception as e:
                            logger.error(
                                f"Error executing function {func_call['name']}: {e}")
                            continue
                else:
                    if llm_chunk_resp.content:
                        yield llm_chunk_resp.content

    async def say(self, message: str) -> None:
        """
        Direct TTS synthesis (used for initial messages)
        """
        if self.tts:
            cascading_metrics_collector.start_new_interaction("")
            cascading_metrics_collector.set_agent_response(message)

            try:
                await self._synthesize_with_tts(message)
            finally:
                cascading_metrics_collector.complete_current_turn()

    async def process_text_input(self, text: str) -> None:
        """
        Process text input directly (for A2A communication).
        This bypasses STT and directly processes the text through the LLM.
        """
        cascading_metrics_collector.start_new_interaction(text)

        self.agent.chat_context.add_message(
            role=ChatRole.USER,
            content=text
        )

        full_response = ""
        async for response_chunk in self.process_with_llm():
            full_response += response_chunk

        if full_response:
            cascading_metrics_collector.set_agent_response(full_response)
            cascading_metrics_collector.complete_current_turn()
            global_event_emitter.emit("text_response", {"text": full_response})

    async def run(self, transcript: str) -> AsyncIterator[str]:
        """
        Main conversation loop: handle a user turn.
        Users should implement this method to preprocess transcripts and yield response chunks.
        """
        async for response in self.process_with_llm():
            yield response

    async def on_turn_start(self, transcript: str) -> None:
        """Called at the start of a user turn."""
        pass

    async def on_turn_end(self) -> None:
        """Called at the end of a user turn."""
        pass

    def on_speech_started_stt(self, event_data: Any) -> None:
        if self.user_speech_callback:
            self.user_speech_callback()

    def on_speech_stopped_stt(self, event_data: Any) -> None:
        pass

    async def on_speech_started(self) -> None:
        cascading_metrics_collector.on_user_speech_start()

        if self.user_speech_callback:
            self.user_speech_callback()

        if self._stt_started:
            self._stt_started = False

        if self.tts:
            await self._interrupt_tts()

    async def _interrupt_tts(self) -> None:
        logger.info("Interrupting TTS and LLM generation")

        self._is_interrupted = True

        if self.tts:
            await self.tts.interrupt()

        if self.llm:
            await self._cancel_llm()

        tasks_to_cancel = []
        if self._current_tts_task and not self._current_tts_task.done():
            tasks_to_cancel.append(self._current_tts_task)
        if self._current_llm_task and not self._current_llm_task.done():
            tasks_to_cancel.append(self._current_llm_task)

        if tasks_to_cancel:
            await graceful_cancel(*tasks_to_cancel)

        cascading_metrics_collector.on_interrupted()

    async def _cancel_llm(self) -> None:
        """Cancel LLM generation"""
        try:
            await self.llm.cancel_current_generation()
        except Exception as e:
            logger.error(f"LLM cancellation failed: {e}")

    def on_speech_stopped(self) -> None:
        if not self._stt_started:
            cascading_metrics_collector.on_stt_start()
            self._stt_started = True

        cascading_metrics_collector.on_user_speech_end()

    async def _synthesize_with_tts(self, response_gen: AsyncIterator[str] | str) -> None:
        """
        Stream LLM response directly to TTS.
        """
        if not self.tts:
            return

        async def on_first_audio_byte():
            cascading_metrics_collector.on_tts_first_byte()
            cascading_metrics_collector.on_agent_speech_start()

        self.tts.on_first_audio_byte(on_first_audio_byte)
        self.tts.reset_first_audio_tracking()

        cascading_metrics_collector.on_tts_start()
        try:
            response_iterator: AsyncIterator[str]
            if isinstance(response_gen, str):
                async def string_to_iterator(text: str):
                    yield text
                response_iterator = string_to_iterator(response_gen)
            else:
                response_iterator = response_gen

            await self.tts.synthesize(response_iterator)

        finally:
            cascading_metrics_collector.on_agent_speech_end()
