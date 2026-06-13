from __future__ import annotations

import asyncio
import av
import json
import logging
import os
from typing import Any, Dict, Optional, Literal, List
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from dotenv import load_dotenv
import uuid
import base64
import aiohttp
import logging
import numpy as np
from videosdk.agents.resampling import resample_fft
import traceback
from videosdk.agents import (
    FunctionTool,
    is_function_tool,
    get_tool_info,
    build_openai_schema,
    CustomAudioStreamTrack,
    ToolChoice,
    RealtimeBaseModel,
    global_event_emitter,
    Agent,
    encode as encode_image,
    EncodeOptions,
    ResizeOptions,
)
from videosdk.agents.metrics import metrics_collector

logger = logging.getLogger(__name__)

load_dotenv()
logger = logging.getLogger(__name__)
from openai.types.beta.realtime.session import InputAudioTranscription, TurnDetection

OPENAI_BASE_URL = "https://api.openai.com/v1"

DEFAULT_TEMPERATURE = 0.8
DEFAULT_TURN_DETECTION = TurnDetection(
    type="server_vad",
    threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=200,
    create_response=True,
    interrupt_response=True,
)
DEFAULT_INPUT_AUDIO_TRANSCRIPTION = InputAudioTranscription(
    model="gpt-4o-mini-transcribe",
)
DEFAULT_TOOL_CHOICE = "auto"

OpenAIEventTypes = Literal["user_speech_started", "text_response", "error"]
DEFAULT_VOICE = "alloy"
DEFAULT_INPUT_AUDIO_FORMAT = "pcm16"
DEFAULT_OUTPUT_AUDIO_FORMAT = "pcm16"

DEFAULT_IMAGE_ENCODE_OPTIONS = EncodeOptions(
    format="JPEG",
    quality=75,
    resize_options=ResizeOptions(width=1024, height=1024),
)

@dataclass
class OpenAIRealtimeConfig:
    """Configuration for the OpenAI realtime API

    Args:
        voice: Voice ID for audio output. Default is 'alloy'
        temperature: Controls randomness in response generation. Higher values (e.g. 0.8) make output more random,
                    lower values make it more deterministic. Default is 0.8
        turn_detection: Configuration for detecting user speech turns. Contains settings for:
                       - type: Detection type ('server_vad')
                       - threshold: Voice activity detection threshold (0.0-1.0)
                       - prefix_padding_ms: Padding before speech start (ms)
                       - silence_duration_ms: Silence duration to mark end (ms)
                       - create_response: Whether to generate response on turn
                       - interrupt_response: Whether to allow interruption
        input_audio_transcription: Configuration for audio transcription. Contains:
                                 - model: Model to use for transcription
        tool_choice: How tools should be selected ('auto' or 'none'). Default is 'auto'
        modalities: List of enabled response types ["text", "audio"]. Default includes both
    """

    voice: str = DEFAULT_VOICE
    temperature: float = DEFAULT_TEMPERATURE
    turn_detection: TurnDetection | None = field(
        default_factory=lambda: DEFAULT_TURN_DETECTION
    )
    input_audio_transcription: InputAudioTranscription | None = field(
        default_factory=lambda: DEFAULT_INPUT_AUDIO_TRANSCRIPTION
    )
    tool_choice: ToolChoice | None = DEFAULT_TOOL_CHOICE
    modalities: list[str] = field(default_factory=lambda: ["text", "audio"])
    
    @property
    def is_text_only_mode(self) -> bool:
        """Check if configured for text-only responses (no audio)"""
        return "audio" not in self.modalities


@dataclass
class OpenAISession:
    """Represents an OpenAI WebSocket session"""

    ws: aiohttp.ClientWebSocketResponse
    msg_queue: asyncio.Queue[Dict[str, Any]]
    tasks: list[asyncio.Task]


class OpenAIRealtime(RealtimeBaseModel[OpenAIEventTypes]):
    """OpenAI's realtime model implementation."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        config: OpenAIRealtimeConfig | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize OpenAI realtime model.

        Args:
            api_key: OpenAI API key. If not provided, will attempt to read from OPENAI_API_KEY env var
            model: The OpenAI model identifier to use (e.g. 'gpt-4', 'gpt-3.5-turbo')
            config: Optional configuration object for customizing model behavior. Contains settings for:
                   - voice: Voice ID to use for audio output
                   - temperature: Sampling temperature for responses
                   - turn_detection: Settings for detecting user speech turns
                   - input_audio_transcription: Settings for audio transcription
                   - tool_choice: How tools should be selected ('auto' or 'none')
                   - modalities: List of enabled modalities ('text', 'audio')
            base_url: Base URL for OpenAI API. Defaults to 'https://api.openai.com/v1'

        Raises:
            ValueError: If no API key is provided and none found in environment variables
        """
        super().__init__()
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or OPENAI_BASE_URL
        if not self.api_key:
            self.emit(
                "error",
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable",
            )
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._session: Optional[OpenAISession] = None
        self._closing = False
        self._instructions: Optional[str] = None
        self._tools: Optional[List[FunctionTool]] = []
        self._formatted_tools: Optional[List[Dict[str, Any]]] = None
        self.config: OpenAIRealtimeConfig = config or OpenAIRealtimeConfig()
        self.input_sample_rate = 48000
        # GA Realtime API: audio/pcm is fixed at 24kHz. This drives both the
        # resample target for outgoing user audio and the rate declared in
        # session.audio.input.format — they must stay equal.
        self.target_sample_rate = 24000
        self._agent_speaking = False
        self._active_response_id: Optional[str] = None

    def set_agent(self, agent: Agent) -> None:
        self._agent = agent
        self._instructions = agent.instructions
        self._tools = agent.tools
        self.tools_formatted = self._format_tools_for_session(self._tools)
        self._formatted_tools = self.tools_formatted

    async def connect(self) -> None:
        headers = {"Agent": "VideoSDK Agents"}
        headers["Authorization"] = f"Bearer {self.api_key}"
        # GA Realtime API — do NOT send "OpenAI-Beta: realtime=v1". That header
        # opts into the retired Beta API, which the server now rejects with
        # "The Realtime Beta API is no longer supported."

        url = self.process_base_url(self.base_url, self.model)

        if "audio" in self.config.modalities:
            self.reframe_audio_track(self.target_sample_rate)

        try:
            self._session = await self._create_session(url, headers)
            await self._handle_websocket(self._session)
            await self.send_first_session_update()
        except aiohttp.WSServerHandshakeError as e:
            # Bad/expired API key, wrong URL, or rejected model fail here —
            # before the WebSocket opens, so the receive loop (and
            # _handle_error) never run. Surface it on the error channel.
            message = (
                f"OpenAI Realtime connection rejected (HTTP {e.status}): {e.message}"
            )
            if e.status in (401, 403):
                message += " — verify OPENAI_API_KEY is set and valid."
            logger.error(message)
            self.emit("error", message)
            raise
        except Exception as e:
            message = f"OpenAI Realtime connection failed: {e}"
            logger.error(message)
            self.emit("error", message)
            raise

    async def handle_audio_input(self, audio_data: bytes) -> None:
        """Handle incoming audio data from the user"""
        if self._session and not self._closing and "audio" in self.config.modalities:
            if self.current_utterance and not self.current_utterance.is_interruptible:
                logger.info("Interruption is disabled for the current utterance. Not processing audio input.")
                return
            # WebRTC source (aiortc) delivers 48 kHz s16 stereo-interleaved
            # frames flattened to bytes — _input_stream's frame.to_ndarray()[0]
            # is one row of L,R,L,R samples, NOT mono. Mix channels to mono
            # BEFORE resampling: without this, the buffer is twice the true
            # mono length, and once we declare GA's required rate=24000 the
            # server reads it at half real-time speed → both the transcription
            # model and the realtime LLM hear slowed-down speech and hallucinate
            # random-language tokens. (Gemini Live papers over this by declaring
            # rate=48000; GA OpenAI cannot — audio/pcm is fixed at 24 kHz.)
            raw = np.frombuffer(audio_data, dtype=np.int16)
            if raw.size >= 2 and raw.size % 2 == 0:
                mono = raw.reshape(-1, 2).astype(np.float32).mean(axis=1)
            else:
                mono = raw.astype(np.float32)
            resampled = resample_fft(
                mono,
                int(len(mono) * self.target_sample_rate / self.input_sample_rate),
            )
            audio_data = np.clip(resampled, -32767, 32767).astype(np.int16).tobytes()
            base64_audio_data = base64.b64encode(audio_data).decode("utf-8")
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio_data,
            }
            await self.send_event(audio_event)

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        """Ensure we have an HTTP session"""
        if not self._http_session:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def _create_session(self, url: str, headers: dict) -> OpenAISession:
        """Create a new WebSocket session"""

        http_session = await self._ensure_http_session()
        ws = await http_session.ws_connect(
            url,
            headers=headers,
            autoping=True,
            heartbeat=10,
            autoclose=False,
            timeout=30,
        )
        msg_queue: asyncio.Queue = asyncio.Queue()
        tasks: list[asyncio.Task] = []

        self._closing = False

        return OpenAISession(ws=ws, msg_queue=msg_queue, tasks=tasks)


    async def send_message(self, message: str) -> None:
        """Send a message to the OpenAI realtime API"""
        await self.send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            # GA: assistant/output message content is "output_text"
                            # (the Beta API's "text" is rejected).
                            "type": "output_text",
                            "text": "Repeat the user's exact message back to them:"
                            + message
                            + "DO NOT ADD ANYTHING ELSE",
                        }
                    ],
                },
            }
        )
        await self.create_response()

    async def handle_video_input(self, video_data: av.VideoFrame) -> None:
        if not self._session or self._closing:
            return

        try:
            if not video_data or not video_data.planes:
                return

            processed_jpeg = encode_image(video_data, DEFAULT_IMAGE_ENCODE_OPTIONS)

            if not processed_jpeg or len(processed_jpeg) < 100:
                logger.warning("Invalid JPEG data generated")
                return

            base64_url = self.bytes_to_base64_url(processed_jpeg)

            content = [{"type": "input_image", "image_url": base64_url}]

            conversation_event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": content,
                },
            }
            await self.send_event(conversation_event)

        except Exception as e:
            self.emit("error", f"Video processing error: {str(e)}")

    async def send_message_with_frames(
        self, message: Optional[str], frames: list[av.VideoFrame]
    ) -> None:
        content = []
        if message:
            content.append({"type": "input_text", "text": message})

        for frame in frames:
            try:
                processed_jpeg = encode_image(frame, DEFAULT_IMAGE_ENCODE_OPTIONS)

                if not processed_jpeg or len(processed_jpeg) < 100:
                    logger.warning("Invalid JPEG data generated")
                    continue

                base64_url = self.bytes_to_base64_url(processed_jpeg)
                content.append({"type": "input_image", "image_url": base64_url})
            except Exception as e:
                logger.error(f"Error processing frame: {e}")

        if not any(
            item.get("type") == "input_image" or item.get("type") == "input_text"
            for item in content
        ):
            logger.warning("No content to send.")
            return

        conversation_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": content,
            },
        }
        await self.send_event(conversation_event)

        await self.create_response()
    
    async def create_response(self) -> None:
        """Create a response to the OpenAI realtime API"""
        if not self._session:
            self.emit("error", "No active WebSocket session")
            raise RuntimeError("No active WebSocket session")

        response_event = {
            "type": "response.create",
            "event_id": str(uuid.uuid4()),
            "response": {
                "instructions": self._instructions,
                "metadata": {"client_event_id": str(uuid.uuid4())},
            },
        }

        await self.send_event(response_event)

    async def _handle_websocket(self, session: OpenAISession) -> None:
        """Start WebSocket send/receive tasks"""
        session.tasks.extend(
            [
                asyncio.create_task(self._send_loop(session), name="send_loop"),
                asyncio.create_task(self._receive_loop(session), name="receive_loop"),
            ]
        )

    async def _send_loop(self, session: OpenAISession) -> None:
        """Send messages from queue to WebSocket"""
        try:
            while not self._closing:
                msg = await session.msg_queue.get()
                if isinstance(msg, dict):
                    await session.ws.send_json(msg)
                else:
                    await session.ws.send_str(str(msg))
        except asyncio.CancelledError:
            pass
        except ConnectionError as e:
            # The WebSocket was closed underneath us. Don't leak an unretrieved
            # task exception, but do log it — a mid-session close here is a
            # symptom, not just teardown noise.
            logger.warning("OpenAI Realtime send loop stopped — connection closed: %s", e)
        finally:
            await self._cleanup_session(session)

    async def _receive_loop(self, session: OpenAISession) -> None:
        """Receive and process WebSocket messages"""
        try:
            while not self._closing:
                msg = await session.ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    # Server (or aiohttp's heartbeat) closed the socket. Log the
                    # close code so the cause is visible — 1000 clean, 1006
                    # abnormal/heartbeat, 1011 server error, 4xxx app-specific.
                    logger.error(
                        "OpenAI Realtime WebSocket closed by server "
                        "(msg_type=%s close_code=%s reason=%s)",
                        msg.type.name, session.ws.close_code, msg.extra,
                    )
                    self.emit(
                        "error",
                        f"OpenAI Realtime WebSocket closed: "
                        f"code={session.ws.close_code} reason={msg.extra}",
                    )
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("OpenAI Realtime WebSocket error: %s", msg.data)
                    self.emit("error", f"WebSocket error: {msg.data}")
                    break
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(json.loads(msg.data))
        except Exception as e:
            logger.error("OpenAI Realtime receive loop crashed: %s", e, exc_info=True)
            self.emit("error", f"WebSocket receive error: {str(e)}")
        finally:
            await self._cleanup_session(session)

    async def _handle_message(self, data: dict) -> None:
        """Handle incoming WebSocket messages"""
        try:
            event_type = data.get("type")

            if event_type == "input_audio_buffer.speech_started":
                await self._handle_speech_started(data)

            elif event_type == "input_audio_buffer.speech_stopped":
                await self._handle_speech_stopped(data)

            elif event_type == "response.created":
                await self._handle_response_created(data)

            elif event_type == "response.output_item.added":
                await self._handle_output_item_added(data)

            elif event_type == "response.content_part.added":
                await self._handle_content_part_added(data)

            elif event_type in ("response.text.delta", "response.output_text.delta"):
                await self._handle_text_delta(data)

            elif event_type in ("response.audio.delta", "response.output_audio.delta"):
                await self._handle_audio_delta(data)

            elif event_type in (
                "response.audio_transcript.delta",
                "response.output_audio_transcript.delta",
            ):
                await self._handle_audio_transcript_delta(data)

            elif event_type == "response.done":
                await self._handle_response_done(data)

            elif event_type == "error":
                await self._handle_error(data)

            elif event_type == "response.function_call_arguments.delta":
                await self._handle_function_call_arguments_delta(data)

            elif event_type == "response.function_call_arguments.done":
                await self._handle_function_call_arguments_done(data)

            elif event_type == "response.output_item.done":
                await self._handle_output_item_done(data)

            elif event_type == "conversation.item.input_audio_transcription.completed":
                await self._handle_input_audio_transcription_completed(data)

            elif event_type == "response.text.done":
                await self._handle_text_done(data)

        except Exception as e:
            self.emit("error", f"Error handling event {event_type}: {str(e)}")

    async def _handle_speech_started(self, data: dict) -> None:
        """Handle speech detection start"""
        if "audio" in self.config.modalities:
            self.emit("user_speech_started", {"type": "done"})
            logger.info("Interrupting on speech start.>>")
            if self.current_utterance and not self.current_utterance.is_interruptible:
                logger.info("Interruption is disabled for the current utterance. Not interrupting on speech start.")
                return
            await self.interrupt()
            if self.audio_track:
                self.audio_track.interrupt()
        metrics_collector.on_user_speech_start()
        metrics_collector.start_turn()

    async def _handle_speech_stopped(self, data: dict) -> None:
        """Handle speech detection end"""
        metrics_collector.on_user_speech_end()
        self.emit("user_speech_ended", {})

    async def _handle_response_created(self, data: dict) -> None:
        """Handle initial response creation"""
        self._active_response_id = data.get("response", {}).get("id")

    async def _handle_output_item_added(self, data: dict) -> None:
        """Handle new output item addition"""

    async def _handle_output_item_done(self, data: dict) -> None:
        """Handle output item done"""
        try:
            item = data.get("item", {})
            if (
                item.get("type") == "function_call"
                and item.get("status") == "completed"
            ):
                name = item.get("name")
                arguments = json.loads(item.get("arguments", "{}"))

                if name and self._tools:
                    for tool in self._tools:
                        tool_info = get_tool_info(tool)
                        if tool_info.name == name:
                            try:
                                metrics_collector.add_function_tool_call(tool_name=name)
                                result = await tool(**arguments)
                                self.emit(
                                    "realtime_model_function_executed",
                                    {
                                        "name": name,
                                        "arguments": item.get("arguments", "{}"),
                                        "call_id": item.get("call_id"),
                                        "output": result if isinstance(result, str) else json.dumps(result),
                                        "is_error": False,
                                    },
                                )
                                await self.send_event(
                                    {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": item.get("call_id"),
                                            "output": json.dumps(result),
                                        },
                                    }
                                )

                                await self.send_event(
                                    {
                                        "type": "response.create",
                                        "event_id": str(uuid.uuid4()),
                                        "response": {
                                            "instructions": self._instructions,
                                            "metadata": {
                                                "client_event_id": str(uuid.uuid4())
                                            },
                                        },
                                    }
                                )

                            except Exception as e:
                                self.emit(
                                    "realtime_model_function_executed",
                                    {
                                        "name": name,
                                        "arguments": item.get("arguments", "{}"),
                                        "call_id": item.get("call_id"),
                                        "output": str(e),
                                        "is_error": True,
                                    },
                                )
                                self.emit(
                                    "error", f"Error executing function {name}: {e}"
                                )
                            break
        except Exception as e:
            self.emit("error", f"Error handling output item done: {e}")

    async def _handle_content_part_added(self, data: dict) -> None:
        """Handle new content part"""

    async def _handle_text_delta(self, data: dict) -> None:
        """Handle text delta chunk (for text-only mode)"""
        delta_content = data.get("delta", "")
        
        if not hasattr(self, "_current_text_response"):
            self._current_text_response = ""
        
        if not self._agent_speaking and delta_content:
            metrics_collector.on_agent_speech_start()
            self._agent_speaking = True
            self.emit("agent_speech_started", {})

        self._current_text_response += delta_content
        
        self.emit("realtime_model_text_delta", {
            "role": "assistant",
            "delta": delta_content,
            "text": self._current_text_response,
        })

    async def _handle_audio_delta(self, data: dict) -> None:
        """Handle audio chunk"""
        if "audio" not in self.config.modalities:
            return

        try:
            if not self._agent_speaking:
                metrics_collector.on_agent_speech_start()
                self._agent_speaking = True
                self.emit("agent_speech_started", {})
            base64_audio_data = base64.b64decode(data.get("delta"))
            if base64_audio_data:
                if self.audio_track and self.loop:
                    asyncio.create_task(
                        self.audio_track.add_new_bytes(base64_audio_data)
                    )
        except Exception as e:
            self.emit("error", f"Error handling audio delta: {e}")
            traceback.print_exc()

    async def interrupt(self) -> None:
        """Interrupt the current response and flush audio"""
        if self._session and not self._closing:
            if self.current_utterance and not self.current_utterance.is_interruptible:
                logger.info("Interruption is disabled for the current utterance. Not interrupting OpenAI realtime session.")
                return
            # Only cancel when a response is actually in flight — GA rejects a
            # stray response.cancel with "no active response found".
            if self._active_response_id:
                cancel_event = {"type": "response.cancel", "event_id": str(uuid.uuid4())}
                await self.send_event(cancel_event)
                self._active_response_id = None
                metrics_collector.on_interrupted()
        if self.audio_track:
            self.audio_track.interrupt()
        if self._agent_speaking:
            if self.audio_track:
                self.audio_track.mark_synthesis_complete()
            self._agent_speaking = False

    async def _handle_audio_transcript_delta(self, data: dict) -> None:
        """Handle transcript chunk"""
        delta_content = data.get("delta", "")
        if not hasattr(self, "_current_audio_transcript"):
            self._current_audio_transcript = ""
        self._current_audio_transcript += delta_content

    async def _handle_input_audio_transcription_completed(self, data: dict) -> None:
        """Handle input audio transcription completion for user transcript"""
        transcript = data.get("transcript", "")
        if transcript:
            metrics_collector.set_user_transcript(transcript)
            try:
                self.emit(
                    "realtime_model_transcription",
                    {"role": "user", "text": transcript, "is_final": True},
                )
            except Exception:
                pass

    async def _handle_response_done(self, data: dict) -> None:
        """Handle response completion for agent transcript"""
        usage_metadata = self.get_realtime_tokens(data)
        metrics_collector.set_realtime_usage(usage_metadata)
        if (
            hasattr(self, "_current_audio_transcript")
            and self._current_audio_transcript
        ):
            metrics_collector.set_agent_response(
                self._current_audio_transcript
            )
            
            self.emit("llm_text_output", {"text": self._current_audio_transcript})
            
            global_event_emitter.emit(
                "text_response",
                {"text": self._current_audio_transcript, "type": "done"},
            )
            try:
                self.emit(
                    "realtime_model_transcription",
                    {
                        "role": "agent",
                        "text": self._current_audio_transcript,
                        "is_final": True,
                    },
                )
            except Exception:
                pass
            self._current_audio_transcript = ""
        self._active_response_id = None
        self.audio_track.mark_synthesis_complete()
        # self.emit("agent_speech_ended", {})
        # metrics_collector.on_agent_speech_end()
        # metrics_collector.schedule_turn_complete(timeout=1.0)
        self._agent_speaking = False
        pass

    async def _handle_function_call_arguments_delta(self, data: dict) -> None:
        """Handle function call arguments delta"""

    async def _handle_function_call_arguments_done(self, data: dict) -> None:
        """Handle function call arguments done"""

    async def _handle_error(self, data: dict) -> None:
        """Handle error events from the OpenAI Realtime API.

        Previously a silent no-op, which made every API-side failure
        (invalid model, bad session config, rejected items) invisible.
        """
        error = data.get("error", data)
        if isinstance(error, dict):
            message = error.get("message") or error.get("code") or str(error)
        else:
            message = str(error)
        logger.error(f"OpenAI Realtime API error: {message}")
        self.emit("error", f"OpenAI Realtime API error: {message}")

    async def _cleanup_session(self, session: OpenAISession) -> None:
        """Clean up session resources"""
        if self._closing:
            return

        logger.info(
            "OpenAI Realtime session teardown — closing send/receive loops "
            "(ws_closed=%s)", session.ws.closed,
        )
        self._closing = True

        for task in session.tasks:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)  # Add timeout
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        if not session.ws.closed:
            try:
                await session.ws.close()
            except Exception:
                pass

    async def send_event(self, event: Dict[str, Any]) -> None:
        """Send an event to the WebSocket"""
        if self._session and not self._closing:
            await self._session.msg_queue.put(event)

    async def aclose(self) -> None:
        """Cleanup all resources"""
        if self._closing:
            return

        self._closing = True

        if self._session:
            await self._cleanup_session(self._session)

        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

        await super().aclose()

    async def send_first_session_update(self) -> None:
        """Send the initial session.update using the GA Realtime API schema.

        The GA ``session`` object differs from the retired Beta schema:
        - ``modalities`` → ``output_modalities``
        - flat ``voice`` / ``input_audio_format`` / ``output_audio_format`` /
          ``turn_detection`` / ``input_audio_transcription`` are nested under
          ``audio.input`` / ``audio.output``
        - audio formats are objects (``{"type": "audio/pcm", "rate": N}``)
        - ``session.type`` is ``"realtime"``; the model is set via the
          connect URL, not the session object.
        """
        if not self._session:
            return

        audio_mode = "audio" in self.config.modalities

        session: Dict[str, Any] = {
            "type": "realtime",
            "instructions": self.instructions_with_context(self._instructions),
            "output_modalities": ["audio"] if audio_mode else ["text"],
            "tools": self._formatted_tools or [],
            "tool_choice": self.config.tool_choice,
        }

        if audio_mode:
            audio: Dict[str, Any] = {
                "input": {
                    "format": {"type": "audio/pcm", "rate": self.target_sample_rate},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": self.config.voice,
                },
            }
            if self.config.turn_detection:
                audio["input"]["turn_detection"] = self.config.turn_detection.model_dump(
                    by_alias=True, exclude_unset=True, exclude_defaults=True,
                )
            if self.config.input_audio_transcription:
                audio["input"]["transcription"] = (
                    self.config.input_audio_transcription.model_dump(
                        by_alias=True, exclude_unset=True, exclude_defaults=True,
                    )
                )
            session["audio"] = audio

        await self.send_event({"type": "session.update", "session": session})


    def process_base_url(self, url: str, model: str) -> str:
        if url.startswith("http"):
            url = url.replace("http", "ws", 1)

        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)

        if not parsed_url.path or parsed_url.path.rstrip("/") in ["", "/v1", "/openai"]:
            path = parsed_url.path.rstrip("/") + "/realtime"
        else:
            path = parsed_url.path

        if "model" not in query_params:
            query_params["model"] = [model]

        new_query = urlencode(query_params, doseq=True)
        new_url = urlunparse(
            (parsed_url.scheme, parsed_url.netloc, path, "", new_query, "")
        )

        return new_url

    def _format_tools_for_session(
        self, tools: List[FunctionTool]
    ) -> List[Dict[str, Any]]:
        """Format tools for OpenAI session update"""
        oai_tools = []
        for tool in tools:
            if not is_function_tool(tool):
                continue

            try:
                tool_schema = build_openai_schema(tool)
                oai_tools.append(tool_schema)
            except Exception as e:
                self.emit("error", f"Failed to format tool {tool}: {e}")
                continue

        return oai_tools

    async def send_text_message(self, message: str) -> None:
        """Send a text message to the OpenAI realtime API"""
        if not self._session:
            self.emit("error", "No active WebSocket session")
            raise RuntimeError("No active WebSocket session")

        await self.send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": message}],
                },
            }
        )
        await self.create_response()


    def bytes_to_base64_url(self, image_bytes: bytes, fmt: str = "jpeg") -> str:
            mime = f"image/{fmt.lower()}"
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:{mime};base64,{encoded}"

    def get_realtime_tokens(self, event: dict) -> dict:
        """
        Extract and flatten all token details needed for pricing from a
        OpenAI Realtime response.done event into a single-level dictionary.

        Parameters:
            event (dict): Full Realtime event payload

        Returns:
            dict: Single-level dictionary with token counts
        """
        usage = event.get("response", {}).get("usage", {})
        input_details = usage.get("input_token_details", {})
        cached_details = input_details.get("cached_tokens_details", {})
        output_details = usage.get("output_token_details", {})

        token_dict = {
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),

            "input_text_tokens": input_details.get("text_tokens", 0),
            "input_audio_tokens": input_details.get("audio_tokens", 0),
            "input_image_tokens": input_details.get("image_tokens", 0),
            "input_cached_tokens": input_details.get("cached_tokens", 0),

            "cached_text_tokens": cached_details.get("text_tokens", 0),
            "cached_audio_tokens": cached_details.get("audio_tokens", 0),
            "cached_image_tokens": cached_details.get("image_tokens", 0),

            "output_text_tokens": output_details.get("text_tokens", 0),
            "output_audio_tokens": output_details.get("audio_tokens", 0),
            "output_image_tokens": output_details.get("image_tokens", 0)
        }

        return token_dict            
