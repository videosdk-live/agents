from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional, Literal, List
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from dotenv import load_dotenv
import uuid
import base64
import aiohttp
import numpy as np
from scipy import signal
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
)
from videosdk.agents import realtime_metrics_collector


load_dotenv()
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
        model: str,
        config: OpenAIRealtimeConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize OpenAI realtime model.

        Args:
            model: The OpenAI model identifier to use (e.g. 'gpt-4', 'gpt-3.5-turbo')
            config: Optional configuration object for customizing model behavior. Contains settings for:
                   - voice: Voice ID to use for audio output
                   - temperature: Sampling temperature for responses
                   - turn_detection: Settings for detecting user speech turns
                   - input_audio_transcription: Settings for audio transcription
                   - tool_choice: How tools should be selected ('auto' or 'none')
                   - modalities: List of enabled modalities ('text', 'audio')
            api_key: OpenAI API key. If not provided, will attempt to read from OPENAI_API_KEY env var
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
        self.loop = None
        self.audio_track: Optional[CustomAudioStreamTrack] = None
        self._formatted_tools: Optional[List[Dict[str, Any]]] = None
        self.config: OpenAIRealtimeConfig = config or OpenAIRealtimeConfig()
        self.input_sample_rate = 48000
        self.target_sample_rate = 16000
        self._agent_speaking = False

    def set_agent(self, agent: Agent) -> None:
        self._instructions = agent.instructions
        self._tools = agent.tools
        self.tools_formatted = self._format_tools_for_session(self._tools)
        self._formatted_tools = self.tools_formatted

    async def connect(self) -> None:
        headers = {"Agent": "VideoSDK Agents"}
        headers["Authorization"] = f"Bearer {self.api_key}"
        headers["OpenAI-Beta"] = "realtime=v1"

        url = self.process_base_url(self.base_url, self.model)

        self._session = await self._create_session(url, headers)
        await self._handle_websocket(self._session)
        await self.send_first_session_update()

    async def handle_audio_input(self, audio_data: bytes) -> None:
        """Handle incoming audio data from the user"""
        if self._session and not self._closing and "audio" in self.config.modalities:
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
            audio_data = signal.resample(
                audio_data,
                int(len(audio_data) * self.target_sample_rate / self.input_sample_rate),
            )
            audio_data = audio_data.astype(np.int16).tobytes()
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
                            "type": "text",
                            "text": "Repeat the user's exact message back to them:"
                            + message
                            + "DO NOT ADD ANYTHING ELSE",
                        }
                    ],
                },
            }
        )
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
        finally:
            await self._cleanup_session(session)

    async def _receive_loop(self, session: OpenAISession) -> None:
        """Receive and process WebSocket messages"""
        try:
            while not self._closing:
                msg = await session.ws.receive()

                if msg.type == aiohttp.WSMsgType.CLOSED:
                    self.emit("error", f"WebSocket closed with reason: {msg.extra}")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.emit("error", f"WebSocket error: {msg.data}")
                    break
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(json.loads(msg.data))
        except Exception as e:
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

            elif event_type == "response.text.delta":
                await self._handle_text_delta(data)

            elif event_type == "response.audio.delta":
                await self._handle_audio_delta(data)

            elif event_type == "response.audio_transcript.delta":
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
            await self.interrupt()
            if self.audio_track:
                self.audio_track.interrupt()
        await realtime_metrics_collector.set_user_speech_start()

    async def _handle_speech_stopped(self, data: dict) -> None:
        """Handle speech detection end"""
        await realtime_metrics_collector.set_user_speech_end()

    async def _handle_response_created(self, data: dict) -> None:
        """Handle initial response creation"""
        response_id = data.get("response", {}).get("id")

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
                                await realtime_metrics_collector.add_tool_call(name)
                                result = await tool(**arguments)
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
                                    "error", f"Error executing function {name}: {e}"
                                )
                            break
        except Exception as e:
            self.emit("error", f"Error handling output item done: {e}")

    async def _handle_content_part_added(self, data: dict) -> None:
        """Handle new content part"""

    async def _handle_text_delta(self, data: dict) -> None:
        """Handle text delta chunk"""
        pass

    async def _handle_audio_delta(self, data: dict) -> None:
        """Handle audio chunk"""
        if "audio" not in self.config.modalities:
            return

        try:
            if not self._agent_speaking:
                await realtime_metrics_collector.set_agent_speech_start()
                self._agent_speaking = True
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
            cancel_event = {"type": "response.cancel", "event_id": str(uuid.uuid4())}
            await self.send_event(cancel_event)
            await realtime_metrics_collector.set_interrupted()
        if self.audio_track:
            self.audio_track.interrupt()
        if self._agent_speaking:
            await realtime_metrics_collector.set_agent_speech_end(timeout=1.0)
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
            await realtime_metrics_collector.set_user_transcript(transcript)
            try:
                self.emit(
                    "realtime_model_transcription",
                    {"role": "user", "text": transcript, "is_final": True},
                )
            except Exception:
                pass

    async def _handle_response_done(self, data: dict) -> None:
        """Handle response completion for agent transcript"""
        if (
            hasattr(self, "_current_audio_transcript")
            and self._current_audio_transcript
        ):
            await realtime_metrics_collector.set_agent_response(
                self._current_audio_transcript
            )
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
        await realtime_metrics_collector.set_agent_speech_end(timeout=1.0)
        self._agent_speaking = False
        pass

    async def _handle_function_call_arguments_delta(self, data: dict) -> None:
        """Handle function call arguments delta"""

    async def _handle_function_call_arguments_done(self, data: dict) -> None:
        """Handle function call arguments done"""

    async def _handle_error(self, data: dict) -> None:
        """Handle error events"""

    async def _cleanup_session(self, session: OpenAISession) -> None:
        """Clean up session resources"""
        if self._closing:
            return

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

    async def send_first_session_update(self) -> None:
        """Send initial session update with default values after connection"""
        if not self._session:
            return

        turn_detection = None
        input_audio_transcription = None

        if "audio" in self.config.modalities:
            turn_detection = (
                self.config.turn_detection.model_dump(
                    by_alias=True,
                    exclude_unset=True,
                    exclude_defaults=True,
                )
                if self.config.turn_detection
                else None
            )
            input_audio_transcription = (
                self.config.input_audio_transcription.model_dump(
                    by_alias=True,
                    exclude_unset=True,
                    exclude_defaults=True,
                )
                if self.config.input_audio_transcription
                else None
            )

        session_update = {
            "type": "session.update",
            "session": {
                "model": self.model,
                "instructions": self._instructions
                or "You are a helpful assistant that can answer questions and help with tasks.",
                "temperature": self.config.temperature,
                "tool_choice": self.config.tool_choice,
                "tools": self._formatted_tools or [],
                "modalities": self.config.modalities,
                "max_response_output_tokens": "inf",
            },
        }

        if "audio" in self.config.modalities:
            session_update["session"]["voice"] = self.config.voice
            session_update["session"]["input_audio_format"] = DEFAULT_INPUT_AUDIO_FORMAT
            session_update["session"][
                "output_audio_format"
            ] = DEFAULT_OUTPUT_AUDIO_FORMAT
            if turn_detection:
                session_update["session"]["turn_detection"] = turn_detection
            if input_audio_transcription:
                session_update["session"][
                    "input_audio_transcription"
                ] = input_audio_transcription

        # Send the event
        await self.send_event(session_update)

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
