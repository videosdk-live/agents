from __future__ import annotations

import asyncio
import os
import logging
import json
from typing import Any, Optional, Literal
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv
from videosdk.agents import (
    Agent,
    CustomAudioStreamTrack,
    RealtimeBaseModel,
    is_function_tool,
    get_tool_info,
    build_openai_schema,
    realtime_metrics_collector,
    global_event_emitter
)
import websockets
from websockets.client import WebSocketClientProtocol
import aiohttp

load_dotenv()

logger = logging.getLogger(__name__)

UltravoxEventTypes = Literal["user_speech_started", "text_response", "error"]

DEFAULT_MODEL = "fixie-ai/ultravox"
DEFAULT_BASE_URL = "https://api.ultravox.ai/api/"
DEFAULT_LANGUAGE_HINT = "en"
DEFAULT_INPUT_SAMPLE_RATE = 48000
DEFAULT_OUTPUT_SAMPLE_RATE = 24000
DEFAULT_CLIENT_BUFFER_SIZE_MS = 30000
DEFAULT_VAD_TURN_ENDPOINT_DELAY = 800
DEFAULT_VAD_MINIMUM_TURN_DURATION = 600
DEFAULT_VAD_FRAME_ACTIVATION_THRESHOLD = 0.4
DEFAULT_FIRST_SPEAKER = "FIRST_SPEAKER_USER"

Voice = str 

@dataclass
class UltravoxLiveConfig:
    """Configuration for the Ultravox Live API"""
    voice: Voice | None = None
    language_hint: str | None = DEFAULT_LANGUAGE_HINT
    temperature: float | None = None
    max_duration: str | None = None
    time_exceeded_message: str | None = None
    input_sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE
    output_sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE
    client_buffer_size_ms: int = DEFAULT_CLIENT_BUFFER_SIZE_MS
    vad_turn_endpoint_delay: int | None = DEFAULT_VAD_TURN_ENDPOINT_DELAY
    vad_minimum_turn_duration: int | None = DEFAULT_VAD_MINIMUM_TURN_DURATION
    vad_minimum_interruption_duration: int | None = None
    vad_frame_activation_threshold: float | None = DEFAULT_VAD_FRAME_ACTIVATION_THRESHOLD
    first_speaker: str | None = DEFAULT_FIRST_SPEAKER
    enable_greeting_prompt: bool = False 


@dataclass
class UltravoxSession:
    """Represents an Ultravox WebSocket session"""
    websocket: WebSocketClientProtocol
    call_id: str
    tasks: list[asyncio.Task]


class UltravoxRealtime(RealtimeBaseModel[UltravoxEventTypes]):
    """Ultravox's realtime model for audio-only communication"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        config: UltravoxLiveConfig | None = None,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("ULTRAVOX_API_KEY")
        if not self.api_key:
            raise ValueError("ULTRAVOX_API_KEY required")
        
        self.model = model
        self.base_url = base_url.rstrip('/')
        self._session: Optional[UltravoxSession] = None
        self._closing = False
        self._session_should_close = asyncio.Event()
        self._main_task = None
        self.loop = None
        self.audio_track = None
        self.tools = []
        self._instructions: str = "You are a helpful voice assistant."
        self.config: UltravoxLiveConfig = config or UltravoxLiveConfig()
        self._user_speaking = False
        self._agent_speaking = False
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._audio_queue = asyncio.Queue(maxsize=100)
        self.formatted_tools = []
        
    def set_agent(self, agent: Agent) -> None:
        self._instructions = agent.instructions
        self.tools = agent.tools
        self.formatted_tools = self._convert_tools_to_ultravox_format(self.tools)

    async def connect(self) -> None:
        """Connect to the Ultravox API"""
        if self._session:
            await self._cleanup_session(self._session)
            self._session = None

        self._closing = False
        self._session_should_close.clear()

        try:
            if not self.audio_track and self.loop:
                self.audio_track = CustomAudioStreamTrack(self.loop)
            elif not self.loop:
                raise RuntimeError("Event loop not initialized")

            if not self._http_session:
                self._http_session = aiohttp.ClientSession()

            try:
                self._session = await self._create_session()
            except Exception as e:
                self.emit("error", f"Initial session creation failed: {e}")

            if not self._main_task or self._main_task.done():
                self._main_task = asyncio.create_task(
                    self._session_loop(), name="ultravox-main-loop"
                )

        except Exception as e:
            self.emit("error", f"Error connecting to Ultravox API: {e}")
            raise
    
    async def _create_session(self) -> UltravoxSession:
        """Create a new Ultravox WebSocket session"""
        try:
            call_payload = {
                "systemPrompt": self._instructions,
                "model": self.model,
                "medium": {
                    "serverWebSocket": {
                        "inputSampleRate": self.config.input_sample_rate,
                        "outputSampleRate": self.config.output_sample_rate,
                        "clientBufferSizeMs": self.config.client_buffer_size_ms,
                    }
                },
            }

            if self.config.voice:
                call_payload["voice"] = self.config.voice
            if self.config.language_hint:
                call_payload["languageHint"] = self.config.language_hint
            if self.config.temperature is not None:
                call_payload["temperature"] = self.config.temperature
            if self.config.max_duration:
                call_payload["maxDuration"] = self.config.max_duration
            if self.config.time_exceeded_message:
                call_payload["timeExceededMessage"] = self.config.time_exceeded_message
            if self.config.first_speaker:
                call_payload["firstSpeaker"] = self.config.first_speaker
            if self.formatted_tools:
                call_payload["selectedTools"] = self.formatted_tools

            vad_settings = {}
            if self.config.vad_turn_endpoint_delay is not None:
                vad_settings["turnEndpointDelay"] = f"{self.config.vad_turn_endpoint_delay / 1000}s"
            if self.config.vad_minimum_turn_duration is not None:
                vad_settings["minimumTurnDuration"] = f"{self.config.vad_minimum_turn_duration / 1000}s"
            if self.config.vad_minimum_interruption_duration is not None:
                vad_settings["minimumInterruptionDuration"] = f"{self.config.vad_minimum_interruption_duration / 1000}s"
            if self.config.vad_frame_activation_threshold is not None:
                vad_settings["frameActivationThreshold"] = self.config.vad_frame_activation_threshold
            
            if vad_settings:
                call_payload["vadSettings"] = vad_settings

            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            }

            url = f"{self.base_url}/calls"
            if not self.config.enable_greeting_prompt:
                url += "?enableGreetingPrompt=false"

            async with self._http_session.post(
                url,
                json=call_payload,
                headers=headers,
            ) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise Exception(f"Failed to create call: {response.status} - {error_text}")
                
                data = await response.json()
                join_url = data.get("joinUrl")
                call_id = data.get("callId")

                if not join_url:
                    raise Exception("No joinUrl in response")

            websocket = await websockets.connect(join_url)
            logger.info(f"Connected to Ultravox call: {call_id}")
            
            return UltravoxSession(websocket=websocket, call_id=call_id, tasks=[])

        except Exception as e:
            self.emit("error", f"Connection error: {e}")
            raise

    async def _session_loop(self) -> None:
        """Main processing loop for Ultravox sessions"""
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        reconnect_delay = 1

        while not self._closing:
            if not self._session:
                try:
                    self._session = await self._create_session()
                    reconnect_attempts = 0
                    reconnect_delay = 1
                except Exception as e:
                    reconnect_attempts += 1
                    reconnect_delay = min(30, reconnect_delay * 2)
                    self.emit("error", f"Reconnection attempt {reconnect_attempts} failed: {e}")
                    if reconnect_attempts >= max_reconnect_attempts:
                        self.emit("error", "Max reconnection attempts reached")
                        break
                    await asyncio.sleep(reconnect_delay)
                    continue

            session = self._session

            recv_task = asyncio.create_task(self._receive_loop(session), name="ultravox_receive")
            send_task = asyncio.create_task(self._send_audio_loop(session), name="ultravox_send")
            keep_alive_task = asyncio.create_task(self._keep_alive(session), name="ultravox_keepalive")
            session.tasks.extend([recv_task, send_task, keep_alive_task])

            try:
                await self._session_should_close.wait()
            finally:
                for task in session.tasks:
                    if not task.done():
                        task.cancel()
                try:
                    await asyncio.gather(*session.tasks, return_exceptions=True)
                except Exception as e:
                    logger.error(f"Error during task cleanup: {e}")

            if not self._closing:
                await self._cleanup_session(session)
                self._session = None
                await asyncio.sleep(reconnect_delay)
                self._session_should_close.clear()

    async def _send_audio_loop(self, session: UltravoxSession) -> None:
        """Continuously send audio chunks to Ultravox"""
        try:
            logger.info("Audio send loop started")
            
            while not self._closing:
                audio_data = None
                try:
                    audio_data = await asyncio.wait_for(self._audio_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass
                    
                if audio_data and len(audio_data) >= 2:
                        await session.websocket.send(audio_data)
        except asyncio.CancelledError:
            logger.info("Audio send loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in audio send loop: {e}")
            self._session_should_close.set()
            
    async def send_message(self, message: str) -> None:
        """Send a message to get audio response"""
        if not self._session:
            logger.warning("Ultravox: No active session for send_message")
            return

        msg = {
            "type": "user_text_message",
            "text": f"<instruction>Please say exactly: \"{message}\". Do not add any other text.</instruction>",
            "defer_response": False
        }
        try:
            await self._session.websocket.send(json.dumps(msg))
        except Exception as e:
            logger.error(f"Ultravox: Error sending text message: {e}")

    async def send_text_message(self, message: str) -> None:
        """Send a text message for text-only communication"""
        if not self._session:
            logger.warning("Ultravox: No active session for send_text_message")
            return
            
        msg = {
            "type": "user_text_message",
            "text": message,
            "defer_response": False
        }
        try:
            await self._session.websocket.send(json.dumps(msg))
        except Exception as e:
            logger.error(f"Ultravox: Error sending text message: {e}")

    async def _handle_tool_calls(self, tool_invocation: dict) -> None:
        invocation_id = tool_invocation.get("invocationId") or tool_invocation.get("invocation_id")
        tool_name = tool_invocation.get("toolName") or tool_invocation.get("name")
        parameters = tool_invocation.get("parameters", {})

        if not tool_name or not invocation_id:
            logger.error("Ultravox: Invalid tool invocation payload")
            return

        for tool in self.tools:
            if not is_function_tool(tool):
                continue

            tool_info = get_tool_info(tool)
            if tool_info.name == tool_name:
                try:
                    await realtime_metrics_collector.add_tool_call(tool_info.name)
                    result = await tool(**parameters)
                    await self._send_tool_result(invocation_id, result)
                except Exception as e:
                    logger.error(f"Ultravox: Tool execution failed: {e}")
                    await self._send_tool_result(
                        invocation_id, None, error_message=str(e)
                    )
                return

        logger.error(f"Ultravox: Tool not found: {tool_name}")

    async def _send_tool_result(
        self, invocation_id: str, result: Any, error_message: str | None = None
    ) -> None:
        """Send tool result back to Ultravox"""
        if not self._session:
            return

        message = {
            "type": "client_tool_result",
            "invocationId": invocation_id,
        }
        
        message["invocation_id"] = invocation_id

        if error_message:
            message["result"] = None
            message["errorMessage"] = error_message
            
        else:
            if isinstance(result, (dict, list)):
                message["result"] = json.dumps(result)
            else:
                message["result"] = str(result)
            
        try:
            await self._session.websocket.send(json.dumps(message))
        except Exception as e:
            self.emit("error", f"Error sending tool result: {e}")

    async def _receive_loop(self, session: UltravoxSession) -> None:
        """Process incoming messages from Ultravox"""
        try:
            accumulated_transcript = ""

            while not self._closing:
                try:
                    message = await session.websocket.recv()

                    if self._closing:
                        break

                    if isinstance(message, bytes):
                        if len(message) < 2:
                            continue
                            
                        if not self._agent_speaking:
                            self.emit("agent_speech_started", {})
                            await realtime_metrics_collector.set_agent_speech_start()
                            self._agent_speaking = True

                        if self.audio_track and self.loop:
                            asyncio.create_task(
                                self.audio_track.add_new_bytes(message),
                                name=f"audio_output",
                            )
                        continue
                    
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type")

                    if msg_type == "transcript":
                        role = data.get("role")
                        text = data.get("text", "")
                        delta = data.get("delta", "")
                        is_final = data.get("final", False)

                        if role == "user":
                            if not self._user_speaking and (text or delta):
                                self.emit("user_speech_started", {})
                                await realtime_metrics_collector.set_user_speech_start()
                                self._user_speaking = True
                            
                            display_text = text if text else delta
                            if display_text:
                                global_event_emitter.emit(
                                    "input_transcription",
                                    {"text": display_text, "is_final": is_final},
                                )
                            
                            if is_final and text:
                                await realtime_metrics_collector.set_user_transcript(text)
                                self.emit(
                                    "realtime_model_transcription",
                                    {"role": "user", "text": text, "is_final": True},
                                )
                                await realtime_metrics_collector.set_user_speech_end()
                                self._user_speaking = False

                        elif role == "agent":
                            if not self._agent_speaking and (text or delta):
                                self.emit("agent_speech_started", {})
                                await realtime_metrics_collector.set_agent_speech_start()
                                self._agent_speaking = True
                                
                            display_text = text if text else delta
                            if display_text:
                                accumulated_transcript = text if text else (accumulated_transcript + delta)
                                global_event_emitter.emit(
                                    "output_transcription",
                                    {"text": accumulated_transcript, "is_final": is_final},
                                )
                            
                            if is_final and text:
                                await realtime_metrics_collector.set_agent_response(text)
                                self.emit(
                                    "realtime_model_transcription",
                                    {"role": "agent", "text": text, "is_final": True},
                                )
                                global_event_emitter.emit(
                                    "text_response",
                                    {"type": "done", "text": text},
                                )
                                accumulated_transcript = ""

                    elif msg_type == "state":
                        state = data.get("state")
                        if state == "speaking":
                            if not self._agent_speaking:
                                self.emit("agent_speech_started", {})
                                await realtime_metrics_collector.set_agent_speech_start()
                                self._agent_speaking = True
                        elif state in ["idle", "listening"]:
                            if self._agent_speaking:
                                self.emit("agent_speech_ended", {})
                                await realtime_metrics_collector.set_agent_speech_end()
                                self._agent_speaking = False

                    elif msg_type in ["clientToolInvocation", "client_tool_invocation"]:
                        await self._handle_tool_calls(data)

                    elif msg_type in ["playbackClearBuffer", "playback_clear_buffer"]:
                        if self.audio_track:
                            self.audio_track.interrupt()
                            

                except websockets.exceptions.ConnectionClosed:
                    self._session_should_close.set()
                    break
                except Exception as e:
                    logger.error(f"Error in receive loop: {e}")

                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Fatal error in receive loop: {e}")
            self._session_should_close.set()

    async def _keep_alive(self, session: UltravoxSession) -> None:
        """Send periodic keep-alive messages"""
        try:
            while not self._closing:
                await asyncio.sleep(10)
                if self._closing:
                    break
                try:
                    ping_msg = {"type": "ping", "timestamp": int(time.time() * 1000)}
                    await session.websocket.send(json.dumps(ping_msg))
                except Exception as e:
                    if "closed" in str(e).lower():
                        self._session_should_close.set()
                        break
        except asyncio.CancelledError:
            pass

    def _convert_to_s16le_pcm_mono(self, audio_bytes: bytes) -> bytes:
        """Convert audio to s16le PCM mono format"""
        try:
            if not audio_bytes or len(audio_bytes) < 2:
                return b''

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            if audio_array.size == 0:
                return b''

            if len(audio_array) % 2 == 0:
                try:
                    stereo = audio_array.reshape(-1, 2)
                    mono = stereo.mean(axis=1).astype(np.int16)
                except ValueError:
                    mono = audio_array
            else:
                mono = audio_array

            return mono.tobytes()
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return b''

    async def handle_audio_input(self, audio_data: bytes) -> None:
        """Handle incoming audio data from the user"""
        if not self._session or self._closing:
            return

        if self.current_utterance and not self.current_utterance.is_interruptible:
            return

        try:
            converted_audio = self._convert_to_s16le_pcm_mono(audio_data)
            
            if converted_audio and len(converted_audio) >= 2:
                try:
                    self._audio_queue.put_nowait(converted_audio)
                except asyncio.QueueFull:
                    try:
                        self._audio_queue.get_nowait()
                        self._audio_queue.put_nowait(converted_audio)
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error queuing audio: {e}")

    async def interrupt(self) -> None:
        """Interrupt current response"""
        if not self._session or self._closing:
            return
        
        if self.current_utterance and not self.current_utterance.is_interruptible:
            return
        
        try:
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            clear_msg = {"type": "playbackClearBuffer"}
            await self._session.websocket.send(json.dumps(clear_msg))
            
            self.emit("agent_speech_ended", {})
            await realtime_metrics_collector.set_interrupted()
            
            if self.audio_track:
                self.audio_track.interrupt()
        except Exception as e:
            logger.error(f"Interrupt error: {e}")

    async def _cleanup_session(self, session: UltravoxSession) -> None:
        """Clean up a session's resources"""
        for task in session.tasks:
            if not task.done():
                task.cancel()

        try:
            if session.websocket:
                is_closed = getattr(session.websocket, 'closed', True)
                if not is_closed:
                    await session.websocket.close()
        except Exception as e:
            logger.error(f"Error closing websocket: {e}")

    async def aclose(self) -> None:
        """Clean up all resources"""
        if self._closing:
            return

        self._closing = True
        self._session_should_close.set()

        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                await asyncio.wait_for(self._main_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if self._session:
            await self._cleanup_session(self._session)
            self._session = None

        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        if hasattr(self.audio_track, "cleanup") and self.audio_track:
            try:
                await self.audio_track.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up audio track: {e}")

    def _convert_tools_to_ultravox_format(self, tools):
        """
        Ultravox-compatible tool format.
        IMPORTANT:
        - client MUST be {}
        - parameters must be in dynamicParameters list
        """

        formatted_tools = []

        for tool in tools:
            if not is_function_tool(tool):
                continue

            info = get_tool_info(tool)
            openai_schema = build_openai_schema(tool)
            parameters = openai_schema.get("parameters", {})
            properties = parameters.get("properties", {})
            required = parameters.get("required", [])

            dynamic_parameters = []
            for name, schema in properties.items():
                if "title" in schema:
                    del schema["title"]
                
                dynamic_parameters.append({
                    "name": name,
                    "location": "PARAMETER_LOCATION_BODY",
                    "schema": schema,
                    "required": name in required
                })

            ultravox_tool = {
                "temporaryTool": {
                    "modelToolName": info.name,
                    "description": info.description or "",
                    "dynamicParameters": dynamic_parameters,
                    "precomputable": True,
                    "client": {}
                }
            }

            formatted_tools.append(ultravox_tool)

        return formatted_tools