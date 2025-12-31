from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
import base64
from typing import Any, Dict, Optional, Literal, List
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from scipy import signal
from dotenv import load_dotenv

from videosdk.agents import (
    FunctionTool,
    is_function_tool,
    get_tool_info,
    build_openai_schema, 
    CustomAudioStreamTrack,
    RealtimeBaseModel,
    global_event_emitter,
    Agent,
    realtime_metrics_collector,
)

load_dotenv()
logger = logging.getLogger(__name__)

XAI_BASE_URL = "wss://api.x.ai/v1/realtime"
DEFAULT_XAI_VOICE = "Ara"
DEFAULT_SAMPLE_RATE = 24000
INPUT_SAMPLE_RATE = 48000 

XAIEventTypes = Literal["user_speech_started", "text_response", "error"]
XAIVoice = Literal["Ara", "Rex", "Sal", "Eve", "Leo"]

@dataclass
class XAITurnDetection:
    type: Literal["server_vad"] | None = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 200

@dataclass
class XAIRealtimeConfig:
    """Configuration for the xAI (Grok) Realtime API
    
    Args:
        voice: The voice identifier. Options: 'Ara', 'Rex', 'Sal', 'Eve', 'Leo'. Default: 'Ara'
        instructions: System instructions for the agent.
        turn_detection: Configuration for server-side VAD.
        tools: List of specific xAI tools (e.g., web_search, x_search). 
               Standard function tools are handled via the Agent class.
    """
    voice: XAIVoice = DEFAULT_XAI_VOICE
    instructions: str | None = None
    turn_detection: XAITurnDetection | None = field(default_factory=XAITurnDetection)
    modalities: List[str] = field(default_factory=lambda: ["text", "audio"])
    enable_web_search: bool = False
    enable_x_search: bool = False
    allowed_x_handles: List[str] | None = None
    collection_id: str | None = None
    max_num_results: int = 10

@dataclass
class XAISession:
    """Represents an xAI WebSocket session"""
    ws: aiohttp.ClientWebSocketResponse
    msg_queue: asyncio.Queue[Dict[str, Any]]
    tasks: list[asyncio.Task]

class XAIRealtime(RealtimeBaseModel[XAIEventTypes]):
    """xAI's Grok realtime model implementation"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "grok-4-1-fast-non-reasoning",
        config: XAIRealtimeConfig | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.base_url = base_url or XAI_BASE_URL
        
        if not self.api_key:
            self.emit("error", "XAI_API_KEY is required")
            raise ValueError("XAI_API_KEY is required")

        self.config: XAIRealtimeConfig = config or XAIRealtimeConfig()
        
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._session: Optional[XAISession] = None
        self._closing = False
        self._instructions: str = "You are a helpful assistant."
        self._tools: List[FunctionTool] = []
        self._formatted_tools: List[Dict[str, Any]] = []
        
        self.loop = None
        self.audio_track: Optional[CustomAudioStreamTrack] = None
        self.input_sample_rate = INPUT_SAMPLE_RATE
        self.target_sample_rate = DEFAULT_SAMPLE_RATE
        self._agent_speaking = False
        self._current_response_id: str | None = None
        self._is_configured = False
        self._session_ready = asyncio.Event()
        self._has_unprocessed_tool_outputs = False
        self._generated_text_in_current_response = False

    def set_agent(self, agent: Agent) -> None:
        if agent.instructions:
            self._instructions = agent.instructions
        self._tools = agent.tools
        self._formatted_tools = self._format_tools_for_session(self._tools)

    async def connect(self) -> None:
        """Establish WebSocket connection to xAI"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self._session = await self._create_session(self.base_url, headers)
        await self._handle_websocket(self._session)
        
        if self.audio_track:
            from fractions import Fraction
            self.audio_track.sample_rate = self.target_sample_rate
            self.audio_track.time_base_fraction = Fraction(1, self.target_sample_rate)
            self.audio_track.samples = int(0.02 * self.target_sample_rate)
            self.audio_track.chunk_size = int(self.audio_track.samples * getattr(self.audio_track, "channels", 1) * getattr(self.audio_track, "sample_width", 2))
        
        try:
            await asyncio.wait_for(self._session_ready.wait(), timeout=10.0)
            logger.info("xAI session configuration complete")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for xAI session configuration")

    async def _create_session(self, url: str, headers: dict) -> XAISession:
        if not self._http_session:
            self._http_session = aiohttp.ClientSession()
            
        try:
            ws = await self._http_session.ws_connect(
                url,
                headers=headers,
                autoping=True,
                heartbeat=10,
                timeout=30,
            )
        except Exception as e:
            self.emit("error", f"Connection failed: {e}")
            raise

        msg_queue: asyncio.Queue = asyncio.Queue()
        tasks: list[asyncio.Task] = []
        self._closing = False

        return XAISession(ws=ws, msg_queue=msg_queue, tasks=tasks)

    async def _send_initial_config(self) -> None:
        """Send session.update to configure voice and audio"""
        if not self._session:
            return

        tools_config = []
        
        if self._formatted_tools:
            tools_config.extend(self._formatted_tools)
            
        if self.config.enable_web_search:
            tools_config.append({"type": "web_search"})
        
        if self.config.enable_x_search or self.config.allowed_x_handles:
            x_search_config = {"type": "x_search"}
            if self.config.allowed_x_handles:
                logger.info(f"Allowed xAI handles: {self.config.allowed_x_handles}")
                x_search_config["allowed_x_handles"] = self.config.allowed_x_handles
            tools_config.append(x_search_config)

        if self.config.collection_id:
            tools_config.append({
                "type": "file_search",
                "vector_store_ids": [self.config.collection_id],
                "max_num_results": self.config.max_num_results,
            })

        session_update = {
            "type": "session.update",
            "session": {
                "instructions": self._instructions,
                "voice": self.config.voice,
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.target_sample_rate
                        }
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.target_sample_rate
                        }
                    }
                },
                "turn_detection": {
                    "type": "server_vad"
                },
                "tools": tools_config if tools_config else None
            }
        }

        await self.send_event(session_update)

    async def handle_audio_input(self, audio_data: bytes) -> None:
        """Process incoming audio: Resample 48k -> target (usually 24k) and send to xAI"""
        if not self._session or self._closing:
            return

        if "audio" not in self.config.modalities:
            return

        if self.current_utterance and not self.current_utterance.is_interruptible:
            return

        try:
            raw_audio = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(raw_audio) >= 1920 and len(raw_audio) % 2 == 0:
                raw_audio = (raw_audio.reshape(-1, 2).astype(np.int32).mean(axis=1)).astype(np.int16)
            
            if self.input_sample_rate != self.target_sample_rate:
                num_samples = int(len(raw_audio) * self.target_sample_rate / self.input_sample_rate)
                float_audio = raw_audio.astype(np.float32)
                resampled_audio = signal.resample(float_audio, num_samples).astype(np.int16)
            else:
                resampled_audio = raw_audio

            base64_audio = base64.b64encode(resampled_audio.tobytes()).decode("utf-8")
            
            if not hasattr(self, "_audio_log_counter"):
                self._audio_log_counter = 0
            self._audio_log_counter += 1
            if self._audio_log_counter % 100 == 0:
                rms = np.sqrt(np.mean(resampled_audio.astype(np.float32)**2))
                logger.info(f"xAI Audio: Sent chunk {self._audio_log_counter}, samples={len(resampled_audio)}, RMS={rms:.2f}")

            await self.send_event({
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            })
        except Exception as e:
            logger.error(f"Error processing audio input: {e}")

    async def handle_video_input(self, video_data: Any) -> None:
        """xAI Voice API currently does not document direct video stream support in this endpoint."""
        pass

    async def send_message(self, message: str) -> None:
        """Send text message to trigger audio response"""
        await self.send_event({"type": "input_audio_buffer.commit"})
        await self.send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text", 
                    "text": message
                }],
            }
        })
        await self.create_response()

    async def create_response(self) -> None:
        """Trigger a response from the model"""
        await self.send_event({
            "type": "response.create"
        })

    async def send_text_message(self, message: str) -> None:
        """Send text message (same as send_message for xAI flow)"""
        await self.send_message(message)

    async def interrupt(self) -> None:
        """Interrupt current generation"""
        if self._session and not self._closing:
            if self.current_utterance and not self.current_utterance.is_interruptible:
                return
            await realtime_metrics_collector.set_interrupted()
            
        if self.audio_track:
            self.audio_track.interrupt()
        
        if self._agent_speaking:
            self.emit("agent_speech_ended", {})
            self._agent_speaking = False

    async def _handle_websocket(self, session: XAISession) -> None:
        session.tasks.extend([
            asyncio.create_task(self._send_loop(session), name="xai_send"),
            asyncio.create_task(self._receive_loop(session), name="xai_recv"),
        ])

    async def _send_loop(self, session: XAISession) -> None:
        try:
            while not self._closing:
                msg = await session.msg_queue.get()
                if isinstance(msg, dict):
                    logger.debug(f"Sending xAI event: {msg.get('type')}")
                    await session.ws.send_json(msg)
                else:
                    await session.ws.send_str(str(msg))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"xAI Send loop error: {e}")
            self.emit("error", f"Send loop error: {e}")

    async def _receive_loop(self, session: XAISession) -> None:
        try:
            while not self._closing:
                msg = await session.ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_event(data)
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("xAI WebSocket closed")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"xAI WebSocket error: {session.ws.exception()}")
                    break
        except Exception as e:
            logger.error(f"xAI Receive loop error: {e}")
            self.emit("error", f"Receive loop error: {e}")
        finally:
            logger.info("xAI Receive loop finished, closing session")
            await self.aclose()

    async def _handle_event(self, data: dict) -> None:
        event_type = data.get("type")
        try:
            if event_type == "conversation.created":
                if not self._is_configured:
                    await self._send_initial_config()
                    self._is_configured = True
            elif event_type == "input_audio_buffer.speech_started":
                await self._handle_speech_started()
            elif event_type == "input_audio_buffer.speech_stopped":
                await self._handle_speech_stopped()
            elif event_type == "session.updated":
                logger.info("xAI Session updated successfully")
                self._session_ready.set()
            elif event_type == "response.created":
                logger.info(f"Response created: {data.get('response', {}).get('id')}")
                self._generated_text_in_current_response = False
            elif event_type == "response.output_item.added":
                logger.info(f"Output item added: {data.get('item', {}).get('id')}")
            elif event_type == "response.output_audio.delta":
                await self._handle_audio_delta(data)
            elif event_type == "response.output_audio_transcript.delta":
                await self._handle_transcript_delta(data)
            elif event_type == "response.output_audio_transcript.done":
                 await self._handle_transcript_done(data)
            elif event_type == "conversation.item.input_audio_transcription.completed":
                await self._handle_input_audio_transcription_completed(data)
            elif event_type == "response.function_call_arguments.done":
                await self._handle_function_call(data)
            elif event_type == "response.done":
                await self._handle_response_done()
            elif event_type == "error":
                logger.error(f"xAI Error: {data}")
                
        except Exception as e:
            logger.error(f"Error handling event {event_type}: {e}")
            traceback.print_exc()

    async def _handle_speech_started(self) -> None:
        logger.info("xAI User speech started")
        self.emit("user_speech_started", {"type": "done"})
        await realtime_metrics_collector.set_user_speech_start()
        
        if self.current_utterance and not self.current_utterance.is_interruptible:
            return
            
        await self.interrupt()

    async def _handle_speech_stopped(self) -> None:
        logger.info("xAI User speech stopped")
        await realtime_metrics_collector.set_user_speech_end()
        self.emit("user_speech_ended", {})

    async def _handle_audio_delta(self, data: dict) -> None:
        delta = data.get("delta")
        if not delta:
            return

        if not self._agent_speaking:
            await realtime_metrics_collector.set_agent_speech_start()
            self._agent_speaking = True
            self.emit("agent_speech_started", {})

        if self.audio_track and self.loop:
            audio_bytes = base64.b64decode(delta)
            asyncio.create_task(self.audio_track.add_new_bytes(audio_bytes))

    async def _handle_transcript_delta(self, data: dict) -> None:
        delta = data.get("delta", "")
        if delta:
            self._generated_text_in_current_response = True
            if not hasattr(self, "_current_transcript"):
                self._current_transcript = ""
            self._current_transcript += delta

    async def _handle_transcript_done(self, data: dict) -> None:
        pass

    async def _handle_input_audio_transcription_completed(self, data: dict) -> None:
        """Handle input audio transcription completion for user transcript"""
        transcript = data.get("transcript", "")
        if transcript:
            logger.info(f"xAI User transcript: {transcript}")
            await realtime_metrics_collector.set_user_transcript(transcript)
            try:
                self.emit(
                    "realtime_model_transcription",
                    {"role": "user", "text": transcript, "is_final": True},
                )
            except Exception:
                pass

    async def _handle_response_done(self) -> None:
        if hasattr(self, "_current_transcript") and self._current_transcript:
             logger.info(f"xAI Agent response: {self._current_transcript}")
             await realtime_metrics_collector.set_agent_response(self._current_transcript)
             global_event_emitter.emit(
                "text_response",
                {"text": self._current_transcript, "type": "done"},
            )
             self._current_transcript = ""

        logger.info("xAI Agent speech ended")
        self.emit("agent_speech_ended", {})
        await realtime_metrics_collector.set_agent_speech_end(timeout=1.0)
        self._agent_speaking = False

        if self._has_unprocessed_tool_outputs and not self._generated_text_in_current_response:
            logger.info("xAI: Triggering follow-up response for tool outputs")
            self._has_unprocessed_tool_outputs = False
            await self.create_response()
        else:
            self._has_unprocessed_tool_outputs = False

    async def _handle_function_call(self, data: dict) -> None:
        """Handle tool execution flow for xAI"""
        name = data.get("name")
        call_id = data.get("call_id")
        args_str = data.get("arguments")
        
        if not name or not args_str:
            return

        try:
            arguments = json.loads(args_str)
            logger.info(f"Executing tool: {name} with args: {arguments}")
            await realtime_metrics_collector.add_tool_call(name)
            result = None
            found = False
            for tool in self._tools:
                info = get_tool_info(tool)
                if info.name == name:
                    result = await tool(**arguments)
                    found = True
                    break
            
            if not found:
                logger.warning(f"Tool {name} not found")
                result = {"error": "Tool not found"}

            await self.send_event({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result)
                }
            })

            if found:
                self._has_unprocessed_tool_outputs = True

        except Exception as e:
            logger.error(f"Error executing function {name}: {e}")

    async def send_event(self, event: Dict[str, Any]) -> None:
        if self._session and not self._closing:
            await self._session.msg_queue.put(event)

    def _format_tools_for_session(self, tools: List[FunctionTool]) -> List[Dict[str, Any]]:
        """Format tools using OpenAI schema builder (xAI is compatible)"""
        formatted = []
        for tool in tools:
            if is_function_tool(tool):
                try:
                    schema = build_openai_schema(tool)
                    formatted.append(schema)
                except Exception as e:
                    logger.error(f"Failed to format tool {tool}: {e}")
        return formatted

    async def aclose(self) -> None:
        """Cleanup resources"""
        if self._closing:
            return
        
        self._closing = True
        
        if self._session:
            for task in self._session.tasks:
                if not task.done():
                    task.cancel()
            
            if not self._session.ws.closed:
                await self._session.ws.close()
                
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

        if hasattr(self.audio_track, "cleanup") and self.audio_track:
            await self.audio_track.cleanup()