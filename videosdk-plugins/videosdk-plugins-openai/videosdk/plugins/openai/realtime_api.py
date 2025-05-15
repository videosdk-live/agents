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
import traceback
from videosdk.agents import (
    FunctionTool,
    is_function_tool,
    get_tool_info,
    build_openai_schema,
    CustomAudioStreamTrack,
    ToolChoice,
    RealtimeBaseModel
)

load_dotenv()
from openai.types.beta.realtime.session import InputAudioTranscription, TurnDetection

OPENAI_BASE_URL = "https://api.openai.com/v1"
SAMPLE_RATE = 24000
NUM_CHANNELS = 1

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

OpenAIEventTypes = Literal[
    "instructions_updated",
    "tools_updated"
]
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
    turn_detection: TurnDetection | None = field(default_factory=lambda: DEFAULT_TURN_DETECTION)
    input_audio_transcription: InputAudioTranscription | None = field(default_factory=lambda: DEFAULT_INPUT_AUDIO_TRANSCRIPTION)
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
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._session: Optional[OpenAISession] = None
        self._closing = False
        self._instructions: Optional[str] = None
        self._tools: Optional[List[FunctionTool]] = []
        self.loop = None
        self.audio_track: Optional[CustomAudioStreamTrack] = None
        self._formatted_tools: Optional[List[Dict[str, Any]]] = None
        self.config: OpenAIRealtimeConfig = config or OpenAIRealtimeConfig()
        self.on("instructions_updated", self._handle_instructions_updated)
        self.on("tools_updated", self._handle_tools_updated) 
    
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
        if self._session and not self._closing:
            base64_audio_data = base64.b64encode(audio_data).decode("utf-8")
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio_data
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
        ws = await http_session.ws_connect(url, headers=headers, autoping=True, heartbeat=10, autoclose=False, timeout=30)
        msg_queue: asyncio.Queue = asyncio.Queue()
        tasks: list[asyncio.Task] = []
        
        self._closing = False
        
        return OpenAISession(ws=ws, msg_queue=msg_queue, tasks=tasks)
    
    async def send_message(self, message: str) -> None:
        """Send a message to the OpenAI realtime API"""
        await self.send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Repeat the user's exact message back to them:" + message + "DO NOT ADD ANYTHING ELSE",
                    }
                ]
            }
        })
        await self.create_response()
        
    async def create_response(self) -> None:
        """Create a response to the OpenAI realtime API"""
        if not self._session:
            raise RuntimeError("No active WebSocket session")
            
        # Create response event
        response_event = {
            "type": "response.create",
            "event_id": str(uuid.uuid4()),
            "response": {
                "instructions": self._instructions, 
                "metadata": {
                    "client_event_id": str(uuid.uuid4()) 
                }
            }
        }
        
        # Send the event through our message queue
        await self.send_event(response_event)
        
        # session_update = {
        #     "type": "session.update",
        #     "session": {
        #         "instructions": self._instructions
        #     }
        # }
        
        # await self.send_event(session_update)

    async def _handle_websocket(self, session: OpenAISession) -> None:
        """Start WebSocket send/receive tasks"""
        session.tasks.extend([
            asyncio.create_task(self._send_loop(session), name="send_loop"),
            asyncio.create_task(self._receive_loop(session), name="receive_loop")
        ])

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
                    print("WebSocket closed with reason:", msg.extra)
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print("WebSocket error:", msg.data)
                    break
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(json.loads(msg.data))
        except Exception as e:
            print("WebSocket receive error:", str(e))
        finally:
            await self._cleanup_session(session)

    async def _handle_message(self, data: dict) -> None:
        """Handle incoming WebSocket messages"""
        try:
            event_type = data.get('type')

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
                
            elif event_type == "response.audio.delta":
                await self._handle_audio_delta(data)
                
            elif event_type == "response.audio_transcript.delta":
                await self._handle_transcript_delta(data)
                
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

        except Exception as e:
            self.emit_error(f"Error handling event {event_type}: {str(e)}")

    async def _handle_speech_started(self, data: dict) -> None:
        """Handle speech detection start"""
        await self.interrupt()
        self.audio_track.interrupt()

    async def _handle_speech_stopped(self, data: dict) -> None:
        """Handle speech detection end"""

    async def _handle_response_created(self, data: dict) -> None:
        """Handle initial response creation"""
        response_id = data.get("response", {}).get("id")
        
        self.emit("response_created", {"response_id": response_id})

    async def _handle_output_item_added(self, data: dict) -> None:
        """Handle new output item addition"""
    
    async def _handle_output_item_done(self, data: dict) -> None:
        """Handle output item done"""
        try:
            item = data.get("item", {})
            if item.get("type") == "function_call" and item.get("status") == "completed":
                name = item.get("name")
                arguments = json.loads(item.get("arguments", "{}"))
                
                if name and self._tools:
                    for tool in self._tools:
                        tool_info = get_tool_info(tool)
                        if tool_info.name == name:
                            try:
                                result = await tool(**arguments)
                                await self.send_event({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": item.get("call_id"),
                                        "output": json.dumps(result)
                                    }
                                })
                                
                                await self.send_event({
                                    "type": "response.create",
                                    "event_id": str(uuid.uuid4()),
                                    "response": {
                                        "instructions": self._instructions,
                                        "metadata": {
                                            "client_event_id": str(uuid.uuid4())
                                        }
                                    }
                                })
                                
                            except Exception as e:
                                print(f"Error executing function {name}: {e}")
                            break
        except Exception as e:
            print(f"Error handling output item done: {e}")

    async def _handle_content_part_added(self, data: dict) -> None:
        """Handle new content part"""

    async def _handle_audio_delta(self, data: dict) -> None:
        """Handle audio chunk"""
        try:
            base64_audio_data = base64.b64decode(data.get("delta"))
            if base64_audio_data:
                if self.audio_track and self.loop:
                    self.loop.create_task(self.audio_track.add_new_bytes(base64_audio_data))
        except Exception as e:
            print(f"[ERROR] Error handling audio delta: {e}")
            traceback.print_exc()
    
    async def interrupt(self) -> None:
        """Interrupt the current response and flush audio"""
        if self._session and not self._closing:
            cancel_event = {
                "type": "response.cancel",
                "event_id": str(uuid.uuid4())
            }
            await self.send_event(cancel_event)
            
    async def _handle_transcript_delta(self, data: dict) -> None:
        """Handle transcript chunk"""
    
    async def _handle_input_audio_transcription_completed(self, data: dict) -> None:
        """Handle input audio transcription completion"""
        # if "transcript" in data:
            # self.emit("transcription_event", {"text": data["transcript"]})

    async def _handle_response_done(self, data: dict) -> None:
        """Handle response completion"""
    
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

        # Close WebSocket
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

        session_update = {
            "type": "session.update",
            "session": {
                "model": self.model,
                "voice": self.config.voice,
                "instructions": self._instructions or  "You are a helpful voice assistant that can answer questions and help with tasks.",
                "temperature": self.config.temperature,
                "turn_detection": self.config.turn_detection.model_dump(
                    by_alias=True,
                    exclude_unset=True,
                    exclude_defaults=True,
                ),
                "input_audio_transcription": self.config.input_audio_transcription.model_dump(
                    by_alias=True,
                    exclude_unset=True,
                    exclude_defaults=True,
                ),
                "tool_choice": self.config.tool_choice,
                "tools": self._formatted_tools or [],
                "modalities": self.config.modalities,
                "input_audio_format": DEFAULT_INPUT_AUDIO_FORMAT,
                "output_audio_format": DEFAULT_OUTPUT_AUDIO_FORMAT,
                "max_response_output_tokens": "inf"
            }
        }
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
        new_url = urlunparse((parsed_url.scheme, parsed_url.netloc, path, "", new_query, ""))

        return new_url
    
    def _handle_instructions_updated(self, data: Dict[str, Any]) -> None:
        """Handle instructions_updated event"""
        self._instructions = data.get("instructions")

    def _format_tools_for_session(self, tools: List[FunctionTool]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI session update"""
        oai_tools = []
        for tool in tools:
            if not is_function_tool(tool):
                continue
                
            try:
                tool_schema = build_openai_schema(tool)
                oai_tools.append(tool_schema)
            except Exception as e:
                print(f"Failed to format tool {tool}: {e}")
                continue
                
        return oai_tools

    def _handle_tools_updated(self, data: Dict[str, Any]) -> None:
        """Handle tools_updated event"""
        tools = data.get("tools", [])
        self._tools = tools
        self.tools_formatted = self._format_tools_for_session(tools)
        self._formatted_tools = self.tools_formatted