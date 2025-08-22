from __future__ import annotations

import os
import asyncio
import base64
import json
import uuid
from typing import Optional, Literal, List, Dict, Any
from dataclasses import dataclass
import numpy as np
from scipy import signal


from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
)
from aws_sdk_bedrock_runtime.config import (
    Config,
    HTTPAuthSchemeResolver,
    SigV4AuthScheme,
)
from smithy_aws_core.credentials_resolvers.environment import (
    EnvironmentCredentialsResolver,
)

from videosdk.agents import (
    Agent,
    RealtimeBaseModel,
    build_nova_sonic_schema,
    get_tool_info,
    is_function_tool,
    FunctionTool,
    realtime_metrics_collector,
)

from videosdk.agents import realtime_metrics_collector

NOVA_INPUT_SAMPLE_RATE = 16000
NOVA_OUTPUT_SAMPLE_RATE = 24000

# Event types
NovaSonicEventTypes = Literal["user_speech_started", "text_response", "error"]


@dataclass
class NovaSonicConfig:
    """Configuration for Nova Sonic API

    Args:
        model_id: The Nova Sonic model ID to use. Default is 'amazon.nova-sonic-v1:0'
        voice: Voice ID for audio output. Default is 'matthew'
        temperature: Controls randomness in responses. Default is 0.7
        top_p: Nucleus sampling parameter. Default is 0.9
        max_tokens: Maximum tokens in response. Default is 1024
    """

    model_id: str = "amazon.nova-sonic-v1:0"
    voice: str = "tiffany"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024


class NovaSonicRealtime(RealtimeBaseModel[NovaSonicEventTypes]):
    """Nova Sonic's realtime model implementation"""

    def __init__(
        self,
        *,
        model: str,
        config: NovaSonicConfig | None = None,
        region: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        """
        Initialize Nova Sonic realtime model.

        Args:
            model: The Nova Sonic model identifier
            config: Optional configuration object for customizing model behavior
            region: AWS region for Bedrock
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
        """
        super().__init__()
        self.model = model
        self.config = config or NovaSonicConfig()
        self.region = region or os.getenv("AWS_DEFAULT_REGION")
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )

        if not self.region:
            self.emit(
                "error",
                "AWS region is required (pass as parameter or set AWS_DEFAULT_REGIONenvironment variable)",
            )
            raise ValueError(
                "AWS region is required (pass as parameter or set AWS_DEFAULT_REGIONenvironment variable)"
            )
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            self.emit(
                "error",
                "AWS credentials required (pass as parameters or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment variables)",
            )
            raise ValueError(
                "AWS credentials required (pass as parameters or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment variables)"
            )

        self.bedrock_client = None
        self.stream = None
        self._closing = False
        self._instructions = "You are a helpful assistant. The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation. Keep your responses short, generally two or three sentences for chatty scenarios."
        self._tools = []
        self.tools_formatted = []
        self.loop = asyncio.get_event_loop()
        self.audio_track = None
        self.prompt_name = str(uuid.uuid4())
        self.system_content_name = f"system_{str(uuid.uuid4())}"
        self.audio_content_name = f"audio_{str(uuid.uuid4())}"
        self.is_active = False
        self.response_task = None
        self._agent_speaking = False
        self._initialize_bedrock_client()
        self.input_sample_rate = 48000
        self.target_sample_rate = 16000

    def set_agent(self, agent: Agent) -> None:
        self._instructions = agent.instructions
        self._tools = agent.tools
        self.tools_formatted = [
            build_nova_sonic_schema(tool)
            for tool in self._tools
            if is_function_tool(tool)
        ]
        self.formatted_tools = self.tools_formatted

    def _initialize_bedrock_client(self):
        """Initialize the Bedrock client with manual credential handling"""
        try:
            if self.region:
                os.environ["AWS_REGION"] = self.region
            if self.aws_access_key_id:
                os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
            if self.aws_secret_access_key:
                os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key

            config = Config(
                endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
                region=self.region,
                aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
                http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
                http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
            )
            self.bedrock_client = BedrockRuntimeClient(config=config)

        except Exception as e:
            self.emit("error", f"Error initializing Bedrock client: {e}")
            raise

    async def connect(self) -> None:
        """Initialize connection to Nova Sonic"""
        if self.is_active:
            await self._cleanup()

        self._closing = False

        try:
            self.loop = asyncio.get_event_loop()
            self.stream = (
                await self.bedrock_client.invoke_model_with_bidirectional_stream(
                    InvokeModelWithBidirectionalStreamOperationInput(
                        model_id=self.config.model_id
                    )
                )
            )
            self.is_active = True

            session_start_payload = {
                "event": {
                    "sessionStart": {
                        "inferenceConfiguration": {
                            "maxTokens": self.config.max_tokens,
                            "topP": self.config.top_p,
                            "temperature": self.config.temperature,
                        }
                    }
                }
            }
            await self._send_event(json.dumps(session_start_payload))

            prompt_start_event_dict = {
                "event": {
                    "promptStart": {
                        "promptName": self.prompt_name,
                        "textOutputConfiguration": {"mediaType": "text/plain"},
                        "audioOutputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": NOVA_OUTPUT_SAMPLE_RATE,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "voiceId": self.config.voice,
                            "encoding": "base64",
                            "audioType": "SPEECH",
                        },
                    }
                }
            }

            if self.tools_formatted:
                prompt_start_event_dict["event"]["promptStart"][
                    "toolUseOutputConfiguration"
                ] = {"mediaType": "application/json"}
                prompt_start_event_dict["event"]["promptStart"]["toolConfiguration"] = {
                    "tools": self.tools_formatted
                }

            await self._send_event(json.dumps(prompt_start_event_dict))

            system_content_start_payload = {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.system_content_name,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "SYSTEM",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            }
            await self._send_event(json.dumps(system_content_start_payload))

            system_instructions = (
                self._instructions
                or "You are a helpful voice assistant. Keep your responses short and conversational."
            )
            text_input_payload = {
                "event": {
                    "textInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.system_content_name,
                        "content": system_instructions,
                    }
                }
            }
            await self._send_event(json.dumps(text_input_payload))

            content_end_payload = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.system_content_name,
                    }
                }
            }
            await self._send_event(json.dumps(content_end_payload))

            self.response_task = asyncio.create_task(self._process_responses())

            await self._start_audio_input()

        except Exception as e:
            await self._cleanup()
            raise

    async def _send_event(self, event_json: str):
        """Send an event to the bidirectional stream"""
        if not self.is_active or not self.stream:
            return

        try:
            event = InvokeModelWithBidirectionalStreamInputChunk(
                value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
            )

            await self.stream.input_stream.send(event)

        except Exception as e:
            self.emit("error", f"Error sending event: {e}")

    async def _start_audio_input(self):
        """Start audio input stream"""
        if not self.is_active:
            return

        audio_content_start_payload = {
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": NOVA_INPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64",
                    },
                }
            }
        }
        await self._send_event(json.dumps(audio_content_start_payload))

    async def handle_audio_input(self, audio_data: bytes) -> None:
        """Handle incoming 48kHz audio from VideoSDK"""
        if not self.is_active or self._closing:
            return

        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            if len(audio_array) % 2 == 0:
                audio_array = audio_array.reshape(-1, 2)
                audio_array = np.mean(audio_array, axis=1).astype(np.int16)

            target_length = int(
                len(audio_array) * self.target_sample_rate / self.input_sample_rate
            )
            resampled_float = signal.resample(
                audio_array.astype(np.float32), target_length
            )

            resampled_int16 = np.clip(resampled_float, -32768, 32767).astype(np.int16)
            resampled_bytes = resampled_int16.tobytes()

            encoded_audio = base64.b64encode(resampled_bytes).decode("utf-8")

            audio_event_payload = {
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "content": encoded_audio,
                    }
                }
            }

            await self._send_event(json.dumps(audio_event_payload))

        except Exception as e:
            self.emit("error", f"Resampling error: {e}")

    async def _process_responses(self):
        """Process responses from the bidirectional stream"""
        try:
            while self.is_active and not self._closing:
                try:
                    output = await self.stream.await_output()
                    result = await output[1].receive()

                    if result.value and result.value.bytes_:
                        response_data = result.value.bytes_.decode("utf-8")

                        try:
                            json_data = json.loads(response_data)

                            if "event" in json_data:
                                event_keys = list(json_data["event"].keys())

                                if "completionStart" in json_data["event"]:
                                    completion_start = json_data["event"][
                                        "completionStart"
                                    ]

                                elif "contentStart" in json_data["event"]:
                                    content_start = json_data["event"]["contentStart"]

                                    if "additionalModelFields" in content_start:
                                        try:
                                            additional_fields = json.loads(
                                                content_start["additionalModelFields"]
                                            )
                                        except (json.JSONDecodeError, KeyError) as e:
                                            self.emit(
                                                "error",
                                                f"Error parsing additionalModelFields: {e}",
                                            )
                                elif "textOutput" in json_data["event"]:
                                    text_output = json_data["event"]["textOutput"]
                                    if "content" in text_output:
                                        transcript = text_output["content"]
                                        role = text_output.get("role", "UNKNOWN")
                                        if role == "USER":
                                            await realtime_metrics_collector.set_user_speech_start()
                                            await realtime_metrics_collector.set_user_transcript(
                                                transcript
                                            )
                                            await realtime_metrics_collector.set_user_speech_end()
                                            self._safe_emit(
                                                "user_speech_started", {"type": "done"}
                                            )
                                            try:
                                                await self.emit(
                                                    "realtime_model_transcription",
                                                    {
                                                        "role": "user",
                                                        "text": transcript,
                                                        "is_final": True,
                                                    },
                                                )
                                            except Exception:
                                                pass
                                        elif role == "ASSISTANT":
                                            skip_emit = False
                                            try:
                                                parsed = json.loads(transcript)
                                                if (
                                                    isinstance(parsed, dict)
                                                    and parsed.get("interrupted")
                                                    is True
                                                ):
                                                    skip_emit = True
                                            except Exception:
                                                pass
                                            if not skip_emit:
                                                await realtime_metrics_collector.set_agent_response(
                                                    transcript
                                                )
                                                try:
                                                    await self.emit(
                                                        "realtime_model_transcription",
                                                        {
                                                            "role": "agent",
                                                            "text": transcript,
                                                            "is_final": True,
                                                        },
                                                    )
                                                except Exception:
                                                    pass

                                elif "audioOutput" in json_data["event"]:
                                    audio_output = json_data["event"]["audioOutput"]
                                    if "content" not in audio_output:
                                        continue

                                    audio_content = audio_output["content"]
                                    if not audio_content:
                                        continue

                                    try:
                                        audio_bytes = base64.b64decode(audio_content)
                                        if not self._agent_speaking:
                                            await realtime_metrics_collector.set_agent_speech_start()
                                            self._agent_speaking = True

                                        if (
                                            self.audio_track
                                            and self.loop
                                            and not self._closing
                                        ):
                                            asyncio.create_task(
                                                self.audio_track.add_new_bytes(
                                                    audio_bytes
                                                )
                                            )

                                    except Exception as e:
                                        self.emit(
                                            "error", f"AUDIO PROCESSING ERROR: {e}"
                                        )

                                elif "contentEnd" in json_data["event"]:
                                    content_end = json_data["event"]["contentEnd"]
                                    if (
                                        content_end.get("stopReason", "") == "END_TURN"
                                        and self._agent_speaking
                                    ):
                                        await realtime_metrics_collector.set_agent_speech_end(
                                            timeout=1.0
                                        )
                                        self._agent_speaking = False

                                elif "usageEvent" in json_data["event"]:
                                    pass

                                elif "toolUse" in json_data["event"]:
                                    tool_use = json_data["event"]["toolUse"]
                                    await realtime_metrics_collector.add_tool_call(
                                        tool_use["toolName"]
                                    )
                                    asyncio.create_task(
                                        self._execute_tool_and_send_result(tool_use)
                                    )

                                elif "completionEnd" in json_data["event"]:
                                    completion_end = json_data["event"]["completionEnd"]
                                    print(
                                        f"Nova completionEnd received: {json.dumps(completion_end, indent=2)}"
                                    )
                                    await realtime_metrics_collector.set_agent_speech_end(
                                        timeout=1.0
                                    )
                                    self._agent_speaking = False

                                else:
                                    print(
                                        f"Unhandled event type from Nova: {event_keys} - {json.dumps(json_data['event'], indent=2)}"
                                    )
                            else:
                                print(f"Non-event response: {json_data}")

                        except json.JSONDecodeError as e:
                            self.emit("error", f"Failed to parse response: {e}")
                            self.emit("error", f"Raw data: {response_data[:200]}...")

                except Exception as e:
                    self.emit("error", f"Error processing response: {e}")
                    if not self.is_active or self._closing:
                        break

        except Exception as e:
            print(f"Unexpected error in response processing: {e}")

    async def send_message(self, message: str) -> None:
        """Send a text message to the model"""
        if not self.is_active or self._closing:
            return

        try:
            text_content_name = f"text_{str(uuid.uuid4())}"

            text_content_start_payload = {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": text_content_name,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "USER",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            }
            await self._send_event(json.dumps(text_content_start_payload))

            text_input_payload = {
                "event": {
                    "textInput": {
                        "promptName": self.prompt_name,
                        "contentName": text_content_name,
                        "content": message,
                    }
                }
            }
            await self._send_event(json.dumps(text_input_payload))

            content_end_payload = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": text_content_name,
                    }
                }
            }
            await self._send_event(json.dumps(content_end_payload))

        except Exception as e:
            self.emit("error", f"Error sending message: {e}")

    async def emit(self, event_type: NovaSonicEventTypes, data: Dict[str, Any]) -> None:
        """Emit an event to subscribers"""
        try:
            super().emit(event_type, data)
        except Exception as e:
            self.emit("error", f"Error in emit for {event_type}: {e}")

    def _safe_emit(self, event_type: NovaSonicEventTypes, data: Dict[str, Any]) -> None:
        """Safely emit an event without requiring await"""
        try:
            if self.loop and not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(self.emit(event_type, data), self.loop)
        except Exception as e:
            self.emit("error", f"Error safely emitting event {event_type}: {e}")

    async def interrupt(self) -> None:
        """Interrupt current response"""
        if not self.is_active or self._closing:
            return

        if self.audio_track:
            self.audio_track.interrupt()
        await realtime_metrics_collector.set_interrupted()
        if self._agent_speaking:
            print("Interrupting agent speech, calling set_agent_speech_end")
            await realtime_metrics_collector.set_agent_speech_end(timeout=1.0)
            self._agent_speaking = False

        content_end_payload = {
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                }
            }
        }
        await self._send_event(json.dumps(content_end_payload))
        print(f"Sent contentEnd for {self.audio_content_name}")

        self.audio_content_name = f"audio_{str(uuid.uuid4())}"
        await self._start_audio_input()

    async def _cleanup(self) -> None:
        """Clean up resources"""
        if not self.is_active:
            return

        try:
            audio_content_end_payload = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                    }
                }
            }
            await self._send_event(json.dumps(audio_content_end_payload))

            prompt_end_payload = {
                "event": {"promptEnd": {"promptName": self.prompt_name}}
            }
            await self._send_event(json.dumps(prompt_end_payload))

            session_end_payload = {"event": {"sessionEnd": {}}}
            await self._send_event(json.dumps(session_end_payload))

            if self.stream and hasattr(self.stream, "input_stream"):
                await self.stream.input_stream.close()
        except Exception as e:
            self.emit("error", f"Error during cleanup: {e}")
        finally:
            self.is_active = False

            if self.response_task and not self.response_task.done():
                self.response_task.cancel()
                try:
                    await self.response_task
                except asyncio.CancelledError:
                    pass
                print("Cancelled response task")

            self.stream = None

    async def aclose(self) -> None:
        """Clean up all resources"""
        if self._closing:
            return

        self._closing = True

        await self._cleanup()

        if self.audio_track:
            if hasattr(self.audio_track, "cleanup"):
                try:
                    await self.audio_track.cleanup()
                except Exception as e:
                    self.emit("error", f"Error cleaning up audio track: {e}")
            self.audio_track = None

    async def _execute_tool_and_send_result(
        self, tool_use_event: Dict[str, Any]
    ) -> None:
        """Executes a tool and sends the result back to Nova Sonic."""
        tool_name = tool_use_event.get("toolName")
        tool_use_id = tool_use_event.get("toolUseId")
        tool_input_str = tool_use_event.get("content", "{}")

        if not tool_name or not tool_use_id:
            self.emit(
                "error",
                f"Error: Missing toolName or toolUseId in toolUse event: {tool_use_event}",
            )
            return

        try:
            tool_input_args = json.loads(tool_input_str)
        except json.JSONDecodeError as e:
            self.emit(
                "error",
                f"Error decoding tool input JSON: {e}. Input string: {tool_input_str}",
            )
            return

        target_tool: Optional[FunctionTool] = None
        for tool in self._tools:
            if is_function_tool(tool):
                tool_info = get_tool_info(tool)
                if tool_info.name == tool_name:
                    target_tool = tool
                    break

        if not target_tool:
            self.emit(
                "error", f"Error: Tool '{tool_name}' not found in registered tools."
            )
            return

        try:
            result = await target_tool(**tool_input_args)
            result_content_str = json.dumps(result)

            tool_content_name = f"tool_result_{str(uuid.uuid4())}"

            tool_content_start_dict = {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": tool_content_name,
                        "interactive": False,
                        "type": "TOOL",
                        "role": "TOOL",
                        "toolResultInputConfiguration": {
                            "toolUseId": tool_use_id,
                            "type": "TEXT",
                            "textInputConfiguration": {"mediaType": "text/plain"},
                        },
                    }
                }
            }
            await self._send_event(json.dumps(tool_content_start_dict))

            tool_result_event_dict = {
                "event": {
                    "toolResult": {
                        "promptName": self.prompt_name,
                        "contentName": tool_content_name,
                        "content": result_content_str,
                    }
                }
            }
            await self._send_event(json.dumps(tool_result_event_dict))

            tool_content_end_payload = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": tool_content_name,
                    }
                }
            }
            await self._send_event(json.dumps(tool_content_end_payload))

        except Exception as e:
            self.emit(
                "error", f"Error executing tool {tool_name} or sending result: {e}"
            )
