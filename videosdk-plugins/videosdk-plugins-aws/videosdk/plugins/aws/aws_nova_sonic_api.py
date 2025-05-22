from __future__ import annotations

import os
import asyncio
import base64
import json
import uuid
from typing import Optional, Literal, List, Dict, Any
from dataclasses import dataclass
import librosa
import numpy as np


from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart
)
from aws_sdk_bedrock_runtime.config import (
    Config,
    HTTPAuthSchemeResolver,
    SigV4AuthScheme
)
from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver

from videosdk.agents import RealtimeBaseModel
from videosdk.agents.utils import build_nova_sonic_schema, get_tool_info, is_function_tool, FunctionTool

NOVA_INPUT_SAMPLE_RATE = 16000  
NOVA_OUTPUT_SAMPLE_RATE = 24000 

# Event types
NovaSonicEventTypes = Literal[
    "audio_output",
    "transcription",
    "error"
]

@dataclass
class NovaSonicConfig:
    """Configuration for Nova Sonic API
    
    Args:
        model_id: The Nova Sonic model ID to use. Default is 'amazon.nova-sonic-v1:0'
        region: AWS region for Bedrock. Default is 'us-east-1'
        voice: Voice ID for audio output. Default is 'matthew'
        temperature: Controls randomness in responses. Default is 0.7
        top_p: Nucleus sampling parameter. Default is 0.9
        max_tokens: Maximum tokens in response. Default is 1024
    """
    model_id: str = "amazon.nova-sonic-v1:0"
    region: str = "us-east-1"
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
    ) -> None:
        """
        Initialize Nova Sonic realtime model.
        
        Args:
            model: The Nova Sonic model identifier
            config: Optional configuration object for customizing model behavior
        """
        super().__init__()
        self.model = model
        self.config = config or NovaSonicConfig()
        
        # Initialize state
        self.bedrock_client = None
        self.stream = None
        self._closing = False
        self._instructions = "You are a helpful assistant. The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation. Keep your responses short, generally two or three sentences for chatty scenarios."
        self._tools = []
        self.tools_formatted = [] 
        self.loop = asyncio.get_event_loop()
        self.audio_track = None
        
        # Session information
        self.prompt_name = str(uuid.uuid4())
        self.system_content_name = f"system_{str(uuid.uuid4())}"
        self.audio_content_name = f"audio_{str(uuid.uuid4())}"
        
        # For response handling
        self.is_active = False
        self.response_task = None
        
        # Initialize Bedrock client
        self._initialize_bedrock_client()

        # Listen for agent events
        self.on("instructions_updated", self._handle_instructions_updated)
        self.on("tools_updated", self._handle_tools_updated)

    def _initialize_bedrock_client(self):
        """Initialize the Bedrock client with aws_sdk_bedrock_runtime."""
        try:
            config = Config(
                endpoint_uri=f"https://bedrock-runtime.{self.config.region}.amazonaws.com",
                region=self.config.region,
                aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
                http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
                http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()}
            )
            self.bedrock_client = BedrockRuntimeClient(config=config)
            
            # Check for AWS credentials
            aws_vars = [k for k in os.environ.keys() if k.startswith('AWS_')]
            if not aws_vars:
                print("WARNING: No AWS environment variables found. Authentication may fail.")
            else:
                found_keys = ["AWS_ACCESS_KEY_ID" in os.environ, "AWS_SECRET_ACCESS_KEY" in os.environ]
                print(f"AWS credential presence check: {', '.join([f'{k}: {v}' for k, v in zip(['ACCESS_KEY', 'SECRET_KEY'], found_keys)])}")
                
        except Exception as e:
            print(f"Error initializing Bedrock client: {e}")
            raise

    async def connect(self) -> None:
        """Initialize connection to Nova Sonic"""
        if self.is_active:
            await self._cleanup()
        
        self._closing = False
        
        try:
            # Store loop reference for creating tasks
            self.loop = asyncio.get_event_loop()
            # Initialize Bedrock stream
            self.stream = await self.bedrock_client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(
                    model_id=self.config.model_id
                )
            )
            self.is_active = True
            
            # Step 1: Send session start event -            
            session_start_payload = {
              "event": {
                "sessionStart": {
                  "inferenceConfiguration": {
                    "maxTokens": self.config.max_tokens,
                    "topP": self.config.top_p,
                    "temperature": self.config.temperature
                  }
                }
              }
            }
            await self._send_event(json.dumps(session_start_payload))
            
            # Step 2: Send prompt start event -            
            prompt_start_event_dict = {
              "event": {
                "promptStart": {
                  "promptName": self.prompt_name,
                  "textOutputConfiguration": {
                    "mediaType": "text/plain"
                  },
                  "audioOutputConfiguration": {
                    "mediaType": "audio/lpcm",
                    "sampleRateHertz": NOVA_OUTPUT_SAMPLE_RATE,
                    "sampleSizeBits": 16,
                    "channelCount": 1,
                    "voiceId": self.config.voice,
                    "encoding": "base64",
                    "audioType": "SPEECH"
                  }
                }
              }
            }

            if self.tools_formatted:
                prompt_start_event_dict["event"]["promptStart"]["toolUseOutputConfiguration"] = {
                    "mediaType": "application/json"
                }
                prompt_start_event_dict["event"]["promptStart"]["toolConfiguration"] = {
                    "tools": self.tools_formatted
                }
            
            await self._send_event(json.dumps(prompt_start_event_dict))
            
            # Step 3: Send system content start -            
            system_content_start_payload = {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.system_content_name,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "SYSTEM",
                        "textInputConfiguration": {
                            "mediaType": "text/plain"
                        }
                    }
                }
            }
            await self._send_event(json.dumps(system_content_start_payload))
            
            # Step 4: Send system content text -            
            system_instructions = self._instructions or "You are a helpful voice assistant. Keep your responses short and conversational."
            text_input_payload = {
                "event": {
                    "textInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.system_content_name,
                        "content": system_instructions
                    }
                }
            }
            await self._send_event(json.dumps(text_input_payload))
            
            # Step 5: End system content -            
            content_end_payload = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.system_content_name
                    }
                }
            }
            await self._send_event(json.dumps(content_end_payload))         

            # Start response processing task
            self.response_task = asyncio.create_task(self._process_responses())
            
            # Start audio input
            await self._start_audio_input()

        except Exception as e:
            await self._cleanup()
            raise

    async def _send_event(self, event_json: str):
        """Send an event to the bidirectional stream"""
        if not self.is_active or not self.stream:
            return
            
        try:
            # Create the input chunk - exactly like     
            event = InvokeModelWithBidirectionalStreamInputChunk(
                value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
            )

            # Send the event
            await self.stream.input_stream.send(event)
            
        except Exception as e:
            print(f"Error sending event: {e}")

    async def _start_audio_input(self):
        """Start audio input stream"""
        if not self.is_active:
            return
        
        # Create fresh audio content start event
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
                        "encoding": "base64"
                    }
                }
            }
        }
        await self._send_event(json.dumps(audio_content_start_payload))

    async def handle_audio_input(self, audio_data: bytes) -> None:
        """Handle incoming 24kHz audio from VideoSDK"""
        if not self.is_active or self._closing:
            return
            
        try:
            # Get audio array from bytes
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Resample from 24kHz to 16kHz
            resampled = librosa.resample(
                audio_array.astype(np.float32),
                orig_sr=24000,
                target_sr=16000
            ).astype(np.int16)
            
            # Convert to bytes
            resampled_bytes = resampled.tobytes()
            
            # Encode in base64 as expected by Nova Sonic
            encoded_audio = base64.b64encode(resampled_bytes).decode('utf-8')
            
            # Format exactly as in the     
            audio_event_payload = {
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "content": encoded_audio
                    }
                }
            }
            
            # Send using the raw event
            await self._send_event(json.dumps(audio_event_payload))

        except Exception as e:
            print(f"Resampling error: {e}")

    async def _process_responses(self):
        """Process responses from the bidirectional stream"""
        try:
            while self.is_active and not self._closing:
                try:
                    output = await self.stream.await_output()
                    result = await output[1].receive()
                    
                    if result.value and result.value.bytes_:
                        response_data = result.value.bytes_.decode('utf-8')
                        
                        try:
                            json_data = json.loads(response_data)
                            
                            if 'event' in json_data:
                                event_keys = list(json_data['event'].keys())
                                
                                if 'completionStart' in json_data['event']:
                                    completion_start = json_data['event']['completionStart']
                                
                                # Handle content start - similar to     
                                elif 'contentStart' in json_data['event']:
                                    content_start = json_data['event']['contentStart']
 
                                    # Check for speculative content like in     
                                    if 'additionalModelFields' in content_start:
                                        try:
                                            additional_fields = json.loads(content_start['additionalModelFields'])
                                        except (json.JSONDecodeError, KeyError) as e:
                                            print(f"Error parsing additionalModelFields: {e}")
                                elif 'textOutput' in json_data['event']:
                                    pass

                                elif 'audioOutput' in json_data['event']:                                    
                                    # Extract audio content -  
                                    audio_output = json_data['event']['audioOutput']
                                    if 'content' not in audio_output:
                                        continue
                                    
                                    audio_content = audio_output['content']
                                    if not audio_content:
                                        continue
                                    
                                    try:
                                        # Decode audio like in     
                                        audio_bytes = base64.b64decode(audio_content)
                                        
                                        # Queue for playback
                                        if self.audio_track and not self._closing:
                                            await self.audio_track.add_new_bytes(audio_bytes)

                                    except Exception as e:
                                        print(f"AUDIO PROCESSING ERROR: {e}")
                                
                                elif 'contentEnd' in json_data['event']: 
                                    pass

                                elif 'toolUse' in json_data['event']:
                                     tool_use = json_data['event']['toolUse']
                                     asyncio.create_task(self._execute_tool_and_send_result(tool_use))

                                elif 'completionEnd' in json_data['event']:
                                     completion_end = json_data['event']['completionEnd']
                                     print(f"Nova completionEnd: {json.dumps(completion_end, indent=2)}")

                                # Handle other event types
                                else:
                                    print(f"Unhandled event type from Nova: {event_keys} - {json.dumps(json_data['event'], indent=2)}")
                            else:
                                print(f"Non-event response: {json_data}")
                                
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse response: {e}")
                            print(f"Raw data: {response_data[:200]}...")
                        
                except Exception as e:
                    print(f"Error processing response: {e}")
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
            
            # Step 1: Start text content 
            text_content_start_payload = {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": text_content_name,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "USER",
                        "textInputConfiguration": {
                            "mediaType": "text/plain"
                        }
                    }
                }
            }
            await self._send_event(json.dumps(text_content_start_payload))
            
            # Step 2: Send text input
            text_input_payload = {
                "event": {
                    "textInput": {
                        "promptName": self.prompt_name,
                        "contentName": text_content_name,
                        "content": message
                    }
                }
            }
            await self._send_event(json.dumps(text_input_payload))
            
            # Step 3: End text content
            content_end_payload = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": text_content_name
                    }
                }
            }
            await self._send_event(json.dumps(content_end_payload))
            
            
        except Exception as e:
            print(f"Error sending message: {e}")

    async def emit(self, event_type: NovaSonicEventTypes, data: Dict[str, Any]) -> None:
        """Emit an event to subscribers"""
        try:
            # Call parent class emit method
            await super().emit(event_type, data)
        except Exception as e:
            print(f"Error in emit for {event_type}: {e}")

    def _safe_emit(self, event_type: NovaSonicEventTypes, data: Dict[str, Any]) -> None:
        """Safely emit an event without requiring await"""
        try:
            # Create a future to hold the result of the emit coroutine
            if self.loop and not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self.emit(event_type, data),
                    self.loop
                )
        except Exception as e:
            print(f"Error safely emitting event {event_type}: {e}")

    async def interrupt(self) -> None:
        """Interrupt current response"""
        if not self.is_active or self._closing:
            return
            
        # Interrupt audio track if available
        if self.audio_track:
            self.audio_track.interrupt()
        
        # End the current audio content stream
        content_end_payload = {
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name
                }
            }
        }
        await self._send_event(json.dumps(content_end_payload))
        print(f"Sent contentEnd for {self.audio_content_name}")

        # Generate new content name for fresh audio session
        self.audio_content_name = f"audio_{str(uuid.uuid4())}"
        await self._start_audio_input()

    async def _cleanup(self) -> None:
        """Clean up resources"""
        if not self.is_active:
            return
            
        try:
            # End audio content
            audio_content_end_payload = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name
                    }
                }
            }
            await self._send_event(json.dumps(audio_content_end_payload))
            
            # End prompt
            prompt_end_payload = {
                "event": {
                    "promptEnd": {
                        "promptName": self.prompt_name
                    }
                }
            }
            await self._send_event(json.dumps(prompt_end_payload))
            
            # End session
            session_end_payload = {
                "event": {
                    "sessionEnd": {}
                }
            }
            await self._send_event(json.dumps(session_end_payload))
            
            # Close the stream
            if self.stream and hasattr(self.stream, 'input_stream'):
                await self.stream.input_stream.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            self.is_active = False
            
            # Cancel the response task
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
        
        # Clean up session
        await self._cleanup()
        
        # Clean up audio track
        if self.audio_track:
            if hasattr(self.audio_track, 'cleanup'):
                try:
                    await self.audio_track.cleanup()
                except Exception as e:
                    print(f"Error cleaning up audio track: {e}")
            self.audio_track = None
        

    def _handle_instructions_updated(self, data: Dict[str, Any]) -> None:
        """Handle instructions updated event"""
        self._instructions = data.get("instructions")

    def _handle_tools_updated(self, data: Dict[str, Any]) -> None:
        """Handle tools updated event"""
        tools = data.get("tools", [])
        self._tools = tools 
        self.tools_formatted = [build_nova_sonic_schema(tool) for tool in tools if is_function_tool(tool)]
        
    async def _execute_tool_and_send_result(self, tool_use_event: Dict[str, Any]) -> None:
        """Executes a tool and sends the result back to Nova Sonic."""
        tool_name = tool_use_event.get("toolName")
        tool_use_id = tool_use_event.get("toolUseId")
        tool_input_str = tool_use_event.get("content", "{}")

        if not tool_name or not tool_use_id:
            print(f"Error: Missing toolName or toolUseId in toolUse event: {tool_use_event}")
            return

        try:
            tool_input_args = json.loads(tool_input_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding tool input JSON: {e}. Input string: {tool_input_str}")
            return

        target_tool: Optional[FunctionTool] = None
        for tool in self._tools:
            if is_function_tool(tool):
                tool_info = get_tool_info(tool)
                if tool_info.name == tool_name:
                    target_tool = tool
                    break
        
        if not target_tool:
            print(f"Error: Tool '{tool_name}' not found in registered tools.")
            return

        try:
            result = await target_tool(**tool_input_args)
            result_content_str = json.dumps(result)

            # Send the result back to Nova Sonic
            tool_content_name = f"tool_result_{str(uuid.uuid4())}"

            # 1. Send contentStart for tool result
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
                            "textInputConfiguration": {
                                "mediaType": "text/plain"
                            }
                        }
                    }
                }
            }
            await self._send_event(json.dumps(tool_content_start_dict))

            # 2. Send toolResult
            tool_result_event_dict = {
                "event": {
                    "toolResult": {
                        "promptName": self.prompt_name,
                        "contentName": tool_content_name,
                        "content": result_content_str
                    }
                }
            }
            await self._send_event(json.dumps(tool_result_event_dict))
            
            # 3. Send contentEnd for tool result
            tool_content_end_payload = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": tool_content_name
                    }
                }
            }
            await self._send_event(json.dumps(tool_content_end_payload))

        except Exception as e:
            print(f"Error executing tool {tool_name} or sending result: {e}")