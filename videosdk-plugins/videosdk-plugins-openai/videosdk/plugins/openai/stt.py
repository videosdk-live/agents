from __future__ import annotations

import asyncio
import base64
import os
import time
from typing import Any, Optional
from urllib.parse import urlencode
import io
import wave
from scipy import signal
import aiohttp
import httpx
import openai
import numpy as np
from videosdk.agents import STT as BaseSTT, STTResponse, SpeechEventType, SpeechData, global_event_emitter

class OpenAISTT(BaseSTT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini-transcribe",
        base_url: str | None = None,
        prompt: str | None = None,
        language: str = "en",
        turn_detection: dict | None = None,
        enable_streaming: bool = True,
        silence_threshold: float = 0.01,
        silence_duration: float = 0.8,
    ) -> None:
        """Initialize the OpenAI STT plugin.

        Args:
            api_key (Optional[str], optional): OpenAI API key. Defaults to None.
            model (str): The model to use for the STT plugin. Defaults to "whisper-1".
            base_url (Optional[str], optional): The base URL for the OpenAI API. Defaults to None.
            prompt (Optional[str], optional): The prompt for the STT plugin. Defaults to None.
            language (str): The language to use for the STT plugin. Defaults to "en".
            turn_detection (dict | None): The turn detection for the STT plugin. Defaults to None.
        """
        super().__init__()
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either through api_key parameter or OPENAI_API_KEY environment variable")
        
        self.model = model
        self.language = language
        self.prompt = prompt
        self.turn_detection = turn_detection or {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
        }
        self.enable_streaming = enable_streaming
        
        # Custom VAD parameters for non-streaming mode
        self.silence_threshold_bytes = int(silence_threshold * 32767)
        self.silence_duration_frames = int(silence_duration * 48000)  # input_sample_rate
        
        self.client = openai.AsyncClient(
            max_retries=0,
            api_key=self.api_key,
            base_url=base_url or None,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._current_text = ""
        self._last_interim_at = 0
        self.input_sample_rate = 48000
        self.target_sample_rate = 16000
        self._audio_buffer = bytearray()
        
        # Custom VAD state for non-streaming mode
        self._is_speaking = False
        self._silence_frames = 0
        
    @staticmethod
    def azure(
        *,
        model: str = "gpt-4o-mini-transcribe",
        language: str = "en",
        prompt: str | None = None,
        turn_detection: dict | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        enable_streaming: bool = False,
        timeout: httpx.Timeout | None = None,
    ) -> "OpenAISTT":
        """
        Create a new instance of Azure OpenAI STT.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `AZURE_OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
        - `api_version` from `OPENAI_API_VERSION`
        - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
        - `azure_deployment` from `AZURE_OPENAI_DEPLOYMENT` (if not provided, uses `model` as deployment name)
        """
        
        # Get values from environment variables if not provided
        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = api_version or os.getenv("OPENAI_API_VERSION")
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        azure_ad_token = azure_ad_token or os.getenv("AZURE_OPENAI_AD_TOKEN")
        organization = organization or os.getenv("OPENAI_ORG_ID")
        project = project or os.getenv("OPENAI_PROJECT_ID")
        
        # If azure_deployment is not provided, use model as the deployment name
        if not azure_deployment:
            azure_deployment = model
        
        if not azure_endpoint:
            raise ValueError("Azure endpoint must be provided either through azure_endpoint parameter or AZURE_OPENAI_ENDPOINT environment variable")
        
        if not api_key and not azure_ad_token:
            raise ValueError("Either API key or Azure AD token must be provided")
        
        azure_client = openai.AsyncAzureOpenAI(
            max_retries=0,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout
            if timeout
            else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
        )
        
        instance = OpenAISTT(
            model=model,
            language=language,
            prompt=prompt,
            turn_detection=turn_detection,
            enable_streaming=enable_streaming,
        )
        instance.client = azure_client
        return instance
        
    async def process_audio(
        self,
        audio_frames: bytes,
        language: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Process audio frames and send to OpenAI based on enabled mode"""
        
        if not self.enable_streaming:
            await self._transcribe_non_streaming(audio_frames)
            return
        
        if not self._ws:
            await self._connect_ws()
            self._ws_task = asyncio.create_task(self._listen_for_responses())
            
        try:
            audio_data = np.frombuffer(audio_frames, dtype=np.int16)
            audio_data = signal.resample(audio_data, int(len(audio_data) * self.target_sample_rate / self.input_sample_rate))
            audio_data = audio_data.astype(np.int16).tobytes()
            audio_data = base64.b64encode(audio_data).decode("utf-8")
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_data,
            }
            await self._ws.send_json(message)
        except Exception as e:
            print(f"Error in process_audio: {str(e)}")
            self.emit("error", str(e))
            if self._ws:
                await self._ws.close()
                self._ws = None
                if self._ws_task:
                    self._ws_task.cancel()
                    self._ws_task = None

    async def _transcribe_non_streaming(self, audio_frames: bytes) -> None:
        """HTTP-based transcription using OpenAI audio/transcriptions API with custom VAD"""
        if not audio_frames:
            return
            
        self._audio_buffer.extend(audio_frames)
        
        # Custom VAD logic similar to other STT implementations
        is_silent_chunk = self._is_silent(audio_frames)
        
        if not is_silent_chunk:
            if not self._is_speaking:
                self._is_speaking = True
                global_event_emitter.emit("speech_started")
            self._silence_frames = 0
        else:
            if self._is_speaking:
                self._silence_frames += len(audio_frames) // 4  # Approximate frame count
                if self._silence_frames > self.silence_duration_frames:
                    global_event_emitter.emit("speech_stopped")
                    await self._process_audio_buffer()
                    self._is_speaking = False
                    self._silence_frames = 0

    def _is_silent(self, audio_chunk: bytes) -> bool:
        """Simple VAD: check if the max amplitude is below a threshold."""
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        return np.max(np.abs(audio_data)) < self.silence_threshold_bytes



    async def _process_audio_buffer(self) -> None:
        """Process the accumulated audio buffer with OpenAI transcription"""
        if not self._audio_buffer:
            return
            
        audio_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        
        wav_bytes = self._audio_frames_to_wav_bytes(audio_data)
        
        try:
            resp = await self.client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes, "audio/wav"),
                model=self.model,
                language=self.language,
                prompt=self.prompt or openai.NOT_GIVEN,
            )
            text = getattr(resp, "text", "")
            if text and self._transcript_callback:
                await self._transcript_callback(STTResponse(
                    event_type=SpeechEventType.FINAL,
                    data=SpeechData(text=text, language=self.language),
                    metadata={"model": self.model}
                ))
        except Exception as e:
            print(f"OpenAI transcription error: {str(e)}")
            self.emit("error", str(e))

    def _audio_frames_to_wav_bytes(self, audio_frames: bytes) -> bytes:
        """Convert audio frames to WAV bytes"""
        pcm = np.frombuffer(audio_frames, dtype=np.int16)
        resampled = signal.resample(pcm, int(len(pcm) * self.target_sample_rate / self.input_sample_rate))
        resampled = resampled.astype(np.int16)
        
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(self.target_sample_rate)
            wf.writeframes(resampled.tobytes())
        
        return buf.getvalue()

    async def _listen_for_responses(self) -> None:
        """Background task to listen for WebSocket responses"""
        if not self._ws:
            return
            
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = msg.json()
                    responses = self._handle_ws_message(data)
                    for response in responses:
                        if self._transcript_callback:
                            await self._transcript_callback(response)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    error = f"WebSocket error: {self._ws.exception()}"
                    print(error)
                    self.emit("error", error)
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    print("WebSocket connection closed")
                    break
        except Exception as e:
            error = f"Error in WebSocket listener: {str(e)}"
            print(error)
            self.emit("error", error)
        finally:
            if self._ws:
                await self._ws.close()
                self._ws = None
                
    async def _connect_ws(self) -> None:
        """Establish WebSocket connection with OpenAI's Realtime API"""
        
        if not self._session:
            self._session = aiohttp.ClientSession()
            
        config = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": self.model,
                    "prompt": self.prompt or "",
                    "language": self.language if self.language else None,
                },
                "turn_detection": self.turn_detection,
                "input_audio_noise_reduction": {
                    "type": "near_field"
                },
                "include": ["item.input_audio_transcription.logprobs"]
            }
        }
        
        query_params = {
            "intent": "transcription",
        }
        headers = {
            "User-Agent": "VideoSDK",
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        
        base_url = str(self.client.base_url).rstrip('/')
        ws_url = f"{base_url}/realtime?{urlencode(query_params)}"
        if ws_url.startswith("http"):
            ws_url = ws_url.replace("http", "ws", 1)

        try:
            self._ws = await self._session.ws_connect(ws_url, headers=headers)
            
            initial_response = await self._ws.receive_json()
            
            if initial_response.get("type") != "transcription_session.created":
                raise Exception(f"Expected session creation, got: {initial_response}")
            
            await self._ws.send_json(config)
            
            update_response = await self._ws.receive_json()
            
            if update_response.get("type") != "transcription_session.updated":
                raise Exception(f"Configuration update failed: {update_response}")
            
        except Exception as e:
            print(f"Error connecting to WebSocket: {str(e)}")
            if self._ws:
                await self._ws.close()
                self._ws = None
            raise
        
    def _handle_ws_message(self, msg: dict) -> list[STTResponse]:
        """Handle incoming WebSocket messages and generate STT responses"""
        responses = []
        
        try:
            msg_type = msg.get("type")
            if msg_type == "conversation.item.input_audio_transcription.delta":
                delta = msg.get("delta", "")
                if delta:
                    self._current_text += delta
                    current_time = asyncio.get_event_loop().time()
                    
                    if current_time - self._last_interim_at > 0.5:
                        responses.append(STTResponse(
                            event_type=SpeechEventType.INTERIM,
                            data=SpeechData(
                                text=self._current_text,
                                language=self.language,
                            ),
                            metadata={"model": self.model}
                        ))
                        self._last_interim_at = current_time
                        
            elif msg_type == "conversation.item.input_audio_transcription.completed":
                transcript = msg.get("transcript", "")
                if transcript:
                    responses.append(STTResponse(
                        event_type=SpeechEventType.FINAL,
                        data=SpeechData(
                            text=transcript,
                            language=self.language,
                        ),
                        metadata={"model": self.model}
                    ))
                    self._current_text = ""
            
            elif msg_type == "input_audio_buffer.speech_started":
                global_event_emitter.emit("speech_started")
            
            elif msg_type == "input_audio_buffer.speech_stopped":
                global_event_emitter.emit("speech_stopped")
                
        except Exception as e:
            print(f"Error handling WebSocket message: {str(e)}")
        
        return responses

    async def aclose(self) -> None:
        """Cleanup resources"""
        self._audio_buffer.clear()
        
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None
            
        if self._ws:
            await self._ws.close()
            self._ws = None
            
        if self._session:
            await self._session.close()
            self._session = None
            
        await self.client.close()

    async def _ensure_ws_connection(self):
        """Ensure WebSocket is connected, reconnect if necessary"""
        if not self._ws or self._ws.closed:
            await self._connect_ws()