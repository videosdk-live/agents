from __future__ import annotations

import asyncio
from typing import Any, Optional
import os
from urllib.parse import urlencode
import aiohttp
from videosdk.agents import STT as BaseSTT, STTResponse, SpeechEventType, SpeechData, global_event_emitter
import logging

logger = logging.getLogger(__name__)


class DeepgramSTT(BaseSTT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "flux-general-en",
        sample_rate: int = 16000,
        eager_eot_threshold:float=0.6,
        eot_threshold:float=0.8,
        eot_timeout_ms:int=7000,
        base_url: str = "wss://api.deepgram.com/v2/listen",
    ) -> None:
        super().__init__()

        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key must be provided either through api_key parameter or DEEPGRAM_API_KEY environment variable")

        self.model = model
        self.sample_rate = sample_rate
        self.eager_eot_threshold = eager_eot_threshold
        self.eot_threshold=eot_threshold
        self.eot_timeout_ms = eot_timeout_ms
        self.base_url = base_url

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._last_transcript: str = ""

    async def process_audio(
        self,
        audio_frames: bytes,
        **kwargs: Any
    ) -> None:
        """Process audio frames and send to Deepgram's Streaming API"""

        if not self._ws:
            await self._connect_ws()
            self._ws_task = asyncio.create_task(self._listen_for_responses())

        try:
            await self._ws.send_bytes(audio_frames)
        except Exception as e:
            logger.error(f"Error in process_audio: {str(e)}")
            self.emit("error", str(e))
            if self._ws:
                await self._ws.close()
                self._ws = None
                if self._ws_task:
                    self._ws_task.cancel()
                    self._ws_task = None

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
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    self.emit(
                        "error", f"WebSocket error: {self._ws.exception()}")
                    break
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {str(e)}")
            self.emit("error", f"Error in WebSocket listener: {str(e)}")
        finally:
            if self._ws:
                await self._ws.close()
                self._ws = None

    async def _connect_ws(self) -> None:
        """Establish WebSocket connection with Deepgram's Streaming API"""
        if not self._session:
            self._session = aiohttp.ClientSession()

        query_params = {
            "model": self.model,
            "encoding": "linear16",
            "sample_rate": self.sample_rate,
            "eot_threshold": self.eot_threshold,
            "eot_timeout_ms": self.eot_timeout_ms,
            "eager_eot_threshold": self.eager_eot_threshold,
            # "channels": 1,
            # "vad_events": "true",
            # "no_delay": "true",
        }
        headers = {"Authorization": f"Token {self.api_key}"}
        ws_url = f"{self.base_url}?{urlencode(query_params)}"

        try:
            self._ws = await self._session.ws_connect(ws_url, headers=headers)
            logger.info("Connected to Deepgram Flux WebSocket.")
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            raise

    def _handle_ws_message(self, msg: dict) -> list[STTResponse]:
        """Handle incoming WebSocket messages and generate STT responses"""
        responses = []

        try:
            if msg.get("type") != "TurnInfo":
                return responses

            event = msg.get("event")
            transcript = msg.get("transcript", "")
            logger.info(f"{event} and {transcript}")
            start_time = msg.get("audio_window_start", 0.0)
            end_time = msg.get("audio_window_end", 0.0)
            confidence = msg.get("end_of_turn_confidence", 0.0)

            self._last_transcript = transcript

            # Emit turn-related events
            if event == "StartOfTurn":
                global_event_emitter.emit("speech_started")
            elif event == "EagerEndOfTurn":
                global_event_emitter.emit("speech_eager_end")
            elif event == "EndOfTurn":
                global_event_emitter.emit("speech_finalized")
                if transcript and self._transcript_callback:
                    responses.append(
                        STTResponse(
                            event_type=SpeechEventType.FINAL,
                            data=SpeechData(
                                text=transcript,
                                confidence=confidence,
                                start_time=start_time,
                                end_time=end_time,
                            ),
                            metadata={"model": self.model},
                        )
                    )
            elif event == "TurnResumed":
                global_event_emitter.emit("speech_resumed")

            # Send interim transcript for ongoing turn
            if transcript and event not in ("EndOfTurn",):
                responses.append(
                    STTResponse(
                        event_type=SpeechEventType.INTERIM,
                        data=SpeechData(
                            text=transcript,
                            confidence=confidence,
                            start_time=start_time,
                            end_time=end_time,
                        ),
                        metadata={"model": self.model},
                    )
                )

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")

        return responses

    async def aclose(self) -> None:
        """Cleanup resources"""
        if self._ws_task:
            self._ws_task.cancel()
            logger.info("DeepgramSTT WebSocket task cancelled")
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None
            logger.info("DeepgramSTT WebSocket task cleared")

        if self._ws:
            await self._ws.close()
            logger.info("DeepgramSTT WebSocket closed")
            self._ws = None

        if self._session:
            await self._session.close()
            logger.info("DeepgramSTT cleaned up")
            self._session = None
        
        # Call base class cleanup
        await super().aclose()