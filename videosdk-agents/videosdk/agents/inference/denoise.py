# File: videosdk-agents/videosdk/agents/inference/denoise.py

from __future__ import annotations

import asyncio
import base64
import json
import os
import logging
from typing import Any, Optional, Dict

import aiohttp

from videosdk.agents import (
    Denoise as BaseDenoise,
)
import numpy as np

try:
    import resampy

    RESAMPY_AVAILABLE = True
except ImportError:
    RESAMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default inference gateway URL
VIDEOSDK_INFERENCE_URL = "wss://inference-gateway.videosdk.live"


class Denoise(BaseDenoise):
    """
    VideoSDK Inference Gateway Denoise Plugin.

    A lightweight noise cancellation client that connects to VideoSDK's Inference Gateway.
    Supports SANAS and AI-Coustics noise cancellation through a unified interface.

    Example:
        # Using factory methods (recommended)
        denoise = Denoise.sanas(model_name="ASR_NC3.0_VAD0.6")
        denoise = Denoise.aicoustics(model_name="sparrow-xxs-48khz")

        # Using generic constructor
        denoise = Denoise(
            provider="aicoustics",
            model_name="sparrow-xxs-48khz",
            config={"sample_rate": 48000}
        )

        # Use in pipeline
        pipeline = CascadingPipeline(
            stt=DeepgramSTT(sample_rate=48000),
            llm=GoogleLLM(),
            tts=ElevenLabsTTS(),
            vad=SileroVAD(input_sample_rate=48000),
            turn_detector=TurnDetector(),
            denoise=denoise
        )
    """

    def __init__(
        self,
        *,
        provider: str,
        model_name: str,
        sample_rate: int = 48000,
        config: Dict[str, Any] | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize the VideoSDK Inference Denoise plugin.

        Args:
            provider: Denoise provider name (e.g., "sanas", "aicoustics")
            model_name: Model identifier for the provider
            sample_rate: Audio sample rate in Hz (default: 48000)
            config: Provider-specific configuration dictionary
            base_url: Custom inference gateway URL (default: production gateway)
        """
        super().__init__()

        self._videosdk_token = os.getenv("VIDEOSDK_AUTH_TOKEN")
        if not self._videosdk_token:
            raise ValueError(
                "VIDEOSDK_AUTH_TOKEN environment variable must be set for authentication"
            )

        self.provider = provider
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.config = config or {}
        self.base_url = base_url or VIDEOSDK_INFERENCE_URL

        # WebSocket state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._config_sent: bool = False

        # Audio buffering for responses
        self._audio_buffer: asyncio.Queue = asyncio.Queue()
        self._processing: bool = False

        # Statistics
        self._stats = {
            "chunks_processed": 0,
            "bytes_processed": 0,
            "bytes_denoised": 0,
            "errors": 0,
        }

        logger.info(
            f"[InferenceDenoise] Initialized: provider={provider}, "
            f"model={model_name}, sample_rate={sample_rate}Hz"
        )

    # ==================== Factory Methods ====================

    # @staticmethod
    # def sanas(
    #     *,
    #     model_name: str = "ASR_NC3.0_VAD0.6",
    #     sample_rate: int = 48000,
    #     secure_media: bool = False,
    #     base_url: str | None = None,
    # ) -> "Denoise":
    #     """
    #     Create a Denoise instance configured for SANAS.

    #     Args:
    #         model_name: SANAS model (default: "ASR_NC3.0_VAD0.6")
    #             Options: "ASR_NC3.0_VAD0.6", "ASR_NC2.0", etc.
    #         sample_rate: Audio sample rate in Hz (default: 48000)
    #         secure_media: Whether to use secure media connection (default: False)
    #         base_url: Custom inference gateway URL

    #     Returns:
    #         Configured Denoise instance for SANAS

    #     Example:
    #         >>> denoise = Denoise.sanas(model_name="ASR_NC3.0_VAD0.6")
    #     """
    #     config = {
    #         "model_name": model_name,
    #         "sample_rate": sample_rate,
    #         "secure_media": secure_media,
    #     }

    #     return Denoise(
    #         provider="sanas",
    #         model_name=model_name,
    #         sample_rate=sample_rate,
    #         config=config,
    #         base_url=base_url,
    #     )

    @staticmethod
    def aicoustics(
        *,
        model_name: str = "sparrow-xxs-48khz",
        sample_rate: int = 48000,
        channels: int = 1,
        base_url: str | None = None,
    ) -> "Denoise":
        """
        Create a Denoise instance configured for AI-Coustics.

        Args:
            model_name: AI-Coustics model (default: "sparrow-xxs-48khz")
                Sparrow family (human-to-human):
                - "sparrow-xxs-48khz": Ultra-fast, 10ms latency, 1MB
                - "sparrow-s-48khz": Small, 30ms latency, 8.96MB
                - "sparrow-l-48khz": Large, best quality, 30ms latency, 35.1MB

                Quail family (human-to-machine, voice AI):
                - "quail-vf-l-16khz": Voice focus + STT optimization, 35MB
                - "quail-l-16khz": General purpose, 35MB
                - "quail-s-16khz": Faster, 8.88MB

            sample_rate: Audio sample rate in Hz (default: 48000)
                - Sparrow models: 48000 Hz
                - Quail models: 16000 Hz
            channels: Number of audio channels (default: 1 for mono)
            base_url: Custom inference gateway URL

        Returns:
            Configured Denoise instance for AI-Coustics

        Example:
            >>> # Ultra-fast for real-time calls
            >>> denoise = Denoise.aicoustics(model_name="sparrow-xxs-48khz")
            >>>
            >>> # Best quality for recordings
            >>> denoise = Denoise.aicoustics(model_name="sparrow-l-48khz")
            >>>
            >>> # Voice AI / STT optimization
            >>> denoise = Denoise.aicoustics(
            ...     model_name="quail-vf-l-16khz",
            ...     sample_rate=16000
            ... )
        """
        config = {
            "model": model_name,
            "sample_rate": sample_rate,
            "channels": channels,
        }

        return Denoise(
            provider="aicoustics",
            model_name=model_name,
            sample_rate=sample_rate,
            config=config,
            base_url=base_url,
        )

    # ==================== Core Denoise Methods ====================

    async def denoise(
        self,
        audio_frames: bytes,
        **kwargs: Any,
    ) -> bytes:
        """
        Process audio frames through noise cancellation.

        Args:
            audio_frames: Raw PCM audio bytes (int16 format)
            **kwargs: Additional arguments (unused)

        Returns:
            Denoised PCM audio bytes (int16 format)
        """
        try:
            # Ensure WebSocket connection
            if not self._ws or self._ws.closed:
                await self._connect_ws()
                if not self._ws_task or self._ws_task.done():
                    self._ws_task = asyncio.create_task(self._listen_for_responses())

            if not self._config_sent:
                await self._send_config()

            await self._send_audio(audio_frames)

            # Try to get denoised audio (non-blocking)
            try:
                denoised_audio = await asyncio.wait_for(
                    self._audio_buffer.get(), timeout=0.01
                )
                self._stats["bytes_denoised"] += len(denoised_audio)
                return denoised_audio
            except asyncio.TimeoutError:
                # No denoised audio available yet, return original
                # This can happen during initial buffering
                return audio_frames

        except Exception as e:
            logger.error(f"[InferenceDenoise] Error in denoise: {e}")
            self._stats["errors"] += 1
            self.emit("error", str(e))
            # On error, return original audio
            return audio_frames

    # ==================== WebSocket Communication ====================

    async def _connect_ws(self) -> None:
        """Establish WebSocket connection to the inference gateway."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        ws_url = (
            f"{self.base_url}/v1/denoise"
            f"?provider={self.provider}"
            f"&secret={self._videosdk_token}"
            f"&modelId={self.model_name}"
        )

        try:
            logger.info(
                f"[InferenceDenoise] Connecting to {self.base_url} "
                f"(provider={self.provider}, model={self.model_name})"
            )
            self._ws = await self._session.ws_connect(ws_url)
            self._config_sent = False
            logger.info("[InferenceDenoise] Connected successfully")
        except Exception as e:
            logger.error(f"[InferenceDenoise] Connection failed: {e}")
            raise

    async def _send_config(self) -> None:
        """Send configuration message to the inference server."""
        if not self._ws:
            return

        config_message = {
            "type": "config",
            "data": {
                **self.config,
                "sample_rate": self.sample_rate,
            },
        }

        try:
            await self._ws.send_str(json.dumps(config_message))
            self._config_sent = True
            logger.info(f"[InferenceDenoise] Config sent: {config_message['data']}")
        except Exception as e:
            logger.error(f"[InferenceDenoise] Failed to send config: {e}")
            raise

    async def _send_audio(self, audio_bytes: bytes) -> None:
        """Send audio data to the inference server."""
        if not self._ws:
            return

        audio_message = {
            "type": "audio",
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
        }

        try:
            await self._ws.send_str(json.dumps(audio_message))
            self._stats["bytes_processed"] += len(audio_bytes)
            self._stats["chunks_processed"] += 1
        except Exception as e:
            logger.error(f"[InferenceDenoise] Failed to send audio: {e}")
            raise

    async def _listen_for_responses(self) -> None:
        """Background task to listen for WebSocket responses from the server."""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        f"[InferenceDenoise] WebSocket error: {self._ws.exception()}"
                    )
                    self.emit("error", f"WebSocket error: {self._ws.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("[InferenceDenoise] WebSocket closed by server")
                    break

        except asyncio.CancelledError:
            logger.debug("[InferenceDenoise] WebSocket listener cancelled")
        except Exception as e:
            logger.error(f"[InferenceDenoise] Error in WebSocket listener: {e}")
            self.emit("error", str(e))
        finally:
            await self._cleanup_connection()

    async def _handle_message(self, raw_message: str) -> None:
        """
        Handle incoming messages from the inference server.

        Args:
            raw_message: Raw JSON message string from server
        """
        try:
            data = json.loads(raw_message)
            msg_type = data.get("type")

            if msg_type == "event":
                # Handle event messages (AI-Coustics format)
                event_data = data.get("data", {})
                event_type = event_data.get("eventType")

                if event_type == "DENOISE_AUDIO":
                    audio_data = event_data.get("audio", "")
                    denoised_bytes = base64.b64decode(audio_data)
                    await self._audio_buffer.put(denoised_bytes)

            elif msg_type == "audio":
                # Handle audio messages (SANAS format)
                audio_data = data.get("data", "")
                denoised_bytes = base64.b64decode(audio_data)
                await self._audio_buffer.put(denoised_bytes)

            elif msg_type == "error":
                error_msg = data.get("data", {}).get("error") or data.get(
                    "message", "Unknown error"
                )
                logger.error(f"[InferenceDenoise] Server error: {error_msg}")
                self.emit("error", error_msg)

            elif msg_type == "stats":
                # Optional: handle statistics from server
                stats_data = data.get("data", {})
                logger.debug(f"[InferenceDenoise] Server stats: {stats_data}")

        except json.JSONDecodeError as e:
            logger.error(f"[InferenceDenoise] Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"[InferenceDenoise] Error handling message: {e}")

    async def _cleanup_connection(self) -> None:
        """Clean up WebSocket connection resources."""
        if self._ws and not self._ws.closed:
            try:
                # Send stop message before closing
                stop_message = {"type": "stop"}
                await self._ws.send_str(json.dumps(stop_message))
                await asyncio.sleep(0.1)
            except Exception:
                pass

            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._config_sent = False

    # ==================== Utility Methods ====================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return {
            **self._stats,
            "buffer_size": self._audio_buffer.qsize(),
            "provider": self.provider,
            "model": self.model_name,
        }

    async def aclose(self) -> None:
        """Clean up all resources."""
        logger.info(
            f"[InferenceDenoise] Closing Denoise (provider={self.provider}). "
            f"Final stats: {self.get_stats()}"
        )

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

        await self._cleanup_connection()

        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        while not self._audio_buffer.empty():
            try:
                self._audio_buffer.get_nowait()
            except asyncio.QueueEmpty:
                break

        await super().aclose()

        logger.info("[InferenceDenoise] Closed successfully")

    # ==================== Properties ====================

    @property
    def label(self) -> str:
        """Get a descriptive label for this Denoise instance."""
        return f"videosdk.inference.Denoise.{self.provider}.{self.model_name}"
