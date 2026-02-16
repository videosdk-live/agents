from __future__ import annotations

import asyncio
import base64
import json
import os
import logging
from typing import Any, Optional, Dict

import aiohttp

from videosdk.agents.denoise import (
    Denoise as BaseDenoise,
)

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
        denoise = Denoise.aicoustics(model_id="sparrow-xxs-48khz")

        # Using generic constructor
        denoise = Denoise(
            provider="aicoustics",
            model_id="sparrow-xxs-48khz",
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
        model_id: str,
        sample_rate: int = 48000,
        channels: int = 1,
        config: Dict[str, Any] | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize the VideoSDK Inference Denoise plugin.

        Args:
            provider: Denoise provider name (e.g., "aicoustics")
            model_id: Model identifier for the provider
            sample_rate: Audio sample rate in Hz (default: 48000)
            channels: Number of audio channels (default: 1 for mono)
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
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.config = config or {}
        self.base_url = base_url or VIDEOSDK_INFERENCE_URL

        # WebSocket state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._config_sent: bool = False
        self._reconnecting: bool = False

        # Audio buffering for responses
        self._audio_buffer: asyncio.Queue = asyncio.Queue()
        self._processing: bool = False

        # Statistics
        self._stats = {
            "chunks_sent": 0,
            "bytes_sent": 0,
            "chunks_received": 0,
            "bytes_received": 0,
            "errors": 0,
            "reconnections": 0,
        }

        logger.info(
            f"[InferenceDenoise] Initialized: provider={provider}, "
            f"model={model_id}, sample_rate={sample_rate}Hz, channels={channels}"
        )

    # ==================== Factory Methods ====================
    @staticmethod
    def aicoustics(
        *,
        model_id: str = "sparrow-xxs-48khz",
        sample_rate: int = 48000,
        channels: int = 1,
        base_url: str | None = None,
    ) -> "Denoise":
        """
        Create a Denoise instance configured for AI-Coustics.

        Args:
            model_id: AI-Coustics model (default: "sparrow-xxs-48khz")
                Sparrow family (human-to-human, 48kHz):
                - "sparrow-xxs-48khz": Ultra-fast, 10ms latency, 1MB
                - "sparrow-s-48khz": Small, 30ms latency, 8.96MB
                - "sparrow-l-48khz": Large, best quality, 30ms latency, 35.1MB

                Quail family (human-to-machine, voice AI, 16kHz):
                - "quail-vf-l-16khz": Voice focus + STT optimization, 35MB
                - "quail-l-16khz": General purpose, 35MB
                - "quail-s-16khz": Faster, 8.88MB

            sample_rate: Audio sample rate in Hz
                - Sparrow models: 48000 Hz (default)
                - Quail models: 16000 Hz
            channels: Number of audio channels (default: 1 for mono)
            base_url: Custom inference gateway URL

        Returns:
            Configured Denoise instance for AI-Coustics

        Example:
            >>> # Ultra-fast for real-time calls
            >>> denoise = Denoise.aicoustics(model_id="sparrow-xxs-48khz")
            >>>
            >>> # Best quality for recordings
            >>> denoise = Denoise.aicoustics(model_id="sparrow-l-48khz")
            >>>
            >>> # Voice AI / STT optimization (16kHz)
            >>> denoise = Denoise.aicoustics(
            ...     model_id="quail-vf-l-16khz",
            ...     sample_rate=16000
            ... )
        """
        config = {
            "model": model_id,
            "sample_rate": sample_rate,
            "channels": channels,
        }

        return Denoise(
            provider="aicoustics",
            model_id=model_id,
            sample_rate=sample_rate,
            channels=channels,
            config=config,
            base_url=base_url or VIDEOSDK_INFERENCE_URL,
        )

    @staticmethod
    def krisp(
        *,
        model_id: str = "krisp-viva-tel-v2",
        sample_rate: int = 16000,
        channels: int = 1,
        base_url: str | None = None,
    ) -> "Denoise":

        """
            Create a Denoise instance configured for Krisp.

            Args:
                model_id: Krisp model (default: "krisp-viva-tel-v2")

                sample_rate: Audio sample rate in Hz
                    - krisp-viva-tel-v2 - 16000 for noise cancellation
                channels: Number of audio channels (default: 1 for mono)
                base_url: Custom inference gateway URL

            Returns:
                Configured Denoise instance for Krisp

            Example:
                >>> # Ultra-fast for real-time calls
                >>> denoise = Denoise.krisp(model_id="krisp-viva-tel-v2")
                >>>
                >>> # Best quality for recordings
                >>> denoise = Denoise.krisp(model_id="krisp-viva-tel-v2")
                >>>
                >>> # Voice AI / STT optimization (16kHz)
                >>> denoise = Denoise.krisp(
                ...     model_id="krisp-viva-tel-v2",
                ...     sample_rate=16000
                ... )
        """

        return Denoise(
            provider="krisp",
            model_id=model_id,
            sample_rate=sample_rate,
            channels=channels,
            base_url=base_url or VIDEOSDK_INFERENCE_URL,
        )
    
    @staticmethod
    def sanas(
        *,
        model_id: str = "VI_G_NC3.0",
        sample_rate: int = 16000,
        channels: int = 1,
        base_url: str | None = None,
    ) -> "Denoise":

        """
            Create a Denoise instance configured for Sanas.

            Args:
                model_id: Sanas model (default: "VI_G_NC3.0")

                sample_rate: Audio sample rate in Hz
                    - VI_G_NC3.0 - 16000 for noise cancellation
                channels: Number of audio channels (default: 1 for mono)
                base_url: Custom inference gateway URL

            Returns:
                Configured Denoise instance for Sanas

            Example:
                >>> # Ultra-fast for real-time calls
                >>> denoise = Denoise.aicoustics(model_id="VI_G_NC3.0")
                >>>
                >>> # Best quality for recordings
                >>> denoise = Denoise.aicoustics(model_id="VI_G_NC3.0")
                >>>
                >>> # Voice AI / STT optimization (16kHz)
                >>> denoise = Denoise.sanas(
                ...     model_id="VI_G_NC3.0",
                ...     sample_rate=16000
                ... )
            """

        return Denoise(
            provider="sanas",
            model_id=model_id,
            sample_rate=sample_rate,
            channels=channels,
            base_url=base_url or VIDEOSDK_INFERENCE_URL,
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
            Denoised PCM audio bytes (int16 format), or original audio if processing fails
        """
        try:
            if not self._ws or self._ws.closed:
                await self._connect_ws()
                if not self._ws_task or self._ws_task.done():
                    self._ws_task = asyncio.create_task(self._listen_for_responses())

            if not self._config_sent:
                await self._send_config()

            await self._send_audio(audio_frames)

            # Try to get denoised audio (with timeout to prevent blocking)
            # This returns immediately if no audio is ready
            try:
                denoised_audio = await asyncio.wait_for(
                    self._audio_buffer.get(), timeout=0.05
                )
                self._stats["chunks_received"] += 1
                self._stats["bytes_received"] += len(denoised_audio)
                return denoised_audio
            except asyncio.TimeoutError:
                # no denoised audio available yet
                # this is normal during initial buffering or low latency scenarios and return empty bytes to maintain stream flow
                return b""

        except Exception as e:
            logger.error(f"[InferenceDenoise] Error in denoise: {e}", exc_info=True)
            self._stats["errors"] += 1
            self.emit("error", str(e))

            # Attempt reconnection on persistent errors
            if self._stats["errors"] % 10 == 0:
                logger.warning(
                    f"[InferenceDenoise] {self._stats['errors']} errors detected, "
                    "attempting reconnection..."
                )
                await self._reconnect()

            # Return empty bytes on error (don't pass through original to avoid glitches)
            return b""

    # ==================== WebSocket Communication ====================

    async def _connect_ws(self) -> None:
        """Establish WebSocket connection to the inference gateway."""
        if self._reconnecting:
            return  # Prevent concurrent reconnection attempts

        try:
            if not self._session or self._session.closed:
                self._session = aiohttp.ClientSession()

            ws_url = (
                f"{self.base_url}/v1/denoise"
                f"?provider={self.provider}"
                f"&secret={self._videosdk_token}"
                f"&modelId={self.model_id}"
            )

            logger.info(
                f"[InferenceDenoise] Connecting to {self.base_url} "
                f"(provider={self.provider}, model={self.model_id})"
            )

            timeout = aiohttp.ClientTimeout(total=10)
            self._ws = await self._session.ws_connect(ws_url, timeout=timeout)
            self._config_sent = False

            logger.info("[InferenceDenoise] Connected successfully")

        except Exception as e:
            logger.error(f"[InferenceDenoise] Connection failed: {e}", exc_info=True)
            raise

    async def _reconnect(self) -> None:
        """Attempt to reconnect to the inference gateway."""
        if self._reconnecting:
            return

        self._reconnecting = True
        try:
            logger.info("[InferenceDenoise] Reconnecting...")

            await self._cleanup_connection()

            # wait a bit before reconnecting
            await asyncio.sleep(1)

            await self._connect_ws()

            # restart listener
            if self._ws_task:
                self._ws_task.cancel()
            self._ws_task = asyncio.create_task(self._listen_for_responses())

            self._stats["reconnections"] += 1
            logger.info(
                f"[InferenceDenoise] Reconnected successfully "
                f"(attempt {self._stats['reconnections']})"
            )

        except Exception as e:
            logger.error(f"[InferenceDenoise] Reconnection failed: {e}")
        finally:
            self._reconnecting = False

    async def _send_config(self) -> None:
        """Send configuration message to the inference server."""
        if not self._ws or self._ws.closed:
            raise ConnectionError("WebSocket not connected")

        config_message = {
            "type": "config",
            "data": {
                "model": self.model_id,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                **self.config,
            },
        }

        try:
            await self._ws.send_str(json.dumps(config_message))
            self._config_sent = True
            logger.info(
                f"[InferenceDenoise] Config sent: "
                f"model={self.model_id}, "
                f"sample_rate={self.sample_rate}Hz, "
                f"channels={self.channels}"
            )
        except Exception as e:
            logger.error(f"[InferenceDenoise] Failed to send config: {e}")
            raise

    async def _send_audio(self, audio_bytes: bytes) -> None:
        """Send audio data to the inference server."""
        if not self._ws or self._ws.closed:
            raise ConnectionError("WebSocket not connected")

        audio_message = {
            "type": "audio",
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
        }

        try:
            await self._ws.send_str(json.dumps(audio_message))
            self._stats["chunks_sent"] += 1
            self._stats["bytes_sent"] += len(audio_bytes)
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
            logger.error(
                f"[InferenceDenoise] Error in WebSocket listener: {e}", exc_info=True
            )
            self.emit("error", str(e))

        finally:
            pass

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
                    if audio_data:
                        denoised_bytes = base64.b64decode(audio_data)
                        await self._audio_buffer.put(denoised_bytes)

                else:
                    logger.debug(f"[InferenceDenoise] Unknown event type: {event_type}")

            elif msg_type == "audio":
                # handle audio messages (alternative format)
                audio_data = data.get("data", "")
                if audio_data:
                    denoised_bytes = base64.b64decode(audio_data)
                    await self._audio_buffer.put(denoised_bytes)

            elif msg_type == "error":
                error_data = data.get("data", {})
                error_msg = error_data.get("error") or error_data.get(
                    "message", "Unknown error"
                )
                logger.error(f"[InferenceDenoise] Server error: {error_msg}")
                self.emit("error", error_msg)
                self._stats["errors"] += 1

            elif msg_type == "stats":
                # Optional: handle statistics from server
                stats_data = data.get("data", {})
                logger.debug(f"[InferenceDenoise] Server stats: {stats_data}")

            else:
                logger.debug(f"[InferenceDenoise] Unknown message type: {msg_type}")

        except json.JSONDecodeError as e:
            logger.error(
                f"[InferenceDenoise] Failed to parse message: {e} | "
                f"raw_message: {raw_message[:100]}"
            )
        except Exception as e:
            logger.error(
                f"[InferenceDenoise] Error handling message: {e}", exc_info=True
            )

    async def _cleanup_connection(self) -> None:
        """Clean up WebSocket connection resources."""
        if self._ws and not self._ws.closed:
            try:
                stop_message = {"type": "stop"}
                await asyncio.wait_for(
                    self._ws.send_str(json.dumps(stop_message)), timeout=1.0
                )
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.debug(f"[InferenceDenoise] Error sending stop message: {e}")

            try:
                await self._ws.close()
            except Exception as e:
                logger.debug(f"[InferenceDenoise] Error closing WebSocket: {e}")

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
            "model": self.model_id,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "connected": self._ws is not None and not self._ws.closed,
        }

    async def flush(self) -> None:
        """
        Flush any remaining buffered audio.

        Note: This sends a stop message to ensure all audio is processed.
        """
        if self._ws and not self._ws.closed:
            try:
                stop_message = {"type": "stop"}
                await self._ws.send_str(json.dumps(stop_message))
                logger.info("[InferenceDenoise] Flush requested (stop message sent)")

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"[InferenceDenoise] Error during flush: {e}")

    async def aclose(self) -> None:
        """Clean up all resources."""
        logger.info(
            f"[InferenceDenoise] Closing Denoise (provider={self.provider}). "
            f"Final stats: {self.get_stats()}"
        )

        # Cancel WebSocket listener task
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await asyncio.wait_for(self._ws_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
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
        return f"videosdk.inference.Denoise.{self.provider}.{self.model_id}"
