from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Callable, Optional

import httpx
import numpy as np
from av import AudioFrame
from videosdk import ReliabilityModes, MeetingEventHandler
from videosdk.meeting import Meeting

from .avatar_auth import (
    AvatarAuthCredentials,
    AvatarJoinInfo,
)
from .avatar_schema import AvatarInput, AudioSegmentEnd

logger = logging.getLogger(__name__)

MSG_INTERRUPT = b"INTERRUPT"
MSG_TYPE_SEGMENT_END = "segment_end"
MSG_TYPE_STREAM_ENDED = "stream_ended"


class _AvatarAckHandler(MeetingEventHandler):
    """
    Listens for ``stream_ended`` JSON messages sent back by the Avatar Server
    and forwards them to a callback on the agent side.
    """

    def __init__(self, on_stream_ended: Callable[[float, bool], None]):
        super().__init__()
        self._on_stream_ended = on_stream_ended

    def on_data(self, data: dict) -> None:
        payload = data.get("payload", b"")
        if not isinstance(payload, (str, bytes, bytearray, memoryview)):
            return
        if isinstance(payload, (bytes, bytearray, memoryview)):
            try:
                payload = bytes(payload).decode("utf-8")
            except Exception:
                return
        try:
            msg = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return
        if msg.get("type") == MSG_TYPE_STREAM_ENDED:
            inner = msg.get("data", {})
            self._on_stream_ended(
                float(inner.get("playback_position", 0.0)),
                bool(inner.get("interrupted", False)),
            )


class AvatarAudioOut:
    """
    Agent-side handle for the avatar data channel.

    Responsibilities:
    - Spin up the Avatar Server via an HTTP dispatcher.
    - Stream raw PCM audio chunks to the worker (UNRELIABLE).
    - Send ``segment_end`` control messages (RELIABLE) so the worker knows
      when a TTS turn has finished — this is what allows ``notify_stream_ended``
      to fire on the worker side.
    - Send ``INTERRUPT`` (RELIABLE) when the agent interrupts its output.
    - Receive ``stream_ended`` acks from the worker via an on_data listener.
    """

    def __init__(
        self,
        *,
        credentials: AvatarAuthCredentials,
        avatar_dispatcher_url: Optional[str] = None,
        room_id: Optional[str] = None,
    ):
        self._credentials = credentials
        self._avatar_dispatcher_url = avatar_dispatcher_url
        self._room_id = room_id
        self._meeting: Meeting | None = None
        self._ack_handler: _AvatarAckHandler | None = None
        self._participant_id: str = credentials.participant_id
        self.video_track = None
        self.audio_track = None
        
    def set_room_id(self, room_id: str) -> None:
        self._room_id = room_id

    @property
    def participant_id(self) -> str:
        return self._participant_id

    async def connect(self) -> None:
        """Call the avatar dispatcher so the worker process joins the room."""
        await self._avatar_spinup()

    def _set_meeting(self, meeting: Meeting) -> None:
        """
        Inject the live Meeting object. Called by the framework after the agent
        has joined the room. Also registers the ack listener.
        """
        self._meeting = meeting
        self._ack_handler = _AvatarAckHandler(on_stream_ended=self._on_stream_ended)
        self._meeting.add_event_listener(self._ack_handler)
        logger.info("AvatarAudioOut attached to meeting: %s", meeting.id)

    async def _avatar_spinup(self) -> None:
        if not self._avatar_dispatcher_url:
            logger.info("AvatarAudioOut: No dispatcher URL provided, skipping local avatar spinup.")
            return

        if not self._room_id:
            raise ValueError("room_id must be set before calling connect()")

        join_info = AvatarJoinInfo(
            room_name=self._room_id,
            token=self._credentials.token,
            participant_id=self._credentials.participant_id,
        )
        logger.info(
            "Sending connection info to avatar dispatcher %s (participant_id=%s)",
            self._avatar_dispatcher_url,
            self._credentials.participant_id,
        )
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._avatar_dispatcher_url, json=asdict(join_info)
            )
            response.raise_for_status()
        logger.info("Avatar handshake completed")


    async def handle_audio_input(self, audio_data: bytes) -> None:
        """
        Chunk and send raw PCM bytes to the Avatar Server via data channel.
        Uses UNRELIABLE mode for low-latency streaming.
        """
        if not self._meeting:
            return

        MAX_CHUNK = 15_000
        for i in range(0, len(audio_data), MAX_CHUNK):
            chunk = audio_data[i : i + MAX_CHUNK]
            if not chunk:
                continue
            if len(chunk) % 2 != 0:
                chunk = chunk + b"\x00"
            try:
                await self._meeting.send(
                    chunk, {"reliability": ReliabilityModes.UNRELIABLE.value}
                )
            except Exception:
                # Data channel closed (e.g. participant left) — stop sending
                logger.debug("AvatarAudioOut: data channel closed, dropping remaining audio")
                self._meeting = None
                return

    async def send_segment_end(self) -> None:
        """
        Notify the Avatar Server that the current TTS segment has finished.
        This causes the receiver to enqueue AudioSegmentEnd so the controller
        can call notify_stream_ended(interrupted=False).
        """
        if not self._meeting:
            return
        try:
            payload = json.dumps({"type": MSG_TYPE_SEGMENT_END})
            await self._meeting.send(
                payload, {"reliability": ReliabilityModes.RELIABLE.value}
            )
            logger.debug("AvatarAudioOut: sent segment_end")
        except Exception:
            self._meeting = None

    async def interrupt(self) -> None:
        """Tell the Avatar Server to immediately stop playback."""
        if not self._meeting:
            return
        try:
            await self._meeting.send(
                MSG_INTERRUPT, {"reliability": ReliabilityModes.RELIABLE.value}
            )
            logger.info("AvatarAudioOut: sent INTERRUPT to Avatar Server")
        except Exception:
            self._meeting = None


    def _on_stream_ended(self, playback_position: float, interrupted: bool) -> None:
        logger.info(
            "AvatarAudioOut: stream_ended ack received (pos=%.3fs, interrupted=%s)",
            playback_position,
            interrupted,
        )

    async def aclose(self) -> None:
        if self._meeting and self._ack_handler:
            try:
                self._meeting.remove_event_listener(self._ack_handler)
            except Exception:
                pass
        self._ack_handler = None


class _AvatarDataHandler(MeetingEventHandler):
    """Bridges VideoSDK on_data events to the receiver callback."""
    def __init__(self, callback: Callable[[dict], None]):
        super().__init__()
        self._callback = callback

    def on_data(self, data: dict) -> None:
        self._callback(data)


class AvatarAudioIn(AvatarInput):
    """
    Avatar-worker-side receiver.

    Listens for data-channel messages from the agent and exposes them as an
    async iterator of AudioFrame / AudioSegmentEnd items. Control messages:
    - Raw bytes  → reconstruct AudioFrame and enqueue it.
    - INTERRUPT  → clear the queue and emit ``reset_stream``.
    - segment_end JSON → enqueue AudioSegmentEnd.

    Note: VideoSDK broadcasts data-channel messages to all participants, so
    every participant in the room sees every message.
    """

    _INTERRUPT_COOLDOWN = 0.3

    def __init__(
        self,
        meeting: Meeting | None,
        *,
        channels: int = 1,
        sample_rate: int = 24000,
    ):
        super().__init__()
        self._channels = channels
        self._sample_rate = sample_rate
        self._data_ch: asyncio.Queue[AudioFrame | AudioSegmentEnd] = asyncio.Queue()
        self._handler: _AvatarDataHandler | None = None
        self._meeting: Meeting | None = None
        self._interrupt_until: float = 0.0 
        if meeting:
            self.set_meeting(meeting)

    def set_meeting(self, meeting: Meeting) -> None:
        if self._meeting and self._handler:
            try:
                self._meeting.remove_event_listener(self._handler)
            except Exception:
                pass
            self._handler = None

        self._meeting = meeting
        self._handler = _AvatarDataHandler(callback=self._on_data)
        self._meeting.add_event_listener(self._handler)
        logger.info("AvatarAudioIn attached to meeting")

    def notify_stream_ended(self, playback_position: float, interrupted: bool) -> None:
        asyncio.create_task(self._send_stream_ended(playback_position, interrupted))

    def __aiter__(self):
        return self

    async def __anext__(self) -> AudioFrame | AudioSegmentEnd:
        try:
            return await self._data_ch.get()
        except asyncio.CancelledError:
            raise StopAsyncIteration

    async def aclose(self) -> None:
        if self._meeting and self._handler:
            try:
                self._meeting.remove_event_listener(self._handler)
            except Exception:
                pass
            self._handler = None

    def _on_data(self, data: dict) -> None:
        payload = data.get("payload", b"")
        try:
            if isinstance(payload, memoryview):
                payload = payload.tobytes()

            if payload == MSG_INTERRUPT or payload == "INTERRUPT":
                self._handle_interrupt()
                return

            if isinstance(payload, (bytes, bytearray)):
                self._handle_audio_bytes(payload)
                return

            if isinstance(payload, str):
                self._handle_text_payload(payload)
        except Exception as e:
            logger.error("AvatarAudioIn: error processing message: %s", e)

    def _handle_audio_bytes(self, raw: bytes) -> None:
        import time as _time
        if _time.monotonic() < self._interrupt_until:
            return
        if len(raw) % 2 != 0:
            raw = raw + b"\x00"
        array = np.frombuffer(raw, dtype=np.int16)
        mono = array.reshape(-1, 1)
        if self._channels == 2:
            stereo = np.column_stack([mono[:, 0], mono[:, 0]])
            array_out = stereo
        else:
            array_out = mono
        frame = AudioFrame.from_ndarray(
            array_out.T,
            format="s16",
            layout="mono" if self._channels == 1 else "stereo",
        )
        frame.sample_rate = self._sample_rate
        self._data_ch.put_nowait(frame)

    def _handle_interrupt(self) -> None:
        import time as _time
        self._interrupt_until = _time.monotonic() + self._INTERRUPT_COOLDOWN
        while not self._data_ch.empty():
            try:
                self._data_ch.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.emit("reset_stream")
        logger.info("AvatarAudioIn: INTERRUPT received, buffer cleared, cooldown %.1fs", self._INTERRUPT_COOLDOWN)

    def _handle_text_payload(self, payload: str) -> None:
        if not payload:
            return
        try:
            msg = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return

        if msg.get("type") == MSG_TYPE_SEGMENT_END:
            self._data_ch.put_nowait(AudioSegmentEnd())
            logger.debug("AvatarAudioIn: segment_end received")
        elif msg.get("type") == MSG_TYPE_STREAM_ENDED:
            pass

    async def _send_stream_ended(self, playback_position: float, interrupted: bool) -> None:
        if not self._meeting:
            return
        payload = json.dumps(
            {
                "type": MSG_TYPE_STREAM_ENDED,
                "data": {
                    "playback_position": playback_position,
                    "interrupted": interrupted,
                },
            }
        )
        await self._meeting.send(
            payload, {"reliability": ReliabilityModes.RELIABLE.value}
        )
        logger.debug(
            "AvatarAudioIn: sent stream_ended (pos=%.3fs, interrupted=%s)",
            playback_position,
            interrupted,
        )