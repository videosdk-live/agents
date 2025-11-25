from __future__ import annotations
import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Callable, Optional, Union
import httpx
import numpy as np
import os
import uuid
import warnings

from videosdk.meeting import Meeting
from videosdk import ReliabilityModes
from ..utils import generate_videosdk_token
from .avatar_auth import (
    AvatarJoinInfo,
    AvatarAuthCredentials,
    DEFAULT_AVATAR_IDENTITY_PREFIX,
)
import logging
from .avatar_schema import AvatarInput, AudioSegmentEnd
from av import AudioFrame
from videosdk import ReliabilityModes, MeetingEventHandler
logger = logging.getLogger(__name__)


RPC_RESET_STREAM = "reset_stream"
RPC_STREAM_ENDED = "stream_ended"


class AvatarDataChannel:
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        room_id: Optional[str] = None,
        avatar_dispatcher_url: str = "http://localhost:8089/launch",
        *,
        credentials: Optional[AvatarAuthCredentials] = None,
        credential_factory: Optional[Callable[[], AvatarAuthCredentials]] = None,
        identity_prefix: str = DEFAULT_AVATAR_IDENTITY_PREFIX,
    ):
        self._api_key = api_key
        self._secret_key = secret_key
        self._room_id = room_id
        self._avatar_dispatcher_url = avatar_dispatcher_url
        self._credential_factory = credential_factory
        self._credentials = credentials
        self._identity_prefix = identity_prefix
        self._meeting: Meeting | None = None 
        self._participant_id: str | None = (
            credentials.participant_id if credentials else None
        )
        self._chunk_size: int | None = None 
        self.video_track = None
        self.audio_track = None

    async def connect(self):
        await self._avatar_spinup()

    def set_room_id(self, room_id: str) -> None:
        self._room_id = room_id

    @property
    def participant_id(self) -> Optional[str]:
        return self._participant_id

    def _set_meeting(self, meeting: Meeting):
        """
        Injects the active meeting instance. 
        Can be called multiple times if the room reconnects.
        """
        self._meeting = meeting
        logger.info(f"AvatarDataChannel attached to meeting: {meeting.id}")


    async def _avatar_spinup(self):
        logger.info(f"Sending connection info to avatar dispatcher {self._avatar_dispatcher_url}")
        if not self._room_id:
            raise ValueError("Room ID must be set before launching the avatar worker")

        credentials = self._resolve_credentials()
        logger.info(f"Launching avatar with forced ID: {credentials.participant_id}")

        connection_info = AvatarJoinInfo(
            room_name=self._room_id,
            token=credentials.token,
            participant_id=credentials.participant_id,
        )
        async with httpx.AsyncClient() as client:
            response = await client.post(self._avatar_dispatcher_url, json=asdict(connection_info))
            response.raise_for_status()
        logger.info("Avatar handshake completed")

    def _resolve_credentials(self) -> AvatarAuthCredentials:
        if self._credentials:
            self._participant_id = self._credentials.participant_id
            return self._credentials

        if self._credential_factory:
            creds = self._credential_factory()
            if not isinstance(creds, AvatarAuthCredentials):
                raise TypeError("credential_factory must return AvatarAuthCredentials")
            self._credentials = creds
            self._participant_id = creds.participant_id
            return creds

        api_key = self._api_key or os.getenv("VIDEOSDK_API_KEY")
        secret_key = self._secret_key or os.getenv("VIDEOSDK_SECRET_KEY")
        if not api_key or not secret_key:
            raise ValueError(
                "Avatar credentials are missing. Either provide pre-minted credentials, "
                "set VIDEOSDK_API_KEY/VIDEOSDK_SECRET_KEY, or pass credential_factory."
            )

        warnings.warn(
            "Generating avatar credentials inside AvatarDataChannel is deprecated. "
            "Call JobContext.create_avatar_credentials() and pass them via the "
            "'credentials' or 'credential_factory' arguments instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        participant_id = f"{self._identity_prefix}_{uuid.uuid4().hex[:8]}"
        token = generate_videosdk_token(
            api_key=api_key,
            secret_key=secret_key,
            participant_id=participant_id,
        )
        creds = AvatarAuthCredentials(participant_id=participant_id, token=token)
        self._credentials = creds
        self._participant_id = participant_id
        return creds
    
    async def handle_audio_input(self, audio_data: bytes):
        if not self._meeting:
            logger.warning("AvatarDataChannel: Dropping audio - Meeting not connected yet.")
            return

        MAX_CHUNK_SIZE = 15000
        chunk_size = self._chunk_size or MAX_CHUNK_SIZE
        
        if chunk_size > MAX_CHUNK_SIZE:
            chunk_size = MAX_CHUNK_SIZE
        
        payloads: list[bytes] = [
            audio_data[i : i + chunk_size]
            for i in range(0, len(audio_data), chunk_size)
        ]

        for chunk in payloads:
            if not chunk:
                continue
            if len(chunk) % 2 != 0:
                chunk = chunk + b"\x00"
            await self._meeting.send(
                chunk,
                {"reliability": ReliabilityModes.UNRELIABLE.value}
            )

    async def interrupt(self):
        if not self._meeting:
            return
        await self._meeting.send(
            b"INTERRUPT",
            {"reliability": ReliabilityModes.RELIABLE.value}
        )

    async def aclose(self):
        pass


class AvatarDataHandler(MeetingEventHandler):
    """
    Dedicated handler to bridge VideoSDK Meeting events to the AvatarInput.
    """
    def __init__(self, callback: Callable[[dict], None]):
        super().__init__()
        self._callback = callback

    def on_data(self, data: dict):
        self._callback(data)


class AvatarDataChannelReceiver(AvatarInput):
    def __init__(
        self,
        meeting: Meeting | None,
        *,
        sender_identity: str | None = None,
        channels: int = 1,
    ):
        super().__init__()
        self._sender_identity = sender_identity
        self._channels = channels
        self._data_ch = asyncio.Queue[Union[AudioFrame, AudioSegmentEnd]]()
        self._handler = None
        self._meeting: Meeting | None = None
        if meeting:
            self.set_meeting(meeting)

    def set_meeting(self, meeting: Meeting):
        if self._meeting and self._handler:
            self._meeting.remove_event_listener(self._handler)
            self._handler = None

        self._meeting = meeting
        self._handler = AvatarDataHandler(callback=self.on_data)        
        self._meeting.add_event_listener(self._handler)
        
        logger.info("AvatarDataChannelReceiver attached to meeting")


    def on_data(self, data):
        payload = data.get('payload', b'')

        try:
            if isinstance(payload, str):
                self._handle_control_payload(payload)
                return
            if isinstance(payload, memoryview):
                payload = payload.tobytes()
            if not isinstance(payload, (bytes, bytearray)):
                logger.debug("Ignoring non-bytes payload on data channel")
                return
            if payload == b"INTERRUPT":
                self.emit("reset_stream")
                return
            frame_data = payload
            numpy_array = np.frombuffer(frame_data, dtype=np.int16)
            samples = len(numpy_array)
            mono_array = numpy_array.reshape(-1, 1)

            if self._channels == 2:
                stereo_array = np.column_stack([mono_array[:, 0], mono_array[:, 0]])
                numpy_array = stereo_array
            else:
                numpy_array = mono_array

            frame = AudioFrame.from_ndarray(
                numpy_array.T,
                format='s16',
                layout='mono' if self._channels == 1 else 'stereo'
            )
            frame.sample_rate = 24000

            self._data_ch.put_nowait(frame)

        except Exception as e:
            logger.error(f"Failed to process data channel message: {e}")
    
    def _handle_control_payload(self, payload: str) -> None:
        if not payload:
            return
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            logger.debug("Received non-audio string payload on data channel")
            return

        if message.get("type") == RPC_STREAM_ENDED:
            logger.debug("Received playback finished ack from avatar channel")
            return
    
    def notify_stream_ended(self, playback_position: float, interrupted: bool) -> None:
        asyncio.create_task(self._send_stream_ended(playback_position, interrupted))

    async def _send_stream_ended(self, playback_position: float, interrupted: bool):
        payload = {
            "type": RPC_STREAM_ENDED,
            "data": {
                "playback_position": playback_position,
                "interrupted": interrupted
            }
        }
        await self._meeting.send(
            json.dumps(payload),
            {"reliability": ReliabilityModes.RELIABLE.value}
        )

    def __aiter__(self) -> AsyncIterator[AudioFrame | AudioSegmentEnd]:
        return self

    async def __anext__(self) -> AudioFrame | AudioSegmentEnd:
        try:
            return await self._data_ch.get()
        except asyncio.CancelledError:
            raise StopAsyncIteration

    async def aclose(self) -> None:
        if self._meeting and self._handler:
            self._meeting.remove_event_listener(self._handler)
            self._handler = None