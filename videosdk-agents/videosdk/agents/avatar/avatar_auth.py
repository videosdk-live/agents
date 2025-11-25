from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AvatarAuthCredentials:
    """Pre-signed credentials that allow an avatar worker to join a VideoSDK room."""
    participant_id: str
    token: str
    attributes: Optional[dict[str, str]] = None


@dataclass(frozen=True)
class AvatarJoinInfo:
    """Payload sent to avatar dispatcher services so they can join the room."""

    room_name: str
    token: str
    participant_id: Optional[str] = None
    signaling_base_url: Optional[str] = None

AVATAR_PROXY_FOR = "videosdk.publish_on_behalf"
DEFAULT_AVATAR_IDENTITY_PREFIX = "avatar"

