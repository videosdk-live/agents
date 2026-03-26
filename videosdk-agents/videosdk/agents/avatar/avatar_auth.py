from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AvatarAuthCredentials:
    """Pre-signed credentials that allow an Avatar Server to join a VideoSDK room."""

    participant_id: str
    token: str
    attributes: Optional[dict[str, str]] = None


@dataclass(frozen=True)
class AvatarJoinInfo:
    """Payload sent to the avatar dispatcher so it can join the room."""

    room_name: str
    token: str
    participant_id: Optional[str] = None
    signaling_base_url: Optional[str] = None


LINKED_AGENT_ID = "videosdk.linked_agent_id"


def generate_avatar_credentials(
    api_key: str,
    secret: str,
    *,
    participant_id: Optional[str] = None,
    ttl_seconds: int = 3600,
) -> AvatarAuthCredentials:
    """
    Generate a pre-signed VideoSDK token for an Avatar Server participant.

    Args:
        api_key: Your VideoSDK API key.
        secret: Your VideoSDK secret key.
        participant_id: Optional fixed participant ID. A random one is generated if omitted.
        ttl_seconds: Token validity in seconds (default 1 hour).

    Returns:
        AvatarAuthCredentials with participant_id and signed token.
    """
    try:
        import jwt
    except ImportError as exc:
        raise ImportError(
            "PyJWT is required for generate_avatar_credentials(). "
            "Install it with: pip install PyJWT"
        ) from exc

    pid = participant_id or f"{"avatar"}_{uuid.uuid4().hex[:8]}"
    now = int(time.time())
    payload = {
        "apikey": api_key,
        "permissions": ["allow_join"],
        "version": 2,
        "iat": now,
        "exp": now + ttl_seconds,
        "participantId": pid,
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return AvatarAuthCredentials(participant_id=pid, token=token)