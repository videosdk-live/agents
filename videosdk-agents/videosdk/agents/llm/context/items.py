from __future__ import annotations

import base64
import time
import uuid
from enum import Enum
from typing import Any, List, Literal, Optional, Union

import av
from pydantic import BaseModel, ConfigDict, Field

from ... import images
from ...images import EncodeOptions, ResizeOptions


class ChatRole(str, Enum):
    """Roles used in chat conversations."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"


class ImageContent(BaseModel):
    """Image content in a chat message."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: f"img_{uuid.uuid4().hex[:12]}")
    type: Literal["image"] = "image"
    image: Union[av.VideoFrame, str]
    inference_detail: Literal["auto", "high", "low"] = "auto"
    encode_options: EncodeOptions = Field(
        default_factory=lambda: EncodeOptions(
            format="JPEG",
            quality=90,
            resize_options=ResizeOptions(width=320, height=240),
        )
    )

    def to_data_url(self) -> str:
        """Convert the image to a data URL string."""
        if isinstance(self.image, str):
            return self.image
        encoded_image = images.encode(self.image, self.encode_options)
        b64_image = base64.b64encode(encoded_image).decode("utf-8")
        return f"data:image/{self.encode_options.format.lower()};base64,{b64_image}"


ChatContent = Union[str, ImageContent]


class _ChatItemBase(BaseModel):
    """Shared base for every conversation item.

    Provides creation time and multi-agent attribution. ``agent_id`` is the
    agent that produced the item; it is nullable for backward compatibility.
    """

    created_at: float = Field(default_factory=time.time)
    agent_id: Optional[str] = None


class FunctionCall(_ChatItemBase):
    """A tool invocation initiated by the language model."""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    type: Literal["function_call"] = "function_call"
    name: str
    arguments: str
    call_id: str
    metadata: Optional[dict] = None


class FunctionCallOutput(_ChatItemBase):
    """The result of a tool execution."""

    id: str = Field(default_factory=lambda: f"output_{uuid.uuid4().hex[:12]}")
    type: Literal["function_call_output"] = "function_call_output"
    name: str
    call_id: str
    output: str
    is_error: bool = False


class ChatMessage(_ChatItemBase):
    """A user, assistant, system, or developer utterance."""

    role: ChatRole
    content: Union[str, List[ChatContent]]
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    type: Literal["message"] = "message"
    interrupted: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = None
    metrics: Optional[dict] = None
    audio_instructions: Optional[str] = None
    text_instructions: Optional[str] = None

    def instructions_for_modality(
        self, modality: Literal["audio", "text"]
    ) -> Union[str, List[ChatContent]]:
        """Return the instruction variant for the given input modality.

        Falls back to ``content`` when no modality-specific variant is set.
        """
        if modality == "audio" and self.audio_instructions is not None:
            return self.audio_instructions
        if modality == "text" and self.text_instructions is not None:
            return self.text_instructions
        return self.content


class AgentHandoff(_ChatItemBase):
    """Records a transfer of control between agents.

    Structural item — excluded from provider conversion.
    """

    id: str = Field(default_factory=lambda: f"handoff_{uuid.uuid4().hex[:12]}")
    type: Literal["agent_handoff"] = "agent_handoff"
    from_agent: Optional[str] = None
    to_agent: str
    reason: Optional[str] = None


class AgentConfigUpdate(_ChatItemBase):
    """Records a mid-conversation change to an agent's instructions or tools.

    Structural item — excluded from provider conversion; feeds active-config
    resolution.
    """

    id: str = Field(default_factory=lambda: f"cfgupd_{uuid.uuid4().hex[:12]}")
    type: Literal["agent_config_update"] = "agent_config_update"
    instructions: Optional[str] = None
    tools: Optional[List[str]] = None


ChatItem = Union[
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    AgentHandoff,
    AgentConfigUpdate,
]
