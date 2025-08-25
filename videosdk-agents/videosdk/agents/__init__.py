import logging
import sys


# Configure logging for the videosdk-agents module
def setup_logging(level=logging.INFO):
    """Setup logging configuration for videosdk-agents."""
    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Get the logger for videosdk.agents
    logger = logging.getLogger("videosdk.agents")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add our handler
    logger.addHandler(console_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


# Note: Logging is now configured automatically when creating a Worker instance
# based on the log_level field in WorkerOptions. No manual setup required.

from .agent import Agent
from .agent_session import AgentSession
from .conversation_flow import ConversationFlow
from .realtime_base_model import RealtimeBaseModel
from .realtime_pipeline import RealTimePipeline
from .metrics import realtime_metrics_collector
from .utils import (
    function_tool,
    is_function_tool,
    get_tool_info,
    FunctionTool,
    FunctionToolInfo,
    build_openai_schema,
    build_gemini_schema,
    ToolChoice,
    build_nova_sonic_schema,
    segment_text,
)
from .room.audio_stream import CustomAudioStreamTrack, TeeCustomAudioStreamTrack
from .event_emitter import EventEmitter
from .job import WorkerJob, JobContext, RoomOptions, Options
from .worker import Worker, WorkerOptions, WorkerType, DirectRoomOptions

# New execution module exports
from .execution import (
    ExecutorType,
    ResourceType,
    TaskType,
    ResourceConfig,
    TaskConfig,
    TaskResult,
    TaskStatus,
    ResourceStatus,
    ResourceInfo,
    HealthMetrics,
    ResourceManager,
    ProcessResource,
    ThreadResource,
    TaskExecutor,
)
from .execution.inference_resource import DedicatedInferenceResource

from .llm.llm import LLM, LLMResponse
from .llm.chat_context import (
    ChatContext,
    ChatRole,
    ChatMessage,
    ChatContent,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
)
from .stt.stt import STT, STTResponse, SpeechEventType, SpeechData
from .tts.tts import TTS
from .vad import VAD, VADResponse, VADEventType
from .cascading_pipeline import CascadingPipeline
from .mcp.mcp_server import MCPServerStdio, MCPServerHTTP
from .eou import EOU
from .event_bus import global_event_emitter, EventTypes
from .a2a.card import AgentCard
from .a2a.protocol import A2AMessage
from .images import EncodeOptions, ResizeOptions, encode

__all__ = [
    "Agent",
    "AgentSession",
    "ConversationFlow",
    "RealtimeBaseModel",
    "RealTimePipeline",
    "function_tool",
    "is_function_tool",
    "get_tool_info",
    "FunctionTool",
    "FunctionToolInfo",
    "CustomAudioStreamTrack",
    "TeeCustomAudioStreamTrack",
    "build_openai_schema",
    "build_gemini_schema",
    "ToolChoice",
    "WorkerJob",
    "LLM",
    "ChatContext",
    "ChatRole",
    "ChatMessage",
    "ChatContent",
    "FunctionCall",
    "FunctionCallOutput",
    "LLMResponse",
    "STT",
    "STTResponse",
    "SpeechEventType",
    "SpeechData",
    "TTS",
    "VAD",
    "VADResponse",
    "VADEventType",
    "EventEmitter",
    "global_event_emitter",
    "EventTypes",
    "CascadingPipeline",
    "build_nova_sonic_schema",
    "MCPServerStdio",
    "MCPServerHTTP",
    "ConversationFlow",
    "EOU",
    "AgentCard",
    "A2AMessage",
    "EncodeOptions",
    "ResizeOptions",
    "encode",
    "JobContext",
    "RoomOptions",
    "Options",
    "realtime_metrics_collector",
    "ImageContent",
    "segment_text",
    "Worker",
    "WorkerOptions",
    "WorkerType",
    "DirectRoomOptions",
    # New execution module exports
    "ExecutorType",
    "ResourceType",
    "TaskType",
    "ResourceConfig",
    "TaskConfig",
    "TaskResult",
    "TaskStatus",
    "ResourceStatus",
    "ResourceInfo",
    "HealthMetrics",
    "ResourceManager",
    "ProcessResource",
    "ThreadResource",
    "TaskExecutor",
    "DedicatedInferenceResource",
    "setup_logging",
]
