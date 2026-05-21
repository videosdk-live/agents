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
from .base_instructions import BASE_VOICE_INSTRUCTIONS
from .agent_session import AgentSession
from .utils import UserState, AgentState, PipelineMode, RealtimeMode, PipelineComponent, PipelineConfig
from .pipeline import Pipeline, EOUConfig, InterruptConfig
from .realtime_base_model import RealtimeBaseModel
from .realtime_llm_adapter import RealtimeLLMAdapter

from .metrics import metrics_collector
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
from .room.output_stream import CustomAudioStreamTrack, TeeCustomAudioStreamTrack, TeeMixingCustomAudioStreamTrack
from .event_emitter import EventEmitter
from .job import WorkerJob, JobContext, RoomOptions, RecordingOptions, Options, WebSocketConfig, WebRTCConfig, TracesOptions, MetricsOptions, LoggingOptions, ObservabilityOptions
from .worker import Worker, WorkerOptions, WorkerType
from .utterance_handle import UtteranceHandle

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
from .graph_adapter import ConversationalGraphResponse, GraphPipelineAdapter
from .llm.chat_context import (
    ChatContext,
    ChatRole,
    ChatMessage,
    ChatContent,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
    AgentHandoff,
    AgentConfigUpdate,
    ReadOnlyChatContext,
)
from .stt.stt import STT, STTResponse, SpeechEventType, SpeechData
from .tts.tts import TTS, FlushMarker
from .vad import VAD, VADResponse, VADEventType
from .mcp.mcp_server import MCPServerStdio, MCPServerHTTP
from .eou import EOU
from .event_bus import global_event_emitter, EventTypes
from .a2a.card import AgentCard
from .a2a.protocol import A2AMessage
from .images import EncodeOptions, ResizeOptions, encode
from .knowledge_base import KnowledgeBaseConfig, KnowledgeBase
from .dtmf_handler import DTMFHandler
from .voice_mail_detector import VoiceMailDetector
from .stt import FallbackSTT
from .llm import FallbackLLM
from .llm.context_window import ContextWindow
from .tts import FallbackTTS
from .utils import run_stt, run_tts
from .tokenize import (
    BasicSentenceChunker,
    BasicTextFilter,
    BufferedSentenceChunkStream,
    EnglishHyphenator,
    INDIC_LANGS,
    IndicScriptTransliterator,
    IndicSentenceChunker,
    SentenceChunkStream,
    SentenceChunker,
    TextFilter,
    detect_script,
    hyphenate_english,
    normalize_lang_code,
    pre_warm_tokenizer,
)

__all__ = [
    "Agent",
    "BASE_VOICE_INSTRUCTIONS",
    "AgentSession",
    "UserState",
    "AgentState",
    "PipelineMode",
    "RealtimeMode",
    "PipelineComponent",
    "PipelineConfig",
    "Pipeline",
    "EOUConfig",
    "InterruptConfig",
    "RealtimeBaseModel",
    "RealtimeLLMAdapter",
    "function_tool",
    "is_function_tool",
    "get_tool_info",
    "FunctionTool",
    "FunctionToolInfo",
    "CustomAudioStreamTrack",
    "TeeCustomAudioStreamTrack",
    "TeeMixingCustomAudioStreamTrack",
    "build_openai_schema",
    "build_gemini_schema",
    "ToolChoice",
    "WorkerJob",
    "LLM",
    "ChatContext",
    "AgentHandoff",
    "AgentConfigUpdate",
    "ReadOnlyChatContext",
    "ChatRole",
    "ChatMessage",
    "ChatContent",
    "FunctionCall",
    "FunctionCallOutput",
    "LLMResponse",
    "ConversationalGraphResponse",
    "GraphPipelineAdapter",
    "STT",
    "STTResponse",
    "SpeechEventType",
    "SpeechData",
    "TTS",
    "FlushMarker",
    "VAD",
    "VADResponse",
    "VADEventType",
    "EventEmitter",
    "global_event_emitter",
    "EventTypes",
    "build_nova_sonic_schema",
    "MCPServerStdio",
    "MCPServerHTTP",
    "EOU",
    "AgentCard",
    "A2AMessage",
    "EncodeOptions",
    "ResizeOptions",
    "encode",
    "JobContext",
    "RoomOptions",
    "RecordingOptions",
    "Options",
    "WebSocketConfig",
    "WebRTCConfig",
    "TracesOptions",
    "MetricsOptions",
    "LoggingOptions",
    "ObservabilityOptions",
    "metrics_collector",
    "ImageContent",
    "segment_text",
    "Worker",
    "WorkerOptions",
    "WorkerType",
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
    "BackgroundAudioHandlerConfig",
    "UtteranceHandle",
    "KnowledgeBaseConfig",
    "KnowledgeBase",
    "DTMFHandler",
    "VoiceMailDetector",
    "FallbackSTT",
    "FallbackLLM",
    "FallbackTTS",
    "run_stt",
    "run_tts",
    "ContextWindow",
    "SentenceChunker",
    "SentenceChunkStream",
    "BufferedSentenceChunkStream",
    "BasicSentenceChunker",
    "IndicSentenceChunker",
    "IndicScriptTransliterator",
    "TextFilter",
    "BasicTextFilter",
    "EnglishHyphenator",
    "hyphenate_english",
    "detect_script",
    "normalize_lang_code",
    "INDIC_LANGS",
    "pre_warm_tokenizer",
]
