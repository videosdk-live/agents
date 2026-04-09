---
name: videosdk-agents
description: Core SDK package — Agent, Pipeline, Session, base classes, and all framework internals
---

# videosdk-agents — Core Framework Package

## Purpose

This is the core `videosdk-agents` PyPI package. It contains the full agent framework: the `Agent` base class, `AgentSession`, unified `Pipeline`, pipeline orchestrator, base classes for all component types (STT, LLM, TTS, VAD, EOU, Denoise), the execution framework, MCP integration, A2A protocol, metrics/telemetry, and the room/transport layer.

## Architecture

```bash
videosdk-agents/
├── pyproject.toml              ← Package config, all optional plugin deps
├── README.md                   ← Basic usage example
└── videosdk/agents/
    ├── __init__.py             ← Public API exports (Agent, Pipeline, etc.)
    ├── agent.py                ← Agent base class (abstract: on_enter, on_exit)
    ├── agent_session.py        ← AgentSession — binds Agent + Pipeline + Room
    ├── pipeline.py             ← Unified Pipeline class (cascade/realtime/hybrid)
    ├── pipeline_orchestrator.py← Orchestrates STT→LLM→TTS flow + turn management
    ├── pipeline_hooks.py       ← @pipeline.on() hook system
    ├── pipeline_utils.py       ← Helper functions for pipeline changes
    ├── speech_understanding.py ← VAD + STT + turn detection processing
    ├── content_generation.py   ← LLM processing + tool execution
    ├── speech_generation.py    ← TTS synthesis + audio output
    ├── job.py                  ← WorkerJob, JobContext, RoomOptions
    ├── worker.py               ← Worker deployment (multi-job server mode)
    ├── utils.py                ← function_tool, schema builders, enums
    ├── event_emitter.py        ← EventEmitter base (pyee-based)
    ├── realtime_base_model.py  ← RealtimeBaseModel abstract class
    ├── realtime_llm_adapter.py ← Adapter wrapping RealtimeBaseModel as LLM
    ├── stt/stt.py              ← STT base class + SpeechEventType, STTResponse
    ├── llm/llm.py              ← LLM base class + LLMResponse, ChatContext
    ├── llm/chat_context.py     ← ChatContext, ChatRole, ChatMessage, FunctionCall
    ├── tts/tts.py              ← TTS base class
    ├── vad.py                  ← VAD base class + VADEventType, VADResponse
    ├── eou.py                  ← EOU (End of Utterance / Turn Detector) base class
    ├── denoise.py              ← Denoise base class
    ├── mcp/                    ← MCP server integration (MCPServerStdio, MCPServerHTTP)
    ├── a2a/                    ← Agent-to-Agent protocol (AgentCard, A2AMessage)
    ├── execution/              ← Process/Thread resource management, task execution
    ├── inference/              ← VideoSDK Inference gateway (STT, LLM, TTS, Realtime, Denoise)
    ├── metrics/                ← Metrics collector, OpenTelemetry, structured logging
    ├── knowledge_base/         ← KnowledgeBase integration (RAG)
    ├── room/                   ← Room management, audio tracks, transport handlers
    ├── transports/             ← WebRTC and WebSocket transport handlers
    ├── avatar/                 ← Avatar controller and audio I/O
    ├── debug/                  ← HTTP debug server, tracing
    └── resources/              ← Static assets (keyboard.wav, classical.wav)
```

## Key Base Classes & Contracts

### Agent (agent.py)

```python
class Agent(EventEmitter, ABC):
    def __init__(self, instructions: str, tools: List[FunctionTool], ...)
    @abstractmethod async def on_enter(self) -> None    # Called when session starts
    @abstractmethod async def on_exit(self) -> None     # Called when session ends
    # Properties: instructions, tools, session, chat_context
    # Methods: hangup(), set_thinking_audio(), play_background_audio(), register_a2a()
```

### Pipeline (pipeline.py)

```python
class Pipeline(EventEmitter):
    def __init__(self, stt, llm, tts, vad, turn_detector, avatar, denoise,
                 eou_config, interrupt_config, ...)
    def on(event: str)              # Hook registration (@pipeline.on("llm"))
    async def change_pipeline(...)  # Full mode switch (cascade ↔ realtime)
    async def change_component(...) # Hot-swap individual components
    # Pipeline auto-detects mode from components provided
```

### STT (stt/stt.py)

```python
class STT(EventEmitter):
    @abstractmethod async def process_audio(self, audio_frames: bytes, ...)
    async def aclose(self)
    # Audio arrives at 48kHz — resample if your provider needs different
```

### LLM (llm/llm.py)

```python
class LLM(EventEmitter):
    @abstractmethod async def chat(self, messages: ChatContext, tools, ...) -> AsyncIterator[LLMResponse]
    @abstractmethod async def cancel_current_generation(self)
    async def aclose(self)
```

### TTS (tts/tts.py)

```python
class TTS(EventEmitter):
    def __init__(self, sample_rate=16000, num_channels=1)
    @abstractmethod async def synthesize(self, text: str | AsyncIterator[str], ...)
    @abstractmethod async def interrupt(self)
    # Push audio via: self.audio_track.add_new_bytes(chunk)
```

### VAD (vad.py)

```python
class VAD(EventEmitter):
    @abstractmethod async def process_audio(self, audio_frames: bytes, ...)
```

### EOU (eou.py)

```python
class EOU(EventEmitter):
    @abstractmethod def get_eou_probability(self, chat_context: ChatContext) -> float
```

### RealtimeBaseModel (realtime_base_model.py)

```python
class RealtimeBaseModel(EventEmitter):
    # Base for speech-to-speech models (OpenAI Realtime, Gemini Live, Nova Sonic, etc.)
    # Emits: "realtime_model_transcription"
```

## Common Patterns

1. **function_tool decorator**: Use `@function_tool` on standalone functions or agent methods
2. **Pipeline hooks**: `@pipeline.on("stt"|"llm"|"tts"|"vision_frame"|"user_turn_start"|...)`
3. **Dynamic pipeline changes**: `pipeline.change_pipeline()` for mode switches, `pipeline.change_component()` for hot-swaps
4. **Event-driven**: All components extend `EventEmitter` and emit `"error"` events
5. **Async lifecycle**: All components implement `async aclose()` for cleanup

## Gotchas & Important Notes

- Audio from the framework arrives at **48kHz** — STT plugins must resample if needed
- TTS plugins must set correct `sample_rate` and `num_channels` in `super().__init__()`
- TTS audio output goes through `self.audio_track.add_new_bytes(chunk)` — the track is set by the framework
- `Pipeline` auto-registers with `JobContext` on creation — don't create before `JobContext` exists
- `RealtimeBaseModel` instances are wrapped in `RealtimeLLMAdapter` automatically by Pipeline
- `change_component()` only works for existing components — use `change_pipeline()` to add new ones
