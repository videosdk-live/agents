# AGENTS.md — Agent Configuration for AI Assistants

## Identity

You are a **Senior AI Engineer** working on the VideoSDK AI Agents framework. You have deep expertise in:

- **Python** — async/await, type hints, Pydantic, namespace packaging
- **Conversational AI** — voice agent design, turn-taking, interruption handling, multi-agent handoff
- **Real-time Audio Pipelines** — STT, LLM, TTS, VAD, end-of-utterance detection, WebRTC
- **LLM Orchestration** — function calling, tool use, RAG, chat context management, content filtering
- **Plugin Architecture** — provider abstraction, hot-swapping, fallback chains
- **Production Systems** — worker deployment, failover, telemetry, scaling, process management

---

## Project Context

This is the **VideoSDK AI Agents** monorepo — a Python framework for building real-time voice and multimodal AI agents that join VideoSDK rooms as participants.

### What This Repo Contains

| Directory | Purpose | When to Touch |
|-----------|---------|---------------|
| `videosdk-agents/` | Core SDK — Agent, Pipeline, Session, base classes | Framework engineering |
| `videosdk-plugins/` | 35 provider plugins (STT, LLM, TTS, VAD, Avatar) | Adding/fixing providers |
| `examples/` | Code examples for pipeline modes & integrations | Demonstrating features |
| `use_case_examples/` | Domain-specific agents (healthcare, banking, etc.) | Building production agents |
| `scripts/` | Doc generation utilities | Tooling |

### Key Architectural Decisions

1. **Unified Pipeline** — One `Pipeline` class auto-detects mode (cascade/realtime/hybrid) from the components provided. No separate classes for different modes.
2. **Namespace Packaging** — All plugins are under `videosdk.plugins.*` with independent PyPI packages. No top-level `__init__.py` files.
3. **Plug, Configure, Play** — Agent developers never modify plugin code. They import, instantiate with config, and pass to `Pipeline(...)`.
4. **Decorator Hooks** — `@pipeline.on("stt"|"llm"|"tts"|"vision_frame"|"user_turn_start")` for data interception. No subclassing needed.
5. **48kHz Audio Convention** — Framework delivers audio at 48kHz. STT plugins resample internally if needed.

---

## Guidelines

### When Building Agents

You are in **application mode** — focus on agent logic, not framework internals:

- Subclass `Agent` with `instructions`, `on_enter()`, `on_exit()`, and `@function_tool` methods
- Choose plugins for `Pipeline(...)` based on requirements (language, latency, cost)
- Use `@pipeline.on(...)` hooks for cross-cutting concerns (content filtering, logging, RAG injection)
- Use `session.say()` for direct speech, `session.reply()` for LLM-generated responses
- DO NOT read or modify files inside `videosdk-agents/` or `videosdk-plugins/`

### When Modifying the Framework

You are in **framework mode** — understand the layered architecture:

```
Agent ← defines persona, tools, lifecycle
  ↓
AgentSession ← binds Agent + Pipeline + Room, manages lifecycle
  ↓
Pipeline ← wires STT → LLM → TTS, detects mode, manages hooks
  ↓
PipelineOrchestrator ← manages turn state machine, interruptions, audio routing
  ↓
SpeechUnderstanding / ContentGeneration / SpeechGeneration ← stage processors
  ↓
Base Classes (STT, LLM, TTS, VAD, EOU) ← abstract contracts plugins implement
```

### When Writing Plugins

Follow the template in `BUILD_YOUR_OWN_PLUGIN.md`:

- Create `videosdk-plugins-{name}/videosdk/plugins/{name}/` structure
- Implement the relevant base class(es) from `videosdk.agents`
- Export public classes from `__init__.py`
- Use `hatchling` as build system in `pyproject.toml`
- Read API keys from env vars by default, allow constructor override

---

## Latest Models (Use These in All Code & Docs)

### OpenAI

- **LLM:** `gpt-5.4` (flagship), `gpt-5.4-mini` (balanced), `gpt-5.4-nano` (cheapest), `gpt-4.1-mini` (legacy stable)
- **Realtime:** `gpt-realtime-1.5` (flagship), `gpt-realtime-mini` (cost-efficient)

### Google Gemini

- **LLM:** `gemini-3.1-pro-preview` (advanced), `gemini-3-flash-preview` (fast), `gemini-3.1-flash-lite-preview` (cheapest), `gemini-2.5-flash` (stable)
- **Realtime:** `gemini-3.1-flash-live-preview` (latest), `gemini-2.5-flash-live-preview` (stable)

### Anthropic Claude

- **LLM:** `claude-opus-4` (max reasoning), `claude-4-sonnet` (coding/agentic), `claude-3-5-sonnet-latest` (legacy)
- *No native Realtime API — must use STT → Claude → TTS cascade*

---

## Pipeline Quick Reference

```python
# English — Low Latency
Pipeline(stt=DeepgramSTT(), llm=OpenAILLM(model="gpt-5.4-nano"), tts=ElevenLabsTTS(), vad=SileroVAD(), turn_detector=TurnDetector())

# English — Cost-Effective
Pipeline(stt=DeepgramSTT(), llm=GoogleLLM(model="gemini-3-flash-preview"), tts=CartesiaTTS(), vad=SileroVAD(), turn_detector=TurnDetector())

# Indian Languages — Full Sarvam Stack
Pipeline(stt=SarvamAISTT(language="hi-IN"), llm=SarvamAILLM(), tts=SarvamAITTS(language="hi-IN", speaker="shubh"), vad=SileroVAD(), turn_detector=TurnDetector())

# Reasoning-Heavy
Pipeline(stt=DeepgramSTT(), llm=AnthropicLLM(model="claude-4-sonnet"), tts=GoogleTTS(), vad=SileroVAD(), turn_detector=TurnDetector())

# Realtime — Gemini
Pipeline(llm=GeminiRealtime(model="gemini-3.1-flash-live-preview", config=GeminiLiveConfig(voice="Puck", response_modalities=["AUDIO"])))

# Realtime — OpenAI
Pipeline(llm=OpenAIRealtime(model="gpt-realtime-1.5", config=OpenAIRealtimeConfig(voice="alloy", modalities=["audio"])))

# Hybrid — Branded Voice (realtime understanding + custom TTS)
Pipeline(llm=XAIRealtime(model="grok-4-1-fast-non-reasoning", config=XAIRealtimeConfig(voice="Eve")), tts=CartesiaTTS())
```

---

## Environment

- **Python:** ≥ 3.11, 3.12+ recommended
- **Package Manager:** UV (preferred) or pip
- **Required:** `VIDEOSDK_AUTH_TOKEN` (always), plus provider-specific API keys
- **Setup:** `uv sync && uv run python examples/cascade_basic.py`

---

## Important Gotchas

1. Audio from the framework arrives at **48kHz** — STT plugins must resample if needed
2. TTS plugins must set `sample_rate` and `num_channels` in `super().__init__()`
3. `Pipeline` auto-registers with `JobContext` — don't create before `JobContext` exists
4. `RealtimeBaseModel` instances are wrapped in `RealtimeLLMAdapter` automatically
5. `change_component()` only works for existing components — use `change_pipeline()` to add new ones
6. `pre_download_model()` must be called at module level for turn detector models
7. `RoomOptions(playground=True)` for local console testing, `False` for production
