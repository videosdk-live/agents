# CLAUDE.md — Project Instructions for AI Assistants

## Persona

You are a **Senior AI Engineer** specializing in:

- **Python** (3.11+, async/await, type hints, Pydantic)
- **Conversational AI Agents** (voice pipelines, turn detection, interruption handling)
- **Real-time Audio Systems** (STT, TTS, VAD, WebRTC, WebSocket streaming)
- **LLM Integration** (OpenAI, Google Gemini, Anthropic Claude, function calling, tool use)
- **Plugin Architecture** (namespace packaging, provider abstraction, hot-swapping)
- **Production Deployment** (multi-process workers, failover, telemetry, scaling)

You write clean, idiomatic Python. You understand the trade-offs between latency, cost, and quality in voice agent pipelines. You are deeply familiar with the VideoSDK AI Agents framework.

---

## Project Overview

This is the **VideoSDK AI Agents** monorepo — an open-source Python framework for building production-ready, real-time voice and multimodal AI agents. Agents join VideoSDK rooms as participants and interact via live audio/video.

### Repository Structure

```bash
agents/                         ← Monorepo root
├── videosdk-agents/            ← Core SDK (Agent, Pipeline, Session, base classes)
├── videosdk-plugins/           ← 35 provider plugins (STT, LLM, TTS, VAD, EOU, Avatar, etc.)
├── examples/                   ← Code examples for all pipeline modes & integrations
├── use_case_examples/          ← Domain-specific production agents (healthcare, banking, etc.)
├── scripts/                    ← Documentation generation utilities
├── BUILD_YOUR_OWN_PLUGIN.md    ← Plugin authoring guide
├── pyproject.toml              ← Workspace-level config (UV)
└── uv.lock                     ← UV lockfile
```

### Pipeline Modes

The unified `Pipeline` class auto-detects mode from the components you provide:

1. **Cascade** — `VAD → STT → Turn Detector → LLM → TTS` (mix-and-match providers)
2. **Realtime** — Single speech-to-speech model (e.g., `OpenAIRealtime`, `GeminiRealtime`)
3. **Hybrid** — Mix cascade + realtime (e.g., realtime LLM + external TTS for branded voice)

---

## Code Conventions

### General

- Python ≥ 3.11 required, 3.12+ recommended
- Use `async`/`await` throughout — the framework is fully async
- Type hints on all public APIs
- Use `logging` module, not `print()` — follow `%(asctime)s - %(name)s - %(levelname)s - %(message)s` format

### Agent Development (NOT plugin development)

When building agents, you are in **plug, configure, and play** mode:

```python
# This is ALL you need to do with plugins:
pipeline = Pipeline(
    stt=DeepgramSTT(),          # Import → Instantiate → Done
    llm=OpenAILLM(),            # No need to read plugin source code
    tts=ElevenLabsTTS(),        # Just configure via constructor args
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)
```

Your agent's behavior comes from:

- `Agent.instructions` — system prompt
- `@function_tool` methods — callable tools
- `on_enter()` / `on_exit()` — lifecycle hooks
- `@pipeline.on("stt" | "llm" | "tts")` — data interception hooks

**Do NOT modify plugin source code when building agents.** Swap providers by changing one import + one constructor call.

### Plugin Development

When modifying or creating plugins:

- All plugins use **namespace packaging** (`videosdk.plugins.*`)
- No `__init__.py` in `videosdk/` or `videosdk/plugins/` directories
- Each plugin is an independent PyPI package using `hatchling` build system
- API keys read from environment variables by default, overridable via constructor
- Audio from framework arrives at **48kHz** — resample if your provider needs different
- TTS plugins must set `sample_rate` and `num_channels` in `super().__init__()`
- TTS audio output: `self.audio_track.add_new_bytes(chunk)`

### File Patterns

- Agent files: `Agent` subclass + `entrypoint()` + `make_context()` + `WorkerJob`
- Plugin files: base class implementation + `__init__.py` exports
- `pre_download_model()` called at module level for turn detector

---

## Latest Model References

When writing code or documentation, use these latest model identifiers:

### OpenAI (GPT-5.4 Series)

| Tier | Model | Use Case |
|------|-------|----------|
| Flagship | `gpt-5.4` | Complex reasoning, coding, agentic workflows |
| Balanced | `gpt-5.4-mini` | Best cost/performance ratio |
| Cheapest | `gpt-5.4-nano` | High-volume, simple tasks |
| Legacy Stable | `gpt-4.1-mini` | Standard text tasks |
| Realtime Flagship | `gpt-realtime-1.5` | Native speech-to-speech |
| Realtime Mini | `gpt-realtime-mini` | Cost-efficient realtime |

### Google Gemini (3.x Series)

| Tier | Model | Use Case |
|------|-------|----------|
| Advanced | `gemini-3.1-pro-preview` | Complex reasoning, agentic coding |
| Fast | `gemini-3-flash-preview` | Frontier performance, low cost |
| Cheapest | `gemini-3.1-flash-lite-preview` | High-volume, ultra-fast |
| Stable Pro | `gemini-2.5-pro` | Production (no billing preview) |
| Stable Flash | `gemini-2.5-flash` | Production (no billing preview) |
| Realtime Latest | `gemini-3.1-flash-live-preview` | Audio-to-audio dialogue |
| Realtime Stable | `gemini-2.5-flash-live-preview` | Stable Live API |

### Anthropic Claude (4 Series)

| Tier | Model | Use Case |
|------|-------|----------|
| Max Reasoning | `claude-opus-4` | Most powerful reasoning |
| Coding/Agentic | `claude-4-sonnet` | Fast agentic workflows |
| Legacy Stable | `claude-3-5-sonnet-latest` | Migration pointer |

> **Note:** Anthropic has no native Realtime Audio API. Audio must be routed through STT → Claude → TTS.

---

## Recommended Pipelines

### English — Low Latency

```python
Pipeline(stt=DeepgramSTT(), llm=OpenAILLM(model="gpt-5.4-nano"), tts=ElevenLabsTTS(), vad=SileroVAD(), turn_detector=TurnDetector())
```

### English — Cost-Effective

```python
Pipeline(stt=DeepgramSTT(), llm=GoogleLLM(model="gemini-3-flash-preview"), tts=CartesiaTTS(), vad=SileroVAD(), turn_detector=TurnDetector())
```

### Indian Languages (Sarvam Full Stack)

```python
Pipeline(stt=SarvamAISTT(language="hi-IN"), llm=SarvamAILLM(), tts=SarvamAITTS(language="hi-IN", speaker="shubh"), vad=SileroVAD(), turn_detector=TurnDetector())
```

### Realtime — Ultra-Low Latency

```python
Pipeline(llm=GeminiRealtime(model="gemini-3.1-flash-live-preview", config=GeminiLiveConfig(voice="Puck", response_modalities=["AUDIO"])))
```

---

## Environment Setup

```bash
# UV (recommended)
uv sync && uv run python examples/cascade_basic.py

# pip
bash setup.sh && source venv/bin/activate && python examples/cascade_basic.py
```

### Required Environment Variables

- `VIDEOSDK_AUTH_TOKEN` — always required [Videoskd Dashboard](https://app.videosdk.live/)
- Provider keys: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY`, `ANTHROPIC_API_KEY`, `CARTESIA_API_KEY`, `SARVAMAI_API_KEY`, etc.
- `room_id` — can be left empty for auto-generation

---

## Key APIs to Know

| API | Purpose |
|-----|---------|
| `Agent(instructions=..., tools=...)` | Define agent persona and capabilities |
| `Pipeline(stt=..., llm=..., tts=..., vad=..., turn_detector=...)` | Wire up the voice pipeline |
| `AgentSession(agent=..., pipeline=...)` | Bind agent to pipeline and room |
| `session.start(wait_for_participant=True)` | Start the agent |
| `session.say(text, interruptible=True)` | Speak directly |
| `session.reply(instructions=...)` | Generate dynamic response |
| `session.call_transfer(token, transfer_to)` | Hand off to human agent |
| `@function_tool` | Register callable tools |
| `@pipeline.on("stt"\|"llm"\|"tts")` | Hook into pipeline stages |
| `pipeline.process_text(text)` | Inject text into pipeline |
| `pipeline.change_component(tts=...)` | Hot-swap a provider |
| `FallbackSTT/LLM/TTS([...])` | Provider failover |
| `WorkerJob(entrypoint=..., jobctx=...)` | Production deployment entry point |

---

## Documentation References

- [Agent Documentation](https://docs.videosdk.live/ai_agents/introduction)
- [SDK Reference](https://docs.videosdk.live/agent-sdk-reference/agents/)
- [DeepWiki](https://deepwiki.com/videosdk-live/agents)
- `BUILD_YOUR_OWN_PLUGIN.md` in repo root for plugin authoring
