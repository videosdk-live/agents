---
name: videosdk-plugins
description: Parent directory for all 35 VideoSDK agent plugin packages (STT, LLM, TTS, VAD, EOU, Denoise, Avatar, Realtime)
---

# videosdk-plugins — Plugin System

## Purpose

Contains all provider-specific plugin packages for the VideoSDK AI Agents framework. Each plugin integrates a third-party AI service by implementing the framework's base classes. **Plugins are interchangeable components** — you pick one per slot (STT, LLM, TTS, VAD, EOU) and plug it into a `Pipeline`. No pipeline-level code changes are needed to swap providers.

> **Note for Agent Developers:** If you are building an agent (not modifying the framework), you do **not** need to read or modify plugin source code. Plugins are **plug, configure, and play** — just import the class, pass your API key, and hand it to `Pipeline(...)`. All the wiring is handled automatically. Your work lives in the `Agent` subclass (instructions, tools, lifecycle hooks), not in the plugins.

---

## Quick Start — Building a Pipeline

A cascade pipeline requires exactly **5 components**: STT + LLM + TTS + VAD + Turn Detector. Pick one provider per slot:

```python
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector

pipeline = Pipeline(
    stt=DeepgramSTT(),          # ← Swap for any STT provider
    llm=OpenAILLM(),            # ← Swap for any LLM provider
    tts=ElevenLabsTTS(),        # ← Swap for any TTS provider
    vad=SileroVAD(),            # ← Always SileroVAD (only option)
    turn_detector=TurnDetector() # ← Always TurnDetector (default)
)
```

For **realtime mode**, you only need the realtime model — no separate STT/TTS/VAD:

```python
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig

pipeline = Pipeline(
    llm=GeminiRealtime(model="gemini-3.1-flash-live-preview",
                       config=GeminiLiveConfig(voice="Puck", response_modalities=["AUDIO"]))
)
```

---

## Recommended Pipeline Combinations

Choose a pipeline based on your latency, quality, language, and cost requirements.

### Best for English — Low Latency

| Slot | Provider | Import | Config |
|------|----------|--------|--------|
| **STT** | Deepgram | `DeepgramSTT()` | `model="nova-3"`, `language="en"` |
| **LLM** | OpenAI | `OpenAILLM()` | `model="gpt-5.4-nano"` (cheapest), `gpt-5.4-mini` (balanced), `gpt-5.4` (flagship) |
| **TTS** | ElevenLabs | `ElevenLabsTTS()` | Voice ID via dashboard |
| **VAD** | Silero | `SileroVAD()` | — |
| **EOU** | TurnDetector | `TurnDetector()` | `threshold=0.5` |

```python
# Best for: English-only, low latency, premium voice quality
pipeline = Pipeline(
    stt=DeepgramSTT(model="nova-3", language="en"),
    llm=OpenAILLM(model="gpt-5.4-nano"),
    tts=ElevenLabsTTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)
```

### Best for English — Cost-Effective

| Slot | Provider | Import | Config |
|------|----------|--------|--------|
| **STT** | Deepgram | `DeepgramSTT()` | — |
| **LLM** | Google | `GoogleLLM()` | `model="gemini-3-flash-preview"` (latest), `gemini-2.5-flash` (stable) |
| **TTS** | Cartesia | `CartesiaTTS()` | `speed`, `volume` via `GenerationConfig` |
| **VAD** | Silero | `SileroVAD()` | — |
| **EOU** | TurnDetector | `TurnDetector()` | — |

```python
# Best for: English, cost-effective, fast LLM
pipeline = Pipeline(
    stt=DeepgramSTT(),
    llm=GoogleLLM(model="gemini-3-flash-preview"),
    tts=CartesiaTTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)
```

### Best for Indian Languages (Hindi, Tamil, etc.)

| Slot | Provider | Import | Config |
|------|----------|--------|--------|
| **STT** | Sarvam AI | `SarvamAISTT()` | `language="hi-IN"` |
| **LLM** | Google | `GoogleLLM()` | `model="gemini-3.1-flash-lite-preview"` |
| **TTS** | Sarvam AI | `SarvamAITTS()` | `language="hi-IN"`, `speaker="anushka"` |
| **VAD** | Silero | `SileroVAD()` | — |
| **EOU** | TurnDetector | `TurnDetector()` | — |

```python
# Best for: Hindi, Indian English, and regional Indian languages
pipeline = Pipeline(
    stt=SarvamAISTT(language="hi-IN"),
    llm=GoogleLLM(model="gemini-3.1-flash-lite-preview"),
    tts=SarvamAITTS(language="hi-IN", speaker="shubh"),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)
```

### Best for Full Sarvam AI Stack (Single-Vendor Indian Languages)

| Slot | Provider | Import | Config |
|------|----------|--------|--------|
| **STT** | Sarvam AI | `SarvamAISTT()` | `language="hi-IN"` |
| **LLM** | Sarvam AI | `SarvamAILLM()` | Indian language–optimized LLM |
| **TTS** | Sarvam AI | `SarvamAITTS()` | `language="hi-IN"`, `speaker="shubh"` |
| **VAD** | Silero | `SileroVAD()` | — |
| **EOU** | TurnDetector | `TurnDetector()` | — |

```python
# Best for: Full single-vendor Sarvam stack — Hindi, Indian English, regional languages
# All three providers (STT, LLM, TTS) from Sarvam AI — only SARVAMAI_API_KEY needed
from videosdk.plugins.sarvamai import SarvamAISTT, SarvamAILLM, SarvamAITTS

pipeline = Pipeline(
    stt=SarvamAISTT(language="hi-IN"),
    llm=SarvamAILLM(),
    tts=SarvamAITTS(language="hi-IN", speaker="shubh"),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)
```

### Best for Reasoning-Heavy Tasks

| Slot | Provider | Import | Config |
|------|----------|--------|--------|
| **STT** | Deepgram | `DeepgramSTT()` | — |
| **LLM** | Anthropic | `AnthropicLLM()` | `model="claude-4-sonnet"` (coding/agentic), `claude-opus-4` (max reasoning) |
| **TTS** | Google | `GoogleTTS()` | — |
| **VAD** | Silero | `SileroVAD()` | — |
| **EOU** | TurnDetector | `TurnDetector()` | — |

```python
# Best for: Complex reasoning, code generation, nuanced instructions
pipeline = Pipeline(
    stt=DeepgramSTT(),
    llm=AnthropicLLM(model="claude-4-sonnet"),
    tts=GoogleTTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)
```

### Best for Ultra-Low Latency (Realtime Mode)

```python
# Gemini Realtime — single model, no separate STT/TTS
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
pipeline = Pipeline(
    llm=GeminiRealtime(model="gemini-3.1-flash-live-preview",  # Latest live model
                       config=GeminiLiveConfig(voice="Puck", response_modalities=["AUDIO"]))
)
# Stable alternative: model="gemini-2.5-flash-live-preview"

# OpenAI Realtime — single model, no separate STT/TTS
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
pipeline = Pipeline(
    llm=OpenAIRealtime(model="gpt-realtime-1.5",  # Flagship realtime
                       config=OpenAIRealtimeConfig(voice="alloy", modalities=["audio"]))
)
# Cost-efficient alternative: model="gpt-realtime-mini"
```

### Best for Branded Voice (Hybrid Mode)

```python
# XAI handles STT + LLM (understanding), Cartesia provides custom branded TTS
from videosdk.plugins.xai import XAIRealtime, XAIRealtimeConfig
from videosdk.plugins.cartesia import CartesiaTTS

pipeline = Pipeline(
    llm=XAIRealtime(model="grok-4-1-fast-non-reasoning",
                    config=XAIRealtimeConfig(voice="Eve")),
    tts=CartesiaTTS(),  # Your custom branded voice overrides the realtime model's audio
)
```

---

## Plugin Directory Convention

Every plugin follows this exact structure:

```bash
videosdk-plugins-{service-name}/
├── pyproject.toml          ← Package metadata, dependencies
├── README.md               ← Brief description + installation
└── videosdk/
    └── plugins/
        └── {service_name}/ ← Underscores, not hyphens (Python module)
            ├── __init__.py ← Public exports
            ├── version.py  ← __version__ = "x.y.z"
            ├── stt.py      ← (optional) STT implementation
            ├── llm.py      ← (optional) LLM implementation
            ├── tts.py      ← (optional) TTS implementation
            └── ...         ← Realtime API, VAD, Avatar, Denoise, etc.
```

---

## Base Class Contracts (from videosdk-agents)

| Category | Base Class | Required Methods | Key Notes |
|----------|-----------|-----------------|-----------| 
| **STT** | `STT` | `process_audio()`, `aclose()` | Audio at 48kHz — resample if needed |
| **LLM** | `LLM` | `chat()`, `cancel_current_generation()`, `aclose()` | Must return `AsyncIterator[LLMResponse]` |
| **TTS** | `TTS` | `synthesize()`, `interrupt()`, `aclose()` | Set `sample_rate` + `num_channels` in `__init__` |
| **VAD** | `VAD` | `process_audio()`, `aclose()` | Emit via `_vad_callback` |
| **EOU** | `EOU` | `get_eou_probability()`, `aclose()` | Returns 0.0–1.0 probability |
| **Denoise** | `Denoise` | Provider-specific | Used for audio preprocessing |
| **Realtime** | `RealtimeBaseModel` | Provider-specific | Speech-to-speech models |
| **Avatar** | Protocol-based | `connect()`, `aclose()` | No formal base class |

---

## All 35 Plugins — Complete Reference

### Realtime Models (Speech-to-Speech)

These are **all-in-one models** that handle STT + LLM + TTS in a single connection. Use `Pipeline(llm=model)` — no separate STT/TTS/VAD needed.

| Plugin | Exports | Env Var | Notes |
|--------|---------|---------|-------|
| `videosdk-plugins-openai` | `OpenAIRealtime`, `OpenAIRealtimeConfig` | `OPENAI_API_KEY` | Models: `gpt-realtime-1.5` (flagship), `gpt-realtime-mini` (cost-efficient). Voices: alloy, echo, fable, onyx, nova, shimmer. Supports `tool_choice="auto"`. |
| `videosdk-plugins-google` | `GeminiRealtime`, `GeminiLiveConfig` | `GOOGLE_API_KEY` | Models: `gemini-3.1-flash-live-preview` (latest), `gemini-2.5-flash-live-preview` (stable). Voices: Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr. Set `response_modalities=["AUDIO"]`. |
| `videosdk-plugins-xai` | `XAIRealtime`, `XAIRealtimeConfig`, `XAITurnDetection` | `XAI_API_KEY` | Grok models. Supports `enable_web_search=True`. |
| `videosdk-plugins-aws` | `NovaSonicRealtime`, `NovaSonicConfig` | AWS credentials | Amazon Nova Sonic. |
| `videosdk-plugins-azure` | `AzureVoiceLive`, `AzureVoiceLiveConfig` | `AZURE_API_KEY` | Azure AI Voice Live. |
| `videosdk-plugins-ultravox` | `UltravoxRealtime`, `UltravoxLiveConfig` | `ULTRAVOX_API_KEY` | Ultravox speech model. |

### STT Providers (Speech-to-Text)

| Plugin | Export | Env Var | Key Config |
|--------|--------|---------|------------|
| `videosdk-plugins-deepgram` | `DeepgramSTT`, `DeepgramSTTV2` | `DEEPGRAM_API_KEY` | `model="nova-3"`, `language="en"`, `enable_diarization=True` |
| `videosdk-plugins-openai` | `OpenAISTT` | `OPENAI_API_KEY` | Whisper-based |
| `videosdk-plugins-google` | `GoogleSTT` | `GOOGLE_API_KEY` | `VoiceActivityConfig` for tune |
| `videosdk-plugins-cartesia` | `CartesiaSTT` | `CARTESIA_API_KEY` | — |
| `videosdk-plugins-elevenlabs` | `ElevenLabsSTT` | `ELEVENLABS_API_KEY` | — |
| `videosdk-plugins-assemblyai` | `AssemblyAISTT` | `ASSEMBLYAI_API_KEY` | — |
| `videosdk-plugins-nvidia` | `NvidiaSTT` | NVIDIA credentials | — |
| `videosdk-plugins-navana` | `NavanaSTT` | `NAVANA_API_KEY` | — |
| `videosdk-plugins-gladia` | `GladiaSTT` | `GLADIA_API_KEY` | — |
| `videosdk-plugins-sarvamai` | `SarvamAISTT` | `SARVAMAI_API_KEY` | `language="hi-IN"` — Indian languages |
| `videosdk-plugins-cometapi` | `CometAPISTT` | `COMETAPI_API_KEY` | — |
| `videosdk-plugins-cambai` | (STT not exported) | — | TTS only via CambAI |
| `videosdk-plugins-azure` | `AzureSTT` | `AZURE_API_KEY` | — |

### LLM Providers (Large Language Models)

| Plugin | Export | Env Var | Key Config |
|--------|--------|---------|------------|
| `videosdk-plugins-openai` | `OpenAILLM` | `OPENAI_API_KEY` | `gpt-5.4` (flagship), `gpt-5.4-mini` (balanced), `gpt-5.4-nano` (cheapest), `gpt-4.1-mini` (legacy stable) |
| `videosdk-plugins-google` | `GoogleLLM` | `GOOGLE_API_KEY` | `gemini-3.1-pro-preview` (advanced), `gemini-3-flash-preview` (fast), `gemini-3.1-flash-lite-preview` (cheapest), `gemini-2.5-flash` (stable). `VertexAIConfig` for Vertex AI. |
| `videosdk-plugins-anthropic` | `AnthropicLLM` | `ANTHROPIC_API_KEY` | `claude-opus-4` (max reasoning), `claude-4-sonnet` (coding/agentic), `claude-3-5-sonnet-latest` (legacy stable) |
| `videosdk-plugins-cerebras` | `CerebrasLLM` | `CEREBRAS_API_KEY` | Fast inference |
| `videosdk-plugins-sarvamai` | `SarvamAILLM` | `SARVAMAI_API_KEY` | Indian language focus |
| `videosdk-plugins-cometapi` | `CometAPILLM` | `COMETAPI_API_KEY` | — |
| `videosdk-plugins-xai` | `XAILLM` | `XAI_API_KEY` | Grok models |
| `videosdk-plugins-langchain` | `LangChainLLM`, `LangGraphLLM` | Depends on wrapped LLM | Wraps any LangChain `BaseChatModel` or LangGraph `StateGraph` |

### TTS Providers (Text-to-Speech)

| Plugin | Export | Env Var | Key Config |
|--------|--------|---------|------------|
| `videosdk-plugins-openai` | `OpenAITTS` | `OPENAI_API_KEY` | `voice="alloy"` |
| `videosdk-plugins-google` | `GoogleTTS` | `GOOGLE_API_KEY` | `GoogleVoiceConfig` |
| `videosdk-plugins-deepgram` | `DeepgramTTS` | `DEEPGRAM_API_KEY` | — |
| `videosdk-plugins-cartesia` | `CartesiaTTS` | `CARTESIA_API_KEY` | `GenerationConfig(speed=1.3, volume=1)` |
| `videosdk-plugins-elevenlabs` | `ElevenLabsTTS` | `ELEVENLABS_API_KEY` | `VoiceSettings` for stability, similarity |
| `videosdk-plugins-aws` | `AWSPollyTTS` | AWS credentials | Amazon Polly |
| `videosdk-plugins-azure` | `AzureTTS` | `AZURE_API_KEY` | `VoiceTuning`, `SpeakingStyle` |
| `videosdk-plugins-groq` | `GroqTTS` | `GROQ_API_KEY` | — |
| `videosdk-plugins-humeai` | `HumeAITTS` | `HUMEAI_API_KEY` | Emotionally expressive |
| `videosdk-plugins-inworldai` | `InworldAITTS` | `INWORLDAI_API_KEY` | Gaming & interactive |
| `videosdk-plugins-lmnt` | `LMNTTTS` | `LMNT_API_KEY` | — |
| `videosdk-plugins-murfai` | `MurfAITTS` | `MURFAI_API_KEY` | `MurfAIVoiceSettings` |
| `videosdk-plugins-neuphonic` | `NeuphonicTTS` | `NEUPHONIC_API_KEY` | — |
| `videosdk-plugins-nvidia` | `NvidiaTTS` | NVIDIA credentials | — |
| `videosdk-plugins-papla` | `PaplaTTS` | `PAPLA_API_KEY` | — |
| `videosdk-plugins-resemble` | `ResembleTTS` | `RESEMBLE_API_KEY` | Voice cloning |
| `videosdk-plugins-rime` | `RimeTTS` | `RIME_API_KEY` | — |
| `videosdk-plugins-sarvamai` | `SarvamAITTS` | `SARVAMAI_API_KEY` | `language="hi-IN"`, `speaker="shubh"` |
| `videosdk-plugins-smallestai` | `SmallestAITTS` | `SMALLESTAI_API_KEY` | — |
| `videosdk-plugins-speechify` | `SpeechifyTTS` | `SPEECHIFY_API_KEY` | — |
| `videosdk-plugins-cambai` | `CambAITTS` | `CAMBAI_API_KEY` | `InferenceOptions`, `OutputConfiguration`, `VoiceSettings` |
| `videosdk-plugins-cometapi` | `CometAPITTS` | `COMETAPI_API_KEY` | — |

### VAD / Turn Detection / Denoise / Avatar

| Plugin | Export | Role | Notes |
|--------|--------|------|-------|
| `videosdk-plugins-silero` | `SileroVAD` | **VAD** | Only VAD option. Silence/speech detection for cascade mode. |
| `videosdk-plugins-turn-detector` | `TurnDetector`, `VideoSDKTurnDetector`, `NamoTurnDetectorV1` | **EOU** | End-of-utterance detection. `pre_download_model()` at startup. `threshold=0.5` default. V2 and V3 variants available. |
| `videosdk-plugins-rnnoise` | `RNNoise` | **Denoise** | Background noise suppression. Native `.so`/`.dylib` dependency. |
| `videosdk-plugins-simli` | `SimliAvatar`, `SimliConfig` | **Avatar** | Visual avatar rendering. `faceId`, `maxSessionLength`, `maxIdleTime`. Pass as `Pipeline(avatar=SimliAvatar(...))`. |
| `videosdk-plugins-anam` | `AnamAvatar` | **Avatar** | Visual avatar rendering. `api_key`, `avatar_id`. Pass as `Pipeline(avatar=AnamAvatar(...))`. |

---

## Multi-Provider Plugins

Several plugins export classes for **multiple roles**. Here's a summary:

| Plugin | STT | LLM | TTS | Realtime |
|--------|:---:|:---:|:---:|:--------:|
| `openai` | ✅ `OpenAISTT` | ✅ `OpenAILLM` | ✅ `OpenAITTS` | ✅ `OpenAIRealtime` |
| `google` | ✅ `GoogleSTT` | ✅ `GoogleLLM` | ✅ `GoogleTTS` | ✅ `GeminiRealtime` |
| `deepgram` | ✅ `DeepgramSTT` | — | ✅ `DeepgramTTS` | — |
| `cartesia` | ✅ `CartesiaSTT` | — | ✅ `CartesiaTTS` | — |
| `elevenlabs` | ✅ `ElevenLabsSTT` | — | ✅ `ElevenLabsTTS` | — |
| `sarvamai` | ✅ `SarvamAISTT` | ✅ `SarvamAILLM` | ✅ `SarvamAITTS` | — |
| `nvidia` | ✅ `NvidiaSTT` | — | ✅ `NvidiaTTS` | — |
| `azure` | ✅ `AzureSTT` | — | ✅ `AzureTTS` | ✅ `AzureVoiceLive` |
| `xai` | — | ✅ `XAILLM` | — | ✅ `XAIRealtime` |
| `cometapi` | ✅ `CometAPISTT` | ✅ `CometAPILLM` | ✅ `CometAPITTS` | — |
| `aws` | — | — | ✅ `AWSPollyTTS` | ✅ `NovaSonicRealtime` |

---

## Agent Developer Notes

> **This section is for developers building agents, not modifying plugins.**

### You Don't Need to Touch Plugin Code

Plugins are **black boxes**. Your agent code only interacts with them through `Pipeline(...)`:

```python
# This is ALL you need to do with plugins:
pipeline = Pipeline(
    stt=DeepgramSTT(),       # Import → Instantiate → Done
    llm=OpenAILLM(),
    tts=CartesiaTTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)
```

Your agent's behavior comes from:

- `Agent.instructions` — system prompt
- `@function_tool` methods — callable tools
- `on_enter()` / `on_exit()` — lifecycle hooks
- `@pipeline.on("stt" | "tts" | "llm")` — data interception hooks

### Swapping Providers

To switch from Deepgram STT to Sarvam STT, change **one line**:

```diff
-from videosdk.plugins.deepgram import DeepgramSTT
+from videosdk.plugins.sarvamai import SarvamAISTT

 pipeline = Pipeline(
-    stt=DeepgramSTT(),
+    stt=SarvamAISTT(language="hi-IN"),
     llm=GoogleLLM(),
     ...
 )
```

No other code changes needed. The pipeline handles all the plumbing.

### API Key Configuration

All plugins read API keys from **environment variables** by default. You can also pass them explicitly:

```python
# Via environment variable (recommended):
# Set DEEPGRAM_API_KEY in your .env file
stt = DeepgramSTT()  # Reads DEEPGRAM_API_KEY automatically

# Via constructor (explicit):
stt = DeepgramSTT(api_key="your-key-here")
```

### Fallback Providers (High Availability)

Wrap any provider in `FallbackSTT`, `FallbackLLM`, or `FallbackTTS` for automatic failover:

```python
from videosdk.agents import FallbackSTT, FallbackLLM, FallbackTTS

pipeline = Pipeline(
    stt=FallbackSTT([OpenAISTT(), DeepgramSTT()], temporary_disable_sec=30.0),
    llm=FallbackLLM([OpenAILLM(), CerebrasLLM()], permanent_disable_after_attempts=3),
    tts=FallbackTTS([ElevenLabsTTS(), CartesiaTTS()]),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)
```

### VideoSDK Inference Gateway (No Third-Party Keys)

Use `videosdk.agents.inference` to route STT/TTS through VideoSDK's gateway — only `VIDEOSDK_AUTH_TOKEN` required:

```python
from videosdk.agents.inference import STT, TTS, Realtime

pipeline = Pipeline(
    stt=STT.sarvam(language="en-IN"),
    llm=GoogleLLM(),
    tts=TTS.sarvam(model_id="bulbul:v2", speaker="anushka"),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)
```

> **Note:** Inference Gateway requires Pay As You Go account type in the VideoSDK Dashboard.

---

## Common Patterns

- All plugins use **namespace packaging** (`videosdk.plugins.*`) — no top-level `__init__.py` in `videosdk/` or `videosdk/plugins/`
- Plugin `pyproject.toml` uses `hatchling` build system
- API keys are read from environment variables (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`)
- `pre_download_model()` must be called at module level for turn detector model pre-caching
- See `BUILD_YOUR_OWN_PLUGIN.md` in the repo root for the full authoring guide
