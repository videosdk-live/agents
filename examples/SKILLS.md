---
name: examples
description: Code examples demonstrating all pipeline modes, integrations, and advanced features
---

# examples — Agent Code Examples

## Purpose

Contains runnable Python examples demonstrating how to use the VideoSDK AI Agents framework across all pipeline modes (cascade, realtime, hybrid) and major integrations.

---

## Core Pipeline Modes

### `cascade_basic.py`

A foundational cascade-mode voice agent using **DeepgramSTT → GoogleLLM → CartesiaTTS** with SileroVAD and TurnDetector. Demonstrates:

- Defining a custom `Agent` subclass with `on_enter`/`on_exit` lifecycle hooks
- Registering external `@function_tool` (weather API via aiohttp) and agent-method `@function_tool` (horoscope lookup)
- Creating a full cascade `Pipeline` and wiring it with `AgentSession`
- Using `WorkerJob` with `make_context()` / `RoomOptions(playground=True)` for local testing

**Plugins used:** `deepgram` (STT), `google` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `cascade_advanced.py`

Extends the cascade pattern with **advanced EOU and interruption configuration**. Demonstrates:

- `EOUConfig(mode='ADAPTIVE', min_max_speech_wait_timeout=[0.5, 0.8])` — adaptive end-of-utterance with confidence-based timeout
- `InterruptConfig(mode='HYBRID', ...)` — hybrid interruption using both VAD and STT signals
- `interrupt_min_duration`, `interrupt_min_words`, `false_interrupt_pause_duration`, `resume_on_false_interrupt` parameters
- Non-interruptible speech via `session.say(..., interruptible=False)`

**Plugins used:** `deepgram` (STT), `google` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `realtime_basic.py`

Minimal realtime-mode agent using **GeminiRealtime** (Gemini 3.1 Flash Live Preview). Demonstrates:

- Single-model pipeline: `Pipeline(llm=model)` with no separate STT/TTS/VAD needed
- `GeminiLiveConfig(voice="Leda", response_modalities=["AUDIO"])` configuration
- Available Gemini voices: Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr

**Plugins used:** `google` (GeminiRealtime)

### `voice_pipeline_hooks.py`

Comprehensive demo of the **`@pipeline.on()` hook system** for intercepting and modifying data at every stage. Demonstrates:

- **STT hook** (`@pipeline.on("stt")`) — filters small audio frames, removes filler words ("uh", "um"), applies word replacements, normalizes transcript text
- **TTS hook** (`@pipeline.on("tts")`) — preprocesses text before synthesis (e.g., expanding "AM"→"A M"), uses `run_tts()` helper
- **LLM hook** (`@pipeline.on("llm")`) — logs generated text
- **Turn event hooks** — `user_turn_start`, `user_turn_end`, `agent_turn_start`, `agent_turn_end`
- **Vision hook** (`@pipeline.on("vision_frame")`) — passthrough frame processing
- Uses `run_stt()` and `run_tts()` helpers imported from `videosdk.agents`

**Plugins used:** `deepgram` (STT), `google` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

---

## Subdirectories

### `hybrid_mode(cascade+realtime)/`

Demonstrates **hybrid pipeline mode** — combining cascade components with a realtime model.

| File | Description |
|------|-------------|
| `hybrid_custom_stt_realtime.py` | Uses **SarvamAI STT** (cascade) + **GeminiRealtime** (realtime LLM) + SileroVAD. Also integrates a custom `KnowledgeBase` with trigger-phrase-based retrieval. Pipeline: external STT feeds transcripts into the realtime model. |
| `hybrid_realtime_custom_tts.py` | Uses **XAIRealtime** (Grok for STT+LLM) + **CartesiaTTS** (cascade TTS). The pipeline intercepts the realtime model's audio output and replaces it with CartesiaTTS, giving full voice customization. Demonstrates `XAIRealtimeConfig` with `enable_web_search=True`. |

### `composable_pipelines/`

Shows how to build **partial pipelines** with only the components you need— no voice required.

| File | Description |
|------|-------------|
| `agent_llm.py` | **LLM-only** — text in/out via PubSub. No STT, no TTS, no VAD. User sends text on topic `CHAT`, agent responds on topic `AGENT_RESPONSE`. Uses `pipeline.process_text()` for text injection. |
| `agent_voice_to_text.py` | **STT + LLM only** (no TTS) — voice in, text out via PubSub. User speaks, agent sends text responses through `@pipeline.on("llm")` hook to PubSub topic `CHAT`. |
| `agent_text_to_voice.py` | **LLM + TTS only** (no STT) — text in via PubSub, voice out. User sends text, agent speaks the response. Uses `pipeline.process_text()`. |
| `agent_multimodal.py` | **Full voice + text** — accepts both voice input (STT) and text input (PubSub), responds with voice (TTS). Demonstrates `session.interrupt()` via PubSub message and `pipeline.process_text()` for text injection. |

### `a2a/`
**Agent-to-Agent (A2A) protocol** — multi-agent coordination within the same room.

| File | Description |
|------|-------------|
| `main.py` | Orchestrator that creates two agent sessions (CustomerServiceAgent + LoanAgent), both running in the same room. Uses `register_a2a()` / `unregister_a2a()` for discovery. Manages concurrent sessions with `asyncio.create_task`. |
| `agents/customer_agent.py` | Customer-facing agent that handles general inquiries and delegates loan questions to the specialist via A2A. |
| `agents/loan_agent.py` | Loan specialist agent that handles loan-specific queries. |
| `session_manager.py` | Helper functions (`create_pipeline`, `create_session`) for consistent pipeline/session construction. |

### `langchain/`

**LangChain integration** — uses `LangChainLLM` to wrap a LangChain `BaseChatModel`.

| File | Description |
|------|-------------|
| `agent.py` | Voice-controlled **Slack assistant** powered by LangChain. Uses `ChatOpenAI(model="gpt-4o-mini")` wrapped in `LangChainLLM`. Agent has `@function_tool` methods for posting messages to Slack channels. Demonstrates LangChain + VideoSDK tool bridging. |
| `requirements.txt` | Dependencies: `langchain-openai`, `slack-sdk` |
| `README.md` | Setup instructions |

### `langgraph/`

**LangGraph integration** — uses `LangGraphLLM` to run a compiled `StateGraph`.

| File | Description |
|------|-------------|
| `agent.py` | Voice-driven **blog writer** powered by a LangGraph pipeline with 5 nodes: `coordinator_node` (extracts topic/audience/tone) → `planner_node` (plans 4 blog sections) → `write_sections_node` (writes each section sequentially) → `compiler_node` (saves to markdown file) → `synthesizer_node` (spoken output). Uses `output_node="synthesizer_node"` to control which node's output reaches TTS. All nodes use Gemini via `langchain_google_genai`. |
| `requirements.txt` | Dependencies: `langgraph`, `langchain-google-genai` |
| `README.md` | Setup instructions |

### `mcp_server_examples/`

**MCP server implementations** used by `mcp_example.py`.

| File | Description |
|------|-------------|
| `mcp_server_example.py` | Financial data MCP server using `FastMCP`. Exposes tools: `get_nifty50_price`, `get_stock_quote`, `get_exchange_rate`, `get_company_info`, `search_with_time`. Uses Alpha Vantage API. |
| `mcp_current_time_example.py` | Simple MCP server that returns the current time. |

### `mem0/`

**Persistent memory** across sessions using Mem0 API.

| File | Description |
|------|-------------|
| `memory_agent.py` | Personal assistant that remembers user name, preferences, and past conversations across sessions. Uses `Mem0Memory` class with HTTP API to store/retrieve/search memories. Injects relevant memories into LLM context via `@pipeline.on("user_turn_start")` hook. Keyword-based trigger for memory storage (e.g., "remember", "my name", "i like"). |

### `human_in_the_loop/`

**Human escalation** — agent defers to a human for certain decisions.

| File | Description |
|------|-------------|
| `customer_agent.py` | Customer-facing agent that uses a **Discord MCP server** to ask a human supervisor for discount percentages. The agent uses `MCPServerStdio` to connect to the Discord bot for real-time human input. |
| `discord_mcp_server.py` | MCP server that bridges to Discord — sends questions to a Discord channel and waits for human responses. |
| `DISCORD_BOT.md` | Setup guide for the Discord bot integration. |

### `n8n_workflow/`

**n8n automation** — agent connected to external workflows via n8n MCP server.

| File | Description |
|------|-------------|
| `appointment_telephony.py` | Doctor appointment follow-up agent using telephony. Uses `MCPServerHTTP` to connect to an n8n MCP trigger for retrieving/updating appointment data. Demonstrates `ConversationFlow`, `InterruptConfig`, and `Options(register=True)` for worker mode. |
| `customer_followup_agent.json` | n8n workflow JSON for customer follow-up automation. |
| `APPOINTMENT_TELEPHONY.md` | Setup instructions for the telephony appointment workflow. |
| `images/` | Screenshots for documentation. |

### `vision/`

**Multimodal vision** — agents that can see video frames from participants.

| File | Description |
|------|-------------|
| `vision_cascade.py` | Cascade-mode vision agent. Captures video frames via `agent.capture_frames(num_of_frames=2)` triggered by PubSub messages. Sends frames to LLM via `session.reply(..., frames=frames)` for visual analysis. Requires `RoomOptions(vision=True)`. |
| `vision_hook_example.py` | Vision processing via `@pipeline.on("vision_frame")` hook for continuous frame analysis. |
| `vision_realtime.py` | Realtime-mode vision agent with video frame processing. |

### `avatar/`

**Virtual avatar agents** — visual AI avatars powered by Simli or Anam.

| File | Description |
|------|-------------|
| `simli_cascade_example.py` | Cascade-mode agent with **Simli** avatar rendering. |
| `simli_realtime_example.py` | Realtime-mode agent with Simli avatar. |
| `anam_cascade_example.py` | Cascade-mode agent with **Anam** avatar rendering. |
| `anam_realtime_example.py` | Realtime-mode agent with Anam avatar. |
| `avatar_server_examples/` | Server-side avatar deployment examples including `videosdk_avatar_agent.py`, `videosdk_avatar_launcher.py`, `videosdk_avatar_service.py`, and `waterfall_viz.py` for latency visualization. |

### `browser_transports/`

**Browser-based transport** — connect agents via WebRTC or WebSocket directly from a browser.

| Subdir | Files | Description |
|--------|-------|-------------|
| `webrtc/` | `webrtc_mode.html`, `signaling_server.js`, `package.json` | P2P WebRTC transport. Start the signaling server (`node signaling_server.js`), open the HTML in a browser, and run the Python agent with `RoomOptions(transport_mode="webrtc", webrtc=WebRTCConfig(...))`. |
| `websocket/` | `websocket_mode.html` | Raw PCM WebSocket transport. Open the HTML, run the Python agent with `RoomOptions(transport_mode="websocket", websocket=WebSocketConfig(port=8080, path="/ws"))`. |

---

## Standalone Example Files

### `mcp_example.py`

Agent with **MCP (Model Context Protocol) integration** using OpenAI Realtime. Connects to multiple MCP servers:

- `MCPServerStdio` — starts the financial data and current time MCP servers as subprocesses
- `MCPServerHTTP` — connects to an external MCP server (e.g., Zapier)
Uses `OpenAIRealtime(model="gpt-4o-realtime-preview")` with server-side VAD turn detection.

**Plugins used:** `openai` (OpenAIRealtime)

### `multi_agent_switch.py`

**Multi-agent handoff** using `@function_tool` that returns a new `Agent` instance. Demonstrates:

- `TravelAgent` → `BookingAgent` or `TravelSupportAgent` via tool call
- `inherit_context=True` to carry conversation history to the new agent
- Agent switching is triggered when the LLM calls the transfer tool

**Plugins used:** `deepgram` (STT), `google` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `multi_transport_agent.py`

Demonstrates running the **same agent across different transport modes**: default VideoSDK Room, WebRTC (P2P), and WebSocket (raw PCM). Shows how to configure `RoomOptions` with `transport_mode`, `WebRTCConfig` (signaling URL, ICE servers), and `WebSocketConfig` (port, path).

**Plugins used:** `deepgram` (STT), `google` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `background_audio.py`

Agent with **background audio** capabilities — thinking audio and background music. Demonstrates:

- `self.set_thinking_audio()` — plays audio while the LLM is processing
- `self.play_background_audio(override_thinking=False, looping=True)` — looping background music
- `self.stop_background_audio()` — stops background audio via `@function_tool`
- `RoomOptions(background_audio=True)` to enable the feature

**Plugins used:** `openai` (LLM, TTS), `deepgram` (STT), `silero` (VAD), `turn_detector` (EOU)

### `call_transfer.py`

**Call transfer** — transfers the ongoing call to a different phone number. Uses `self.session.call_transfer(token, transfer_to)`. Demonstrates `Options(agent_id=..., register=True)` for registered worker mode.

**Plugins used:** `deepgram` (STT), `google` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `custom_knowledge_base.py`

**RAG with custom knowledge base** — extends the `KnowledgeBase` base class with custom retrieval logic. Demonstrates:

- `allow_retrieval(transcript)` — trigger-phrase-based gating (only retrieves when user mentions specific keywords)
- `pre_process_query(transcript)` — cleans queries before sending to vector DB
- `format_context(documents)` — formats retrieved documents for LLM context injection
- `KnowledgeBaseConfig(id=kb_id, top_k=3)` — configuration

**Plugins used:** `google` (LLM, TTS), `sarvamai` (STT), `silero` (VAD), `turn_detector` (EOU)

### `dtmf_voicemail.py`

**DTMF tone handling and voicemail detection**. Demonstrates:

- `DTMFHandler(callback)` — registers a callback for DTMF key press events
- `VoiceMailDetector(llm=OpenAILLM(), duration=5.0, callback=callback)` — detects voicemail and triggers `agent.hangup()`
- Both are passed to `AgentSession(dtmf_handler=..., voice_mail_detector=...)`

**Plugins used:** `deepgram` (STT), `openai` (LLM), `elevenlabs` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `enhanced_pronounciation.py`

**Pronunciation correction** via TTS pipeline hook. Uses `@pipeline.on("tts")` to apply a pronunciation map (e.g., "nginx"→"engine x", "API"→"A P I", "SQL"→"sequel") before feeding text to TTS. Uses `run_tts()` helper for re-synthesis.

**Plugins used:** `openai` (LLM), `deepgram` (STT), `elevenlabs` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `fallback_recovery.py`

**Fallback providers** for high availability. Demonstrates:

- `FallbackSTT([OpenAISTT(), DeepgramSTT()], ...)` — primary + backup STT
- `FallbackLLM([OpenAILLM(), CerebrasLLM()], ...)` — primary + backup LLM
- `FallbackTTS([OpenAITTS(), CartesiaTTS()], ...)` — primary + backup TTS
- `temporary_disable_sec=30.0` — cooldown before retrying failed provider
- `permanent_disable_after_attempts=3` — permanently disable after N retries

**Plugins used:** `openai` (STT, LLM, TTS), `deepgram` (STT), `cerebras` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `reply_interrupt_agent.py`

**Programmatic reply and interruption** via PubSub. Listens on `CHAT` topic for:

- `"reply"` — triggers `session.reply("Create a random number...")` to inject a new agent response
- `"interrupt"` — triggers `session.interrupt()` to stop current speech

**Plugins used:** `deepgram` (STT), `anthropic` (LLM), `elevenlabs` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `translator_agent.py`

**Real-time language translation** with dynamic TTS hot-swapping. Detects the user's language via Sarvam AI's `identify_language` API in @pipeline.on("user_turn_start") and dynamically changes the TTS to `SarvamAITTS` with the detected language using `pipeline.change_component()`.

**Plugins used:** `openai` (LLM), `sarvamai` (STT, TTS), `cartesia` (TTS initial), `silero` (VAD)

### `utterance_handle_agent.py`

**UtteranceHandle for sequential speech and interruption-aware tools**. Demonstrates:

- `self.session.current_utterance` — access the current `UtteranceHandle`
- `handle = self.session.say(...); await handle` — sequential TTS (wait for each utterance to finish)
- `utterance.interrupted` — check if user interrupted mid-speech
- Long-running tool with periodic interruption checks

**Plugins used:** `openai` (LLM), `deepgram` (STT), `elevenlabs` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `pubsub_example.py`

**PubSub messaging** — bidirectional text messaging between agent and room participants. Uses `@function_tool` to publish messages via `PubSubPublishConfig` and subscribes with `PubSubSubscribeConfig`.

**Plugins used:** `deepgram` (STT), `anthropic` (LLM), `elevenlabs` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `wakeup_call.py`

**Wake-up call** — periodically prompts the user if they haven't spoken. Uses `AgentSession(wake_up=45)` (seconds) and `session.on_wake_up` callback to trigger `session.say("Hello, are you there?")`.

**Plugins used:** `deepgram` (STT), `anthropic` (LLM), `google` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `worker_agent.py`

**Worker deployment mode** — runs the agent as a multi-process server that accepts incoming jobs. Demonstrates:

- `Options(agent_id=..., register=True, max_processes=3, port="9000", host="localhost", num_idle_processes=2)`
- `RoomOptions(auto_end_session=True, session_timeout_seconds=10)` — automatic session cleanup
- Production-ready deployment pattern

**Plugins used:** `deepgram` (STT), `google` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `test_workflow_pipeline.py`

**Conversational Graph** — structured conversation flow using `videosdk-conversational-graph`. Implements a **loan application workflow** with:

- `ConversationalGraph` with `ConversationalDataModel` (Pydantic) for structured data collection
- State machine: Greeting → Loan Type → Details (Personal/Home/Car) → Amount → Accept/Reject → Confirm → Complete
- `loan_application.state()` / `loan_application.transition()` for defining nodes and edges
- Off-topic handler with `master=True`
- Passed to pipeline as `conversational_graph=loan_application`

**Plugins used:** `deepgram` (STT), `openai` (LLM), `google` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `videosdk_cascade_inference_agent.py`

**VideoSDK Inference Gateway** in cascade mode. Uses the built-in inference gateway (`videosdk.agents.inference`) instead of direct plugin imports:

**Note:** To run the Inference models, your account type must be Pay As You Go in Videosdk Dashboard.

- `STT.sarvam(language="en-IN")` — Sarvam AI STT via gateway
- `LLM.google(model_id="gemini-2.0-flash")` — Google LLM via gateway
- `TTS.sarvam(model_id="bulbul:v2", speaker="anushka")` — Sarvam TTS via gateway

### `videosdk_realtime_inference_agent.py`

**VideoSDK Inference Gateway** in realtime mode. Uses:

- `Realtime.gemini(model="gemini-2.5-flash-native-audio-preview", voice="Puck", response_modalities=["AUDIO"])` — Gemini realtime via gateway

---

## Common Patterns

- All examples follow: create `Agent` subclass → build `Pipeline` → create `AgentSession` → `session.start()`
- Examples use `WorkerJob(entrypoint=..., jobctx=make_context)` with `make_context()` returning `JobContext(room_options=...)`
- Set `playground=True` in `RoomOptions` for local console testing
- Environment variables are loaded from `.env` files
- `pre_download_model()` is called at module level for turn detector model pre-caching
