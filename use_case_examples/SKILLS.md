---
name: use-case-examples
description: Domain-specific agent examples for real-world use cases (healthcare, support, IVR, etc.)
---

# use_case_examples — Domain-Specific Agent Examples

## Purpose

Contains production-oriented agent examples for specific business domains and use cases. Each file is a standalone, runnable agent implementation demonstrating how to build specialized voice agents using different pipeline modes, integrations, and framework features in a real-world context.

---

## Healthcare & Wellness

### `appointment_booking_agent.py`

**Clinic front-desk receptionist** for Sunrise Clinic. Demonstrates:

- Multi-step sequential speech using `UtteranceHandle` — each booking confirmation line is spoken and awaited individually
- `utterance.interrupted` checks between sequential speech steps to bail out gracefully if the patient interrupts
- `@function_tool` methods: `check_availability(date, time)` validates slot against a mock calendar, `book_appointment(patient_name, date, time)` generates a confirmation ID
- Hash-based confirmation ID generation: `f"SC-{abs(hash(...)) % 10000:04d}"`

**Plugins used:** `deepgram` (STT), `openai` (LLM), `elevenlabs` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `medical_triage_agent.py`

**Medical symptom triage with multi-agent handoff** for MedPlus Clinic. Implements a 3-agent system:

- `TriageAgent` — frontline nurse that asks about the chief complaint and routes based on symptom keywords
- `CardiologyAgent` — cardiology intake specialist (chest pain, palpitations, breathlessness)
- `GeneralMedicineAgent` — general medicine intake (fever, body ache, headache)
- Handoff via `@function_tool` returning a new `Agent` instance with `inherit_context=True` to carry conversation history
- Uses `session.reply(instructions=...)` for dynamic greeting generation

**Plugins used:** `deepgram` (STT), `google` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `meditation_guide_agent.py`

**Guided meditation and wellness agent** (Serenity). Demonstrates **background audio** features:

- `self.set_thinking_audio()` — plays a soft sound while the agent is processing
- `self.play_background_audio(override_thinking=False, looping=True)` — starts looping ambient music during meditation sessions
- `self.stop_background_audio()` — stops audio on session end or via `@function_tool`
- `@function_tool` methods: `start_session(duration_minutes, session_type)` and `stop_session()` for meditation lifecycle
- Requires `RoomOptions(background_audio=True)`

**Plugins used:** `deepgram` (STT), `openai` (LLM, TTS — voice "nova"), `silero` (VAD), `turn_detector` (EOU)

---

## Customer Service & Support

### `customer_support_agent.py`

**E-commerce order support agent** (Aria) for ShopEasy. Demonstrates:

- `@function_tool` methods with realistic mock data: `lookup_order_status(order_id)` returns status/carrier/ETA/tracking, `create_support_ticket(issue, order_id)` generates ticket IDs
- Structured tool return shapes (`{"found": True, "order_id": ..., "status": "shipped", ...}`)
- Clean agent design pattern — concise instructions, mock data for demo

**Plugins used:** `deepgram` (STT), `google` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `call_center_agent.py`

**Telecom customer care with live-agent handoff** for ConnectTel. Demonstrates:

- `session.call_transfer(token, transfer_to)` — transfers the call to a live human agent's phone number
- `Options(agent_id=..., register=True, host="localhost", port=8081)` — registered worker deployment mode
- `@function_tool` methods: `get_bill_details(account_id)` for billing lookup, `transfer_to_human_agent()` for escalation
- Emotion-aware instructions — agent immediately transfers frustrated customers without asking

**Plugins used:** `deepgram` (STT), `google` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `product_support_agent.py`

**Documentation-backed support agent** for Novu platform. Demonstrates the **custom `KnowledgeBase` pattern**:

- `NovuKnowledgeBase` subclass with `allow_retrieval(transcript)` — trigger-phrase gating (e.g., "how to", "what is", "configure")
- `pre_process_query(transcript)` — strips trigger phrases to produce clean search queries
- `format_context(documents)` — formats retrieved docs with bullet points for LLM context injection
- `KnowledgeBaseConfig(id=kb_id, top_k=3)` — connects to a VideoSDK-hosted vector KB
- Requires `KNOWLEDGE_BASE_ID` environment variable

**Plugins used:** `sarvamai` (STT), `google` (LLM, TTS), `silero` (VAD), `turn_detector` (EOU)

### `helpdesk_chatbot.py`

**Text-only IT helpdesk bot** for Acme Corp. Demonstrates an **LLM-only pipeline** (no STT/TTS/VAD):

- `Pipeline(llm=GoogleLLM())` — pure text pipeline, no voice components
- Input via PubSub `CHAT` topic using `pipeline.process_text(text)`, output via `HELPDESK_RESPONSE` topic
- `@pipeline.on("llm")` hook publishes LLM responses to PubSub
- Manual session lifecycle with `ctx.connect()`, `wait_for_participant()`, `shutdown_event`

**Plugins used:** `google` (LLM)

---

## Sales & Outreach

### `proactive_outreach_agent.py`

**Subscription renewal outbound call agent** for StreamBox. Demonstrates **wake-up / proactive re-engagement**:

- `AgentSession(wake_up=45)` — if the customer is silent for 45 seconds, re-engages automatically
- `session.on_wake_up` callback uses `session.reply(instructions=...)` for dynamic re-engagement speech
- `session.reply(instructions=...)` in `on_enter` for dynamic greeting (not hardcoded `session.say`)
- `@function_tool` methods: `confirm_renewal(plan)` with loyalty discount logic, `process_cancellation(reason)`

**Plugins used:** `deepgram` (STT), `anthropic` (LLM), `google` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `announcement_agent.py`

**Live sports commentary agent** — receives match events as text, broadcasts them as voice. Demonstrates a **text-to-voice pipeline**:

- `Pipeline(llm=GoogleLLM(), tts=CartesiaTTS())` — no STT, no VAD (LLM + TTS only)
- `pipeline.process_text(event_data)` — injects match events from PubSub into the pipeline
- `@pipeline.on("tts")` hook normalizes commentary: collapses repeated characters ("GOOOOAL" → "Goal"), converts all-caps words to title case for natural TTS pronunciation
- Uses `run_tts()` helper for re-synthesis after normalization

**Plugins used:** `google` (LLM), `cartesia` (TTS)

---

## Financial Services

### `omnichannel_agent.py`

**Banking virtual assistant** for First National Bank — accepts **both voice and text input** in the same session. Demonstrates:

- Combined voice (STT) + text (PubSub) input with `pipeline.process_text(text)` for text injection
- `session.interrupt()` triggered by PubSub "interrupt" command
- `@pipeline.on("llm")` hook mirrors LLM responses to `AGENT_RESPONSE` PubSub topic for chat clients
- Manual session lifecycle with `ctx.connect()`, PubSub subscriptions, and shutdown events

**Plugins used:** `deepgram` (STT), `google` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

### `ivr_agent.py`

**Bank IVR phone system** for Metro Bank. Demonstrates **DTMF handling + voicemail detection**:

- `DTMFHandler(callback)` — routes keypad presses (1=Balance, 2=Transactions, 3=Transfers, 0=Agent)
- `VoiceMailDetector(llm=OpenAILLM(), duration=5.0, callback=callback)` — auto-hangs up if voicemail is detected
- Non-interruptible menu announcements via `session.say(..., interruptible=False)`
- `Options(agent_id=..., max_processes=2, register=True)` for production telephony deployment

**Plugins used:** `deepgram` (STT), `google` (LLM), `cartesia` (TTS), `openai` (LLM for voicemail), `silero` (VAD), `turn_detector` (EOU)

### `high_availability_agent.py`

**Emergency insurance hotline** for SafeGuard Insurance with **full provider failover**. Demonstrates:

- `FallbackSTT([OpenAISTT(), DeepgramSTT()], temporary_disable_sec=30.0, permanent_disable_after_attempts=3)` — auto-fallback STT
- `FallbackLLM([OpenAILLM(), CerebrasLLM()], ...)` — auto-fallback LLM
- `FallbackTTS([OpenAITTS(), CartesiaTTS()], ...)` — auto-fallback TTS
- `@function_tool` methods: `dispatch_roadside_assistance(location, vehicle_issue)`, `create_emergency_claim(caller_name, policy_number, incident_description)`
- Emergency-appropriate instructions — never refuse help, never hold callers

**Plugins used:** `openai` (STT, LLM, TTS), `deepgram` (STT), `cerebras` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

---

## Education & Language

### `language_tutor_agent.py`

**Real-time Spanish language tutor** (Sofia). Demonstrates a **pure realtime pipeline**:

- `OpenAIRealtime(model="gpt-realtime-2025-08-28")` — single model handles STT + LLM + TTS end-to-end
- `OpenAIRealtimeConfig(voice="shimmer", modalities=["audio"])` with server-side VAD turn detection
- `TurnDetection(type="server_vad", threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500)`
- No separate STT/TTS/VAD needed — `Pipeline(llm=model)` only

**Plugins used:** `openai` (OpenAIRealtime)

---

## Productivity & Enterprise

### `meeting_notes_agent.py`

**Silent meeting note-taker** (Fireflies-style). Demonstrates a **listen-only agent** with periodic summarization:

- `Pipeline(stt=DeepgramSTT(enable_diarization=True), vad=SileroVAD(), turn_detector=TurnDetector())` — STT-only pipeline, no LLM or TTS in the pipeline
- `@pipeline.on("stt")` hook captures speaker labels and timestamps, buffers final transcripts with filler word removal
- External `GoogleLLM` called on a 2-minute interval (`asyncio.wait_for` timer) to generate structured meeting notes
- Notes published to PubSub `CHAT` topic with structured format: Transcript Summary, Key Points, Decisions, Action Items, Open Questions
- Silent agent — `on_enter`/`on_exit` are no-ops

**Plugins used:** `deepgram` (STT with diarization), `google` (LLM — external calls), `silero` (VAD), `turn_detector` (EOU)

### `onboarding_agent.py`

**SaaS onboarding wizard** for Taskflow project management. Demonstrates **ConversationalGraph state machine**:

- `ConversationalGraph` with `ConversationalDataModel` (Pydantic) for structured data collection
- State machine: Welcome → Team Size → Primary Use Case → Invite Teammates → Complete
- `onboarding_flow.state()` / `onboarding_flow.transition()` for defining nodes and edges
- `master=True` off-topic handler redirects users back to the flow
- `Pipeline(conversational_graph=onboarding_flow)` — graph passed directly to pipeline
- `@function_tool` method: `complete_onboarding(...)` saves the workspace profile

**Plugins used:** `deepgram` (STT), `google` (LLM, TTS), `silero` (VAD), `turn_detector` (EOU)

---

## Content Moderation & Safety

### `content_filter_agent.py`

**Children's learning assistant** (Buddy) for KidsLearn with **input/output content filtering**. Demonstrates:

- `@pipeline.on("stt")` hook — scans transcripts for blocked words (violence, weapons, drugs, etc.) and redacts them with `***` before they reach the LLM
- `@pipeline.on("tts")` hook — blocks unsafe LLM output from being spoken, replaces with safe redirect phrase
- `contains_blocked_content()` / `redact_text()` helper functions for word-level filtering
- Both `BLOCKED_WORDS` (precise terms) and `BLOCKED_TOPICS` (broader subjects) lists
- Uses `run_stt()` and `run_tts()` helpers for hook pipelines

**Plugins used:** `deepgram` (STT), `google` (LLM), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

---

## RAG & Knowledge

### `custom_rag_agent.py`

**Voice assistant with ChromaDB RAG**. Demonstrates a **full custom vector retrieval pipeline**:

- `chromadb.Client()` — in-process vector store, initialized at startup with 5 VideoSDK-related documents
- `OpenAI().embeddings.create(input=text, model="text-embedding-ada-002")` for embedding generation (sync for init, async for queries)
- `collection.query(query_embeddings=[...], n_results=k)` for similarity search
- `@pipeline.on("user_turn_start")` hook retrieves top-k documents and injects them into `agent.chat_context.add_message(role="system", content=...)` before the LLM runs
- `@pipeline.on("llm")` hook for response logging

**Plugins used:** `deepgram` (STT), `openai` (LLM + embeddings), `cartesia` (TTS), `silero` (VAD), `turn_detector` (EOU)

---

## Branding & Voice Customization

### `branded_voice_agent.py`

**Premium retail shopping assistant** (Luxe) for Prestige Boutique. Demonstrates **hybrid TTS mode**:

- `XAIRealtime(model="grok-4-1-fast-non-reasoning")` handles STT + LLM (understanding)
- `CartesiaTTS()` renders the brand voice (external TTS overrides the realtime model's audio output)
- `Pipeline(llm=llm, tts=CartesiaTTS())` — hybrid pipeline: realtime model for comprehension, external TTS for brand voice
- `XAIRealtimeConfig(voice="Eve", turn_detection=XAITurnDetection(...))` with server-side VAD

**Plugins used:** `xai` (XAIRealtime), `cartesia` (TTS)

---

## Multi-provider & Infrastructure

### `inference_gateway_agent.py`

**Regional language support agent** (Kavya) for RuralPay digital payments. Demonstrates **VideoSDK Inference Gateway**:

- `STT.sarvam(model_id="saarika:v2.5", language="en-IN")` — Sarvam STT via VideoSDK gateway (no third-party STT API key needed)
- `TTS.sarvam(model_id="bulbul:v2", speaker="anushka", language="en-IN")` — Sarvam TTS via gateway
- `GoogleLLM(model="gemini-3.1-flash-lite-preview")` — direct Google LLM
- Only `VIDEOSDK_AUTH_TOKEN` required for STT/TTS — all inference routed through VideoSDK

**Note:** To run Inference models, your account type must be Pay As You Go in the VideoSDK Dashboard.

**Plugins used:** `inference` (STT, TTS via gateway), `google` (LLM), `silero` (VAD), `turn_detector` (EOU)

### `responsive_voice_agent.py`

**Fast-food drive-through ordering agent** for QuickBite. Demonstrates **advanced EOU + interrupt tuning**:

- `EOUConfig(mode="ADAPTIVE", min_max_speech_wait_timeout=[0.3, 0.6])` — sub-second end-of-utterance for rapid ordering
- `InterruptConfig(mode="HYBRID", interrupt_min_duration=0.2, interrupt_min_words=2, false_interrupt_pause_duration=1.5, resume_on_false_interrupt=True)` — handles noisy drive-through environments
- Non-interruptible order confirmation via `session.say(..., interruptible=False)`
- `CartesiaTTS(generation_config=GenerationConfig(volume=1, speed=1.3))` — faster TTS speech rate
- `@function_tool` methods: `confirm_order(items, total)`, `place_order(items, total)`

**Plugins used:** `deepgram` (STT), `openai` (LLM), `cartesia` (TTS with GenerationConfig), `silero` (VAD), `turn_detector` (EOU)

---

## Multilingual

### `multilingual_support_agent.py`

**Auto-detecting multilingual support** for GlobalApp SaaS. Demonstrates **dynamic TTS hot-swapping**:

- `Pipeline(stt=SarvamAISTT(), llm=GeminiRealtime(...), vad=SileroVAD())` — hybrid mode (external STT + realtime LLM)
- `@pipeline.on("user_turn_start")` detects language via `SarvamAI().text.identify_language(input=transcript)`
- `pipeline.change_component(tts=SarvamAITTS(language=detected_language))` — dynamically swaps TTS to match detected language
- `GeminiRealtime(model="gemini-3.1-flash-live-preview")` with `response_modalities=["AUDIO"]`

**Plugins used:** `sarvamai` (STT, TTS), `google` (GeminiRealtime), `silero` (VAD)

---

## Smart Home & IoT

### `smart_assistant_agent.py`

**Smart home assistant** (Aria) with **MCP tool servers**. Demonstrates:

- `MCPServerStdio(executable_path=sys.executable, process_arguments=[...])` — local MCP servers for device controls and time queries
- `MCPServerHTTP(endpoint_url=...)` — remote MCP server (Zapier) for home automations (optional)
- `OpenAIRealtime(model="gpt-realtime-2025-08-28")` with `modalities=["text", "audio"]` and `tool_choice="auto"`
- Tools are auto-discovered from connected MCP servers — no manual `@function_tool` registration

**Plugins used:** `openai` (OpenAIRealtime)

---

## Subdirectories

### `avatar/`

**Visual AI avatar agents** for domain-specific use cases — each includes both cascade (P1) and realtime (P2) pipeline options.

| File | Use Case | Pipeline | Avatar |
|------|----------|----------|--------|
| `anam_avatar_agent.py` | HR behavioral interviewer (Maya) for TalentFirst. Conducts 5 STAR-method questions and internally scores answers with `@function_tool score_answer(question_number, question, answer_summary, score)`. | P1: DeepgramSTT + OpenAILLM + ElevenLabsTTS (with commented P2: OpenAIRealtime alternative) | **AnamAvatar** |
| `simli_avatar_agent.py` | Bank loan advisor (Rohan) for BankEasy. Explains Home/Personal/Car loan options and calculates EMI with `@function_tool calculate_emi(principal, annual_rate, tenure_years)`. | P2: GeminiRealtime (with commented P1: DeepgramSTT + OpenAILLM + ElevenLabsTTS alternative) | **SimliAvatar** with `SimliConfig(faceId=..., maxSessionLength=1800)` |

### `vision/`

**Multimodal vision agents** that analyze video frames for domain-specific scenarios. Both require `RoomOptions(vision=True)`.

| File | Use Case | Key Feature |
|------|----------|-------------|
| `quality_inspection_agent.py` | Manufacturing QA inspector for PrecisionParts Ltd. Uses `@pipeline.on("vision_frame")` hook to passively sample every 50th frame from the production line and save to disk. Manual inspection triggered via PubSub `"analyze_frame"` command → `session.reply(frames=[latest_frame])`. Reports defect location, type, severity (LOW/MEDIUM/HIGH), and recommended action. | **Vision frame hook** + on-demand analysis |
| `visual_assistant_agent.py` | Warehouse inspection assistant for LogiCo. Uses `agent.capture_frames(num_of_frames=2)` triggered by PubSub `"inspect"` command. Identifies damage, missing labels, fallen items, and rates urgency. Includes commented-out P2 GeminiRealtime alternative. | **On-demand frame capture** |

---

## Common Patterns

- All examples extend `Agent` with domain-specific `instructions` and `@function_tool` methods
- Most use cascade mode: `Pipeline(stt=..., llm=..., tts=..., vad=..., turn_detector=...)`
- `@function_tool` methods implement domain-specific actions (booking, lookup, transfer, scoring, etc.)
- Environment variables configured via `.env` files
- `pre_download_model()` is called at module level for turn detector model pre-caching
- `RoomOptions(playground=True)` for local console testing
