<!--BEGIN_BANNER_IMAGE-->
<p align="center">
  <img src="https://assets.videosdk.live/images/github-banner.png" alt="VideoSDK AI Agents Banner" style="width:100%;">
</p>
<!--END_BANNER_IMAGE-->

# VideoSDK AI Agents
Open-source Python framework for building production-ready, real-time voice and multimodal AI agents.

![PyPI - Version](https://img.shields.io/pypi/v/videosdk-agents)
[![PyPI Downloads](https://static.pepy.tech/badge/videosdk-agents/month)](https://pepy.tech/projects/videosdk-agents)
[![Twitter Follow](https://img.shields.io/twitter/follow/video_sdk)](https://x.com/video_sdk)
[![YouTube](https://img.shields.io/badge/YouTube-VideoSDK-red)](https://www.youtube.com/c/VideoSDK)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-VideoSDK-blue)](https://www.linkedin.com/company/video-sdk/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA)](https://discord.com/invite/f2WsNDN9S5)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/videosdk-live/agents)

The **VideoSDK AI Agents framework** is a Python SDK for building AI agents that join VideoSDK rooms as real-time participants. It connects your agent worker, AI models, and user devices into a single low-latency pipeline — handling audio streaming, turn detection, interruptions, and media routing automatically so you can focus on agent logic.

<!-- ![VideoSDK AI Agents High Level Architecture](https://strapi.videosdk.live/uploads/Group_15_1_5610ce9c7e.png) -->
<!-- ![VideoSDK AI Agents High Level Architecture](https://cdn.videosdk.live/website-resources/docs-resources/voice_agent_intro.png) -->

![VideoSDK AI Agents High Level Architecture](https://assets.videosdk.live/images/agent-architecture.png)


## Overview

**VideoSDK AI Agents** is a Python framework that lets you build voice and multimodal AI agents that participate directly in VideoSDK rooms. The framework manages the full agent lifecycle — from joining a room and processing live audio, to running STT → LLM → TTS pipelines or connecting to unified realtime models, to handling turn detection, VAD, interruptions, and clean teardown.

**v1.0.0** introduces a unified `Pipeline` class that replaces the previous `CascadingPipeline` and `RealtimePipeline`. Pass in any combination of components — STT, LLM, TTS, VAD, turn detector, avatar — and the framework wires them together and selects the optimal execution mode automatically. A decorator-based hooks system (`@pipeline.on(...)`) lets you intercept and transform data at any stage without subclassing.

<table width="100%">
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🎙️ <a href="examples/cascade_basic.py" target="_blank">Agent with Cascade Mode</a></h3>
      <p>Build an AI Voice Agent using Cascade Mode (STT → LLM → TTS).</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>⚡ <a href="examples/realtime_basic.py" target="_blank">Agent with Realtime Mode</a></h3>
      <p>Build an AI Voice Agent using a unified Realtime model (e.g. Gemini Live).</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>💻 <a href="https://docs.videosdk.live/ai_agents/introduction" target="_blank">Agent Documentation</a></h3>
      <p>The VideoSDK Agent Official Documentation.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>📚 <a href="https://docs.videosdk.live/agent-sdk-reference/agents/" target="_blank">SDK Reference</a></h3>
      <p>Reference Docs for Agents Framework.</p>
    </td>
  </tr>
</table>

<div style={{marginTop: '1.5rem'}}></div>


| #  | Feature                         | Description                                                                 |
|----|----------------------------------|-----------------------------------------------------------------------------|
| 1  | **🎤 Real-time Communication (Audio/Video)**       | Agents can listen, speak, and interact live in meetings.                   |
| 2  | **📞 SIP & Telephony Integration**   | Seamlessly connect agents to phone systems via SIP for call handling, routing, and PSTN access. |
| 3  | **🧍 Virtual Avatars**               | Build or plug in any avatar provider — the framework handles audio routing, sync, and teardown automatically. |
| 4  | **🤖 Multi-Model Support**           | Integrate with OpenAI, Gemini, AWS NovaSonic, Anthropic, and more.         |
| 5  | **🧩 Cascade Mode**                  | Compose any STT → LLM → TTS chain across providers for full control and flexibility. |
| 6  | **⚡ Realtime Mode**                  | Use unified realtime models (OpenAI Realtime, AWS Nova Sonic, Gemini Live) for lowest latency. |
| 7  | **🔀 Hybrid Mode**                   | Mix cascade and realtime components — custom STT with a realtime model, or realtime with custom TTS. |
| 8  | **🪝 Pipeline Hooks**                | Intercept and transform data at any stage (STT, LLM, TTS, turns) using `@pipeline.on(...)`. |
| 9  | **🛠️ Function Tools**               | Extend agent capabilities with any external tool or API call.               |
| 10 | **🌐 MCP Integration**               | Connect agents to external data sources and tools using Model Context Protocol. |
| 11 | **🔗 A2A Protocol**                  | Reliable agent-to-agent routing with correlation-based request tracking.    |
| 12 | **🦜 LangChain & LangGraph**         | Plug in any LangChain `BaseChatModel` or LangGraph `StateGraph` as the agent's LLM. |
| 13 | **📊 Observability**                 | Built-in metrics, OpenTelemetry tracing, and structured logging per component. |

> \[!IMPORTANT]
>
> **Star VideoSDK Repositories** ⭐️
>
> Get instant notifications for new releases and updates. Your support helps us grow and improve VideoSDK!

---

## Pipeline Modes

All agents are built around a single `Pipeline` class. Pass in your components — the SDK picks the right execution mode automatically.

### Cascade Mode — STT → LLM → TTS

Mix and match any provider for each stage. Best when you need custom STT, specific LLM behaviour, or a particular TTS voice.

```python
async def start_session(context: JobContext):
    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=MyAgent(), pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)
```

### Realtime Mode — Lowest Latency with Unified Models

Use a single realtime model for the entire voice pipeline. Best for sub-500ms response latency.

```python
async def start_session(context: JobContext):
    pipeline = Pipeline(
        llm=GeminiRealtime(
            model="gemini-3.1-flash-live-preview",
            config=GeminiLiveConfig(voice="Leda", response_modalities=["AUDIO"]),
        )
    )
    session = AgentSession(agent=MyAgent(), pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)
```

### Hybrid Mode — Mix & Match

Use an external STT with a Realtime LLM, or a Realtime model with a custom TTS:

```python
# External STT → Realtime LLM
pipeline = Pipeline(stt=DeepgramSTT(), llm=OpenAIRealtime(...))

# Realtime LLM → External TTS
pipeline = Pipeline(llm=OpenAIRealtime(...), tts=ElevenLabsTTS(...))
```

### Pipeline Hooks — Intercept Any Stage

```python
@pipeline.on("stt")
async def clean_transcript(text: str) -> str:
    return text.strip()

@pipeline.on("llm")
async def route_llm(messages):
    if "transfer" in messages[-1].content:
        yield "Transferring you now."  # bypass LLM entirely

@pipeline.on("tts")
async def fix_pronunciation(text: str) -> str:
    return text.replace("VideoSDK", "Video S D K")

@pipeline.on("user_turn_start")
async def on_user_starts():
    print("User is speaking...")
```

Available hook points: `stt` · `tts` · `llm` · `vision_frame` · `user_turn_start` · `user_turn_end` · `agent_turn_start` · `agent_turn_end`

---

## Pre-requisites

Before you begin, ensure you have:

- A VideoSDK authentication token (generate from [app.videosdk.live](https://app.videosdk.live))
   - A VideoSDK meeting ID (you can generate one using the [Create Room API](https://docs.videosdk.live/api-reference/realtime-communication/create-room) or through the VideoSDK dashboard)
- Python 3.12 or higher
- Third-Party API Keys:
   - API keys for the services you intend to use (e.g., OpenAI for LLM/STT/TTS, ElevenLabs for TTS, Google for Gemini etc.).

## Installation

### Using UV (Recommended)

[UV](https://docs.astral.sh/uv/) is a fast Python package manager that handles virtual environments and dependency management automatically.

> If you don't have UV installed, see the [UV installation guide](https://docs.astral.sh/uv/getting-started/installation/).

- Install the core VideoSDK AI Agent package:
  ```bash
  uv add videosdk-agents
  ```

- Install Optional Plugins:
  ```bash
  uv add videosdk-plugins-openai
  uv add videosdk-plugins-deepgram
  ```

- Run your agent:
  ```bash
  uv run python main.py
  ```

### Using pip

- Create and activate a virtual environment with Python 3.12 or higher.
    <details>
    <summary><strong> macOS / Linux</strong></summary>

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    </details>
    <details>
    <summary><strong> Windows</strong></summary>

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
    </details>

- Install the core VideoSDK AI Agent package
  ```bash
  pip install videosdk-agents
  ```
- Install Optional Plugins. Plugins help integrate different providers for Realtime, STT, LLM, TTS, and more. Install what your use case needs:
  ```bash
  # Example: Install the Turn Detector plugin
  pip install videosdk-plugins-turn-detector
  ```
  👉 Supported plugins (Realtime, LLM, STT, TTS, VAD, Avatar, SIP) are listed in the [Supported Libraries](#supported-libraries-and-plugins) section below.

### Development Setup

To set up the project locally, clone the repo and install all packages (core + all plugins) as editable installs:

**Using UV (Recommended):**
```bash
git clone https://github.com/videosdk-live/agents.git
cd agents
uv sync
uv run python examples/cascade_basic.py
```

**Using pip:**
```bash
git clone https://github.com/videosdk-live/agents.git
cd agents
bash setup.sh
source venv/bin/activate
python examples/cascade_basic.py
```


## Generating a VideoSDK Meeting ID

Before your AI agent can join a meeting, you'll need to create a meeting ID. You can generate one using the VideoSDK Create Room API:

### Using cURL

```bash
curl -X POST https://api.videosdk.live/v2/rooms \
  -H "Authorization: YOUR_JWT_TOKEN_HERE" \
  -H "Content-Type: application/json"
```

For more details on the Create Room API, refer to the [VideoSDK documentation](https://docs.videosdk.live/api-reference/realtime-communication/create-room).

## Getting Started: Your First Agent

### Quick Start

Now that you've installed the necessary packages, you're ready to build!

### Step 1: Creating a Custom Agent

First, let's create a custom voice agent by inheriting from the base `Agent` class:

```python title="main.py"
from videosdk.agents import Agent, function_tool

# External Tool
# async def get_weather(self, latitude: str, longitude: str):

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks.",
             tools=[get_weather] # You can register any external tool defined outside of this scope
        )

    async def on_enter(self) -> None:
        """Called when the agent first joins the meeting"""
        await self.session.say("Hi there! How can I help you today?")
    
    async def on_exit(self) -> None:
      """Called when the agent exits the meeting"""
        await self.session.say("Goodbye!")
```

This code defines a basic voice agent with:

- Custom instructions that define the agent's personality and capabilities
- An entry message when joining a meeting
- State change handling to track the agent's current activity

### Step 2: Implementing Function Tools

Function tools allow your agent to perform actions beyond conversation. There are two ways to define tools:

- **External Tools:** Defined as standalone functions outside the agent class and registered via the `tools` argument in the agent's constructor.
- **Internal Tools:** Defined as methods inside the agent class and decorated with `@function_tool`.

Below is an example of both:

```python
import aiohttp

# External Function Tools
@function_tool
def get_weather(latitude: str, longitude: str):
    print(f"Getting weather for {latitude}, {longitude}")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "temperature": data["current"]["temperature_2m"],
                    "temperature_unit": "Celsius",
                }
            else:
                raise Exception(
                    f"Failed to get weather data, status code: {response.status}"
                )

class VoiceAgent(Agent):
# ... previous code ...
# Internal Function Tools
    @function_tool
    async def get_horoscope(self, sign: str) -> dict:
        horoscopes = {
            "Aries": "Today is your lucky day!",
            "Taurus": "Focus on your goals today.",
            "Gemini": "Communication will be important today.",
        }
        return {
            "sign": sign,
            "horoscope": horoscopes.get(sign, "The stars are aligned for you today!"),
        }
```

- Use external tools for reusable, standalone functions (registered via `tools=[...]`).
- Use internal tools for agent-specific logic as class methods.
- Both must be decorated with `@function_tool` for the agent to recognize and use them.


### Step 3: Setting Up the Pipeline

Connect your agent to an AI model using the unified `Pipeline` class. Pass in whichever components you need — the SDK handles the rest.

**Realtime mode** (single model, lowest latency):

```python
async def start_session(context: JobContext):
    pipeline = Pipeline(
        llm=GeminiRealtime(
            model="gemini-3.1-flash-live-preview",
            config=GeminiLiveConfig(voice="Leda", response_modalities=["AUDIO"]),
        )
    )
    session = AgentSession(agent=VoiceAgent(), pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)
```

**Cascade mode** (STT → LLM → TTS, full provider control):

```python
async def start_session(context: JobContext):
    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=VoiceAgent(), pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)
```
### Step 4: Assembling and Starting the Agent Session

```python
from videosdk.agents import AgentSession, WorkerJob, RoomOptions, JobContext

async def start_session(context: JobContext):
    session = AgentSession(
        agent=VoiceAgent(),
        pipeline=pipeline,
    )
    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="<meeting_id>",
        name="Test Agent",
        playground=True,
    )
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()
```
### Step 5: Connecting with VideoSDK Client Applications

After setting up your AI Agent, you'll need a client application to connect with it. You can use any of the VideoSDK quickstart examples to create a client that joins the same meeting:

- [JavaScript](https://github.com/videosdk-live/quickstart/tree/main/js-rtc)
- [React](https://github.com/videosdk-live/quickstart/tree/main/react-rtc)
- [React Native](https://github.com/videosdk-live/quickstart/tree/main/react-native)
- [Android](https://github.com/videosdk-live/quickstart/tree/main/android-rtc)
- [Flutter](https://github.com/videosdk-live/quickstart/tree/main/flutter-rtc)
- [iOS](https://github.com/videosdk-live/quickstart/tree/main/ios-rtc)
- [Unity](http://github.com/videosdk-live/videosdk-rtc-unity-sdk-example)
- [IoT](https://github.com/videosdk-live/videosdk-rtc-iot-sdk-example)

When setting up your client application, make sure to use the same meeting ID that your AI Agent is using.

### Step 6: Running the Project
Once you have completed the setup, you can run your AI Voice Agent project using Python. Make sure your `.env` file is properly configured and all dependencies are installed.

```bash
python main.py
```
> [!TIP]
>
> **Console Mode** — test your agent locally without a meeting room.
> Set `playground=True` in `RoomOptions` and run `python main.py` to interact via your mic and speakers directly from the terminal.


### Step 7: Deployment

For deployment options and guide, checkout the official documentation here: [Deployment](https://docs.videosdk.live/ai_agents/deployments/introduction)

---

<!-- - For detailed guides, tutorials, and API references, check out our official [VideoSDK AI Agents Documentation](https://docs.videosdk.live/ai_agents/introduction).
- To see the framework in action, explore the code in the [Examples](examples/) directory. It is a great place to quickstart. -->

## VideoSDK Inference

VideoSDK Inference provides a **unified gateway** to access STT, LLM, TTS, Denoise, and Realtime models — without managing individual provider API keys. Authentication is handled via your `VIDEOSDK_AUTH_TOKEN` and usage is billed from your VideoSDK account balance.

```python
from videosdk.agents.inference import STT, LLM, TTS, Denoise, Realtime
```

**Cascade Mode with VideoSDK Inference:**

```python
async def start_session(context: JobContext):
    pipeline = Pipeline(
        stt=STT.sarvam(model_id="saarika:v2.5", language="en-IN"),
        llm=LLM.google(model_id="gemini-2.5-flash"),
        tts=TTS.sarvam(model_id="bulbul:v2", speaker="anushka", language="en-IN"),
        denoise=Denoise.sanas(),
        vad=SileroVAD(),
    )
    session = AgentSession(agent=MyAgent(), pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)
```

**Realtime Mode with VideoSDK Inference:**

```python
async def start_session(context: JobContext):
    pipeline = Pipeline(
        llm=Realtime.gemini(
            model_id="gemini-3.1-flash-live-preview",
            voice="Puck",
            language_code="en-US",
            response_modalities=["AUDIO"],
        )
    )
    session = AgentSession(agent=MyAgent(), pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)
```

> See [Inference Pricing](https://docs.videosdk.live/help_docs/pricing-inference) for provider-wise billing details.

---

## Supported Libraries and Plugins

The framework supports integration with various AI models and tools, across multiple categories:


| Category                 | Services |
|--------------------------|----------|
| **Real-time Models**     | [OpenAI](https://docs.videosdk.live/ai_agents/plugins/realtime/openai) &#124; [Gemini](https://docs.videosdk.live/ai_agents/plugins/realtime/google-live-api) &#124; [AWS Nova Sonic](https://docs.videosdk.live/ai_agents/plugins/realtime/aws-nova-sonic) &#124; [Azure Voice Live](https://docs.videosdk.live/ai_agents/plugins/realtime/azure-voice-live)|
| **Speech-to-Text (STT)** | [OpenAI](https://docs.videosdk.live/ai_agents/plugins/stt/openai) &#124; [Google](https://docs.videosdk.live/ai_agents/plugins/stt/google) &#124; [Azure AI Speech](https://docs.videosdk.live/ai_agents/plugins/stt/azure-ai-stt) &#124; [Azure OpenAI](https://docs.videosdk.live/ai_agents/plugins/stt/azureopenai) &#124; [Sarvam AI](https://docs.videosdk.live/ai_agents/plugins/stt/sarvam-ai) &#124; [Deepgram](https://docs.videosdk.live/ai_agents/plugins/stt/deepgram) &#124; [Cartesia](https://docs.videosdk.live/ai_agents/plugins/stt/cartesia-stt) &#124; [AssemblyAI](https://docs.videosdk.live/ai_agents/plugins/stt/assemblyai) &#124; [Navana](https://docs.videosdk.live/ai_agents/plugins/stt/navana) |
| **Language Models (LLM)**| [OpenAI](https://docs.videosdk.live/ai_agents/plugins/llm/openai) &#124; [Azure OpenAI](https://docs.videosdk.live/ai_agents/plugins/llm/azureopenai) &#124; [Google](https://docs.videosdk.live/ai_agents/plugins/llm/google-llm) &#124; [Sarvam AI](https://docs.videosdk.live/ai_agents/plugins/llm/sarvam-ai-llm) &#124; [Anthropic](https://docs.videosdk.live/ai_agents/plugins/llm/anthropic-llm) &#124; [Cerebras](https://docs.videosdk.live/ai_agents/plugins/llm/Cerebras-llm) |
| **Text-to-Speech (TTS)** | [OpenAI](https://docs.videosdk.live/ai_agents/plugins/tts/openai) &#124; [Google](https://docs.videosdk.live/ai_agents/plugins/tts/google-tts) &#124; [AWS Polly](https://docs.videosdk.live/ai_agents/plugins/tts/aws-polly-tts) &#124; [Azure AI Speech](https://docs.videosdk.live/ai_agents/plugins/tts/azure-ai-tts) &#124; [Azure OpenAI](https://docs.videosdk.live/ai_agents/plugins/tts/azureopenai) &#124; [Deepgram](https://docs.videosdk.live/ai_agents/plugins/tts/deepgram) &#124; [Sarvam AI](https://docs.videosdk.live/ai_agents/plugins/tts/sarvam-ai-tts) &#124; [ElevenLabs](https://docs.videosdk.live/ai_agents/plugins/tts/eleven-labs) &#124; [Cartesia](https://docs.videosdk.live/ai_agents/plugins/tts/cartesia-tts) &#124; [Resemble AI](https://docs.videosdk.live/ai_agents/plugins/tts/resemble-ai-tts) &#124; [Smallest AI](https://docs.videosdk.live/ai_agents/plugins/tts/smallestai-tts) &#124; [Speechify](https://docs.videosdk.live/ai_agents/plugins/tts/speechify-tts) &#124; [InWorld](https://docs.videosdk.live/ai_agents/plugins/tts/inworld-ai-tts) &#124; [Neuphonic](https://docs.videosdk.live/ai_agents/plugins/tts/neuphonic-tts) &#124; [Rime AI](https://docs.videosdk.live/ai_agents/plugins/tts/rime-ai-tts) &#124; [Hume AI](https://docs.videosdk.live/ai_agents/plugins/tts/hume-ai-tts) &#124; [Groq](https://docs.videosdk.live/ai_agents/plugins/tts/groq-ai-tts) &#124; [LMNT AI](https://docs.videosdk.live/ai_agents/plugins/tts/lmnt-ai-tts) &#124; [Papla Media](https://docs.videosdk.live/ai_agents/plugins/tts/papla-media) |
| **Voice Activity Detection (VAD)** | [SileroVAD](https://docs.videosdk.live/ai_agents/plugins/silero-vad) |
| **Turn Detection Model** | [Namo Turn Detector](https://docs.videosdk.live/ai_agents/plugins/namo-turn-detector) |
| **Virtual Avatar** | [Simli](https://docs.videosdk.live/ai_agents/core-components/avatar) &#124; [Anam](https://docs.videosdk.live/ai_agents/plugins/avatar/anam) &#124; Custom (implement `connect` / `aclose` protocol) |
| **LLM Orchestration** | [LangChain](https://docs.videosdk.live/ai_agents/plugins/llm/langchain) &#124; [LangGraph](https://docs.videosdk.live/ai_agents/plugins/llm/langgraph) |
| **Denoise** | [RNNoise](https://docs.videosdk.live/ai_agents/core-components/de-noise) |

> [!TIP]
> **Installation Examples**
>
> ```bash
> # Install with specific plugins
> pip install videosdk-agents[openai,elevenlabs,silero]
>
> # Install individual plugins
> pip install videosdk-plugins-anthropic
> pip install videosdk-plugins-deepgram
> ```



## Examples

Explore the following examples to see the framework in action:

### Core Mode Examples

<table width="100%">
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🎙️ <a href="examples/cascade_basic.py" target="_blank">Cascade Mode (Basic)</a></h3>
      <p>Simple STT → LLM → TTS voice agent using Google LLM + Deepgram STT + Cartesia TTS.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🔧 <a href="examples/cascade_advanced.py" target="_blank">Cascade Mode (Advanced)</a></h3>
      <p>Advanced cascade agent with VAD, turn detection, and interruption handling.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>⚡ <a href="examples/realtime_basic.py" target="_blank">Realtime Mode</a></h3>
      <p>Minimal realtime agent using Gemini Live for lowest-latency voice interactions.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🔀 <a href="examples/hybrid_mode(cascade+realtime)/" target="_blank">Hybrid Mode</a></h3>
      <p>Mix cascade and realtime — custom STT with a realtime model, or realtime with custom TTS.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🧩 <a href="examples/composable_pipelines/" target="_blank">Composable Pipelines</a></h3>
      <p>Flexible Pipeline configs — transcription-only, LLM-only, voice+chat, full voice agent.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🪝 <a href="examples/voice_pipeline_hooks.py" target="_blank">Pipeline Hooks</a></h3>
      <p>Intercept and transform STT, LLM, and TTS data at any stage using <code>@pipeline.on(...)</code>.</p>
    </td>
  </tr>
</table>

### Integrations & Advanced Features

<table width="100%">
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🌐 <a href="examples/mcp_server_examples/" target="_blank">Agent with MCP Server</a></h3>
      <p>Stock Market Analyst Agent with real-time market data access via Model Context Protocol.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🤝 <a href="examples/a2a/" target="_blank">Agent-to-Agent (A2A)</a></h3>
      <p>Multi-agent workflow: customer agent that transfers loan queries to a Loan Specialist Agent.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🦜 <a href="examples/langchain/" target="_blank">LangChain Integration</a></h3>
      <p>Use LangChain tools and agents within the VideoSDK agent framework.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🕸️ <a href="examples/langgraph/" target="_blank">LangGraph Integration</a></h3>
      <p>Orchestrate multi-step agent workflows using LangGraph state machines.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🧠 <a href="examples/mem0/" target="_blank">Memory Agent (Mem0)</a></h3>
      <p>Persistent memory across sessions using Mem0 for long-term context retention.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>👁️ <a href="examples/vision/" target="_blank">Vision Agent</a></h3>
      <p>Multimodal agent that processes video frames alongside voice using cascading or realtime pipelines.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🔄 <a href="examples/n8n_workflow/" target="_blank">n8n Workflow Integration</a></h3>
      <p>Trigger n8n automation workflows from within your agent using webhooks.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🧑‍💼 <a href="examples/human_in_the_loop/" target="_blank">Human in the Loop</a></h3>
      <p>Escalate to a human agent mid-conversation via Discord or other channels.</p>
    </td>
  </tr>
</table>

### Use Case Examples

<table width="100%">
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>📞 <a href="https://github.com/videosdk-community/ai-telephony-demo" target="_blank">AI Telephony Agent</a></h3>
      <p>Hospital appointment booking via a voice-enabled telephony agent.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>✈️ <a href="https://github.com/videosdk-community/videosdk-whatsapp-ai-calling-agent" target="_blank">AI WhatsApp Agent</a></h3>
      <p>Ask about available hotel rooms and book on the go.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🛒 <a href="https://github.com/videosdk-live/agents-quickstart/tree/main/RAG" target="_blank">Agent with Knowledge (RAG)</a></h3>
      <p>Agent that answers questions based on documentation knowledge.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🎭 <a href="https://github.com/videosdk-live/agents-quickstart/tree/main/Virtual%20Avatar" target="_blank">Virtual Avatar Agent</a></h3>
      <p>A Virtual Avatar Agent that presents a weather forecast.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🏥 <a href="use_case_examples/appointment_booking_agent.py" target="_blank">Appointment Booking</a></h3>
      <p>Healthcare front-desk receptionist for scheduling clinic appointments.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>📣 <a href="use_case_examples/announcement_agent.py" target="_blank">Announcement Agent</a></h3>
      <p>Proactive outbound agent for broadcasting announcements.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🎧 <a href="use_case_examples/customer_support_agent.py" target="_blank">Customer Support</a></h3>
      <p>AI-powered customer support agent with escalation and knowledge base.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>📂 <a href="use_case_examples/" target="_blank">More Use Cases</a></h3>
      <p>Call center, IVR, medical triage, language tutor, meeting notes, and more.</p>
    </td>
  </tr>
</table>

## Documentation

For comprehensive guides and API references:

<table width="100%">
  <tr>
    <td width="33%" valign="top" style="padding-left: 20px;">
      <h3>📄 <a href="https://docs.videosdk.live/ai_agents/introduction" target="_blank">Official Documentation</a></h3>
      <p>Complete framework documentation</p>
    </td>
    <td width="33%" valign="top" style="padding-left: 20px;">
      <h3>📝 <a href="https://docs.videosdk.live/agent-sdk-reference/agents/" target="_blank">API Reference</a></h3>
      <p>Detailed API documentation</p>
    </td>
    <td width="33%" valign="top" style="padding-left: 20px;">
      <h3>📂 <a href="examples/" target="_blank">Examples Directory</a></h3>
      <p>Additional code examples</p>
    </td>
  </tr>
</table>


## Contributing

We welcome contributions! Here's how you can help:

<table width="100%">
  <tr>
    <td width="25%" valign="top" style="padding-left: 20px;">
      <h3>🐞 <a href="https://github.com/videosdk-live/agents/issues" target="_blank">Report Issues</a></h3>
      <p>Open an issue for bugs or feature requests</p>
    </td>
    <td width="25%" valign="top" style="padding-left: 20px;">
      <h3>🔀 <a href="https://github.com/videosdk-live/agents/pulls" target="_blank">Submit PRs</a></h3>
      <p>Create a pull request with improvements</p>
    </td>
    <td width="25%" valign="top" style="padding-left: 20px;">
      <h3>🛠️ <a href="BUILD_YOUR_OWN_PLUGIN.md" target="_blank">Build Plugins</a></h3>
      <p>Follow our plugin development guide</p>
    </td>
    <td width="25%" valign="top" style="padding-left: 20px;">
      <h3>💬 <a href="https://discord.com/invite/Gpmj6eCq5u" target="_blank">Join Community</a></h3>
      <p>Connect with us on Discord</p>
    </td>
  </tr>
</table>

The framework is under active development, so contributions in the form of new plugins, features, bug fixes, or documentation improvements are highly appreciated.

### 🛠️ Building Custom Plugins

Want to integrate a new AI provider? Check out **[BUILD YOUR OWN PLUGIN](BUILD_YOUR_OWN_PLUGIN.md)** for:

- Step-by-step plugin creation guide  
- Directory structure and file requirements  
- Implementation examples for STT, LLM, and TTS  
- Testing and submission guidelines  

## Community & Support

Stay connected with VideoSDK:

<table width="100%">
  <tr>
    <td width="25%" valign="top" style="padding-left: 20px;">
      <h3>💬 <a href="https://discord.com/invite/Gpmj6eCq5u" target="_blank">Discord</a></h3>
      <p>Join our community</p>
    </td>
    <td width="25%" valign="top" style="padding-left: 20px;">
      <h3>🐦 <a href="https://x.com/video_sdk" target="_blank">Twitter</a></h3>
      <p>@video_sdk</p>
    </td>
    <td width="25%" valign="top" style="padding-left: 20px;">
      <h3>▶️ <a href="https://www.youtube.com/c/VideoSDK" target="_blank">YouTube</a></h3>
      <p>VideoSDK Channel</p>
    </td>
    <td width="25%" valign="top" style="padding-left: 20px;">
      <h3>🔗 <a href="https://www.linkedin.com/company/video-sdk/" target="_blank">LinkedIn</a></h3>
      <p>VideoSDK Company</p>
    </td>
  </tr>
</table>

> [!TIP]
>
> **Support the Project!** ⭐️  
> Star the repository, join the community, and help us improve VideoSDK by providing feedback, reporting bugs, or contributing plugins.

---

<a href="https://github.com/videosdk-live/agents/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=videosdk-live/agents" />
</a>

**<center>Made with ❤️ by The VideoSDK Team</center>**
