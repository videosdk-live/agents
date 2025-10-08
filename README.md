<!--BEGIN_BANNER_IMAGE-->
<p align="center">
  <img src="https://raw.githubusercontent.com/videosdk-community/ai-agent-examples/main/.github/banner.png" alt="VideoSDK AI Agents Banner" style="width:100%;">
  <a href="https://www.producthunt.com/products/video-sdk/launches/voice-agent-sdk?embed=true&utm_source=badge-featured&utm_medium=badge&utm_source=badge-voice&#0045;agent&#0045;sdk" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=991216&theme=light&t=1752567949948" alt="Voice&#0032;Agent&#0032;SDK - The&#0032;Open&#0045;Source&#0032;Framework&#0032;For&#0032;Real&#0045;Time&#0032;AI&#0032;Voice | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
</p>
<!--END_BANNER_IMAGE-->

# VideoSDK AI Agents
Open-source framework for building real-time multimodal conversational AI agents.

![PyPI - Version](https://img.shields.io/pypi/v/videosdk-agents)
[![PyPI Downloads](https://static.pepy.tech/badge/videosdk-agents/month)](https://pepy.tech/projects/videosdk-agents)
[![Twitter Follow](https://img.shields.io/twitter/follow/video_sdk)](https://x.com/video_sdk)
[![YouTube](https://img.shields.io/badge/YouTube-VideoSDK-red)](https://www.youtube.com/c/VideoSDK)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-VideoSDK-blue)](https://www.linkedin.com/company/video-sdk/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA)](https://discord.com/invite/f2WsNDN9S5)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/videosdk-live/agents)


The **VideoSDK AI Agents framework** connects your infrastructure, agent worker, VideoSDK room, and user devices, enabling **real-time, natural voice and multimodal interactions** between users and intelligent agents.

<!-- ![VideoSDK AI Agents High Level Architecture](https://strapi.videosdk.live/uploads/Group_15_1_5610ce9c7e.png) -->
![VideoSDK AI Agents High Level Architecture](https://cdn.videosdk.live/website-resources/docs-resources/voice_agent_intro.png)


## Overview

The AI Agent SDK is a Python framework built on top of the VideoSDK Python SDK that enables AI-powered agents to join VideoSDK rooms as participants. This SDK serves as a real-time bridge between AI models (like OpenAI or Gemini) and your users, facilitating seamless voice and media interactions.

<table width="100%">
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🎙️ <a href="examples/test_cascading_pipeline.py" target="_blank">Agent with Cascading Pipeline</a></h3>
      <p>Test an AI Voice Agent that uses a Cascading Pipeline for STT → LLM → TTS.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>📞 <a href="examples/sip_agent_example.py" target="_blank">AI Telephony Agent</a></h3>
      <p>Test an AI Agent that answers and interacts over phone calls using SIP.</p>
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
| 3  | **🧍 Virtual Avatars**               | Add lifelike avatars to enhance interaction and presence using Simli.     |
| 4  | **🤖 Multi-Model Support**           | Integrate with OpenAI, Gemini, AWS NovaSonic, and more.                    |
| 5  | **🧩 Cascading Pipeline**            | Integrates with different providers of STT, LLM, and TTS seamlessly.       |
| 6  | **⚡ Realtime Pipeline**         | Use unified realtime models (OpenAI Realtime, AWS Nova, Gemini Live) for lowest latency | 
| 7  | **🧠 Conversational Flow**           | Manages turn detection and VAD for smooth interactions.                    |
| 8  | **🛠️ Function Tools**               | Extend agent capabilities with event scheduling, expense tracking, and more. |
| 9  | **🌐 MCP Integration**               | Connect agents to external data sources and tools using Model Context Protocol. |
| 10  | **🔗 A2A Protocol**                  | Enable agent-to-agent interactions for complex workflows.                  |
| 11 | **📊 Observability**             | Built-in OpenTelemetry tracing and metrics collection |  
| 12 | **🚀 CLI Tool**                  | Run agents locally and test with `videosdk` CLI |  



> \[!IMPORTANT]
>
> **Star VideoSDK Repositories** ⭐️
>
> Get instant notifications for new releases and updates. Your support helps us grow and improve VideoSDK!


## Pre-requisites

Before you begin, ensure you have:

- A VideoSDK authentication token (generate from [app.videosdk.live](https://app.videosdk.live))
   - A VideoSDK meeting ID (you can generate one using the [Create Room API](https://docs.videosdk.live/api-reference/realtime-communication/create-room) or through the VideoSDK dashboard)
- Python 3.12 or higher
- Third-Party API Keys:
   - API keys for the services you intend to use (e.g., OpenAI for LLM/STT/TTS, ElevenLabs for TTS, Google for Gemini etc.).

## Installation

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

The pipeline connects your agent to an AI model. Here, we are using Google's Gemini for a [Real-time Pipeline](https://docs.videosdk.live/ai_agents/core-components/realtime-pipeline). You could also use a [Cascading Pipeline](https://docs.videosdk.live/ai_agents/core-components/cascading-pipeline).


```python
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from videosdk.agents import RealTimePipeline, JobContext

async def start_session(context: JobContext):
    # Initialize the AI model
    model = GeminiRealtime(
        model="gemini-2.0-flash-live-001",
        # When GOOGLE_API_KEY is set in .env - DON'T pass api_key parameter
        api_key="AKZSXXXXXXXXXXXXXXXXXXXX",
        config=GeminiLiveConfig(
            voice="Leda", # Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, and Zephyr.
            response_modalities=["AUDIO"]
        )
    )

    pipeline = RealTimePipeline(model=model)

    # Continue to the next steps...
```
### Step 4: Assembling and Starting the Agent Session

Now, let's put everything together and start the agent session:

```python
import asyncio
from videosdk.agents import AgentSession, WorkerJob, RoomOptions, JobContext

async def start_session(context: JobContext):
    # ... previous setup code ...

    # Create the agent session
    session = AgentSession(
        agent=VoiceAgent(),
        pipeline=pipeline
    )

    try:
       await context.connect()
        # Start the session
        await session.start()
        # Keep the session running until manually terminated
        await asyncio.Event().wait()
    finally:
        # Clean up resources when done
        await session.close()
        await context.shutdown()

def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="<meeting_id>", # Replace it with your actual meetingID
        auth_token = "<VIDEOSDK_AUTH_TOKEN>", # When VIDEOSDK_AUTH_TOKEN is set in .env - DON'T include videosdk_auth
        name="Test Agent", 
        playground=True,
        # vision= True # Only available when using the Google Gemini Live API
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
> **Test Your Agent Instantly with the CLI Tool**
>
> Run your agent locally using:
>
> ```bash
> python main.py console
> ```
>
> Experience real-time interactions right from your terminal - no meeting room required!  
> Speak and listen through your system’s mic and speakers for quick testing and rapid development.


### Step 7: Deployment

For deployment options and guide, checkout the official documentation here: [Deployment](https://docs.videosdk.live/ai_agents/deployments/introduction)

---

<!-- - For detailed guides, tutorials, and API references, check out our official [VideoSDK AI Agents Documentation](https://docs.videosdk.live/ai_agents/introduction).
- To see the framework in action, explore the code in the [Examples](examples/) directory. It is a great place to quickstart. -->

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
| **Virtual Avatar** | [Simli](https://docs.videosdk.live/ai_agents/core-components/avatar) |
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

<h2>🤖 AI Voice Agent Usecases</h2>

<table width="100%">
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>📞 <a href="https://github.com/videosdk-community/ai-telephony-demo" target="_blank">AI Telephony Agent Quickstart</a></h3>
      <p>Use case: Hospital appointment booking via a voice-enabled agent.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>✈️ <a href="https://github.com/videosdk-community/videosdk-whatsapp-ai-calling-agent" target="_blank">AI Whatsapp Agent Quickstart</a></h3>
      <p>Use case: Ask about available hotel rooms and book on the go.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>👨‍🏫 <a href="https://github.com/videosdk-live/agents-quickstart/tree/main/A2A" target="_blank">Multi Agent System</a></h3>
      <p>Use case: Customer care agent that transfers loan related to queries to Loan Specialist Agent.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🛒 <a href="https://github.com/videosdk-live/agents-quickstart/tree/main/RAG" target="_blank">Agent with Knowledge (RAG)</a></h3>
      <p>Use case: Agent that answers questions based on documentation knowledge.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>👨‍🏫 <a href="https://github.com/videosdk-live/agents/tree/main/examples/mcp_server_examples" target="_blank">Agent with MCP Server</a></h3>
      <p>Use case: Stock Market Analyst Agent with realtime Market Data Access.</p>
    </td>
    <td width="50%" valign="top" style="padding-left: 20px;">
      <h3>🛒 <a href="https://github.com/videosdk-live/agents-quickstart/tree/main/Virtual%20Avatar" target="_blank">Virtual Avatar Agent</a></h3>
      <p>Use case: A Virtual Avatar Agent that presents weather forecast. </p>
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