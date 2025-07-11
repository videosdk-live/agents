<!--BEGIN_BANNER_IMAGE-->
<p align="center">
  <img src="https://raw.githubusercontent.com/videosdk-community/ai-agent-examples/main/.github/banner.png" alt="VideoSDK AI Agents Banner" style="width:100%;">
</p>
<!--END_BANNER_IMAGE-->

# VideoSDK AI Agents
Open-source framework for developing real-time multimodal conversational AI agents.

![PyPI - Version](https://img.shields.io/pypi/v/videosdk-agents)
[![PyPI Downloads](https://static.pepy.tech/badge/videosdk-agents/month)](https://pepy.tech/projects/videosdk-agents)
[![Twitter Follow](https://img.shields.io/twitter/follow/video_sdk)](https://x.com/video_sdk)
[![YouTube](https://img.shields.io/badge/YouTube-VideoSDK-red)](https://www.youtube.com/c/VideoSDK)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-VideoSDK-blue)](https://www.linkedin.com/company/video-sdk/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA)](https://discord.com/invite/f2WsNDN9S5)

## Overview

The AI Agent SDK is a Python framework built on top of the VideoSDK Python SDK that enables AI-powered agents to join VideoSDK rooms as participants. This SDK serves as a real-time bridge between AI models (like OpenAI and Gemini) and your users, facilitating seamless voice and media interactions.

| #  | Feature                         | Description                                                                 |
|----|----------------------------------|-----------------------------------------------------------------------------|
| 1  | **🎤 Real-time Communication (Audio/Video)**       | Agents can listen, speak, and interact live in meetings.                   |
| 2  | **📞 SIP & Telephony Integration**   | Seamlessly connect agents to phone systems via SIP for call handling, routing, and PSTN access. |
| 3  | **🧍 Virtual Avatars**               | Add lifelike avatars to enhance interaction and presence using Simli.     |
| 4  | **🤖 Multi-Model Support**           | Integrate with OpenAI, Gemini, AWS NovaSonic, and more.                    |
| 5  | **🧩 Cascading Pipeline**            | Integrates with different providers of STT, LLM, and TTS seamlessly.       |
| 6  | **🧠 Conversational Flow**           | Manages turn detection and VAD for smooth interactions.                    |
| 7  | **🛠️ Function Tools**               | Extend agent capabilities with event scheduling, expense tracking, and more. |
| 8  | **🌐 MCP Integration**               | Connect agents to external data sources and tools using Model Context Protocol. |
| 9  | **🔗 A2A Protocol**                  | Enable agent-to-agent interactions for complex workflows.                  |


> \[!IMPORTANT]
>
> **Star VideoSDK Repositories** ⭐️
>
> Get instant notifications for new releases and updates. Your support helps us grow and improve VideoSDK!

## Architecture

This architecture shows how AI voice agents connect to VideoSDK meetings. The system links your backend with VideoSDK's platform, allowing AI assistants to interact with users in real-time.

![VideoSDK AI Agents High Level Architecture](https://strapi.videosdk.live/uploads/Group_15_1_5610ce9c7e.png)

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
        vision: True # Only available when using the Google Gemini Live API
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

When setting up your client application, make sure to use the same meeting ID that your AI Agent is using.

### Step 6: Running the Project
Once you have completed the setup, you can run your AI Voice Agent project using Python. Make sure your `.env` file is properly configured and all dependencies are installed.

```bash
python main.py
```
---

- For detailed guides, tutorials, and API references, check out our official [VideoSDK AI Agents Documentation](https://docs.videosdk.live/ai_agents/introduction).
- To see the framework in action, explore the code in the [Examples](examples/) directory. It is a great place to quickstart.

## Supported Libraries and Plugins

The framework supports integration with various AI models and tools, including:


| Category                 | Services |
|--------------------------|----------|
| **Real-time Models**     | [OpenAI](https://docs.videosdk.live/ai_agents/plugins/realtime/openai) &#124; [Gemini](https://docs.videosdk.live/ai_agents/plugins/realtime/google-live-api) &#124; [AWSNovaSonic](https://docs.videosdk.live/ai_agents/plugins/realtime/aws-nova-sonic) |
| **Speech-to-Text (STT)** | [OpenAI](https://docs.videosdk.live/ai_agents/plugins/stt/openai) &#124; [Google](https://docs.videosdk.live/ai_agents/plugins/stt/google) &#124; [Sarvam AI](https://docs.videosdk.live/ai_agents/plugins/stt/sarvam-ai) &#124; [Deepgram](https://docs.videosdk.live/ai_agents/plugins/stt/deepgram) &#124; [Cartesia](https://docs.videosdk.live/ai_agents/plugins/stt/cartesia-stt)|
| **Language Models (LLM)**| [OpenAI](https://docs.videosdk.live/ai_agents/plugins/llm/openai) &#124; [Google](https://docs.videosdk.live/ai_agents/plugins/llm/google-llm) &#124; [Sarvam AI](https://docs.videosdk.live/ai_agents/plugins/llm/sarvam-ai-llm) &#124; [Anthropic](https://docs.videosdk.live/ai_agents/plugins/llm/anthropic-llm) &#124; [Cerebras](https://docs.videosdk.live/ai_agents/plugins/llm/Cerebras-llm) |
| **Text-to-Speech (TTS)** | [OpenAI](https://docs.videosdk.live/ai_agents/plugins/tts/openai) &#124; [Google](https://docs.videosdk.live/ai_agents/plugins/tts/google-tts) &#124; [AWS Polly](https://docs.videosdk.live/ai_agents/plugins/tts/aws-polly-tts) &#124; [Sarvam AI](https://docs.videosdk.live/ai_agents/plugins/tts/sarvam-ai-tts) &#124; [ElevenLabs](https://docs.videosdk.live/ai_agents/plugins/tts/eleven-labs) &#124; [Cartesia](https://docs.videosdk.live/ai_agents/plugins/tts/cartesia-tts)  &#124; [Resemble AI](https://docs.videosdk.live/ai_agents/plugins/tts/resemble-ai-tts) &#124;[Smallest AI](https://docs.videosdk.live/ai_agents/plugins/tts/smallestai-tts) &#124; [Speechify](https://docs.videosdk.live/ai_agents/plugins/tts/speechify-tts) &#124; [InWorld](https://docs.videosdk.live/ai_agents/plugins/tts/inworld-ai-tts) &#124; [Neuphonic](https://docs.videosdk.live/ai_agents/plugins/tts/neuphonic-tts) &#124; [Rime AI](https://docs.videosdk.live/ai_agents/plugins/tts/rime-ai-tts) &#124; [Hume AI](https://docs.videosdk.live/ai_agents/plugins/tts/hume-ai-tts) &#124; [Groq](https://docs.videosdk.live/ai_agents/plugins/tts/groq-ai-tts) &#124; [LMNT AI](https://docs.videosdk.live/ai_agents/plugins/tts/lmnt-ai-tts) |
| **Voice Activity Detection (VAD)** | [SileroVAD](https://docs.videosdk.live/ai_agents/plugins/silero-vad) |
| **Turn Detection Model** | [Turn Detector](https://docs.videosdk.live/ai_agents/plugins/turn-detector) |
| **Virtual Avatar** | [Simli](https://docs.videosdk.live/ai_agents/plugins/avatar/simli) |
| **SIP Trunking** | [Twilio](https://docs.videosdk.live/ai_agents/sip) |

## Examples

Explore the following examples to see the framework in action:

<h2>🤖 AI Voice Agent Demos</h2>

<table>
  <tr>
    <td width="50%" valign="top" style="padding: 10px;">
      <h3>📞 <a href="https://github.com/videosdk-community/ai-telephony-demo" target="_blank">AI Telephony Agent</a></h3>
      <p>Use case: Hospital appointment booking via a voice-enabled agent.</p>
    </td>
    <td width="50%" valign="top" style="padding: 10px;">
      <h3>✈️ <a href="https://github.com/videosdk-community/a2a-mcp-agent" target="_blank">A2A MCP Agent</a></h3>
      <p>Use case: Ask about available flights & hotels and send email with booking info.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" style="padding: 10px;">
      <h3>👨‍🏫 <a href="https://github.com/videosdk-community/ai-avatar-demo" target="_blank">AI Avatar</a></h3>
      <p>Use case: Answering queries about current weather conditions using an avatar.</p>
    </td>
    <td width="50%" valign="top" style="padding: 10px;">
      <h3>🛒 <a href="https://github.com/videosdk-community/ai-agent-demo/tree/conversational-flow" target="_blank">Conversational Flow Agent</a></h3>
      <p>Use case: E-commerce scenario with turn detection when interrupting the voice agent.</p>
    </td>
  </tr>
</table>

## Contributing

The Agents framework is under active development in a rapidly evolving field. We welcome and appreciate contributions of any kind, be it feedback, bugfixes, features, new plugins and tools, or better documentation. You can file issues under this repo, open a PR, or chat with us in VideoSDK's [Discord community](https://discord.com/invite/f2WsNDN9S5).


When contributing, consider developing new plugins or enhancing existing ones to expand the framework's capabilities. Your contributions can help integrate more AI models and tools, making the framework even more versatile.

We love our contributors! Here's how you can contribute:

- [Open an issue](https://github.com/videosdk-live/agents/issues) if you believe you've encountered a bug.
- Follow the [documentation guide](https://docs.videosdk.live/ai_agents/introduction) to get your local dev environment set up.
- Make a [pull request](https://github.com/videosdk-live/agents/pull) to add new features/make quality-of-life improvements/fix bugs.

<a href="https://github.com/videosdk-live/agents/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=videosdk-live/agents" />
</a>
