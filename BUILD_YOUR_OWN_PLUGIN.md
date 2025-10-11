# Building Your Own VideoSDK Agent Plugins

Welcome, contributor! This guide provides a straightforward path to creating custom Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech (TTS) plugins for the VideoSDK Agent Framework.

## Quick Start

1.  **Fork the repository** on GitHub.
2.  **Study the base classes** in `videosdk-agents/videosdk/agents/`.
3.  **Review existing plugins** (e.g., `videosdk-plugins-openai`) for practical examples.
4.  **Create your plugin** using the directory structure and guidance below.
5.  **Submit a Pull Request** for our team to review.

## Plugin Directory Structure

Your plugin must follow this exact structure to be compatible with the framework.

```
videosdk-plugins/
└── videosdk-plugins-{your-service}/
    ├── pyproject.toml
    |-- README.md
    └── videosdk/
        └── plugins/
            └── {your-service}/
                ├── __init__.py
                ├── version.py
                ├── stt.py (optional)
                ├── llm.py (optional)
                └── tts.py (optional)
```

## Core Plugin Files

These files are essential for your plugin's packaging and initialization.

### `pyproject.toml`

This file manages your plugin's dependencies and packaging.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "videosdk-plugins-{your-service}"
dynamic = ["version"]
description = "VideoSDK Agent Framework plugin for {Your Service}"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.11"
authors = [{ "name": "videosdk" }]
dependencies = [
    "videosdk-agents>=0.0.38",
    # Add any other required dependencies here, e.g., "openai>=1.0.0"
]

[tool.hatch.version]
path = "videosdk/plugins/{your-service}/version.py"

[tool.hatch.build.targets.wheel]
packages = ["videosdk"]
```

### `README.md`

A brief but informative README is crucial.

```markdown
# VideoSDK - {Your Service} Plugin

This plugin integrates {Your Service} with the VideoSDK Agent Framework.

**Implemented Features:**

- [x] STT
- [ ] LLM
- [ ] TTS

**API Key Setup:**
Set the `{YOUR_SERVICE}_API_KEY` environment variable.
```

### `version.py`

```python
__version__ = "0.0.1"
```

### `__init__.py`

This file makes your plugin's classes importable.

```python
# Import and expose only the classes you have implemented
from .stt import YourServiceSTT
from .llm import YourServiceLLM
from .tts import YourServiceTTS

__all__ = [
    'YourServiceSTT',
    'YourServiceLLM',
    'YourServiceTTS'
]
```

## Implementing Your Plugin

Inherit from the base classes and implement the required abstract methods.

### STT Plugin

```python
from typing import Optional
from videosdk.agents import STT as BaseSTT, STTResponse, SpeechEventType, SpeechData

class YourServiceSTT(BaseSTT):
    def __init__(self, api_key: str, **kwargs):
        super().__init__()
        self.api_key = api_key
        # Initialize your WebSocket client and other resources here

    async def process_audio(self, audio_frames: bytes, language: Optional[str] = None, **kwargs):
        """
        Process incoming audio frames. The framework provides audio at 48kHz.
        Resample if your provider requires a different sample rate.
        """
        # 1. Send audio to your service's WebSocket endpoint.
        # 2. Receive transcription data.
        # 3. Call self._transcript_callback(response) with STTResponse objects.
        pass

    async def aclose(self):
        """Clean up all resources, like closing WebSocket connections."""
        # Your cleanup logic here
        pass
```

### LLM Plugin

```python
from typing import Any, AsyncIterator, List, Optional
from videosdk.agents import LLM as BaseLLM, LLMResponse, ChatContext, ChatRole, FunctionTool

class YourServiceLLM(BaseLLM):
    def __init__(self, api_key: str, **kwargs):
        super().__init__()
        self.api_key = api_key
        # Initialize your HTTP client here

    async def chat(
        self,
        messages: ChatContext,
        tools: Optional[List[FunctionTool]] = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMResponse]:
        """
        Process messages and yield responses in a stream.
        """
        # 1. Format the 'messages' and 'tools' for your provider's API.
        # 2. Make a streaming API request.
        # 3. As you receive response chunks, yield LLMResponse objects.
        yield LLMResponse(content="Hello, world!", role=ChatRole.ASSISTANT)
```

### TTS Plugin

```python
from typing import Any, AsyncIterator, Optional, Union
from videosdk.agents import TTS as BaseTTS

class YourServiceTTS(BaseTTS):
    def __init__(self, api_key: str, **kwargs):
        # Set the sample rate and channels your provider uses.
        super().__init__(sample_rate=24000, num_channels=1)
        self.api_key = api_key
        self.audio_track = None # This is set by the framework

    async def synthesize(self, text: Union[AsyncIterator[str], str], voice_id: Optional[str] = None, **kwargs: Any):
        """
        Convert text to speech and stream the audio data.
        """
        # 1. Make a streaming API request to your provider with the text.
        # 2. As you receive audio chunks, push them to the audio track.
        # if self.audio_track:
        #     asyncio.create_task(self.audio_track.add_new_bytes(chunk))
        pass

    async def interrupt(self):
        """Interrupt any ongoing audio synthesis."""
        if self.audio_track:
            self.audio_track.interrupt()
```

## Implementation Checklist

### For All Plugins

- [ ] Inherit from the correct base class (`STT`, `LLM`, or `TTS`).
- [ ] Implement all abstract methods defined in the base class.
- [ ] Emit errors consistently using `self.emit("error", message)`.
- [ ] Clean up all resources (clients, connections) in the `aclose()` method.
- [ ] Use async patterns correctly for all I/O operations.

### STT Specific

- [ ] Handle WebSocket connections gracefully (connect, disconnect, errors).
- [ ] The framework provides 48kHz audio. **Resample audio if your provider requires a different sample rate.**

### LLM Specific

- [ ] Ensure the `chat()` method is a streaming implementation (using `async for` and `yield`).
- [ ] Support function tools if the provider's API allows for it.

### TTS Specific

- [ ] Set the correct `sample_rate` and `num_channels` in the `super().__init__()` call.
- [ ] Push audio chunks to `self.audio_track` for playback.

## Testing Your Plugin

Before submitting, install and test your plugin locally.

```bash
# Install your plugin in editable mode from its root directory
pip install -e .

# Create a test script (e.g., test_my_plugin.py)
# and run it to verify functionality.
python test_my_plugin.py
```

We highly recommend running your plugin with the example scripts in the `examples/` directory to catch breaking changes or integration issues early.

## Submitting a Pull Request

Once your plugin is ready:

1.  Push your branch to your fork.
2.  Create a Pull Request against the main repository.
3.  Fill out the PR template with details about your service, implemented features, and any special configuration notes.

## Reference

- **Base Classes**: `videosdk-agents/videosdk/agents/`
- **Plugin Examples**: `videosdk-plugins/videosdk-plugins-openai/`
