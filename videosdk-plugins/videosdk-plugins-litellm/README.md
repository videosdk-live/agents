# VideoSDK LiteLLM Plugin

Agent Framework plugin for LLM services via [LiteLLM](https://docs.litellm.ai/) - unified interface to 100+ providers.

## Installation

```bash
pip install videosdk-plugins-litellm
```

## Usage

```python
from videosdk.plugins.litellm import LiteLLM

llm = LiteLLM(model="anthropic/claude-sonnet-4-6")
```

LiteLLM resolves credentials from standard provider env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.).

See [LiteLLM provider docs](https://docs.litellm.ai/docs/providers) for supported models.
