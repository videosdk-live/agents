# VideoSDK Agents — LiteLLM Plugin

LLM plugin that routes calls through the [LiteLLM](https://docs.litellm.ai/) SDK,
giving VideoSDK Agents coverage of **100+ providers** (OpenAI, Anthropic, AWS
Bedrock, Vertex AI, Cohere, Mistral, Groq, Perplexity, Together, Fireworks,
Cerebras, Databricks, IBM Watsonx, ...) through a single plugin import.

LiteLLM normalizes every backing's response to OpenAI's chat-completions shape,
so this plugin subclasses `videosdk.plugins.openai.OpenAILLM` and reuses all of
its streaming, tool-call accumulation, and usage-tracking logic.

## Install

```bash
pip install videosdk-plugins-litellm
```

## Quickstart

```python
from videosdk import Pipeline
from videosdk.plugins.litellm import LiteLLMLLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector

pipeline = Pipeline(
    stt=DeepgramSTT(),
    llm=LiteLLMLLM(model="anthropic/claude-sonnet-4-6"),
    tts=ElevenLabsTTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)
```

LiteLLM resolves credentials from each backing's standard env var
(`ANTHROPIC_API_KEY` for `anthropic/...`, `OPENAI_API_KEY` for `openai/...`,
`AWS_*` for `bedrock/...`, etc.). No extra wiring is needed beyond what each
backing already documents.

## Configuration

```python
LiteLLMLLM(
    model="anthropic/claude-sonnet-4-6",   # required
    api_key=None,                           # optional, proxy mode override
    api_base=None,                          # optional, e.g. "http://localhost:4000"
    api_version=None,                       # optional, Azure-style endpoints
    temperature=0.7,
    tool_choice="auto",
    drop_params=True,                       # default; strips kwargs the chosen backing doesn't accept
    extra_kwargs=None,                      # forwarded verbatim to litellm.acompletion
    extra_headers=None,                     # forwarded as extra_headers
    reasoning_effort=None,                  # for reasoning / GPT-5 models
    verbosity=None,                         # for reasoning / GPT-5 models
)
```

### Common model specs

| Backing | Required env var | Example spec |
|---|---|---|
| Anthropic direct | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4-6` |
| OpenAI direct | `OPENAI_API_KEY` | `openai/gpt-4o` |
| Google Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini/gemini-2.5-pro` |
| AWS Bedrock | `AWS_ACCESS_KEY_ID` (+ secret + region) | `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0` |
| Vertex AI | `GOOGLE_APPLICATION_CREDENTIALS` (+ project) | `vertex_ai/claude-sonnet-4-5` |
| Azure OpenAI | `AZURE_API_KEY` (+ base) | `azure/gpt-4o` |
| Cohere | `COHERE_API_KEY` | `cohere/command-r-plus-08-2024` |
| Mistral | `MISTRAL_API_KEY` | `mistral/mistral-large-latest` |
| Groq | `GROQ_API_KEY` | `groq/llama-3.3-70b-versatile` |
| xAI | `XAI_API_KEY` | `xai/grok-4` |

For the full list see the
[LiteLLM provider docs](https://docs.litellm.ai/docs/providers).

## Proxy mode

Run a LiteLLM proxy with model deployments and route through it:

```python
LiteLLMLLM(
    model="anthropic/claude-sonnet-4-6",
    api_base="http://localhost:4000",
    api_key="sk-fastagent-proxy-1234",
)
```

Useful for centralized credential management and audit logging across teams.

## License

Apache-2.0
