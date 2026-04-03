---
name: videosdk-plugins-openai
description: Plugin for OpenAI Realtime API, LLM (GPT-4/5), STT (Whisper), and TTS services
---

# videosdk-plugins-openai

## Purpose
Integrates OpenAI services with the VideoSDK AI Agents framework, providing Realtime (speech-to-speech), LLM, STT, and TTS capabilities.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `OpenAIRealtime` | `RealtimeBaseModel` | OpenAI Realtime API (gpt-4o-realtime-preview) |
| `OpenAIRealtimeConfig` | dataclass | Config for realtime model (modalities, voice, turn detection) |
| `OpenAILLM` | `LLM` | Chat completions (GPT-4o, GPT-5, o-series reasoning) |
| `OpenAISTT` | `STT` | Whisper-based speech-to-text |
| `OpenAITTS` | `TTS` | OpenAI text-to-speech |

## Key Files

| File | Description |
|------|-------------|
| `realtime_api.py` | OpenAI Realtime WebSocket API implementation |
| `llm.py` | Chat completions with streaming, tool calls, reasoning model support |
| `stt.py` | Whisper STT via WebSocket |
| `tts.py` | OpenAI TTS synthesis |
| `__init__.py` | Public exports |
| `version.py` | Package version |

## Environment Variables
- `OPENAI_API_KEY` — Required for all OpenAI services

## Important Notes
- `OpenAILLM` has special handling for GPT-5 and reasoning models (o-series): uses `developer` role instead of `system`, `max_completion_tokens` instead of `max_tokens`, and gates `temperature`/`top_p`
- `OpenAIRealtime` uses WebSocket connection to `wss://api.openai.com/v1/realtime`
- The realtime model supports both text and audio modalities
- STT operates at 16kHz — framework resamples from 48kHz automatically

## Installation
```bash
pip install videosdk-plugins-openai
```
