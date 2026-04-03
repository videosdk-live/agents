---
name: videosdk-plugins-google
description: Plugin for Gemini Live (Realtime), Google LLM, STT (Cloud Speech), and TTS services
---

# videosdk-plugins-google

## Purpose
Integrates Google AI services with the VideoSDK AI Agents framework, providing Gemini Live realtime, Gemini LLM, Cloud Speech STT, and Cloud TTS capabilities.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `GeminiRealtime` | `RealtimeBaseModel` | Gemini Live API (gemini-3.1-flash-live-preview) |
| `GeminiLiveConfig` | dataclass | Config for Gemini Live (voice, response_modalities) |
| `GoogleLLM` | `LLM` | Gemini chat completions (gemini-2.5-flash, etc.) |
| `VertexAIConfig` | dataclass | Vertex AI configuration for enterprise deployments |
| `GoogleSTT` | `STT` | Google Cloud Speech-to-Text |
| `VoiceActivityConfig` | dataclass | VAD config for Google STT |
| `GoogleTTS` | `TTS` | Google Cloud Text-to-Speech |
| `GoogleVoiceConfig` | dataclass | Voice configuration for Google TTS |

## Key Files

| File | Description |
|------|-------------|
| `live_api.py` | Gemini Live WebSocket API (realtime speech-to-speech) |
| `llm.py` | Gemini chat completions with streaming + tool calls |
| `stt.py` | Google Cloud Speech-to-Text (streaming) |
| `tts.py` | Google Cloud TTS synthesis |

## Environment Variables
- `GOOGLE_API_KEY` — Required for Gemini models and Google AI services

## Important Notes
- Gemini 3.x models require `send_realtime_input` instead of `send_client_content` for text
- GeminiRealtime uses `response_modalities=["AUDIO"]` by default for voice
- The `_send_text` helper method handles the protocol differences between Gemini 2.x and 3.x
- Gemini Live natively supports audio features — no separate VAD needed in realtime mode
- GoogleLLM supports both standard Google AI and Vertex AI (enterprise) via `VertexAIConfig`

## Installation
```bash
pip install videosdk-plugins-google
```
