---
name: videosdk-plugins-cometapi
description: Plugin for CometAPI STT, LLM, and TTS services
---

# videosdk-plugins-cometapi

## Purpose
Integrates CometAPI services with the VideoSDK AI Agents framework, providing STT, LLM, and TTS capabilities.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `CometAPISTT` | `STT` | CometAPI speech-to-text |
| `CometAPILLM` | `LLM` | CometAPI language model |
| `CometAPITTS` | `TTS` | CometAPI text-to-speech |

## Key Files

| File | Description |
|------|-------------|
| `stt.py` | CometAPI STT |
| `llm.py` | CometAPI LLM |
| `tts.py` | CometAPI TTS |

## Environment Variables
- `COMETAPI_API_KEY` — Required for CometAPI services

## Installation
```bash
pip install videosdk-plugins-cometapi
```
