---
name: videosdk-plugins-cambai
description: Plugin for CambAI TTS service
---

# videosdk-plugins-cambai

## Purpose
Integrates CambAI TTS with the VideoSDK AI Agents framework.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `CambAITTS` | `TTS` | CambAI text-to-speech |
| `InferenceOptions` | dataclass | Inference configuration |
| `OutputConfiguration` | dataclass | Output format configuration |
| `VoiceSettings` | dataclass | Voice customization settings |

## Key Files

| File | Description |
|------|-------------|
| `tts.py` | CambAI TTS synthesis |

## Environment Variables
- `CAMBAI_API_KEY` — Required for CambAI services

## Installation
```bash
pip install videosdk-plugins-cambai
```
