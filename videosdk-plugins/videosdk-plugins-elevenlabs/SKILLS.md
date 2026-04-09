---
name: videosdk-plugins-elevenlabs
description: Plugin for ElevenLabs TTS and STT services
---

# videosdk-plugins-elevenlabs

## Purpose
Integrates ElevenLabs voice AI services with the VideoSDK AI Agents framework.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `ElevenLabsTTS` | `TTS` | ElevenLabs text-to-speech with voice cloning |
| `VoiceSettings` | dataclass | Voice settings (stability, similarity, style) |
| `ElevenLabsSTT` | `STT` | ElevenLabs speech-to-text |

## Key Files

| File | Description |
|------|-------------|
| `tts.py` | ElevenLabs TTS with WebSocket streaming |
| `stt.py` | ElevenLabs STT |

## Environment Variables
- `ELEVENLABS_API_KEY` — Required for ElevenLabs services

## Installation
```bash
pip install videosdk-plugins-elevenlabs
```
