---
name: videosdk-plugins-cartesia
description: Plugin for Cartesia STT and TTS services
---

# videosdk-plugins-cartesia

## Purpose
Integrates Cartesia AI speech services with the VideoSDK AI Agents framework.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `CartesiaTTS` | `TTS` | Cartesia text-to-speech |
| `GenerationConfig` | dataclass | TTS generation configuration |
| `CartesiaSTT` | `STT` | Cartesia speech-to-text |

## Key Files

| File | Description |
|------|-------------|
| `tts.py` | Cartesia TTS with WebSocket streaming |
| `stt.py` | Cartesia STT |

## Environment Variables
- `CARTESIA_API_KEY` — Required for Cartesia services

## Important Notes
- CartesiaTTS is one of the most commonly used TTS providers in the cascade examples

## Installation
```bash
pip install videosdk-plugins-cartesia
```
