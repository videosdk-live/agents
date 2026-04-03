---
name: videosdk-plugins-murfai
description: Plugin for Murf AI TTS service
---

# videosdk-plugins-murfai

## Purpose
Integrates Murf AI TTS with the VideoSDK AI Agents framework.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `MurfAITTS` | `TTS` | Murf AI text-to-speech |
| `MurfAIVoiceSettings` | dataclass | Voice configuration for Murf AI |

## Key Files

| File | Description |
|------|-------------|
| `tts.py` | Murf AI TTS synthesis |

## Environment Variables
- `MURF_API_KEY` — Required for Murf AI services

## Installation
```bash
pip install videosdk-plugins-murfai
```
