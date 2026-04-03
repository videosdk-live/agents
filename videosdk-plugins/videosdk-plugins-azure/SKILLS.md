---
name: videosdk-plugins-azure
description: Plugin for Azure AI Speech (STT/TTS), Azure Voice Live (Realtime) services
---

# videosdk-plugins-azure

## Purpose
Integrates Azure AI services with the VideoSDK AI Agents framework, providing Azure AI Speech STT/TTS and Azure Voice Live realtime capabilities.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `AzureSTT` | `STT` | Azure AI Speech-to-Text |
| `AzureTTS` | `TTS` | Azure AI Text-to-Speech |
| `VoiceTuning` | dataclass | Voice tuning parameters for Azure TTS |
| `SpeakingStyle` | Enum | Speaking style options for Azure TTS |
| `AzureVoiceLive` | `RealtimeBaseModel` | Azure Voice Live realtime model |
| `AzureVoiceLiveConfig` | dataclass | Configuration for Azure Voice Live |

## Key Files

| File | Description |
|------|-------------|
| `stt.py` | Azure AI Speech STT |
| `tts.py` | Azure AI Speech TTS with voice tuning and speaking styles |
| `voice_live.py` | Azure Voice Live realtime API |

## Environment Variables
- `AZURE_SPEECH_KEY` — Azure Speech service key
- `AZURE_SPEECH_REGION` — Azure region
- `AZURE_OPENAI_API_KEY` — For Azure OpenAI services
- `AZURE_OPENAI_ENDPOINT` — Azure OpenAI endpoint URL

## Installation
```bash
pip install videosdk-plugins-azure
```
