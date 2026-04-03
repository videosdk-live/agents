---
name: videosdk-plugins-deepgram
description: Plugin for Deepgram STT (speech-to-text) and TTS (text-to-speech) services
---

# videosdk-plugins-deepgram

## Purpose
Integrates Deepgram's speech services with the VideoSDK AI Agents framework, providing high-accuracy STT and TTS.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `DeepgramSTT` | `STT` | Deepgram streaming speech-to-text |
| `DeepgramSTTV2` | `STT` | Deepgram STT v2 API |
| `DeepgramTTS` | `TTS` | Deepgram text-to-speech |

## Key Files

| File | Description |
|------|-------------|
| `stt.py` | Deepgram STT v1 (WebSocket streaming) |
| `stt_v2.py` | Deepgram STT v2 API |
| `tts.py` | Deepgram TTS synthesis |

## Environment Variables
- `DEEPGRAM_API_KEY` — Required for Deepgram services

## Important Notes
- DeepgramSTT uses WebSocket streaming for real-time transcription
- One of the most commonly used STT providers in cascade examples
- Supports interim and final transcripts via the `SpeechEventType` enum

## Installation
```bash
pip install videosdk-plugins-deepgram
```
