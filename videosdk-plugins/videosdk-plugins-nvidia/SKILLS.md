---
name: videosdk-plugins-nvidia
description: Plugin for NVIDIA STT and TTS services
---

# videosdk-plugins-nvidia

## Purpose
Integrates NVIDIA AI speech services with the VideoSDK AI Agents framework.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `NvidiaSTT` | `STT` | NVIDIA speech-to-text |
| `NvidiaTTS` | `TTS` | NVIDIA text-to-speech |

## Key Files

| File | Description |
|------|-------------|
| `stt.py` | NVIDIA STT implementation |
| `tts.py` | NVIDIA TTS synthesis |

## Environment Variables
- `NVIDIA_API_KEY` — Required for NVIDIA services

## Installation
```bash
pip install videosdk-plugins-nvidia
```
