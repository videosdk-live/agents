---
name: videosdk-plugins-ultravox
description: Plugin for Ultravox realtime speech-to-speech model
---

# videosdk-plugins-ultravox

## Purpose
Integrates Ultravox realtime model with the VideoSDK AI Agents framework.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `UltravoxRealtime` | `RealtimeBaseModel` | Ultravox realtime speech-to-speech |
| `UltravoxLiveConfig` | dataclass | Configuration for Ultravox realtime |

## Key Files

| File | Description |
|------|-------------|
| `ultravox_realtime.py` | Ultravox realtime API implementation |

## Environment Variables
- `ULTRAVOX_API_KEY` — Required for Ultravox services

## Installation
```bash
pip install videosdk-plugins-ultravox
```
