---
name: videosdk-plugins-gladia
description: Plugin for Gladia STT service
---

# videosdk-plugins-gladia

## Purpose
Integrates Gladia speech-to-text with the VideoSDK AI Agents framework.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `GladiaSTT` | `STT` | Gladia streaming speech-to-text |

## Key Files

| File | Description |
|------|-------------|
| `stt.py` | Gladia STT (WebSocket streaming) |

## Environment Variables
- `GLADIA_API_KEY` — Required for Gladia services

## Installation
```bash
pip install videosdk-plugins-gladia
```
