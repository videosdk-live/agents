---
name: videosdk-plugins-assemblyai
description: Plugin for AssemblyAI STT service
---

# videosdk-plugins-assemblyai

## Purpose
Integrates AssemblyAI speech-to-text with the VideoSDK AI Agents framework.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `AssemblyAISTT` | `STT` | AssemblyAI streaming speech-to-text |

## Key Files

| File | Description |
|------|-------------|
| `stt.py` | AssemblyAI STT (WebSocket streaming) |

## Environment Variables
- `ASSEMBLYAI_API_KEY` — Required for AssemblyAI services

## Installation
```bash
pip install videosdk-plugins-assemblyai
```
