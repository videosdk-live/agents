---
name: videosdk-plugins-xai
description: Plugin for xAI Grok Realtime and LLM services
---

# videosdk-plugins-xai

## Purpose
Integrates xAI (Grok) services with the VideoSDK AI Agents framework, providing realtime speech-to-speech and LLM capabilities.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `XAIRealtime` | `RealtimeBaseModel` | xAI Grok realtime speech-to-speech |
| `XAIRealtimeConfig` | dataclass | Configuration for xAI realtime |
| `XAITurnDetection` | dataclass | Turn detection config for xAI |
| `XAILLM` | `LLM` | xAI Grok LLM chat completions |

## Key Files

| File | Description |
|------|-------------|
| `xai_realtime.py` | xAI Grok realtime API |
| `llm.py` | xAI Grok LLM implementation |

## Environment Variables
- `XAI_API_KEY` — Required for xAI services

## Installation
```bash
pip install videosdk-plugins-xai
```
