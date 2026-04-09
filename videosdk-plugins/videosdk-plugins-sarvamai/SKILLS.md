---
name: videosdk-plugins-sarvamai
description: Plugin for Sarvam AI STT, LLM, and TTS services (Indian languages)
---

# videosdk-plugins-sarvamai

## Purpose
Integrates Sarvam AI services with the VideoSDK AI Agents framework. Sarvam AI specializes in Indian languages.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `SarvamAISTT` | `STT` | Sarvam AI speech-to-text |
| `SarvamAILLM` | `LLM` | Sarvam AI language model |
| `SarvamAITTS` | `TTS` | Sarvam AI text-to-speech |

## Key Files

| File | Description |
|------|-------------|
| `stt.py` | Sarvam AI STT (supports Indian languages) |
| `llm.py` | Sarvam AI LLM |
| `tts.py` | Sarvam AI TTS (supports Indian languages) |

## Environment Variables
- `SARVAM_API_KEY` — Required for Sarvam AI services

## Important Notes
- One of the few providers offering full STT + LLM + TTS stack
- Specialized for Indian language support (Hindi, Tamil, etc.)
- Also available via VideoSDK Inference gateway

## Installation
```bash
pip install videosdk-plugins-sarvamai
```
