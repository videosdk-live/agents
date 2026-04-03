---
name: videosdk-plugins-aws
description: Plugin for AWS Nova Sonic (Realtime) and AWS Polly (TTS) services
---

# videosdk-plugins-aws

## Purpose
Integrates AWS AI services with the VideoSDK AI Agents framework, providing Nova Sonic realtime speech-to-speech and Polly TTS.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `NovaSonicRealtime` | `RealtimeBaseModel` | AWS Nova Sonic realtime speech-to-speech |
| `NovaSonicConfig` | dataclass | Configuration for Nova Sonic |
| `AWSPollyTTS` | `TTS` | AWS Polly text-to-speech |

## Key Files

| File | Description |
|------|-------------|
| `aws_nova_sonic_api.py` | Nova Sonic realtime API implementation |
| `tts.py` | AWS Polly TTS synthesis |

## Environment Variables
- `AWS_ACCESS_KEY_ID` — AWS access key
- `AWS_SECRET_ACCESS_KEY` — AWS secret key
- `AWS_REGION` — AWS region (default: us-east-1)

## Important Notes
- Nova Sonic uses a custom schema format — see `build_nova_sonic_schema()` in the core utils
- Polly TTS supports multiple voices and languages

## Installation
```bash
pip install videosdk-plugins-aws
```
