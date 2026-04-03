---
name: videosdk-agents-repo
description: Root of the VideoSDK AI Agents monorepo — Python framework for real-time voice and multimodal AI agents
---

# VideoSDK AI Agents — Repository Root

## Purpose

This is the monorepo for the **VideoSDK AI Agents** framework — an open-source Python SDK for building production-ready, real-time voice and multimodal AI agents that join VideoSDK rooms as participants.

## Architecture

```bash
agents/                         ← YOU ARE HERE
├── videosdk-agents/            ← Core SDK package (videosdk-agents on PyPI)
│   └── videosdk/agents/        ← Agent, Pipeline, Session, base classes (STT/LLM/TTS/VAD/EOU)
├── videosdk-plugins/           ← All provider plugins (35 packages)
│   └── videosdk-plugins-*/     ← Each is a separate PyPI package
├── examples/                   ← Code examples for all pipeline modes
├── use_case_examples/          ← Domain-specific agent examples
├── scripts/                    ← Doc generation utilities
├── BUILD_YOUR_OWN_PLUGIN.md   ← Plugin authoring guide
├── pyproject.toml              ← Workspace-level config (UV)
└── uv.lock                    ← UV lockfile for monorepo
```

## Pipeline Modes

The unified `Pipeline` class supports three modes (auto-detected from components):

1. **Cascade** — `VAD → STT → Turn Detector → LLM → TTS` (mix-and-match providers)
2. **Realtime** — Single speech-to-speech model (OpenAI Realtime, Gemini Live, Nova Sonic, etc.)
3. **Hybrid** — Mix cascade + realtime (e.g., external STT + realtime LLM, or realtime + external TTS)

## Key Workflows

### Development Setup

#### Note

- Here before running the agent make sure to use appropriate python environment.
- For all example files there is a .env file which contains the environment variables required for the agent to run. Make sure to update the environment variables with the appropriate values.
- For all examples the `room_id=<room_id>` and `auth_token` are required. Make sure to update the `room_id` and `auth_token` in the .env file with the appropriate values.
- If you don't have `room_id` you can leave it empty, it will be automatically generated.
- If you don't have `auth_token` you can generate it from the [VideoSDK Dashboard](https://app.videosdk.live/).

```bash
# UV (recommended)
uv sync && uv run python examples/cascade_basic.py

# pip
bash setup.sh && source venv/bin/activate && python examples/cascade_basic.py
```

## Key Conventions

- All plugins use namespace packaging under `videosdk.plugins.*`
- Python ≥ 3.11 required, 3.12+ recommended
- All plugins follow the same directory structure: `videosdk-plugins-{name}/videosdk/plugins/{name}/`
- Base classes live in `videosdk-agents/videosdk/agents/` (STT, LLM, TTS, VAD, EOU, RealtimeBaseModel)
- The `@function_tool` decorator is used for both internal and external agent tools
- Pipeline hooks use `@pipeline.on("stt"|"llm"|"tts"|...)` decorator pattern
