---
name: videosdk-plugins-simli
description: Plugin for Simli virtual avatar integration
---

# videosdk-plugins-simli

## Purpose
Integrates Simli avatar service with the VideoSDK AI Agents framework for visual avatar output.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `SimliAvatar` | Protocol-based | Simli avatar controller |
| `SimliConfig` | dataclass | Simli configuration |

## Key Files

| File | Description |
|------|-------------|
| `simli.py` | Simli avatar implementation (connect/aclose protocol) |

## Environment Variables
- `SIMLI_API_KEY` — Required for Simli services

## Important Notes
- Avatars follow the `connect()` / `aclose()` protocol (no formal base class)
- Used as `avatar=SimliAvatar(config=SimliConfig(...))` in the Pipeline

## Installation
```bash
pip install videosdk-plugins-simli
```
