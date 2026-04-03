---
name: videosdk-plugins-anam
description: Plugin for Anam virtual avatar integration
---

# videosdk-plugins-anam

## Purpose
Integrates Anam avatar service with the VideoSDK AI Agents framework for visual avatar output.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `AnamAvatar` | Protocol-based | Anam avatar controller |

## Key Files

| File | Description |
|------|-------------|
| `anam.py` | Anam avatar implementation (connect/aclose protocol) |

## Environment Variables
- `ANAM_API_KEY` — Required for Anam services

## Installation
```bash
pip install videosdk-plugins-anam
```
