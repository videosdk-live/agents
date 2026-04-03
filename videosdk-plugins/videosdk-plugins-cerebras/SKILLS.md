---
name: videosdk-plugins-cerebras
description: Plugin for Cerebras LLM integration
---

# videosdk-plugins-cerebras

## Purpose
Integrates Cerebras AI models with the VideoSDK AI Agents framework as an LLM provider.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `CerebrasLLM` | `LLM` | Cerebras LLM chat completions |

## Key Files

| File | Description |
|------|-------------|
| `llm.py` | Cerebras LLM implementation |

## Environment Variables
- `CEREBRAS_API_KEY` — Required for Cerebras services

## Installation
```bash
pip install videosdk-plugins-cerebras
```
