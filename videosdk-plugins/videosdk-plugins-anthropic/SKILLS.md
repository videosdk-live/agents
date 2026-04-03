---
name: videosdk-plugins-anthropic
description: Plugin for Anthropic Claude LLM integration
---

# videosdk-plugins-anthropic

## Purpose
Integrates Anthropic's Claude models with the VideoSDK AI Agents framework as an LLM provider.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `AnthropicLLM` | `LLM` | Anthropic Claude chat completions |

## Key Files

| File | Description |
|------|-------------|
| `llm.py` | Claude LLM implementation with streaming + tool calls |

## Environment Variables
- `ANTHROPIC_API_KEY` — Required for Anthropic services

## Installation
```bash
pip install videosdk-plugins-anthropic
```
