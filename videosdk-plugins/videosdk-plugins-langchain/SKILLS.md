---
name: videosdk-plugins-langchain
description: Plugin for LangChain and LangGraph LLM orchestration integration
---

# videosdk-plugins-langchain

## Purpose
Enables using LangChain `BaseChatModel` or LangGraph `StateGraph` as the LLM in VideoSDK AI Agent pipelines.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `LangChainLLM` | `LLM` | Wraps any LangChain `BaseChatModel` as a VideoSDK LLM |
| `LangGraphLLM` | `LLM` | Wraps a LangGraph `StateGraph` as a VideoSDK LLM |

## Key Files

| File | Description |
|------|-------------|
| `llm.py` | LangChain BaseChatModel adapter |
| `graph.py` | LangGraph StateGraph adapter |

## Important Notes
- `LangChainLLM` translates VideoSDK `ChatContext` to LangChain message format
- `LangGraphLLM` executes a compiled LangGraph state graph on each LLM turn
- Tool calls from LangChain/LangGraph are bridged to VideoSDK's `FunctionTool` system

## Installation
```bash
pip install videosdk-plugins-langchain
```
