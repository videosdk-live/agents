---
name: videosdk-plugins-silero
description: Plugin for Silero Voice Activity Detection (VAD)
---

# videosdk-plugins-silero

## Purpose
Integrates Silero VAD with the VideoSDK AI Agents framework for voice activity detection. This is the primary VAD provider used across the framework.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `SileroVAD` | `VAD` | Silero voice activity detection using ONNX runtime |

## Key Files

| File | Description |
|------|-------------|
| `vad.py` | SileroVAD implementation using ONNX model inference |
| `onnx_runtime.py` | ONNX runtime wrapper for model execution |
| `model/` | Pre-trained Silero VAD ONNX model files |

## Architecture
- Uses ONNX Runtime for efficient model inference
- Processes audio frames and emits `start_of_speech` / `end_of_speech` events
- Configurable threshold, min speech duration, and min silence duration

## Important Notes
- SileroVAD is the **only VAD provider** in the framework — it's used in virtually all cascade examples
- Sensitivity mapping is handled by the factory initializer when building pipelines
- The ONNX model is bundled with the package (no download needed)
- Default sample rate is 16kHz

## Installation
```bash
pip install videosdk-plugins-silero
```
