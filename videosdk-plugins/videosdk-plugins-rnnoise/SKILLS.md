---
name: videosdk-plugins-rnnoise
description: Plugin for RNNoise audio denoising
---

# videosdk-plugins-rnnoise

## Purpose
Provides RNNoise-based audio denoising for the VideoSDK AI Agents framework. Cleans incoming audio before processing by STT/VAD.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `RNNoise` | `Denoise` | RNNoise denoiser implementation |

## Key Files

| File | Description |
|------|-------------|
| `denoise.py` | RNNoise denoise wrapper |
| `rnnoise.py` | Native RNNoise library bindings |
| `build_rnnoise.py` | Build script for native library |

## Important Notes
- Requires a native `.so`/`.dylib` library — platform-dependent
- Doc generation mocks this module due to the native dependency
- Used as `denoise=RNNoise()` in the `Pipeline` constructor

## Installation
```bash
pip install videosdk-plugins-rnnoise
```
