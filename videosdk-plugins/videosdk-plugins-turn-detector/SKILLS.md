---
name: videosdk-plugins-turn-detector
description: Plugin for end-of-utterance (turn) detection using ML models
---

# videosdk-plugins-turn-detector

## Purpose
Provides ML-based turn detection (end-of-utterance) for the VideoSDK AI Agents framework. Determines when a user has finished speaking and the agent should respond.

## Exported Classes

| Class | Base Class | Description |
|-------|-----------|-------------|
| `TurnDetector` | `EOU` | Original turn detector model |
| `VideoSDKTurnDetector` | `EOU` | VideoSDK custom turn detector v2 |
| `NamoTurnDetectorV1` | `EOU` | Namo Turn Detector v1 (latest) |
| `pre_download_model` | function | Pre-download TurnDetector model |
| `pre_download_videosdk_model` | function | Pre-download VideoSDK model |
| `pre_download_namo_turn_v1_model` | function | Pre-download Namo model |

## Key Files

| File | Description |
|------|-------------|
| `turn_detector.py` | Original TurnDetector implementation |
| `turn_detector_v2.py` | VideoSDK custom turn detector |
| `turn_detector_v3.py` | Namo Turn Detector v1 |
| `model.py` | Model loading and inference utilities |
| `download_model.py` | Model download helpers |

## Architecture
- Implements the `EOU` base class from the core framework
- `get_eou_probability(chat_context)` returns 0.0–1.0 probability
- Uses `threshold` (default 0.7) to determine if utterance is complete
- Models are downloaded on first use (use `pre_download_*` for eager loading)

## Important Notes
- The default `TurnDetector` is the most commonly used in cascade examples
- `NamoTurnDetectorV1` is the latest and most accurate version
- Models are cached locally after first download

## Installation
```bash
pip install videosdk-plugins-turn-detector
```
