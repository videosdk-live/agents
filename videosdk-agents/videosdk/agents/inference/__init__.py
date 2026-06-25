"""
VideoSDK Inference Gateway Plugins

Lightweight STT, TTS, and Realtime clients that connect to VideoSDK's Inference Gateway.
All heavy lifting (API connections, resampling, etc.) is handled server-side.

Usage:
    from videosdk.inference import STT, TTS, LLM, Realtime

    # Quick start with factory methods
    stt = STT.google()
    tts = TTS.sarvam(speaker="anushka")
    realtime = Realtime.gemini(model="gemini-2.0-flash-exp")
    denoise = Denoise.sanas()
    
    # Use with CascadingPipeline
    pipeline = CascadingPipeline(stt=stt, llm=llm, tts=tts,denoise=denoise)
    
    # Use with RealTimePipeline
    pipeline = RealTimePipeline(model=realtime)
"""

from .stt import STT
from .tts import TTS
from .realtime import Realtime
from .llm import LLM
from .denoise import Denoise
from .turn import Turn, TurnV2

# STT
GoogleSTT = STT.google
SarvamAISTT = STT.sarvam
DeepgramSTT = STT.deepgram
AssemblyAISTT = STT.assemblyai
CartesiaSTT = STT.cartesia

# LLM
GoogleLLM = LLM.google
SarvamAILLM = LLM.sarvam
BedrockLLM = LLM.bedrock

# TTS
GoogleTTS = TTS.google
SarvamAITTS = TTS.sarvam
CartesiaTTS = TTS.cartesia
DeepgramTTS = TTS.deepgram

# Realtime
GeminiRealtime = Realtime.gemini

# Turn detection
TurnDetector = Turn.turnsense
VideoSDKTurnDetector = Turn.videosdk
NamoTurnDetectorV1 = Turn.namo
# TurnV2 (4-state: Complete / Incomplete / Backchannel / Wait) is exported
# directly; choose a size via TurnV2.echo_small() or TurnV2.echo_large().

# Denoise
AICousticsDenoise = Denoise.aicoustics
SanasDenoise = Denoise.sanas


__all__ = [
    # Prefixed component namespaces
    "STT",
    "TTS",
    "Realtime",
    "LLM",
    "Denoise",
    "Turn",
    "GoogleSTT",
    "SarvamAISTT",
    "DeepgramSTT",
    "AssemblyAISTT",
    "GoogleLLM",
    "SarvamAILLM",
    "BedrockLLM",
    "GoogleTTS",
    "SarvamAITTS",
    "CartesiaTTS",
    "DeepgramTTS",
    "GeminiRealtime",
    "NamoTurnDetectorV1",
    "VideoSDKTurnDetector",
    "TurnDetector",
    "TurnV2",
    "AICousticsDenoise",
    "SanasDenoise",
    "SileroVAD",
]

_LOCAL_REEXPORTS = {
    "SileroVAD": "videosdk.plugins.silero",
}

def __getattr__(name):
    module_path = _LOCAL_REEXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
