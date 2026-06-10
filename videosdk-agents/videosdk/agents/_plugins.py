from __future__ import annotations

import importlib
from typing import Any

_REGISTRY: dict[str, str] = {
    "AWSPollyTTS": "videosdk.plugins.aws",
    "NovaSonicConfig": "videosdk.plugins.aws",
    "NovaSonicRealtime": "videosdk.plugins.aws",
    "AnamAvatar": "videosdk.plugins.anam",
    "AnthropicLLM": "videosdk.plugins.anthropic",
    "AssemblyAISTT": "videosdk.plugins.assemblyai",
    "AzureSTT": "videosdk.plugins.azure",
    "AzureTTS": "videosdk.plugins.azure",
    "AzureVoiceLive": "videosdk.plugins.azure",
    "AzureVoiceLiveConfig": "videosdk.plugins.azure",
    "SpeakingStyle": "videosdk.plugins.azure",
    "VoiceTuning": "videosdk.plugins.azure",
    "CambAITTS": "videosdk.plugins.cambai",
    "InferenceOptions": "videosdk.plugins.cambai",
    "OutputConfiguration": "videosdk.plugins.cambai",
    "CartesiaSTT": "videosdk.plugins.cartesia",
    "CartesiaTTS": "videosdk.plugins.cartesia",
    "GenerationConfig": "videosdk.plugins.cartesia",
    "CerebrasLLM": "videosdk.plugins.cerebras",
    "CometAPILLM": "videosdk.plugins.cometapi",
    "CometAPISTT": "videosdk.plugins.cometapi",
    "CometAPITTS": "videosdk.plugins.cometapi",
    "DeepgramSTT": "videosdk.plugins.deepgram",
    "DeepgramSTTV2": "videosdk.plugins.deepgram",
    "DeepgramTTS": "videosdk.plugins.deepgram",
    "ElevenLabsSTT": "videosdk.plugins.elevenlabs",
    "ElevenLabsTTS": "videosdk.plugins.elevenlabs",
    "GeminiLiveConfig": "videosdk.plugins.google",
    "GeminiRealtime": "videosdk.plugins.google",
    "GoogleLLM": "videosdk.plugins.google",
    "GoogleSTT": "videosdk.plugins.google",
    "GoogleTTS": "videosdk.plugins.google",
    "GoogleVoiceConfig": "videosdk.plugins.google",
    "VertexAIConfig": "videosdk.plugins.google",
    "VoiceActivityConfig": "videosdk.plugins.google",
    "GladiaSTT": "videosdk.plugins.gladia",
    "GroqTTS": "videosdk.plugins.groq",
    "HumeAITTS": "videosdk.plugins.humeai",
    "InworldAITTS": "videosdk.plugins.inworldai",
    "LMNTTTS": "videosdk.plugins.lmnt",
    "LangChainLLM": "videosdk.plugins.langchain",
    "LangGraphLLM": "videosdk.plugins.langchain",
    "MurfAITTS": "videosdk.plugins.murfai",
    "MurfAIVoiceSettings": "videosdk.plugins.murfai",
    "NavanaSTT": "videosdk.plugins.navana",
    "NeuphonicTTS": "videosdk.plugins.neuphonic",
    "NvidiaSTT": "videosdk.plugins.nvidia",
    "NvidiaTTS": "videosdk.plugins.nvidia",
    "OpenAILLM": "videosdk.plugins.openai",
    "OpenAIRealtime": "videosdk.plugins.openai",
    "OpenAIRealtimeConfig": "videosdk.plugins.openai",
    "OpenAISTT": "videosdk.plugins.openai",
    "OpenAITTS": "videosdk.plugins.openai",
    "PaplaTTS": "videosdk.plugins.papla",
    "RNNoise": "videosdk.plugins.rnnoise",
    "ResembleTTS": "videosdk.plugins.resemble",
    "RimeTTS": "videosdk.plugins.rime",
    "SarvamAILLM": "videosdk.plugins.sarvamai",
    "SarvamAISTT": "videosdk.plugins.sarvamai",
    "SarvamAITTS": "videosdk.plugins.sarvamai",
    "SileroVAD": "videosdk.plugins.silero",
    "SimliAvatar": "videosdk.plugins.simli",
    "SimliConfig": "videosdk.plugins.simli",
    "SmallestAITTS": "videosdk.plugins.smallestai",
    "SpeechifyTTS": "videosdk.plugins.speechify",
    "NamoTurnDetectorV1": "videosdk.plugins.turn_detector",
    "TurnDetector": "videosdk.plugins.turn_detector",
    "VideoSDKTurnDetector": "videosdk.plugins.turn_detector",
    "pre_download_model": "videosdk.plugins.turn_detector",
    "pre_download_namo_turn_v1_model": "videosdk.plugins.turn_detector",
    "pre_download_videosdk_model": "videosdk.plugins.turn_detector",
    "UltravoxLiveConfig": "videosdk.plugins.ultravox",
    "UltravoxRealtime": "videosdk.plugins.ultravox",
    "XAILLM": "videosdk.plugins.xai",
    "XAIRealtime": "videosdk.plugins.xai",
    "XAIRealtimeConfig": "videosdk.plugins.xai",
    "XAISTT": "videosdk.plugins.xai",
    "XAITTS": "videosdk.plugins.xai",
    "XAITurnDetection": "videosdk.plugins.xai",
}

__all__ = sorted(_REGISTRY)


def __getattr__(name: str) -> Any:
    """Lazily import and return the plugin symbol ``name`` (PEP 562)."""
    module_path = _REGISTRY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return __all__
