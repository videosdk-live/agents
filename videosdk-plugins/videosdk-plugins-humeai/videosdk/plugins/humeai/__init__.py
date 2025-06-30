"""
Hume AI TTS plugin for videosdk-agents

This plugin provides text-to-speech capabilities using Hume AI's API.
"""

from .tts import HumeAITTS, Utterance, Context

__all__ = ["HumeAITTS", "Utterance", "Context"] 