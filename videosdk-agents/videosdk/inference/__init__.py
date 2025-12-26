"""
VideoSDK Inference Gateway Plugins

Lightweight STT, TTS, LLM, and Realtime clients that connect to VideoSDK's Inference Gateway.
All heavy lifting (API connections, resampling, etc.) is handled server-side.

Usage:
    from videosdk.inference import STT, TTS, LLM, Realtime
    
    # Quick start with factory methods
    stt = STT.google()
    tts = TTS.sarvam(speaker="anushka")
    llm = LLM.google(model="gemini-2.0-flash")
    realtime = Realtime.gemini(model="gemini-2.0-flash-exp")
    
    # Use with CascadingPipeline
    pipeline = CascadingPipeline(stt=stt, llm=llm, tts=tts)
    
    # Use with RealTimePipeline
    pipeline = RealTimePipeline(model=realtime)
"""

from .stt import STT
from .tts import TTS
from .llm import LLM
from .realtime import Realtime

__all__ = [
    "STT",
    "TTS",
    "LLM",
    "Realtime",
]

