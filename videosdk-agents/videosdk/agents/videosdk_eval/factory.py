from typing import Any
from .providers.stt import STTEvalConfig
from .providers.llm import LLMEvalConfig
from .providers.tts import TTSEvalConfig
import os

# Import Plugins
try:
    from videosdk.plugins.deepgram.stt import DeepgramSTT
except ImportError:
    DeepgramSTT = None

try:
    from videosdk.plugins.openai.llm import OpenAILLM
    from videosdk.plugins.openai.tts import OpenAITTS
    from videosdk.plugins.openai.stt import OpenAISTT
except ImportError:
    OpenAILLM = None
    OpenAITTS = None
    OpenAISTT = None

try:
    from videosdk.plugins.google.tts import GoogleTTS, GoogleVoiceConfig
    from videosdk.plugins.google.stt import GoogleSTT
    from videosdk.plugins.google.llm import GoogleLLM
except ImportError:
    GoogleTTS = None
    GoogleSTT = None
    GoogleLLM = None

# Factory Functions
def create_stt(provider_name: str, config: Any):
    if provider_name == "deepgram":
        if not DeepgramSTT: raise ImportError("Deepgram plugin not installed")
        api_key = config.get("api_key") if isinstance(config, dict) else config.api_key
        model = config.get("model") if isinstance(config, dict) else config.model
        
        return DeepgramSTT(
            api_key=api_key or os.getenv("DEEPGRAM_API_KEY"),
            model=model or "nova-2"
        )
    elif provider_name == "openai":
         if not OpenAISTT: raise ImportError("OpenAI plugin not installed")
         api_key = config.get("api_key") if isinstance(config, dict) else config.api_key
         model = config.get("model") if isinstance(config, dict) else config.model

         return OpenAISTT(
             api_key=api_key or os.getenv("OPENAI_API_KEY"),
             model=model or "whisper-1" 
         )
    else:
        raise ValueError(f"Unknown STT provider: {provider_name}")

def create_llm(provider_name: str, config: Any):
    if provider_name == "openai":
        if not OpenAILLM: raise ImportError("OpenAI plugin not installed")
        api_key = config.get("api_key") if isinstance(config, dict) else config.api_key
        model = config.get("model") if isinstance(config, dict) else config.model
        
        return OpenAILLM(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model or "gpt-4o-mini",
        )
    elif provider_name == "google":
        if not GoogleLLM: raise ImportError("Google plugin not installed")
        api_key = config.get("api_key") if isinstance(config, dict) else config.api_key
        model = config.get("model") if isinstance(config, dict) else config.model
        
        return GoogleLLM(
             api_key=api_key or os.getenv("GOOGLE_API_KEY_LLM"),
             model=model
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

def create_tts(provider_name: str, config: Any):
    if provider_name == "openai":
        if not OpenAITTS: raise ImportError("OpenAI plugin not installed")
        api_key = config.get("api_key") if isinstance(config, dict) else config.api_key
        model = config.get("model") if isinstance(config, dict) else config.model
        
        return OpenAITTS(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model or "gpt-4o-mini-tts"
        )
    elif provider_name == "google":
        if not GoogleTTS: raise ImportError("Google plugin not installed")
        api_key = config.get("api_key") if isinstance(config, dict) else config.api_key
        model = config.get("model") if isinstance(config, dict) else config.model
        
        voice_config = None
        if model:
            voice_config = GoogleVoiceConfig(name=model)

        return GoogleTTS(
            api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            voice_config=voice_config
        )
    else:
        raise ValueError(f"Unknown TTS provider: {provider_name}")
