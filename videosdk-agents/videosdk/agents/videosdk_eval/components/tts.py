from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class TTSEvalConfig:
    model: str
    api_key: Optional[str] = None
    use_llm_output: bool = True
    mock_input: Optional[str] = None

class TTSComponent:
    @staticmethod
    def deepgram(config: TTSEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("DEEPGRAM_API_KEY") or config.api_key,
            "use_llm_output": config.use_llm_output,
            "mock_input": config.mock_input
        }   
        return ("deepgram", config)

    @staticmethod
    def openai(config: TTSEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("OPENAI_API_KEY") or config.api_key,
            "use_llm_output": config.use_llm_output,
            "mock_input": config.mock_input
        }
        return ("openai", config)
    
    @staticmethod
    def google(config: TTSEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("GOOGLE_API_KEY") or config.api_key,
            "use_llm_output": config.use_llm_output,
            "mock_input": config.mock_input
        }
        return ("google", config)
    
    @staticmethod
    def cartesia(config: TTSEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("CARTESIA_API_KEY") or config.api_key,
            "use_llm_output": config.use_llm_output,
            "mock_input": config.mock_input
        }
        return ("cartesia", config)
    
    @staticmethod
    def elevenlabs(config: TTSEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("ELEVENLABS_API_KEY") or config.api_key,
            "use_llm_output": config.use_llm_output,
            "mock_input": config.mock_input
        }
        return ("elevenlabs", config)
