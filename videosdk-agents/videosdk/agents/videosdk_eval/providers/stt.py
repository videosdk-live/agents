from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class STTEvalConfig:
    file_path: Optional[str] = None
    model:str = None
    api_key: Optional[str] = None
    sample_rate: int = 16000

class STT:
    @staticmethod
    def deepgram(config: STTEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("DEEPGRAM_API_KEY") or config.api_key,
            "file_path": config.file_path
        }
        return ("deepgram", config)

    @staticmethod
    def openai(config: STTEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("OPENAI_API_KEY") or config.api_key,
            "file_path": config.file_path
        }
        return ("openai", config)
    
    @staticmethod
    def google(config: STTEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("GOOGLE_API_KEY") or config.api_key,
            "file_path": config.file_path
        }
        return ("google", config)
