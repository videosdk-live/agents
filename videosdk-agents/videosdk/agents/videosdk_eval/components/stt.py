from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class STTEvalConfig:
    file_path: Optional[str] = None
    model:str = None
    api_key: Optional[str] = None
    chunk_size: int = 96000
    sample_rate: int = 48000

class STTComponent:
    @staticmethod
    def deepgram(config: STTEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("DEEPGRAM_API_KEY") or config.api_key,
            "file_path": config.file_path,
            "chunk_size": config.chunk_size,
            "sample_rate": config.sample_rate
        }
        return ("deepgram", config)
    def deepgramv2(config: STTEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("DEEPGRAM_API_KEY") or config.api_key,
            "file_path": config.file_path,
            "chunk_size": config.chunk_size,
            "sample_rate": config.sample_rate
        }
        return ("deepgramv2", config)

    @staticmethod
    def openai(config: STTEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("OPENAI_API_KEY") or config.api_key,
            "file_path": config.file_path,
            "chunk_size": config.chunk_size,
            "sample_rate": config.sample_rate
        }
        return ("openai", config)
    
    @staticmethod
    def google(config: STTEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("GOOGLE_API_KEY") or config.api_key,
            "file_path": config.file_path,
            "chunk_size": config.chunk_size,
            "sample_rate": config.sample_rate
        }
        return ("google", config)
