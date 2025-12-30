from dataclasses import dataclass, field
from typing import Optional, List, Any
import os

@dataclass
class LLMEvalConfig:
    model: str
    api_key: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: List[Any] = field(default=None)
    use_stt_output: bool = True
    mock_input: Optional[str] = None

class LLMComponent:
    @staticmethod
    def openai(config: LLMEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("OPENAI_API_KEY") or config.api_key,
            "system_prompt": config.system_prompt,
            "tools": config.tools,
            "use_stt_output": config.use_stt_output,
            "mock_input": config.mock_input
        }
        return ("openai", config)

    @staticmethod
    def anthropic(config: LLMEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("ANTHROPIC_API_KEY") or config.api_key,
            "system_prompt": config.system_prompt,
            "tools": config.tools,
            "use_stt_output": config.use_stt_output,
            "mock_input": config.mock_input
        }
        return ("anthropic", config)
    
    @staticmethod
    def google(config: LLMEvalConfig):
        config = {
            "model": config.model,
            "api_key": os.getenv("GOOGLE_API_KEY_LLM") or config.api_key,
            "system_prompt": config.system_prompt,
            "tools": config.tools,
            "use_stt_output": config.use_stt_output,
            "mock_input": config.mock_input
        }
        return ("google", config)
