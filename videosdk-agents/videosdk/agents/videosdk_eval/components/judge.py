from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class LLMAsJudgeConfig:
    model: str
    prompt: Optional[str] = None
    checks: List[str] = field(default_factory=list)


class LLMAsJudge:
    @staticmethod
    def openai(model: str, prompt: str, checks: List[str]):
        return ("openai", LLMAsJudgeConfig(model=model, prompt=prompt, checks=checks))
    
    @staticmethod
    def anthropic(model: str, prompt: str, checks: List[str]):
        return ("anthropic", LLMAsJudgeConfig(model=model, prompt=prompt, checks=checks))
    
    @staticmethod
    def google(model: str, prompt: str, checks: List[str]):
        return ("google", LLMAsJudgeConfig(model=model, prompt=prompt, checks=checks))
