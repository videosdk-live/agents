from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class LLMJudgeConfig:
    model: str
    prompt: Optional[str] = None
    checks: List[str] = field(default_factory=list)


class LLMJudge:
    @staticmethod
    def openai(model: str, prompt: str, checks: List[str]):
        return ("openai", LLMJudgeConfig(model=model, prompt=prompt, checks=checks))
    
    @staticmethod
    def anthropic(model: str, prompt: str, checks: List[str]):
        return ("anthropic", LLMJudgeConfig(model=model, prompt=prompt, checks=checks))
    
    @staticmethod
    def google(model: str, prompt: str, checks: List[str]):
        return ("google", LLMJudgeConfig(model=model, prompt=prompt, checks=checks))
