from typing import Callable, Optional
from .llm.llm import LLM
from .llm.chat_context import ChatRole, ChatContext
import asyncio
import logging

logger = logging.getLogger(__name__)

class VoiceMailDetector:
    """
    Detects if the initial speech is a voicemail/answering machine.
    """
    
    SYSTEM_PROMPT = """You are a voicemail detection classifier for an OUTBOUND calling system. A bot has called a phone number and you need to determine if a human answered or if the call went to voicemail based on the provided text. Answer in one word yes or no."""

    def __init__(self, llm: LLM, callback: Callable, duration: float = 2.0, custom_prompt: Optional[str] = None) -> None:
        """
        Args:
            llm: The LLM instance to use for classification.
            callback: Callback function to run if voicemail is detected.
            duration: Time in seconds to buffer speech before checking (default 2.0s).
        """
        self.llm = llm
        self.duration = duration
        self.CUSTOM_PROMPT = custom_prompt
        self.callback = callback

    async def detect(self, transcript: str) -> bool:
        if not transcript or not transcript.strip():
            return False

        try:
            context = ChatContext()
            if self.CUSTOM_PROMPT:
                context.add_message(ChatRole.SYSTEM, self.CUSTOM_PROMPT)
            else:
                context.add_message(ChatRole.SYSTEM, self.SYSTEM_PROMPT)
            context.add_message(ChatRole.USER, f"Text to classify: {transcript}")

            response_content = ""
            async for chunk in self.llm.chat(context):
                if chunk.content:
                    response_content += chunk.content

            result = response_content.strip().lower()
            logger.info(f"Voice Mail Detection Result: '{result}' for text: '{transcript}'")
            return "yes" in result
            
        except Exception as e:
            logger.error(f"Error during voice mail detection: {e}")
            return False
    