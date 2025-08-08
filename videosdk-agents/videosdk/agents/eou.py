from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Literal

from .llm.chat_context import ChatContext
from .event_emitter import EventEmitter


class EOU(EventEmitter[Literal["error"]]):
    """Base class for End of Utterance Detection implementations"""
    
    def __init__(self, threshold: float = 0.7) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._threshold = threshold

    @property
    def label(self) -> str:
        """Get the EOU provider label"""
        return self._label

    @property
    def threshold(self) -> float:
        """Get the EOU detection threshold"""
        return self._threshold

    @abstractmethod
    def get_eou_probability(self, chat_context: ChatContext) -> float:
        """
        Get the probability score for end of utterance detection.
        
        Args:
            chat_context: Chat context to analyze
            
        Returns:
            float: Probability score (0.0 to 1.0)
        """
        raise NotImplementedError

    def detect_end_of_utterance(self, chat_context: ChatContext, threshold: Optional[float] = None) -> bool:
        """
        Detect if the given chat context represents an end of utterance.
        
        Args:
            chat_context: Chat context to analyze
            threshold: Optional threshold override
            
        Returns:
            bool: True if end of utterance is detected, False otherwise
        """
        if threshold is None:
            threshold = self._threshold
        
        probability = self.get_eou_probability(chat_context)
        return probability >= threshold

    def set_threshold(self, threshold: float) -> None:
        """Update the EOU detection threshold"""
        self._threshold = threshold

