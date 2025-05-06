from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

class RealtimeBaseModel(ABC):
    """
    Base class for realtime models that can be used in the realtime pipeline.
    """
    
    def __init__(self) -> None:
        """Initialize the realtime model"""
        pass