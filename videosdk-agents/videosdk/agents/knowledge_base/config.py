from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KnowledgeBaseConfig:
    """
    Configuration for managed RAG (Retrieval-Augmented Generation).
    
    Attributes:
        id: The ID of the knowledge base provided by your app dashboard
        top_k: Optional number of documents to retrieve (default: 3)
    """
    id: str
    top_k: int = 3
    
    def __post_init__(self):
        if not self.id:
            raise ValueError("id cannot be empty")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")