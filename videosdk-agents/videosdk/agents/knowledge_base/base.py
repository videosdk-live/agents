from __future__ import annotations

from abc import ABC
from typing import List, Optional
import logging
import aiohttp
import os

from .config import KnowledgeBaseConfig

logger = logging.getLogger(__name__)


class KnowledgeBase(ABC):
    """
    Base class for handling knowledge-base retrieval operations.

    Provides hooks developers can override:
        - allow_retrieval: Decide if the knowledge base should be used.
        - pre_process_query: Preprocess the query before searching.
        - format_context: Format retrieved documents for the final prompt.
    """
    
    def __init__(self, config: KnowledgeBaseConfig):
        """
        Initialize the knowledge base handler.

        Args:
            config (KnowledgeBaseConfig): Configuration for retrieval settings.
        """
        self.config = config

    def allow_retrieval(self, transcript: str) -> bool:
        """
        Decide whether the knowledge base should be used for this message.

        Args:
            transcript (str): User message.

        Returns:
            bool: True to perform retrieval, False otherwise.
        """
        return True
    
    def pre_process_query(self, transcript: str) -> str:
        """
        Preprocess the user message before searching the knowledge base.

        Args:
            transcript (str): Original user message.

        Returns:
            str: Processed query string.
        """
        return transcript
    
    def format_context(self, documents: List[str]) -> str:
        """
        Format retrieved documents into a context string.

        Args:
            documents (List[str]): Retrieved document texts.

        Returns:
            str: Formatted context for the model.
        """
        if not documents:
            return ""
        
        doc_str = "\n".join([f"- {doc}" for doc in documents])
        return f"Use the following context to answer the user:\n{doc_str}\n"
    
    async def retrieve_documents(self, query: str) -> List[str]:
        """
        Fetch documents from the configured knowledge base.

        Args:
            query (str): Search query.

        Returns:
            List[str]: Retrieved document texts.
        """
        api_base_url =  "https://api.videosdk.live/ai/v1"
        auth_token = os.getenv("VIDEOSDK_AUTH_TOKEN")

        if not auth_token:
            logger.warning("VIDEOSDK_AUTH_TOKEN not set, skipping KB retrieval")
            return []
        
        try:
            url = f"{api_base_url}/knowledge-bases/{self.config.id}/search"
            headers = {
                "Authorization": auth_token,
                "Content-Type": "application/json"
            }
            payload = {
                "queryText": query,
                "topK": self.config.top_k
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])
                        
                        # Extract text from each result's payload
                        documents = []
                        for result in results:
                            if isinstance(result, dict):
                                payload = result.get("payload", {})
                                if isinstance(payload, dict):
                                    text = payload.get("text", "")
                                    if text and text.strip():  # Only add non-empty text
                                        documents.append(text.strip())
                        logger.debug(f"Retrieved {len(documents)} documents from knowledge base")
                        
                        return documents
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"KB API error {response.status}: {error_text}"
                        )
                        return []
        except Exception as e:
            logger.error(f"Error retrieving KB documents: {e}")
            return []
    
    async def process_query(self, transcript: str) -> Optional[str]:
        """
        Run the full knowledge-base retrieval flow for a user message.

        Args:
            transcript (str): User message.

        Returns:
            Optional[str]: Formatted context or None if retrieval is skipped.
        """

        # Check if KB should be triggered
        if not self.allow_retrieval(transcript):
            return None
        
        # Transform the query
        query = self.pre_process_query(transcript)
        
        # Retrieve documents
        documents = await self.retrieve_documents(query)
        
        if not documents:
            return None
        
        # Format the prompt
        formatted_context = self.format_context(documents)
        
        return formatted_context 