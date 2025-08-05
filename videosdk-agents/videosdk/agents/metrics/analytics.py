import os
import asyncio
from typing import Dict, Any, Optional
import aiohttp

class AnalyticsClient:
    """Client for sending analytics data to external endpoints"""

    _instance: Optional["AnalyticsClient"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "AnalyticsClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, session_id: Optional[str] = None):
        if self._initialized:
            return
        
        self.session_id = session_id
        self.base_url = os.getenv("ANALYTICS_API_SERVER_URL")
        self._initialized = True
        
    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for analytics tracking"""
        self.session_id = session_id
    
    async def send_interaction_analytics(self, interaction_data: Dict[str, Any]) -> None:
        """Send interaction analytics to the API endpoint"""
        session_id_from_payload = interaction_data.get("sessionId")
        current_session_id = self.session_id or session_id_from_payload
        
        if not current_session_id:
            return
        
        auth_token = os.getenv("VIDEOSDK_AUTH_TOKEN")
        if not auth_token:
            return
        
        url = f"{self.base_url}/v2/sessions/{current_session_id}/agent-analytics"
        
        headers = {
            "Authorization": f"{auth_token}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=interaction_data, headers=headers) as response:
                     if response.status == 200:
                        print(f"Analytics sent successfully")
                     else:
                        response_text = await response.text()
                        print(f"  Failed to send analytics: HTTP {response.status}")
                        print(f"  Response content: {response_text}")

        except Exception as e:
            print(f"  Error sending analytics to API: {e}")
    
    def send_interaction_analytics_safe(self, interaction_data: Dict[str, Any]) -> None:
        """
        Safely send interaction analytics without blocking.
        Creates a task if event loop is running, otherwise ignores.
        """
        try:
            asyncio.create_task(self.send_interaction_analytics(interaction_data))
            pass
        except RuntimeError:
            pass 