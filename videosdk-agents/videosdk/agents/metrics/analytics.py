import os
import asyncio
from typing import Dict, Any, Optional
import aiohttp
import logging

logger = logging.getLogger(__name__)


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
        # Use local analytics server for testing
        self.base_url = "http://localhost:8000"
        self._initialized = True

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for analytics tracking"""
        print(f" seession id is >>>> {session_id}")
        self.session_id = session_id

    async def send_interaction_analytics(
        self, interaction_data: Dict[str, Any]
    ) -> None:
        """Send turn analytics to the API endpoint"""
        session_id_from_payload = interaction_data.get("sessionId")
        current_session_id = self.session_id or session_id_from_payload

        if not current_session_id:
            logger.warning(" No session ID - skipping analytics send")
            return

        # For localhost testing, auth token is optional
        auth_token = os.getenv("VIDEOSDK_AUTH_TOKEN", "")

        url = f"{self.base_url}/v2/sessions/{current_session_id}/agent-analytics"

        # Debug print to see payload
        print(f"\n{'='*80}")
        print(f"SENDING ANALYTICS TO: {url}")
        print(f"{'='*80}")
        print(f">>> Analytics payload being sent:")
        import json
        print(json.dumps(interaction_data, indent=2, default=str))
        print(f"{'='*80}\n")

        # Only include auth header if we have a token (production)
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"{auth_token}"

        try:
            logger.debug(f"Sending turn analytics to {url}")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=interaction_data, headers=headers
                ) as response:
                    if response.status == 200:
                        logger.info(f"✅ Analytics sent successfully (session: {current_session_id[:8]}...)")
                    else:
                        response_text = await response.text()
                        logger.error(
                            f"❌ Failed to send analytics: HTTP {response.status}"
                        )
                        logger.error(f"  Response: {response_text}")

        except aiohttp.ClientConnectorError as e:
            logger.error(f"❌ Cannot connect to analytics server at {self.base_url}")
            logger.error(f"Make sure local_analytics_server.py is running!")
        except Exception as e:
            logger.error(f"❌ Error sending analytics: {e}")

    def send_interaction_analytics_safe(self, interaction_data: Dict[str, Any]) -> None:
        """
        Safely send turn analytics without blocking.
        Creates a task if event loop is running, otherwise ignores.
        """
        try:
            print(f">>> analytics which is being sent {interaction_data}.... ")
            asyncio.create_task(self.send_interaction_analytics(interaction_data))
            pass
        except RuntimeError:
            pass
