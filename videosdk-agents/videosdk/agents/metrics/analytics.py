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
        self.base_url = "https://api.videosdk.live"
        self._initialized = True
        self.turn_count = 0
        self.metrics_options = None

    def configure(self, metrics_options: Any) -> None:
        """Configure analytics client with metrics options"""
        self.metrics_options = metrics_options
        self.turn_count = 0

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for analytics tracking"""
        self.session_id = session_id

    async def send_interaction_analytics(
        self, interaction_data: Dict[str, Any]
    ) -> None:
        """Route analytics to the appropriate endpoint(s)"""
        self.turn_count += 1
        
        metrics_enabled = getattr(self.metrics_options, "enabled", True) if self.metrics_options else True
        custom_url = getattr(self.metrics_options, "export_url", None) if self.metrics_options else None
        custom_headers = getattr(self.metrics_options, "export_headers", None) if self.metrics_options else None

        tasks = []

        if self.turn_count == 1:
            tasks.append(self._send_to_default_endpoint(interaction_data))

        if metrics_enabled:
            if custom_url:
                tasks.append(self._send_to_custom_endpoint(custom_url, custom_headers, interaction_data))
            elif self.turn_count > 1:
                tasks.append(self._send_to_default_endpoint(interaction_data))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_to_default_endpoint(self, interaction_data: Dict[str, Any]) -> None:
        """Send turn analytics to the default VideoSDK API endpoint"""
        session_id_from_payload = interaction_data.get("sessionId")
        current_session_id = self.session_id or session_id_from_payload
        data =  {"data": [interaction_data]}
        if not current_session_id:
            logger.error("Failed sending session data : No session ID")
            return

        auth_token = os.getenv("VIDEOSDK_AUTH_TOKEN")
        if not auth_token:
            logger.error("Failed sending session data : No auth token")
            return

        url = f"{self.base_url}/v2/sessions/{current_session_id}/agent-analytics"
        headers = {"Authorization": f"{auth_token}", "Content-Type": "application/json"}

        await self._execute_request(url, headers, data, "default endpoint")

    async def _send_to_custom_endpoint(
        self, custom_url: str, custom_headers: Optional[Dict[str, str]], interaction_data: Dict[str, Any]
    ) -> None:
        """Send turn analytics to a custom endpoint"""
        data = {"data": [interaction_data]}
        headers = {"Content-Type": "application/json"}
        if custom_headers:
            headers.update(custom_headers)

        await self._execute_request(custom_url, headers, data, "custom endpoint")

    async def _execute_request(self, url: str, headers: Dict[str, str], data: Dict[str, Any], endpoint_name: str) -> None:
        """Execute the HTTP POST request to send analytics data"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status == 200:
                        logger.info(f"Analytics sent successfully to {endpoint_name}")
                    else:
                        response_text = await response.text()
                        logger.error(
                            f"Failed to send analytics to {endpoint_name}: HTTP {response.status}"
                        )
                        logger.error(f"Response content: {response_text}")
        except Exception as e:
            logger.error(f"Error sending analytics to {endpoint_name}: {e}")

    def send_interaction_analytics_safe(self, interaction_data: Dict[str, Any]) -> None:
        """
        Safely send turn analytics without blocking.
        Creates a task if event loop is running, otherwise ignores.
        """
        try:
            asyncio.create_task(self.send_interaction_analytics(interaction_data))
            pass
        except RuntimeError:
            pass
