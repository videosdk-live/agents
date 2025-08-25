"""
Agent initialization configuration utilities.

This module provides functionality to fetch agent initialization configuration
from the VideoSDK API server, including registry URL.
"""

import aiohttp
import logging

logger = logging.getLogger(__name__)


async def fetch_agent_init_config(
    auth_token: str, api_base_url: str = "https://api.videosdk.live"
) -> str:
    """
    Fetch agent initialization configuration from the VideoSDK API server.

    Args:
        auth_token: VideoSDK authentication token
        api_base_url: Base URL for the VideoSDK API server

    Returns:
        Registry URL string

    Raises:
        RuntimeError: If the API call fails or returns invalid data
    """
    url = f"{api_base_url}/v2/agent/init-config"
    headers = {"Authorization": f"{auth_token}", "Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Failed to fetch agent init config. Status: {response.status}, Response: {error_text}"
                    )

                data = await response.json()

                if not data.get("success"):
                    raise RuntimeError(
                        f"API returned error: {data.get('message', 'Unknown error')}"
                    )

                config_data = data.get("data", {})
                registry_url = config_data.get("registryUrl")

                if not registry_url:
                    raise RuntimeError(
                        "Invalid init config response: missing registryUrl"
                    )

                logger.info(f"Fetched agent init config - Registry: {registry_url}")

                return registry_url

    except aiohttp.ClientError as e:
        raise RuntimeError(f"Network error while fetching agent init config: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while fetching agent init config: {e}")
