import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

FETCH_CALL_INFO_URL = "https://api.videosdk.live/v2/sip/call"
TRANSFER_CALL_URL = "https://api.videosdk.live/v2/sip/call/transfer"
END_CALL_URL = "https://api.videosdk.live/v2/sip/call/end"

class SIPManager:
    """
    Handles SIP-related operations such as fetching call info and transferring calls.
    """
    def __init__(self, room_id: str, auth_token: str):
        self.room_id = room_id
        self.auth_token = auth_token

    def fetch_call_info(self, session_id: str):
        """
        Fetch SIP call information for the given room, then match by sessionId.
        """
        try:
            headers = {"Authorization": self.auth_token}
            params = {"roomId": self.room_id}

            logger.info(f"[FETCH CALL INFO] Requesting call info | roomId={self.room_id}")

            response = requests.get(FETCH_CALL_INFO_URL, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            calls = data.get("data", [])

            for call in calls:
                if call.get("sessionId") == session_id:
                    logger.info(f"[FETCH CALL INFO] Matching call found: {call.get('callId')}")
                    return call

            logger.warning("[FETCH CALL INFO] No SIP call matched with sessionId")
            return None

        except requests.RequestException as e:
            logger.error("[FETCH CALL INFO] HTTP request failed", exc_info=True)
            raise

        except Exception as e:
            logger.error("[FETCH CALL INFO] Unexpected error", exc_info=True)
            raise

    def transfer_call(self, call_id: str, transfer_to: str):
        """
        Transfer the call to a new number.
        """
        try:
            logger.info(f"[TRANSFER CALL] Initiating transfer | callId={call_id}, transferTo={transfer_to}")

            headers = {
                "Authorization": self.auth_token,
                "Content-Type": "application/json"
            }

            payload = {
                "callId": call_id,
                "transferTo": transfer_to
            }

            response = requests.post(TRANSFER_CALL_URL, json=payload, headers=headers)
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.error("[TRANSFER CALL] HTTP request failed", exc_info=True)
            raise

        except Exception as e:
            logger.error("[TRANSFER CALL] Unexpected error", exc_info=True)
            raise

    def end_call(self, call_id: str):
        """
        End an ongoing SIP call.
        """
        try:
            logger.info(f"[END CALL] Ending call | callId={call_id}")

            headers = {
                "Authorization": self.auth_token,
                "Content-Type": "application/json",
            }

            payload = {"callId": call_id}

            response = requests.post(END_CALL_URL, json=payload, headers=headers)
            response.raise_for_status()

            return response.json()

        except requests.RequestException:
            logger.error("[END CALL] HTTP request failed", exc_info=True)
            raise

        except Exception:
            logger.error("[END CALL] Unexpected error", exc_info=True)
            raise

    async def end_sip_call(self, session_id: str) -> None:
        """
        High-level SIP call end logic: resolve callId via session_id, then end it.
        """
        if not session_id:
            raise ValueError("Session ID is not set.")

        logger.info(
            f"[END SIP CALL] Fetching SIP call info | roomId={self.room_id}, sessionId={session_id}"
        )

        sip_call = self.fetch_call_info(session_id)

        if not sip_call:
            logger.warning("[END SIP CALL] No active SIP call found for given session ID.")
            return

        call_id = sip_call["callId"]
        logger.info(f"[END SIP CALL] Found SIP Call ID: {call_id}")

        result = self.end_call(call_id=call_id)
        logger.info(f"[END SIP CALL] End successful: {result}")

    async def call_transfer(self, session_id: str, transfer_to: str) -> None:
        """
        High-level call transfer logic.
        """
        if not session_id:
            raise ValueError("Session ID is not set.")

        logger.info(f"[CALL TRANSFER] Fetching SIP call info | roomId={self.room_id}, sessionId={session_id}")

        sip_call = self.fetch_call_info(session_id)

        if not sip_call:
            logger.error("[CALL TRANSFER] No active SIP call found for given session ID.")
            raise RuntimeError("Unable to perform transfer: No active SIP call found.")

        call_id = sip_call["callId"]
        logger.info(f"[CALL TRANSFER] Found SIP Call ID: {call_id}")

        result = self.transfer_call(
            call_id=call_id,
            transfer_to=transfer_to
        )

        logger.info(f"[CALL TRANSFER] Transfer successful: {result}")
