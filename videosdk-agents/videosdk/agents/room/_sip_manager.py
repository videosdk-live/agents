import asyncio
import logging
import requests
from typing import Optional,Any

logger = logging.getLogger(__name__)

FETCH_CALL_INFO_URL = "https://api.videosdk.live/v2/sip/call"
TRANSFER_CALL_URL = "https://api.videosdk.live/v2/sip/call/transfer"
END_CALL_URL = "https://api.videosdk.live/v2/sip/call/end"
MAKE_OUTBOUND_CALL_URL = "https://api.videosdk.live/v2/sip/call"
SWITCH_ROOM_URL = "https://api.videosdk.live/v2/sip/call/switch-room"

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

    def make_outbound_call(
        self,
        *,
        routing_rule_id: str,
        sip_call_to: str,
        sip_call_from: str,
        destination_room_id: Optional[str] = None,
        participant: Optional[dict] = None,
        **extra_options: Any,
    ) -> dict:
        """
        Initiate an outbound SIP call via ``POST /v2/sip/call``.

        The three required body fields (``routingRuleId``, ``sipCallTo``,
        ``sipCallFrom``) are always sent. Every optional field —
        ``destinationRoomId``, ``participant``, ``recordAudio``,
        ``ringingTimeout``, ``headers``, ``metadata``, etc. — is included only
        when explicitly supplied via ``extra_options``.
        """
        try:
            payload: dict[str, Any] = {
                "routingRuleId": routing_rule_id,
                "sipCallTo": sip_call_to,
                "sipCallFrom": sip_call_from,
            }
            
            if destination_room_id is not None:
                payload["destinationRoomId"] = destination_room_id
            if participant is not None:
                payload["participant"] = participant
            for key, value in extra_options.items():
                if value is not None:
                    payload[key] = value

            logger.info(f"[OUTBOUND CALL] payload={payload}")

            headers = {
                "Authorization": self.auth_token,
                "Content-Type": "application/json",
            }
            response = requests.post(MAKE_OUTBOUND_CALL_URL, json=payload, headers=headers)
            if not response.ok:
                body = response.text[:500] if response.text else "<empty>"
                logger.error(
                    f"[OUTBOUND CALL] {response.status_code} {response.reason} "
                    f"url={MAKE_OUTBOUND_CALL_URL} body={body}"
                )
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            logger.error("[OUTBOUND CALL] HTTP request failed", exc_info=True)
            raise
        except Exception:
            logger.error("[OUTBOUND CALL] Unexpected error", exc_info=True)
            raise

    def switch_call_room(
        self,
        *,
        call_id: str,
        room_id: str,
        token: str,
        participant_id: str,
    ) -> dict:
        """
        Move an active SIP call into a different VideoSDK room.

        Wraps ``POST /v2/sip/call/switch-room``. After this succeeds the
        caller appears in ``room_id`` as ``participant_id`` and leaves the
        previous room.
        """
        try:
            logger.info(
                f"[SWITCH ROOM] callId={call_id} → roomId={room_id} "
                f"participantId={participant_id}"
            )

            headers = {
                "Authorization": self.auth_token,
                "Content-Type": "application/json",
            }
            payload = {
                "callId": call_id,
                "roomId": room_id,
                "token": token,
                "participantId": participant_id,
            }
            response = requests.post(SWITCH_ROOM_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            logger.error("[SWITCH ROOM] HTTP request failed", exc_info=True)
            raise
        except Exception:
            logger.error("[SWITCH ROOM] Unexpected error", exc_info=True)
            raise

    async def async_make_outbound_call(self, **kwargs: Any) -> dict:
        """Async wrapper around :meth:`make_outbound_call`."""
        return await asyncio.to_thread(self.make_outbound_call, **kwargs)

    async def async_switch_call_room(self, **kwargs: Any) -> dict:
        """Async wrapper around :meth:`switch_call_room`."""
        return await asyncio.to_thread(self.switch_call_room, **kwargs)
