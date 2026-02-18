import logging
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)

START_RECORDING_URL = "https://api.videosdk.live/v2/recordings/participant/start"
STOP_RECORDING_URL = "https://api.videosdk.live/v2/recordings/participant/stop"
MERGE_RECORDINGS_URL = "https://api.videosdk.live/v2/recordings/participant/merge"

class RecordingManager:
    """
    Handles participant-level recording and merging.
    """
    def __init__(self, room_id: str, auth_token: str):
        self.room_id = room_id
        self.auth_token = auth_token

    async def start_participant_recording(self, participant_id: str):
        """
        Start recording for a specific participant.
        """
        headers = {
            "Authorization": self.auth_token,
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(
                START_RECORDING_URL,
                json={"roomId": self.room_id, "participantId": participant_id},
                headers=headers,
            )
            logger.info(f"[start_participant_recording] Recording successfully started for id {participant_id} and response {response.text}")
            response.raise_for_status()
        except Exception as e:
            logger.error(f"[start_participant_recording] Error starting recording for participant {participant_id}: {e}")

    async def stop_participant_recording(self, participant_id: str):
        """
        Stop recording for a specific participant.
        """
        headers = {
            "Authorization": self.auth_token,
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(
                STOP_RECORDING_URL,
                json={"roomId": self.room_id, "participantId": participant_id},
                headers=headers,
            )
            logger.info(f"[stop_participant_recording] Recording successfully stopped for id {participant_id} and response {response.text}")
            response.raise_for_status()
        except Exception as e:
            logger.error(f"[stop_participant_recording] Error stopping recording for participant {participant_id}: {e}")

    async def merge_participant_recordings(self, session_id: str, local_participant_id: str, participants_data: Dict[str, Any]):
        """
        Merge recordings from all participants.
        """
        headers = {
            "Authorization": self.auth_token,
            "Content-Type": "application/json"
        }
        try:
            payload = {
                "sessionId": session_id,
                "channel1": [{"participantId": local_participant_id}],
                "channel2": [
                    {"participantId": p_id}
                    for p_id in participants_data.keys()
                ],
            }
            response = requests.post(
                MERGE_RECORDINGS_URL,
                json=payload,
                headers=headers,
            )
            logger.info(f"[merge_participant_recordings] Recording successfully merged response: {response.text}")
            response.raise_for_status()
        except Exception as e:
            logger.error(f"[merge_participant_recordings] Error merging recordings: {e}")

    async def stop_and_merge_recordings(self, session_id: str, local_participant_id: str, participants_data: Dict[str, Any]):
        """
        Stop all recordings and merge them.
        """
        # Stop local participant recording
        await self.stop_participant_recording(local_participant_id)
        
        # Stop other participants recording
        for p_id in participants_data.keys():
            logger.info(f"[stop_and_merge_recordings] Stopping participant recording for id {p_id}")
            await self.stop_participant_recording(p_id)
            
        await self.merge_participant_recordings(session_id, local_participant_id, participants_data)
        logger.info("[stop_and_merge_recordings] Stopped and merged recordings")
