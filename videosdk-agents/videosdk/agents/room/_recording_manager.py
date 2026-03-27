import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

START_RECORDING_URL = "https://api.videosdk.live/v2/recordings/participant/start"
STOP_RECORDING_URL = "https://api.videosdk.live/v2/recordings/participant/stop"
MERGE_RECORDINGS_URL = "https://api.videosdk.live/v2/recordings/participant/merge"

START_TRACK_RECORDING_URL = "https://api.videosdk.live/v2/recordings/participant/track/start"
STOP_TRACK_RECORDING_URL = "https://api.videosdk.live/v2/recordings/participant/track/stop"
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
            logger.info(f"starting participant recording response completed for id {participant_id} and response {response.text}")
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error starting recording for participant {participant_id}: {e}")

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
            logger.info(f"stop participant recording response for id {participant_id} and response {response.text}")
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error stopping recording for participant {participant_id}: {e}")

    async def start_track_recording(self, participant_id: str, kind: str) -> None:
        """
        Start recording a specific track kind (audio/video/screen_audio/screen_video).
        """
        headers = {
            "Authorization": self.auth_token,
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                START_TRACK_RECORDING_URL,
                json={"roomId": self.room_id, "participantId": participant_id, "kind": kind},
                headers=headers,
            )
            logger.info(
                "starting track recording kind=%s for participant=%s response=%s",
                kind,
                participant_id,
                response.text,
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(
                "Error starting track recording kind=%s for participant=%s: %s",
                kind,
                participant_id,
                e,
            )

    async def stop_track_recording(self, participant_id: str, kind: str) -> None:
        """
        Stop recording a specific track kind (audio/video/screen_audio/screen_video).
        """
        headers = {
            "Authorization": self.auth_token,
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                STOP_TRACK_RECORDING_URL,
                json={"roomId": self.room_id, "participantId": participant_id, "kind": kind},
                headers=headers,
            )
            logger.info(
                "stopping track recording kind=%s for participant=%s response=%s",
                kind,
                participant_id,
                response.text,
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(
                "Error stopping track recording kind=%s for participant=%s: %s",
                kind,
                participant_id,
                e,
            )

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
            logger.info(f"merging participant recordings completed response: {response.text}")
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error merging recordings: {e}")

    async def stop_and_merge_recordings(
        self,
        session_id: str,
        local_participant_id: str,
        participants_data: Dict[str, Any],
        track_kinds_by_participant: Optional[Dict[str, set[str]]] = None,
        stop_participants_recording: bool = True,
    ):
        """
        Stop all recordings and merge them.
        """
        if stop_participants_recording:
            await self.stop_participant_recording(local_participant_id)

            for p_id in participants_data.keys():
                logger.info(
                    "stopping participant recording for id %s", p_id
                )
                await self.stop_participant_recording(p_id)

        if track_kinds_by_participant:
            for p_id, kinds in track_kinds_by_participant.items():
                for kind in kinds:
                    await self.stop_track_recording(p_id, kind)
            
        await self.merge_participant_recordings(session_id, local_participant_id, participants_data)
        logger.info("stopped and merged recordings")
