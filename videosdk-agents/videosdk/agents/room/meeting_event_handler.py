from videosdk import MeetingEventHandler, Participant
from typing import Callable


class MeetingHandler(MeetingEventHandler):
    def __init__(
        self,
        on_meeting_joined: Callable[[str], None],
        on_meeting_left: Callable[[str], None],
        on_participant_joined: Callable[[Participant], None],
        on_participant_left: Callable[[Participant], None],
    ):
        super().__init__()
        self.on_meeting_joined = on_meeting_joined
        self.on_meeting_left = on_meeting_left
        self.on_participant_joined = on_participant_joined
        self.on_participant_left = on_participant_left