from typing import Callable
from videosdk import (
    Stream,
    ParticipantEventHandler,
)


class ParticipantHandler(ParticipantEventHandler):
    def __init__(
        self,
        participant_id: str,
        on_stream_enabled: Callable[[Stream], None],
        on_stream_disabled: Callable[[Stream], None],
    ):
        self.participant_id = participant_id
        self.on_stream_enabled = on_stream_enabled
        self.on_stream_disabled = on_stream_disabled