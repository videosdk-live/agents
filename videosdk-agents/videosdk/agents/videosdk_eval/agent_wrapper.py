import av
from videosdk.agents import Agent, AgentSession
from videosdk.agents.pipeline import Pipeline
from videosdk.agents.utterance_handle import UtteranceHandle

class MockPipeline(Pipeline):
    async def start(self, **kwargs): pass
    async def on_audio_delta(self, audio_data: bytes): pass
    async def on_video_delta(self, video_data: av.VideoFrame): pass
    async def send_message(self, message: str, handle: UtteranceHandle): pass


class EvalAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock_pipeline = MockPipeline()
        self.session = AgentSession(agent=self, pipeline=self.mock_pipeline)

    async def on_enter(self): pass
    async def on_exit(self): pass