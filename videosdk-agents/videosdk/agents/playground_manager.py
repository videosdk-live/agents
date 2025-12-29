import asyncio
from videosdk import PubSubPublishConfig
import logging
from dataclasses import asdict
import json
logger = logging.getLogger(__name__)

class PlaygroundManager:
    def __init__(self, ctx):
        self.job_context = ctx
        self.job_context.playground_manager = self

    async def start(self):
        """Activates the playground by publishing a message to pubsub.
            Args:
                None
        """
        publish_config = PubSubPublishConfig(
            topic="playground",
            message="Activated playground"
        )

        await self.job_context.room.publish_to_pubsub(publish_config)
    
    def send_cascading_metrics(self, metrics: dict, full_turn_data: bool = False):
        """Sends cascading metrics to the playground.
            Args:
                metrics (dict): The metrics to send.
                full_turn_data (bool): Whether to send full turn data.
        """
        metrics = json.dumps(asdict(metrics))
        publish_config = PubSubPublishConfig(
            topic="AGENT_METRICS",
            message={"type": "cascading", "metrics": metrics, "full_turn_data": full_turn_data}
        )

        asyncio.create_task(self.job_context.room.publish_to_pubsub(publish_config))

    def send_realtime_metrics(self, metrics: dict, full_turn_data: bool = False):
        """Sends realtime metrics to the playground.
            Args:
                metrics (dict): The metrics to send.
                full_turn_data (bool): Whether to send full turn data.
        """
        metrics = json.dumps(asdict(metrics))
        publish_config = PubSubPublishConfig(
            topic="AGENT_METRICS",
            message={"type": "realtime", "metrics": metrics, "full_turn_data": full_turn_data}
        )

        asyncio.create_task(self.job_context.room.publish_to_pubsub(publish_config))
