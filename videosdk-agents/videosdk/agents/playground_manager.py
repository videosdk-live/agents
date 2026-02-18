import asyncio
from videosdk import PubSubPublishConfig
import logging
from dataclasses import asdict
import json
from typing import Literal
logger = logging.getLogger(__name__)

class PlaygroundManager:
    def __init__(self, ctx):
        self.job_context = ctx
        self.job_context.playground_manager = self

    def send_metrics(self, metrics_type: Literal["cascading", "realtime"], metrics: dict, full_turn_data: bool = False):
        """Sends metrics to the playground.
            Args:
                metrics_type (str): The type of metrics to send.
                metrics (dict): The metrics to send.
                full_turn_data (bool): Whether to send full turn data.
        """
        if metrics_type == "cascading":
            self._send_cascading_metrics(metrics, full_turn_data)
        elif metrics_type == "realtime":
            self._send_realtime_metrics(metrics, full_turn_data)
        else:
            logger.error("[send_metrics] Invalid metrics type: {}".format(metrics_type))

    def _send_cascading_metrics(self, metrics: dict, full_turn_data: bool = False):
        """Sends cascading metrics to the playground.
            Args:
                metrics (dict): The metrics to send.
                full_turn_data (bool): Whether to send full turn data.
        """
        if full_turn_data:
            metrics = asdict(metrics)
        metrics = json.dumps(metrics)
        publish_config = PubSubPublishConfig(
            topic="AGENT_METRICS",
            message={"type": "cascading", "metrics": metrics, "full_turn_data": full_turn_data}
        )

        if self.job_context.room:
            asyncio.create_task(self.job_context.room.publish_to_pubsub(publish_config))
        else:
            logger.error("[send_cascading_metrics] Cannot send cascading metrics: room is not available")

    def _send_realtime_metrics(self, metrics: dict, full_turn_data: bool = False):
        """Sends realtime metrics to the playground.
            Args:
                metrics (dict): The metrics to send.
                full_turn_data (bool): Whether to send full turn data.
        """
        if full_turn_data:
            metrics = asdict(metrics)
        metrics = json.dumps(metrics)
        publish_config = PubSubPublishConfig(
            topic="AGENT_METRICS",
            message={"type": "realtime", "metrics": metrics, "full_turn_data": full_turn_data}
        )

        if self.job_context.room:
            asyncio.create_task(self.job_context.room.publish_to_pubsub(publish_config))
