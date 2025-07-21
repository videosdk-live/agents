"""
Inference Executor for VideoSDK Agents IPC.

This module provides the interface for inference execution in the IPC system.
"""

from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class InferenceExecutor:
    """Interface for inference execution in the IPC system."""

    def __init__(self):
        pass

    async def execute(self, inference_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an inference task with the given data."""
        raise NotImplementedError("Subclasses must implement execute")

    async def shutdown(self):
        """Shutdown the inference executor."""
        pass  # Default implementation does nothing

    async def ping(self):
        """Send a ping to check if the inference executor is alive."""
        return True  # Default implementation always returns True

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the inference executor."""
        return {
            "type": "InferenceExecutor",
            "status": "idle",
            "executor_type": "thread",
        }


class InferenceContext:
    """Context for inference execution."""

    def __init__(self, inference_data: Dict[str, Any]):
        self.inference_data = inference_data
        self.task_id = inference_data.get("task_id")
        self.task_type = inference_data.get("task_type")
        self.model_config = inference_data.get("model_config", {})
