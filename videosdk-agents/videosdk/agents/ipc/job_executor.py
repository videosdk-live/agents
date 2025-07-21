"""
Job Executor for VideoSDK Agents IPC.

This module provides the interface for job execution in the IPC system.
"""

from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class JobExecutor:
    """Interface for job execution in the IPC system."""

    def __init__(self):
        pass

    async def execute(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a job with the given data."""
        raise NotImplementedError("Subclasses must implement execute")


class JobContext:
    """Context for job execution."""

    def __init__(self, job_data: Dict[str, Any]):
        self.job_data = job_data
        self.job_id = job_data.get("job_id")
        self.room_id = job_data.get("room_id")
        self.agent_name = job_data.get("agent_name")
