"""
Task executor for high-level task execution.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass

from .resource_manager import ResourceManager
from .types import (
    ResourceConfig,
    TaskConfig,
    TaskType,
    TaskResult,
    TaskStatus,
    ResourceInfo,
)

logger = logging.getLogger(__name__)


class TaskExecutor:
    """
    High-level task executor that manages task execution using resources.

    This class provides a simple interface for executing tasks while
    handling resource management, retries, and monitoring internally.
    """

    def __init__(self, config: ResourceConfig):
        self.config = config
        self.resource_manager = ResourceManager(config)
        self._shutdown = False
        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._total_execution_time = 0.0

    async def start(self):
        """Start the task executor."""
        logger.info("Starting task executor")
        await self.resource_manager.start()
        logger.info("Task executor started")

    async def stop(self):
        """Stop the task executor."""
        logger.info("Stopping task executor")
        self._shutdown = True

        # Stop resource manager
        await self.resource_manager.stop()
        logger.info("Task executor stopped")

    async def execute(
        self,
        entrypoint: Callable,
        task_type: TaskType = TaskType.JOB,
        timeout: float = 300.0,
        retry_count: int = 3,
        priority: int = 0,
        *args,
        **kwargs,
    ) -> TaskResult:
        """
        Execute a task using the resource manager.

        Args:
            entrypoint: Function to execute
            task_type: Type of task (inference, meeting, job)
            timeout: Task timeout in seconds
            retry_count: Number of retries on failure
            priority: Task priority (higher = higher priority)
            *args, **kwargs: Arguments to pass to entrypoint

        Returns:
            TaskResult with execution results
        """
        task_config = TaskConfig(
            task_type=task_type,
            timeout=timeout,
            retry_count=retry_count,
            priority=priority,
        )

        # Execute with retries
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                result = await self.resource_manager.execute_task(
                    task_config, entrypoint, *args, **kwargs
                )

                # Update stats
                self._update_stats(result)

                if result.status == TaskStatus.COMPLETED:
                    return result
                else:
                    last_error = result.error

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Task execution attempt {attempt + 1} failed: {e}")

                if attempt < retry_count:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff

        # All retries failed
        failed_result = TaskResult(
            task_id=task_config.task_type.value,
            status=TaskStatus.FAILED,
            error=f"All {retry_count + 1} attempts failed. Last error: {last_error}",
            execution_time=0.0,
        )

        self._update_stats(failed_result)
        return failed_result

    def _update_stats(self, result: TaskResult):
        """Update execution statistics."""
        self._total_tasks += 1

        if result.status == TaskStatus.COMPLETED:
            self._completed_tasks += 1
            self._total_execution_time += result.execution_time
        elif result.status == TaskStatus.FAILED:
            self._failed_tasks += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        resource_stats = self.resource_manager.get_stats()

        average_execution_time = (
            self._total_execution_time / self._completed_tasks
            if self._completed_tasks > 0
            else 0.0
        )

        return {
            "executor_stats": {
                "total_tasks": self._total_tasks,
                "completed_tasks": self._completed_tasks,
                "failed_tasks": self._failed_tasks,
                "pending_tasks": 0,
                "average_execution_time": average_execution_time,
                "total_execution_time": self._total_execution_time,
            },
            "resource_stats": resource_stats,
        }

    def get_resource_info(self) -> List[ResourceInfo]:
        """Get information about all resources."""
        return self.resource_manager.get_resource_info()
