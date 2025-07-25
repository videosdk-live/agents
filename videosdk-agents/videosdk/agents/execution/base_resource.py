"""
Base resource class for task execution.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable, Coroutine, List
from dataclasses import dataclass

from .types import (
    ResourceType,
    ResourceStatus,
    TaskStatus,
    TaskConfig,
    TaskResult,
    ResourceInfo,
    HealthMetrics,
)
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BaseResource(ABC):
    """
    Abstract base class for all resources (process/thread).

    This class defines the interface that all resources must implement
    for task execution and resource management.
    """

    def __init__(self, resource_id: str, config: Dict[str, Any]):
        self.resource_id = resource_id
        self.config = config
        self.status = ResourceStatus.INITIALIZING
        self.current_load = 0.0
        self.memory_usage_mb = 0.0
        self.cpu_usage_percent = 0.0

        self.total_tasks_processed = 0
        self.last_heartbeat = time.time()
        self.error_count = 0
        self.success_count = 0

        # Health monitoring
        self.health_metrics: List[HealthMetrics] = []
        self.max_health_history = 100

        # Resource-specific state
        self._shutdown = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    @abstractmethod
    def resource_type(self) -> ResourceType:
        """Return the type of this resource."""
        pass

    @property
    def is_available(self) -> bool:
        """Check if the resource is available for new tasks."""
        return self.status == ResourceStatus.IDLE and not self._shutdown

    @property
    def current_load_percentage(self) -> float:
        """Get current load as a percentage."""
        # Resources are either IDLE (0%) or BUSY (100%)
        return 100.0 if self.status == ResourceStatus.BUSY else 0.0

    async def initialize(self) -> None:
        """Initialize the resource."""
        try:
            logger.info(
                f"Initializing {self.resource_type.value} resource {self.resource_id}"
            )
            await self._initialize_impl()
            self.status = ResourceStatus.IDLE
            self.last_heartbeat = time.time()
            logger.info(
                f"Initialized {self.resource_type.value} resource {self.resource_id}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize resource {self.resource_id}: {e}")
            self.status = ResourceStatus.ERROR
            raise

    @abstractmethod
    async def _initialize_impl(self) -> None:
        """Implementation-specific initialization."""
        pass

    async def execute_task(
        self,
        task_id: str,
        config: TaskConfig,
        entrypoint: Callable,
        args: tuple = (),
        kwargs: dict = None,
    ) -> TaskResult:
        """Execute a task and return the result."""
        if not self.is_available:
            raise RuntimeError(f"Resource {self.resource_id} is not available")

        if kwargs is None:
            kwargs = {}

        try:
            self.status = ResourceStatus.BUSY
            self.current_load = self.current_load_percentage

            logger.info(f"Executing task {task_id} on resource {self.resource_id}")
            start_time = time.time()

            # Execute the task
            result = await self._execute_task_impl(
                task_id, config, entrypoint, args, kwargs
            )

            execution_time = time.time() - start_time

            # Update metrics
            self.total_tasks_processed += 1
            self.success_count += 1
            self._update_health_metrics(execution_time)

            return TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                memory_used_mb=self.memory_usage_mb,
            )

        except Exception as e:
            logger.error(f"Task {task_id} failed on resource {self.resource_id}: {e}")
            self.error_count += 1

            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
                memory_used_mb=self.memory_usage_mb,
            )
        finally:
            # Cleanup
            self.current_load = self.current_load_percentage
            self.status = ResourceStatus.IDLE
            self.last_heartbeat = time.time()

    @abstractmethod
    async def _execute_task_impl(
        self,
        task_id: str,
        config: TaskConfig,
        entrypoint: Callable,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Implementation-specific task execution."""
        pass

    async def shutdown(self) -> None:
        """Shutdown the resource gracefully."""
        if self._shutdown:
            return

        logger.info(
            f"Shutting down {self.resource_type.value} resource {self.resource_id}"
        )
        self._shutdown = True
        self.status = ResourceStatus.SHUTTING_DOWN

        try:
            await self._shutdown_impl()
            logger.info(f"Shutdown completed for resource {self.resource_id}")

        except Exception as e:
            logger.error(f"Error during shutdown of resource {self.resource_id}: {e}")
            self.status = ResourceStatus.ERROR
            raise

    @abstractmethod
    async def _shutdown_impl(self) -> None:
        """Implementation-specific shutdown."""
        pass

    def get_info(self) -> ResourceInfo:
        """Get current resource information."""
        return ResourceInfo(
            resource_id=self.resource_id,
            resource_type=self.resource_type,
            status=self.status,
            current_load=self.current_load,
            memory_usage_mb=self.memory_usage_mb,
            cpu_usage_percent=self.cpu_usage_percent,
            active_tasks=0,
            total_tasks_processed=self.total_tasks_processed,
            last_heartbeat=self.last_heartbeat,
            metadata={
                "error_count": self.error_count,
                "success_count": self.success_count,
                "health_metrics_count": len(self.health_metrics),
            },
        )

    def _update_health_metrics(self, response_time_ms: float) -> None:
        """Update health metrics."""
        metrics = HealthMetrics(
            resource_id=self.resource_id,
            timestamp=time.time(),
            memory_usage_mb=self.memory_usage_mb,
            cpu_usage_percent=self.cpu_usage_percent,
            active_tasks=0,
            response_time_ms=response_time_ms,
            error_count=self.error_count,
            success_count=self.success_count,
        )

        self.health_metrics.append(metrics)

        # Keep only recent metrics
        if len(self.health_metrics) > self.max_health_history:
            self.health_metrics = self.health_metrics[-self.max_health_history :]

    async def health_check(self) -> bool:
        """Perform a health check on the resource."""
        try:
            # Basic health check - subclasses can override
            if self._shutdown:
                return False

            # Update heartbeat during health check to show resource is alive
            self.last_heartbeat = time.time()

            # For thread resources, just check if the thread is still alive
            if hasattr(self, "thread") and self.thread:
                return self.thread.is_alive()

            return True

        except Exception as e:
            logger.error(f"Health check failed for resource {self.resource_id}: {e}")
            return False

    def __str__(self) -> str:
        return f"{self.resource_type.value.capitalize()}Resource(id={self.resource_id}, status={self.status.value})"

    def __repr__(self) -> str:
        return self.__str__()
