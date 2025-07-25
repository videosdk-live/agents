"""
Resource manager for task execution.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from .base_resource import BaseResource
from .resources import ProcessResource, ThreadResource
from .inference_resource import DedicatedInferenceResource
from .types import (
    ResourceType,
    ResourceConfig,
    TaskConfig,
    TaskType,
    ResourceStatus,
    TaskStatus,
    TaskResult,
    ResourceInfo,
)

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manages resources for task execution.

    This class handles:
    - Resource creation and lifecycle management
    - Load balancing across resources
    - Health monitoring and recovery
    - Resource allocation for tasks
    - Dedicated inference process management (legacy IPC compatibility)
    """

    def __init__(self, config: ResourceConfig):
        self.config = config
        self.resources: List[BaseResource] = []
        self._shutdown = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._resource_creation_task: Optional[asyncio.Task] = None

        # Dedicated inference resource (legacy IPC compatibility)
        self.dedicated_inference_resource: Optional[DedicatedInferenceResource] = None

    async def start(self):
        """Start the resource manager."""
        logger.info("Starting resource manager")

        # Create dedicated inference resource if enabled
        if self.config.use_dedicated_inference_process:
            await self._create_dedicated_inference_resource()

        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        # Start resource creation
        self._resource_creation_task = asyncio.create_task(
            self._resource_creation_loop()
        )

        # Initialize initial resources
        await self._create_initial_resources()

        logger.info("Resource manager started")

    async def stop(self):
        """Stop the resource manager."""
        logger.info("Stopping resource manager")
        self._shutdown = True

        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._resource_creation_task:
            self._resource_creation_task.cancel()

        # Shutdown all resources
        shutdown_tasks = []
        for resource in self.resources:
            shutdown_tasks.append(resource.shutdown())

        # Shutdown dedicated inference resource
        if self.dedicated_inference_resource:
            shutdown_tasks.append(self.dedicated_inference_resource.shutdown())

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        logger.info("Resource manager stopped")

    async def _create_dedicated_inference_resource(self):
        """Create the dedicated inference resource (legacy IPC compatibility)."""
        logger.info("Creating dedicated inference resource")

        inference_config = {
            "inference_process_timeout": self.config.inference_process_timeout,
            "inference_memory_warn_mb": self.config.inference_memory_warn_mb,
            "ping_interval": self.config.ping_interval,
            "close_timeout": self.config.close_timeout,
        }

        self.dedicated_inference_resource = DedicatedInferenceResource(
            resource_id="dedicated-inference", config=inference_config
        )

        await self.dedicated_inference_resource.initialize()
        logger.info("Dedicated inference resource created")

    async def _create_initial_resources(self):
        """Create initial resources based on configuration."""
        initial_count = self.config.num_idle_resources
        logger.info(
            f"Creating {initial_count} initial {self.config.resource_type.value} resources"
        )

        for i in range(initial_count):
            await self._create_resource(self.config.resource_type)

    async def _create_resource(self, resource_type: ResourceType) -> BaseResource:
        """Create a new resource of the specified type."""
        resource_id = f"{resource_type.value}-{uuid.uuid4().hex[:8]}"

        config = {
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
            "initialize_timeout": self.config.initialize_timeout,
            "close_timeout": self.config.close_timeout,
            "health_check_interval": self.config.health_check_interval,
        }

        if resource_type == ResourceType.PROCESS:
            resource = ProcessResource(resource_id, config)
        elif resource_type == ResourceType.THREAD:
            resource = ThreadResource(resource_id, config)
        else:
            raise ValueError(f"Unsupported resource type: {resource_type}")

        # Initialize the resource
        await resource.initialize()

        # Add to resources list
        self.resources.append(resource)

        logger.info(f"Created {resource_type.value} resource: {resource_id}")
        return resource

    async def _resource_creation_loop(self):
        """Background loop for creating resources as needed."""
        # Wait a bit longer before starting the loop to allow initial resources to stabilize
        await asyncio.sleep(10.0)

        while not self._shutdown:
            try:
                # Check if we need more resources
                available_count = len([r for r in self.resources if r.is_available])
                total_count = len(self.resources)

                # Create more resources if needed
                if (
                    available_count < self.config.num_idle_resources
                    and total_count < self.config.max_resources
                ):
                    logger.info(
                        f"Creating additional {self.config.resource_type.value} resource"
                    )
                    await self._create_resource(self.config.resource_type)

                await asyncio.sleep(10.0)  # Check every 10 seconds instead of 5

            except Exception as e:
                logger.error(f"Error in resource creation loop: {e}")
                await asyncio.sleep(5.0)

    async def _health_check_loop(self):
        """Background loop for health monitoring."""
        while not self._shutdown:
            try:
                # Check job resources
                for resource in self.resources[
                    :
                ]:  # Copy list to avoid modification during iteration
                    try:
                        is_healthy = await resource.health_check()
                        if not is_healthy:
                            logger.warning(
                                f"Unhealthy resource detected: {resource.resource_id}"
                            )
                            # Remove unhealthy resource
                            self.resources.remove(resource)
                            await resource.shutdown()

                            # Create replacement if needed
                            if len(self.resources) < self.config.num_idle_resources:
                                await self._create_resource(self.config.resource_type)

                    except Exception as e:
                        logger.error(
                            f"Health check failed for {resource.resource_id}: {e}"
                        )

                # Check dedicated inference resource
                if self.dedicated_inference_resource:
                    try:
                        is_healthy = (
                            await self.dedicated_inference_resource.health_check()
                        )
                        if not is_healthy:
                            logger.warning(
                                "Unhealthy dedicated inference resource detected"
                            )
                            # Recreate inference resource
                            await self.dedicated_inference_resource.shutdown()
                            await self._create_dedicated_inference_resource()
                    except Exception as e:
                        logger.error(
                            f"Health check failed for dedicated inference resource: {e}"
                        )

                await asyncio.sleep(self.config.health_check_interval)

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5.0)

    async def execute_task(
        self, task_config: TaskConfig, entrypoint: Callable, *args, **kwargs
    ) -> TaskResult:
        """Execute a task using an available resource."""
        task_id = str(uuid.uuid4())

        # Route inference tasks to dedicated inference resource
        if (
            task_config.task_type == TaskType.INFERENCE
            and self.dedicated_inference_resource
        ):
            logger.info(
                f"Routing inference task {task_id} to dedicated inference resource"
            )
            return await self.dedicated_inference_resource.execute_task(
                task_id, task_config, entrypoint, args, kwargs
            )

        # Route other tasks to job resources
        resource = await self._get_available_resource(task_config.task_type)
        if not resource:
            raise RuntimeError("No available resources for task execution")

        # Execute the task
        return await resource.execute_task(
            task_id, task_config, entrypoint, args, kwargs
        )

    async def _get_available_resource(
        self, task_type: TaskType
    ) -> Optional[BaseResource]:
        """Get an available resource for task execution."""
        # For now, use simple round-robin selection
        # In the future, this could be enhanced with load balancing, priority, etc.

        available_resources = [r for r in self.resources if r.is_available]

        if available_resources:
            # Simple round-robin selection
            # In a real implementation, you might want more sophisticated load balancing
            return available_resources[0]

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get resource manager statistics."""
        available_resources = [r for r in self.resources if r.is_available]
        active_resources = [
            r for r in self.resources if r.status != ResourceStatus.IDLE
        ]
        total_resources = len(self.resources)

        average_load = (
            len(active_resources) / total_resources if total_resources > 0 else 0.0
        )

        stats = {
            "total_resources": total_resources,
            "available_resources": len(available_resources),
            "active_resources": len(active_resources),
            "average_load": average_load,
            "resources": [
                {
                    "resource_id": r.get_info().resource_id,
                    "resource_type": r.get_info().resource_type.value,
                    "status": r.get_info().status.value,
                    "current_load": r.get_info().current_load,
                    "memory_usage_mb": r.get_info().memory_usage_mb,
                    "cpu_usage_percent": r.get_info().cpu_usage_percent,
                    "active_tasks": r.get_info().active_tasks,
                    "total_tasks_processed": r.get_info().total_tasks_processed,
                    "last_heartbeat": r.get_info().last_heartbeat,
                    "metadata": r.get_info().metadata,
                }
                for r in self.resources
            ],
            "dedicated_inference": None,
        }

        # Dedicated inference resource stats
        if self.dedicated_inference_resource:
            info = self.dedicated_inference_resource.get_info()
            stats["dedicated_inference"] = {
                "resource_id": info.resource_id,
                "resource_type": info.resource_type.value,
                "status": info.status.value,
                "current_load": info.current_load,
                "memory_usage_mb": info.memory_usage_mb,
                "cpu_usage_percent": info.cpu_usage_percent,
                "active_tasks": info.active_tasks,
                "total_tasks_processed": info.total_tasks_processed,
                "last_heartbeat": info.last_heartbeat,
                "metadata": info.metadata,
            }

        return stats

    def get_resource_info(self) -> List[ResourceInfo]:
        """Get information about all resources."""
        resource_info = []

        # Job resources
        for resource in self.resources:
            resource_info.append(resource.get_info())

        # Dedicated inference resource
        if self.dedicated_inference_resource:
            resource_info.append(self.dedicated_inference_resource.get_info())

        return resource_info
