"""
Execution module for VideoSDK Agents.

This module provides a modern resource management architecture for executing
agent tasks using processes and threads with optimized resource allocation.
"""

from .types import (
    ExecutorType,
    ResourceType,
    TaskType,
    ResourceStatus,
    TaskStatus,
    ResourceConfig,
    TaskConfig,
    TaskResult,
    ResourceInfo,
    HealthMetrics,
)
from .resource_manager import ResourceManager
from .resources import ProcessResource, ThreadResource
from .task_executor import TaskExecutor
from .inference_resource import DedicatedInferenceResource

__all__ = [
    # Core types
    "ExecutorType",
    "ResourceType",
    "TaskType",
    "ResourceStatus",
    "TaskStatus",
    "ResourceConfig",
    "TaskConfig",
    "TaskResult",
    "ResourceInfo",
    "HealthMetrics",
    # Resource management
    "ResourceManager",
    "ProcessResource",
    "ThreadResource",
    "DedicatedInferenceResource",
    # Task execution
    "TaskExecutor",
]
