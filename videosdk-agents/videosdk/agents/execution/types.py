"""
Type definitions for the execution module.
"""

import sys
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, Coroutine


class ExecutorType(Enum):
    """Type of executor for task processing."""

    PROCESS = "process"
    THREAD = "thread"


class ResourceType(Enum):
    """Type of resource for task execution."""

    PROCESS = "process"
    THREAD = "thread"


class TaskType(Enum):
    """Type of task to be executed."""

    INFERENCE = "inference"  # For AI model inference
    MEETING = "meeting"  # For video meeting tasks
    JOB = "job"  # For general job execution


class ResourceStatus(Enum):
    """Status of a resource."""

    IDLE = "idle"
    BUSY = "busy"
    INITIALIZING = "initializing"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


class TaskStatus(Enum):
    """Status of a task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Platform-specific defaults
if sys.platform.startswith("win"):
    _default_executor_type = ExecutorType.THREAD
else:
    _default_executor_type = ExecutorType.PROCESS


@dataclass
class ResourceConfig:
    """Configuration for resource management."""

    # Resource type and count
    resource_type: ResourceType = ResourceType.PROCESS
    num_idle_resources: int = 2
    max_resources: int = 10

    # Timeouts
    initialize_timeout: float = 10.0
    close_timeout: float = 60.0

    # Memory management
    memory_warn_mb: float = 500.0
    memory_limit_mb: float = 0.0

    # Health monitoring
    ping_interval: float = 30.0
    health_check_interval: float = 5.0

    # Load balancing
    load_threshold: float = 0.75
    max_concurrent_tasks: int = 1

    # Platform-specific
    executor_type: ExecutorType = _default_executor_type

    # Legacy IPC compatibility
    use_dedicated_inference_process: bool = (
        True  # New: Enable dedicated inference process
    )
    inference_process_timeout: float = 30.0  # Longer timeout for AI model loading
    inference_memory_warn_mb: float = 1000.0  # Higher threshold for AI models


@dataclass
class TaskConfig:
    """Configuration for task execution."""

    task_type: TaskType
    timeout: float = 300.0  # 5 minutes default
    retry_count: int = 3
    priority: int = 0  # Higher number = higher priority

    # Resource requirements
    required_memory_mb: float = 100.0
    required_cpu_cores: int = 1

    # Task-specific data
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a task execution."""

    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_used_mb: float = 0.0


@dataclass
class ResourceInfo:
    """Information about a resource."""

    resource_id: str
    resource_type: ResourceType
    status: ResourceStatus
    current_load: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_tasks: int = 0
    total_tasks_processed: int = 0
    last_heartbeat: float = 0.0

    # Resource-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthMetrics:
    """Health metrics for resource monitoring."""

    resource_id: str
    timestamp: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_tasks: int
    response_time_ms: float
    error_count: int = 0
    success_count: int = 0
