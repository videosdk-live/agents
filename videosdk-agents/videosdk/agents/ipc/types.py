from enum import Enum
from dataclasses import dataclass


class ExecutorType(Enum):
    """Type of executor for job processing."""

    PROCESS = "process"
    THREAD = "thread"


@dataclass
class ProcPoolConfig:
    """Configuration for process pool."""

    num_idle_processes: int = 2
    initialize_timeout: float = 10.0
    close_timeout: float = 60.0
    memory_warn_mb: float = 500.0
    memory_limit_mb: float = 0.0
    ping_interval: float = 30.0
    max_processes: int = 10
