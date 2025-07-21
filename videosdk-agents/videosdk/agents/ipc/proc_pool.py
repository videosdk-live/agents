import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .process_manager import ProcessManager, ExecutorType, ProcPoolConfig

logger = logging.getLogger(__name__)


@dataclass
class PoolStats:
    """Statistics for the process pool."""

    total_processes: int
    available_processes: int
    busy_processes: int
    failed_processes: int
    memory_usage_mb: float
    uptime_seconds: float


class ProcPool:
    """
    Process pool for agents.

    for agent execution with load balancing and health monitoring.
    """

    def __init__(
        self,
        *,
        initialize_process_fnc: Any,
        job_entrypoint_fnc: Any,
        num_idle_processes: int = 2,
        initialize_timeout: float = 10.0,
        close_timeout: float = 60.0,
        memory_warn_mb: float = 500.0,
        memory_limit_mb: float = 0.0,
        ping_interval: float = 30.0,
        max_processes: int = 10,
        executor_type: ExecutorType = ExecutorType.PROCESS,
    ):
        self.initialize_process_fnc = initialize_process_fnc
        self.job_entrypoint_fnc = job_entrypoint_fnc
        self.executor_type = executor_type
        self.config = ProcPoolConfig(
            num_idle_processes=num_idle_processes,
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
            memory_warn_mb=memory_warn_mb,
            memory_limit_mb=memory_limit_mb,
            ping_interval=ping_interval,
            max_processes=max_processes,
        )

        # Process/Thread pools
        self.job_executors: List[Any] = []
        self._shutdown = False
        self._lock = asyncio.Lock()

        # Single shared inference executor
        self.inference_executor: Optional[Any] = None

        # Health monitoring
        self.health_monitor = None  # Will be set by ProcessManager

    async def initialize(self):
        """Initialize the process pools."""
        logger.info("Initializing process pools")

        # Create single shared inference executor
        await self._create_inference_executor()

        # Create initial idle job executors
        for _ in range(self.config.num_idle_processes):
            await self._create_job_executor()

        logger.info(
            f"Process pools initialized with {self.config.num_idle_processes} idle job {self.executor_type.value}s and 1 shared inference process"
        )

    async def _create_job_executor(self) -> Any:
        """Create a new job executor (process or thread)."""
        if self.executor_type == ExecutorType.THREAD:
            from .job_thread_executor import ThreadJobExecutor

            executor = ThreadJobExecutor(
                initialize_timeout=self.config.initialize_timeout,
                close_timeout=self.config.close_timeout,
                ping_interval=self.config.ping_interval,
                high_ping_threshold=5.0,
                inference_executor=self.inference_executor,  # Pass shared executor
            )
            # Set the entrypoint function
            executor._entrypoint_fnc = self.job_entrypoint_fnc
        else:
            from .job_proc_executor import JobProcExecutor

            executor = JobProcExecutor(
                initialize_timeout=self.config.initialize_timeout,
                close_timeout=self.config.close_timeout,
                memory_warn_mb=self.config.memory_warn_mb,
                memory_limit_mb=self.config.memory_limit_mb,
                ping_interval=self.config.ping_interval,
                # Don't pass inference_executor to avoid pickling issues
                # inference_executor=self.inference_executor,
            )
            # Don't set entrypoint function for process executors to avoid pickling issues
            # The job process will handle job execution internally

        await executor.initialize()
        self.job_executors.append(executor)
        logger.info(
            f"Created job {self.executor_type.value}, total: {len(self.job_executors)}"
        )
        return executor

    async def _create_inference_executor(self) -> Any:
        """Create a single shared inference executor."""
        if self.executor_type == ExecutorType.THREAD:
            from .inference_executor import InferenceExecutor

            # For thread mode, create a simple inference executor
            self.inference_executor = InferenceExecutor()
            logger.info("Created single shared inference executor (thread mode)")
        else:
            from .inference_proc_executor import InferenceProcExecutor

            self.inference_executor = InferenceProcExecutor(
                initialize_timeout=self.config.initialize_timeout,
                close_timeout=self.config.close_timeout,
                memory_warn_mb=self.config.memory_warn_mb,
                memory_limit_mb=self.config.memory_limit_mb,
                ping_interval=self.config.ping_interval,
            )

            await self.inference_executor.initialize()
            logger.info("Created single shared inference executor (process mode)")

        return self.inference_executor

    async def get_job_executor(self) -> Any:
        """Get an available job executor, creating one if needed."""
        async with self._lock:
            # Find available executor
            for executor in self.job_executors:
                if executor.is_available:
                    return executor

            # Create new executor if under limit
            if len(self.job_executors) < self.config.max_processes:
                return await self._create_job_executor()

            # Wait for available executor
            while True:
                for executor in self.job_executors:
                    if executor.is_available:
                        return executor
                await asyncio.sleep(0.1)

    async def get_inference_executor(self) -> Any:
        """Get the single shared inference executor."""
        if not self.inference_executor:
            raise Exception("Inference executor not initialized")
        return self.inference_executor

    def get_by_job_id(self, job_id: str) -> Optional[Any]:
        """Get a job executor by job ID."""
        for executor in self.job_executors:
            if (
                hasattr(executor, "current_job_id")
                and executor.current_job_id == job_id
            ):
                return executor
        return None

    async def shutdown(self):
        """Shutdown all executors."""
        logger.info("Shutting down process pools")
        self._shutdown = True

        # Shutdown job executors
        for executor in self.job_executors:
            try:
                await executor.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down job {self.executor_type.value}: {e}")

        # Shutdown single inference executor
        if self.inference_executor:
            try:
                await self.inference_executor.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down inference executor: {e}")

        self.job_executors.clear()
        self.inference_executor = None
        logger.info("Process pools shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the process pools."""
        return {
            "job_executors": len(self.job_executors),
            "executor_type": self.executor_type.value,
            "inference_executor": "active" if self.inference_executor else "none",
            "max_processes": self.config.max_processes,
            "idle_processes": self.config.num_idle_processes,
        }
