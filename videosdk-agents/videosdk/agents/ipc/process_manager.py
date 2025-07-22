import asyncio
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import logging

from .proc_pool import ProcPool
from .types import ExecutorType, ProcPoolConfig
from .job_executor import JobExecutor
from .inference_executor import InferenceExecutor

logger = logging.getLogger(__name__)


# Automatic platform-based defaults
if sys.platform.startswith("win"):
    # Some python versions on Windows gets a BrokenPipeError when creating a new process
    _default_executor_type = ExecutorType.THREAD
else:
    _default_executor_type = ExecutorType.PROCESS


class ProcessManager:
    """
    Manages all IPC processes/threads for VideoSDK agents.

    This is the main interface for the IPC system, coordinating
    job processes/threads and a single shared inference executor.
    Automatically selects the appropriate executor type based on platform.
    """

    def __init__(
        self,
        *,
        job_entrypoint_fnc: Callable[[Any], Any],
        initialize_process_fnc: Optional[Callable[[Any], Any]] = None,
        config: Optional[ProcPoolConfig] = None,
        executor_type: Optional[ExecutorType] = None,
    ):
        self.job_entrypoint_fnc = job_entrypoint_fnc
        self.initialize_process_fnc = initialize_process_fnc
        self.executor_type = executor_type or _default_executor_type
        self.config = config or ProcPoolConfig()

        # Process pools
        self.proc_pool: Optional[ProcPool] = None

        # Health monitoring
        self.health_monitor_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Log the selected executor type
        logger.info(
            f"ProcessManager initialized with {self.executor_type.value} executor"
        )

    async def initialize(self):
        """Initialize the process manager and all pools."""
        logger.info(
            f"Initializing process manager with {self.executor_type.value} executor"
        )

        # Create process pool
        self.proc_pool = ProcPool(
            initialize_process_fnc=self.initialize_process_fnc,
            job_entrypoint_fnc=self.job_entrypoint_fnc,
            num_idle_processes=self.config.num_idle_processes,
            initialize_timeout=self.config.initialize_timeout,
            close_timeout=self.config.close_timeout,
            memory_warn_mb=self.config.memory_warn_mb,
            memory_limit_mb=self.config.memory_limit_mb,
            ping_interval=self.config.ping_interval,
            max_processes=self.config.max_processes,
            executor_type=self.executor_type,
        )

        # Initialize process pool
        await self.proc_pool.initialize()

        # Start health monitoring
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())

        logger.info("Process manager initialized")

    async def execute_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a job using the process pool."""
        if not self.proc_pool:
            raise Exception("Process manager not initialized")

        # Get available job executor
        job_executor = await self.proc_pool.get_job_executor()

        # Execute job
        return await job_executor.execute_job(job_data)

    async def execute_inference(self, inference_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an inference task using the single shared inference executor."""
        if not self.proc_pool:
            raise Exception("Process manager not initialized")

        # Get the single shared inference executor
        inference_executor = await self.proc_pool.get_inference_executor()

        # Execute inference
        return await inference_executor.execute_inference(inference_data)

    def get_by_job_id(self, job_id: str) -> Optional["JobExecutor"]:
        """Get a job executor by job ID."""
        if not self.proc_pool:
            return None

        return self.proc_pool.get_by_job_id(job_id)

    async def launch_job(self, running_info: "RunningJobInfo") -> None:
        """Launch a job with the given running info."""
        if not self.proc_pool:
            raise Exception("Process manager not initialized")

        # Get available job executor
        job_executor = await self.proc_pool.get_job_executor()

        # Launch the job
        await job_executor.launch_job(running_info)

    async def _health_monitor_loop(self):
        """Monitor health of all executors."""
        while not self._shutdown:
            try:
                if self.proc_pool:
                    # Ping all executors
                    ping_tasks = []

                    for job_executor in self.proc_pool.job_executors:
                        ping_tasks.append(job_executor.ping())

                    # Ping single inference executor
                    if self.proc_pool.inference_executor:
                        ping_tasks.append(self.proc_pool.inference_executor.ping())

                    # Wait for all pings to complete
                    if ping_tasks:
                        await asyncio.gather(*ping_tasks, return_exceptions=True)

                # Wait before next health check
                await asyncio.sleep(self.config.ping_interval)

            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(5.0)  # Wait before retrying

    async def shutdown(self):
        """Shutdown the process manager and all executors."""
        if self._shutdown:
            return

        logger.info("Shutting down process manager")
        self._shutdown = True

        # Stop health monitoring
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass

        # Shutdown process pool
        if self.proc_pool:
            await self.proc_pool.shutdown()

        logger.info("Process manager shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about all executors."""
        if not self.proc_pool:
            return {"status": "not_initialized"}

        stats = {
            "status": "running" if not self._shutdown else "shutdown",
            "executor_type": self.executor_type.value,
            "proc_pool": self.proc_pool.get_stats(),
        }

        # Add individual executor stats
        job_stats = []
        for job_executor in self.proc_pool.job_executors:
            job_stats.append(job_executor.get_stats())

        # Add inference executor stats
        inference_stats = {}
        if self.proc_pool.inference_executor:
            inference_stats = self.proc_pool.inference_executor.get_stats()

        stats["job_executors"] = job_stats
        stats["inference_executor"] = inference_stats

        return stats

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
