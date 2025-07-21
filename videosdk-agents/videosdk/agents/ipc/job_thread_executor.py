import asyncio
import threading
import time
import logging
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class ThreadConfig:
    """Configuration for thread-based job execution."""

    initialize_timeout: float = 10.0
    close_timeout: float = 60.0
    ping_interval: float = 30.0
    high_ping_threshold: float = 5.0


class ThreadJobExecutor:
    """
    Thread-based job executor for VideoSDK agents.

    This executor runs jobs in threads instead of processes, which is useful for:
    - Windows compatibility (avoiding process creation issues)
    - Development and debugging (easier to debug threads)
    - Resource efficiency (lower memory overhead)
    """

    def __init__(
        self,
        *,
        initialize_timeout: float = 10.0,
        close_timeout: float = 60.0,
        ping_interval: float = 30.0,
        high_ping_threshold: float = 5.0,
        inference_executor: Optional[Any] = None,
    ):
        self.config = ThreadConfig(
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
            ping_interval=ping_interval,
            high_ping_threshold=high_ping_threshold,
        )

        self.inference_executor = inference_executor
        self._status = JobStatus.IDLE
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._current_job: Optional[Dict[str, Any]] = None
        self._shutdown = False
        self._lock = asyncio.Lock()
        self._id = f"thread_exec_{id(self)}"

        # Thread communication
        self._job_queue: asyncio.Queue = asyncio.Queue()
        self._result_queue: asyncio.Queue = asyncio.Queue()
        self._ping_queue: asyncio.Queue = asyncio.Queue()

        # Health monitoring
        self._last_ping_time = time.time()
        self._memory_usage = 0.0

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> JobStatus:
        return self._status

    @property
    def is_available(self) -> bool:
        return self._status == JobStatus.IDLE and not self._shutdown

    @property
    def current_job(self) -> Optional[Dict[str, Any]]:
        return self._current_job

    async def initialize(self):
        """Initialize the thread executor."""
        logger.info(f"Initializing thread executor {self.id}")

        # Store the main thread's event loop
        self._loop = asyncio.get_event_loop()

        # Create the worker thread
        self._thread = threading.Thread(
            target=self._thread_worker, name=f"job_thread_{self.id}", daemon=True
        )
        self._thread.start()

        # Wait for thread to be ready
        try:
            await asyncio.wait_for(
                self._wait_for_ready(), timeout=self.config.initialize_timeout
            )
            logger.info(f"Thread executor {self.id} initialized successfully")
        except asyncio.TimeoutError:
            logger.error(f"Thread executor {self.id} initialization timed out")
            raise

    async def _wait_for_ready(self):
        """Wait for the worker thread to be ready."""
        while True:
            try:
                message = await asyncio.wait_for(self._result_queue.get(), timeout=0.1)
                if message.get("type") == "ready":
                    return
            except asyncio.TimeoutError:
                continue

    def _thread_worker(self):
        """Worker function that runs in the thread."""
        try:
            # Set up thread-local event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create tasks after setting the event loop
            async def main():
                # Send ready signal
                await self._send_result({"type": "ready"})

                # Start ping task
                ping_task = asyncio.create_task(self._ping_worker())

                # Main job processing loop
                job_task = asyncio.create_task(self._job_worker())

                # Wait for either task to complete
                await asyncio.gather(ping_task, job_task, return_exceptions=True)

            # Run the main coroutine
            loop.run_until_complete(main())

        except Exception as e:
            logger.error(f"Error in thread worker: {e}")
            try:
                loop.run_until_complete(
                    self._send_result({"type": "error", "error": str(e)})
                )
            except:
                pass
        finally:
            if loop.is_running():
                loop.stop()

    async def _job_worker(self):
        """Process jobs in the thread."""
        while not self._shutdown:
            try:
                # Wait for job
                job_data = await asyncio.wait_for(self._job_queue.get(), timeout=1.0)

                if job_data.get("type") == "shutdown":
                    break
                elif job_data.get("type") == "ping_request":
                    # Handle ping request
                    await self._send_result(
                        {"type": "ping_response", "timestamp": time.time()}
                    )
                    continue

                # Execute job
                await self._execute_job(job_data)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in job worker: {e}")
                await self._send_result({"type": "error", "error": str(e)})

    async def _execute_job(self, job_data: Dict[str, Any]):
        """Execute a job in the thread."""
        try:
            job_id = job_data.get("job_id", "unknown")
            logger.info(f"Executing job {job_id} in thread {self.id}")

            # Update status
            self._status = JobStatus.RUNNING
            self._current_job = job_data

            # Handle different job types
            if job_data.get("type") == "launch_job":
                await self._execute_launch_job(job_data)
            else:
                await self._execute_regular_job(job_data)

        except Exception as e:
            logger.error(f"Error executing job {job_id}: {e}")
            await self._send_result({"type": "error", "error": str(e)})
        finally:
            # Reset status
            self._status = JobStatus.IDLE
            self._current_job = None

    async def _execute_launch_job(self, job_data: Dict[str, Any]):
        """Execute a launch job with running info."""
        running_info = job_data.get("running_info")
        if not running_info:
            raise ValueError("No running_info provided for launch_job")

        # Import here to avoid circular imports
        from videosdk.agents.job import (
            JobContext,
            _set_current_job_context,
            _reset_current_job_context,
        )

        # Create job context from running info
        room_options = running_info.job.get("room")
        if not room_options:
            raise ValueError("No room options in running info")

        job_context = JobContext(room_options=room_options)

        # Set current job context
        token = _set_current_job_context(job_context)

        try:
            # Execute the entrypoint function
            entrypoint_fnc = getattr(self, "_entrypoint_fnc", None)
            if entrypoint_fnc and callable(entrypoint_fnc):
                await entrypoint_fnc(job_context)
            else:
                logger.warning("No entrypoint function available for launch_job")
                # Fallback: just connect and wait
                await job_context.connect()
                await asyncio.sleep(1)  # Keep alive for a bit

        finally:
            # Cleanup
            try:
                await job_context.shutdown()
            except Exception as e:
                logger.error(f"Error during job shutdown: {e}")

            # Reset job context
            _reset_current_job_context(token)

            # Send success result
            await self._send_result(
                {
                    "type": "job_result",
                    "data": {"status": "completed"},
                    "job_id": job_data.get("job_id"),
                }
            )

    async def _execute_regular_job(self, job_data: Dict[str, Any]):
        """Execute a regular job with job data."""
        # This is for the old execute_job method
        result = {"status": "completed", "data": job_data}
        await self._send_result(
            {"type": "job_result", "data": result, "job_id": job_data.get("job_id")}
        )

    async def _ping_worker(self):
        """Send periodic pings to monitor thread health."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.ping_interval)

                # Send ping
                await self._send_result(
                    {
                        "type": "ping",
                        "timestamp": time.time(),
                        "memory_usage": self._get_memory_usage(),
                    }
                )

            except Exception as e:
                logger.error(f"Error in ping worker: {e}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    async def _send_result(self, result: Dict[str, Any]):
        """Send result to the main thread."""
        try:
            # Use call_soon_threadsafe to send to main thread's queue
            if self._loop:
                self._loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._result_queue.put(result))
                )
            else:
                # Fallback: put directly in the queue
                await self._result_queue.put(result)
        except Exception as e:
            logger.error(f"Error sending result: {e}")

    async def execute_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a job with the given data."""
        if self._status != JobStatus.IDLE:
            raise RuntimeError("Executor is not available")

        # Send job to thread
        await self._job_queue.put(job_data)

        # Wait for result
        while True:
            try:
                result = await asyncio.wait_for(self._result_queue.get(), timeout=30.0)
                if result.get("type") == "job_result":
                    return result.get("data", {})
                elif result.get("type") == "error":
                    raise Exception(result.get("error", "Unknown error"))
            except asyncio.TimeoutError:
                raise Exception("Job execution timed out")

    async def launch_job(self, running_info: "RunningJobInfo") -> None:
        """Launch a job with the given running info."""
        if self._status != JobStatus.IDLE:
            raise RuntimeError("Executor is not available")

        # Set current job info
        self._current_job_id = running_info.job.get("id", "unknown")

        # Create job data from running info
        job_data = {
            "type": "launch_job",
            "running_info": running_info,
            "job_id": self._current_job_id,
        }

        # Send job to thread
        await self._job_queue.put(job_data)

        # Update status
        self._status = JobStatus.RUNNING

    async def shutdown(self):
        """Shutdown the thread executor."""
        logger.info(f"Shutting down thread executor {self.id}")
        self._shutdown = True

        # Send shutdown signal to thread
        await self._job_queue.put({"type": "shutdown"})

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, self._thread.join),
                    timeout=self.config.close_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Thread executor {self.id} shutdown timed out")

        self._status = JobStatus.IDLE
        logger.info(f"Thread executor {self.id} shutdown complete")

    async def ping(self):
        """Send a ping to check if the thread executor is alive."""
        try:
            # Send a ping message to the thread
            await self._job_queue.put({"type": "ping_request"})

            # Wait for ping response
            start_time = time.time()
            while time.time() - start_time < 5.0:  # 5 second timeout
                try:
                    message = await asyncio.wait_for(
                        self._result_queue.get(), timeout=0.1
                    )
                    if message.get("type") == "ping_response":
                        return True
                except asyncio.TimeoutError:
                    continue

            return False
        except Exception as e:
            logger.error(f"Error pinging thread executor {self.id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the thread executor."""
        return {
            "id": self.id,
            "status": self.status.value,
            "is_available": self.is_available,
            "current_job": self.current_job,
            "last_ping_time": self._last_ping_time,
            "memory_usage_mb": self._memory_usage,
            "thread_alive": self._thread.is_alive() if self._thread else False,
            "executor_type": "thread",
        }

    def get_health_info(self) -> Dict[str, Any]:
        """Get health information about the thread executor."""
        return {
            "id": self.id,
            "status": self.status.value,
            "is_available": self.is_available,
            "current_job": self.current_job,
            "last_ping_time": self._last_ping_time,
            "memory_usage_mb": self._memory_usage,
            "thread_alive": self._thread.is_alive() if self._thread else False,
        }
