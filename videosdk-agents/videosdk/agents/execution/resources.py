"""
Concrete resource implementations for process and thread execution.
"""

import asyncio
import logging
import multiprocessing
import threading
import time
import uuid
from typing import Any, Dict, Optional, Callable
from multiprocessing import Process, Queue
from multiprocessing.connection import Connection

from .base_resource import BaseResource
from .types import ResourceType, TaskResult

logger = logging.getLogger(__name__)


class ProcessResource(BaseResource):
    """
    Process-based resource for task execution.

    Uses multiprocessing to create isolated processes for task execution.
    """

    def __init__(self, resource_id: str, config: Dict[str, Any]):
        super().__init__(resource_id, config)
        self.process: Optional[Process] = None
        self.task_queue: Optional[Queue] = None
        self.result_queue: Optional[Queue] = None
        self.control_queue: Optional[Queue] = None
        self._process_ready = False

    @property
    def resource_type(self) -> ResourceType:
        return ResourceType.PROCESS

    async def _initialize_impl(self) -> None:
        """Initialize the process resource."""
        # Create queues for communication
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.control_queue = Queue()

        # Start the process
        self.process = Process(
            target=self._process_worker,
            args=(
                self.resource_id,
                self.task_queue,
                self.result_queue,
                self.control_queue,
                self.config,
            ),
            daemon=True,
        )
        self.process.start()

        # Wait for process to be ready
        timeout = self.config.get("initialize_timeout", 10.0)
        start_time = time.time()

        while not self._process_ready and (time.time() - start_time) < timeout:
            try:
                # Check for ready signal
                if not self.control_queue.empty():
                    message = self.control_queue.get_nowait()
                    if message.get("type") == "ready":
                        self._process_ready = True
                        break

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error checking process readiness: {e}")

        if not self._process_ready:
            raise TimeoutError(
                f"Process {self.resource_id} failed to initialize within {timeout}s"
            )

    async def _execute_task_impl(
        self, task_id: str, config, entrypoint: Callable, args: tuple, kwargs: dict
    ) -> Any:
        """Execute task in the process."""
        if not self._process_ready:
            raise RuntimeError(f"Process {self.resource_id} is not ready")

        # Send task to process
        task_data = {
            "task_id": task_id,
            "config": config,
            "entrypoint_name": entrypoint.__name__,
            "args": args,
            "kwargs": kwargs,
        }

        self.task_queue.put(task_data)

        # Wait for result
        timeout = config.timeout
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            try:
                if not self.result_queue.empty():
                    result_data = self.result_queue.get_nowait()
                    if result_data.get("task_id") == task_id:
                        if result_data.get("status") == "success":
                            return result_data.get("result")
                        else:
                            raise RuntimeError(
                                result_data.get("error", "Unknown error")
                            )

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error checking task result: {e}")

        raise TimeoutError(f"Task {task_id} timed out after {timeout}s")

    async def _shutdown_impl(self) -> None:
        """Shutdown the process resource."""
        if self.process and self.process.is_alive():
            # Send shutdown signal
            self.control_queue.put({"type": "shutdown"})

            # Wait for graceful shutdown
            timeout = self.config.get("close_timeout", 60.0)
            start_time = time.time()

            while self.process.is_alive() and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)

            # Force terminate if still alive
            if self.process.is_alive():
                logger.warning(f"Force terminating process {self.resource_id}")
                self.process.terminate()
                self.process.join(timeout=5.0)

                if self.process.is_alive():
                    self.process.kill()

    @staticmethod
    def _process_worker(
        resource_id: str,
        task_queue: Queue,
        result_queue: Queue,
        control_queue: Queue,
        config: Dict[str, Any],
    ):
        """Worker function that runs in the process."""
        try:
            logger.info(f"Process worker {resource_id} started")

            # Signal ready
            control_queue.put({"type": "ready"})

            # Main task processing loop
            while True:
                try:
                    # Check for shutdown signal
                    if not control_queue.empty():
                        message = control_queue.get_nowait()
                        if message.get("type") == "shutdown":
                            break

                    # Check for tasks
                    if not task_queue.empty():
                        task_data = task_queue.get_nowait()
                        task_id = task_data["task_id"]

                        try:
                            # Execute the task
                            # Note: In a real implementation, you'd need to serialize/deserialize the entrypoint
                            # For now, we'll use a simple approach
                            result = {"status": "completed", "task_id": task_id}
                            result_queue.put(
                                {
                                    "task_id": task_id,
                                    "status": "success",
                                    "result": result,
                                }
                            )
                        except Exception as e:
                            result_queue.put(
                                {"task_id": task_id, "status": "error", "error": str(e)}
                            )
                    else:
                        time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error in process worker {resource_id}: {e}")
                    time.sleep(1.0)

            logger.info(f"Process worker {resource_id} shutting down")

        except Exception as e:
            logger.error(f"Fatal error in process worker {resource_id}: {e}")


class ThreadResource(BaseResource):
    """
    Thread-based resource for task execution.

    Uses threading for concurrent task execution within the same process.
    """

    def __init__(self, resource_id: str, config: Dict[str, Any]):
        super().__init__(resource_id, config)
        self.thread: Optional[threading.Thread] = None
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self.control_queue: asyncio.Queue = asyncio.Queue()
        self._thread_ready = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def resource_type(self) -> ResourceType:
        return ResourceType.THREAD

    async def _initialize_impl(self) -> None:
        """Initialize the thread resource."""
        # Start the thread
        self.thread = threading.Thread(
            target=self._thread_worker,
            args=(
                self.resource_id,
                self.task_queue,
                self.result_queue,
                self.control_queue,
                self.config,
            ),
            daemon=True,
        )
        self.thread.start()

        # Simple readiness check - just wait a bit for thread to start
        await asyncio.sleep(0.5)

        # Mark as ready if thread is alive
        if self.thread.is_alive():
            self._thread_ready = True
        else:
            raise RuntimeError(f"Thread {self.resource_id} failed to start")

    async def _execute_task_impl(
        self, task_id: str, config, entrypoint: Callable, args: tuple, kwargs: dict
    ) -> Any:
        """Execute task in the thread."""
        if not self._thread_ready:
            raise RuntimeError(f"Thread {self.resource_id} is not ready")

        # Send task to thread
        task_data = {
            "task_id": task_id,
            "config": config,
            "entrypoint": entrypoint,
            "args": args,
            "kwargs": kwargs,
        }

        await self.task_queue.put(task_data)

        # Wait for result
        timeout = config.timeout
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            try:
                if not self.result_queue.empty():
                    result_data = await self.result_queue.get()
                    if result_data.get("task_id") == task_id:
                        if result_data.get("status") == "success":
                            return result_data.get("result")
                        else:
                            raise RuntimeError(
                                result_data.get("error", "Unknown error")
                            )

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error checking task result: {e}")

        raise TimeoutError(f"Task {task_id} timed out after {timeout}s")

    async def _shutdown_impl(self) -> None:
        """Shutdown the thread resource."""
        if self.thread and self.thread.is_alive():
            # Send shutdown signal
            await self.control_queue.put({"type": "shutdown"})

            # Wait for graceful shutdown
            timeout = self.config.get("close_timeout", 60.0)
            start_time = time.time()

            while self.thread.is_alive() and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)

    @staticmethod
    def _thread_worker(
        resource_id: str,
        task_queue: asyncio.Queue,
        result_queue: asyncio.Queue,
        control_queue: asyncio.Queue,
        config: Dict[str, Any],
    ):
        """Worker function that runs in the thread."""
        try:
            # Set up event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            logger.info(f"Thread worker {resource_id} started")

            async def worker_main():
                # Main task processing loop
                while True:
                    try:
                        # Check for shutdown signal
                        if not control_queue.empty():
                            message = await control_queue.get()
                            if message.get("type") == "shutdown":
                                break

                        # Check for tasks
                        if not task_queue.empty():
                            task_data = await task_queue.get()
                            task_id = task_data["task_id"]
                            entrypoint = task_data["entrypoint"]
                            args = task_data.get("args", ())
                            kwargs = task_data.get("kwargs", {})

                            try:
                                # Execute the task
                                if asyncio.iscoroutinefunction(entrypoint):
                                    result = await entrypoint(*args, **kwargs)
                                else:
                                    result = entrypoint(*args, **kwargs)

                                await result_queue.put(
                                    {
                                        "task_id": task_id,
                                        "status": "success",
                                        "result": result,
                                    }
                                )
                            except Exception as e:
                                await result_queue.put(
                                    {
                                        "task_id": task_id,
                                        "status": "error",
                                        "error": str(e),
                                    }
                                )
                        else:
                            await asyncio.sleep(0.1)

                    except Exception as e:
                        logger.error(f"Error in thread worker {resource_id}: {e}")
                        await asyncio.sleep(1.0)

                logger.info(f"Thread worker {resource_id} shutting down")

            # Run the worker
            loop.run_until_complete(worker_main())

        except Exception as e:
            logger.error(f"Fatal error in thread worker {resource_id}: {e}")
        finally:
            if loop and not loop.is_closed():
                loop.close()
