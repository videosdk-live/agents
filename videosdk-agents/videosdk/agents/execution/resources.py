"""
Concrete resource implementations for process and thread execution.
"""

import asyncio
import logging
import multiprocessing
import threading
import time
import uuid
from functools import partial
from queue import Empty as _QueueEmpty, Queue as _ThreadQueue
from typing import Any, Dict, Optional, Callable
from multiprocessing import Process, Queue
from multiprocessing.connection import Connection

from ._mp_context import get_mp_context
from .base_resource import BaseResource
from .types import ResourceType, TaskResult

# Inner chunk timeout for blocking IPC waits. The outer loop checks process
# health and the overall deadline every chunk; values smaller than ~250ms
# defeat the latency win, values larger than ~2s delay dead-process detection.
_IPC_POLL_CHUNK_SECONDS = 1.0

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
        mp_ctx = get_mp_context()

        # Create queues bound to the same context as the child process so
        # internal feeder threads / locks are not shared with the parent.
        self.task_queue = mp_ctx.Queue()
        self.result_queue = mp_ctx.Queue()
        self.control_queue = mp_ctx.Queue()

        # Start the process via the chosen context (spawn/forkserver — never
        # bare fork — to avoid inheriting held locks from parent threads).
        self.process = mp_ctx.Process(
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
        child_pid = self.process.pid
        logger.info(
            f"New process started | resource_id={self.resource_id} | pid={self.process.pid}"
        )

        # Wait for the child to post its "ready" sentinel. Blocks the executor
        # thread on the mp Queue, not the event loop, so the wake is immediate
        # rather than gated by a 100ms poll cadence.
        timeout = self.config.get("initialize_timeout", 10.0)
        deadline = time.time() + timeout
        loop = asyncio.get_running_loop()

        while time.time() < deadline:
            if not self.process.is_alive():
                raise RuntimeError(
                    f"Process {self.resource_id} exited during initialization "
                    f"(pid={self.process.pid}, exitcode={self.process.exitcode})"
                )

            chunk = min(_IPC_POLL_CHUNK_SECONDS, max(0.05, deadline - time.time()))
            try:
                message = await loop.run_in_executor(
                    None, partial(self.control_queue.get, True, chunk)
                )
            except _QueueEmpty:
                continue
            except Exception as e:
                logger.warning(f"Error reading control queue during init: {e}")
                continue

            if message.get("type") == "ready":
                self._process_ready = True
                break

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
        # Note: entrypoint and args must be picklable
        task_data = {
            "task_id": task_id,
            "config": config,
            "entrypoint": entrypoint,
            "args": args,
            "kwargs": kwargs,
        }

        self.task_queue.put(task_data)

        # Blocks the executor thread on the mp Queue rather than the asyncio
        # loop; the result wakes us immediately instead of within the next
        # 100ms poll tick. Outer loop keeps process-liveness and deadline
        # checks at ``_IPC_POLL_CHUNK_SECONDS`` granularity.
        timeout = config.timeout
        deadline = time.time() + timeout
        loop = asyncio.get_running_loop()

        while time.time() < deadline:
            if self.process and not self.process.is_alive():
                logger.info(
                    f"Process {self.resource_id} exited (pid={self.process.pid}), "
                    f"treating task {task_id} as completed"
                )
                try:
                    result_data = self.result_queue.get_nowait()
                except _QueueEmpty:
                    return None
                except Exception:
                    return None
                if result_data.get("status") == "success":
                    return result_data.get("result")
                return None

            chunk = min(_IPC_POLL_CHUNK_SECONDS, max(0.05, deadline - time.time()))
            try:
                result_data = await loop.run_in_executor(
                    None, partial(self.result_queue.get, True, chunk)
                )
            except _QueueEmpty:
                continue
            except Exception as e:
                logger.warning(f"Error reading result queue: {e}")
                continue

            if result_data.get("task_id") != task_id:
                # Stale result from a prior task — drop it loudly so this is
                # at least visible in logs. See audit finding #8 for the
                # proper fix (route by task_id via futures).
                logger.warning(
                    "Dropping stale result for task %s (expected %s) on resource %s",
                    result_data.get("task_id"),
                    task_id,
                    self.resource_id,
                )
                continue

            if result_data.get("status") == "success":
                return result_data.get("result")
            raise RuntimeError(result_data.get("error", "Unknown error"))

        # Total timeout exhausted. The child likely has an orphaned task
        # still running — terminate the process so it can't keep consuming
        # CPU/memory or post a result the next task would mis-consume. The
        # ProcessResource.health_check override will mark this resource
        # unhealthy on its next tick and the manager will respawn.
        self._terminate_on_timeout(task_id, timeout)
        raise TimeoutError(f"Task {task_id} timed out after {timeout}s")

    def _terminate_on_timeout(self, task_id: str, timeout: float) -> None:
        """Terminate the child so a wedged or orphaned task can't keep running."""
        # Best-effort cancel signal in case the child happens to be between
        # tasks and could honor it before SIGTERM lands. Non-blocking — if
        # the queue is full or the pipe is gone we still terminate.
        try:
            self.control_queue.put_nowait({"type": "cancel", "task_id": task_id})
        except Exception:
            pass

        if self.process and self.process.is_alive():
            logger.warning(
                "Terminating process %s after task %s timed out after %.1fs",
                self.resource_id,
                task_id,
                timeout,
            )
            try:
                self.process.terminate()
            except Exception as e:
                logger.warning(f"terminate() raised: {e}")
        self._process_ready = False


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

    async def health_check(self) -> bool:
        """Return False if the child process is gone.

        The base implementation only knows how to check threads; for processes
        a silently dead child would otherwise hold its slot in the pool forever
        and any job routed to it would hang at the result-queue poll.
        """
        if self._shutdown:
            return False
        if not self.process:
            return False
        if not self.process.is_alive():
            logger.warning(
                "Process %s is dead (pid=%s, exitcode=%s)",
                self.resource_id,
                self.process.pid,
                self.process.exitcode,
            )
            return False
        self.last_heartbeat = time.time()
        return True

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

            # Main task processing loop. Blocking gets — wakes immediately
            # when the parent posts a task instead of within the old
            # 100ms-poll-and-sleep cadence that capped per-task latency.
            while True:
                # Drain any control messages that arrived without blocking;
                # the actual block happens on task_queue.get below so we
                # don't burn CPU when idle.
                try:
                    message = control_queue.get_nowait()
                except _QueueEmpty:
                    pass
                else:
                    if message.get("type") == "shutdown":
                        break
                    # "cancel" messages currently have no in-task effect
                    # (the running asyncio.run can't be interrupted from
                    # outside); the parent does the real work via
                    # process.terminate(). See audit finding #9 for a
                    # cooperative-cancellation follow-up.

                try:
                    task_data = task_queue.get(
                        block=True, timeout=_IPC_POLL_CHUNK_SECONDS
                    )
                except _QueueEmpty:
                    continue
                except Exception as e:
                    logger.error(f"Error reading task queue in worker {resource_id}: {e}")
                    time.sleep(1.0)
                    continue

                task_id = task_data["task_id"]
                entrypoint = task_data["entrypoint"]
                args = task_data.get("args", ())
                kwargs = task_data.get("kwargs", {})

                logger.info(
                    f"Executing task {task_id} on resource {resource_id}"
                )

                try:
                    if asyncio.iscoroutinefunction(entrypoint):
                        # asyncio.run creates a fresh loop and cancels any
                        # remaining tasks on exit before closing the loop.
                        result = asyncio.run(entrypoint(*args, **kwargs))
                    else:
                        result = entrypoint(*args, **kwargs)
                    result_queue.put(
                        {
                            "task_id": task_id,
                            "status": "success",
                            "result": result,
                        }
                    )
                except Exception as e:
                    logger.error(f"Task execution failed: {e}")
                    result_queue.put(
                        {"task_id": task_id, "status": "error", "error": str(e)}
                    )

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
        # Use thread-safe stdlib queues so the parent's event loop and the
        # worker thread can communicate without the cross-loop ``asyncio.Queue``
        # antipattern (audit #3). The parent wraps blocking ``get`` in
        # ``run_in_executor``; ``put`` on an unbounded queue.Queue is
        # non-blocking and safe to call from any thread directly.
        self.task_queue: _ThreadQueue = _ThreadQueue()
        self.result_queue: _ThreadQueue = _ThreadQueue()
        self.control_queue: _ThreadQueue = _ThreadQueue()
        self._thread_ready = False
        self._ready_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def resource_type(self) -> ResourceType:
        return ResourceType.THREAD

    async def _initialize_impl(self) -> None:
        """Initialize the thread resource.

        Waits for the worker thread to signal it has actually entered its
        event loop, rather than the older "sleep 500ms and hope" probe which
        false-positived under CPU pressure (a thread is "alive" from the
        moment ``start()`` returns, before its target function runs).
        """
        self.thread = threading.Thread(
            target=self._thread_worker,
            args=(
                self.resource_id,
                self.task_queue,
                self.result_queue,
                self.control_queue,
                self.config,
                self._ready_event,
            ),
            daemon=True,
        )
        self.thread.start()

        timeout = float(self.config.get("initialize_timeout", 10.0))
        loop = asyncio.get_running_loop()
        ready = await loop.run_in_executor(None, self._ready_event.wait, timeout)

        if not ready:
            raise TimeoutError(
                f"Thread {self.resource_id} did not enter its worker loop within {timeout}s"
            )
        if not self.thread.is_alive():
            raise RuntimeError(
                f"Thread {self.resource_id} exited before signaling ready"
            )
        self._thread_ready = True

    async def _execute_task_impl(
        self, task_id: str, config, entrypoint: Callable, args: tuple, kwargs: dict
    ) -> Any:
        """Execute task in the thread."""
        if not self._thread_ready:
            raise RuntimeError(f"Thread {self.resource_id} is not ready")

        task_data = {
            "task_id": task_id,
            "config": config,
            "entrypoint": entrypoint,
            "args": args,
            "kwargs": kwargs,
        }

        # queue.Queue.put on an unbounded queue is non-blocking and
        # thread-safe — no executor needed.
        self.task_queue.put_nowait(task_data)

        deadline = time.time() + config.timeout
        loop = asyncio.get_running_loop()

        while time.time() < deadline:
            if self.thread and not self.thread.is_alive():
                raise RuntimeError(
                    f"Thread {self.resource_id} died during task {task_id}"
                )

            chunk = min(_IPC_POLL_CHUNK_SECONDS, max(0.05, deadline - time.time()))
            try:
                result_data = await loop.run_in_executor(
                    None, partial(self.result_queue.get, True, chunk)
                )
            except _QueueEmpty:
                continue
            except Exception as e:
                logger.warning(f"Error reading thread result queue: {e}")
                continue

            if result_data.get("task_id") != task_id:
                logger.warning(
                    "Dropping stale result for task %s (expected %s) on resource %s",
                    result_data.get("task_id"),
                    task_id,
                    self.resource_id,
                )
                continue
            if result_data.get("status") == "success":
                return result_data.get("result")
            raise RuntimeError(result_data.get("error", "Unknown error"))

        raise TimeoutError(f"Task {task_id} timed out after {config.timeout}s")

    async def _shutdown_impl(self) -> None:
        """Shutdown the thread resource."""
        if self.thread and self.thread.is_alive():
            self.control_queue.put_nowait({"type": "shutdown"})

            # Join the thread on an executor so we don't block the event loop.
            timeout = self.config.get("close_timeout", 60.0)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.thread.join, timeout)

    @staticmethod
    def _thread_worker(
        resource_id: str,
        task_queue: _ThreadQueue,
        result_queue: _ThreadQueue,
        control_queue: _ThreadQueue,
        config: Dict[str, Any],
        ready_event: threading.Event,
    ):
        """Worker function that runs in the thread.

        Synchronous loop on stdlib ``queue.Queue`` — no per-thread event loop.
        Coroutine entrypoints are run via ``asyncio.run`` per task (matches
        ``ProcessResource._process_worker``). This removes the cross-loop
        ``asyncio.Queue`` antipattern that previously made wakeups unreliable.
        """
        try:
            logger.info(f"Thread worker {resource_id} started")
            ready_event.set()

            while True:
                try:
                    message = control_queue.get(block=False)
                except _QueueEmpty:
                    pass
                else:
                    if message.get("type") == "shutdown":
                        break

                try:
                    task_data = task_queue.get(block=True, timeout=_IPC_POLL_CHUNK_SECONDS)
                except _QueueEmpty:
                    continue

                task_id = task_data["task_id"]
                entrypoint = task_data["entrypoint"]
                args = task_data.get("args", ())
                kwargs = task_data.get("kwargs", {})
                try:
                    if asyncio.iscoroutinefunction(entrypoint):
                        result = asyncio.run(entrypoint(*args, **kwargs))
                    else:
                        result = entrypoint(*args, **kwargs)
                    result_queue.put_nowait(
                        {
                            "task_id": task_id,
                            "status": "success",
                            "result": result,
                        }
                    )
                except Exception as e:
                    logger.error(f"Thread task execution failed: {e}")
                    result_queue.put_nowait(
                        {"task_id": task_id, "status": "error", "error": str(e)}
                    )

            logger.info(f"Thread worker {resource_id} shutting down")

        except Exception as e:
            logger.error(f"Fatal error in thread worker {resource_id}: {e}")
        finally:
            # Unblock the parent on any exit path. The parent uses
            # ``thread.is_alive()`` to distinguish "ready" from "exited
            # before ready" (see ``_initialize_impl``).
            ready_event.set()
