"""Process-wide bounded thread pool for CPU/ONNX inference offload.

Many agent sessions share one process in the gRPC runtime. A per-session
ThreadPoolExecutor (one thread each) does not scale — N sessions = N threads.
This module exposes a single bounded pool that inference offloads opt into, so
thread count stays bounded regardless of concurrent-session count.

Tunable via the VIDEOSDK_INFERENCE_THREADS env var. The shared ONNX
InferenceSessions themselves are cached elsewhere (plugin _session_cache); this
only governs the threads that run inference, not the models.
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

def _default_workers() -> int:
    env = os.getenv("VIDEOSDK_INFERENCE_THREADS")
    if env:
        try:
            n = int(env)
            if n > 0:
                return n
        except ValueError:
            pass
    cpu = os.cpu_count() or 4
    return max(2, min(cpu, 8))


# Eagerly constructed (no OS threads spawn until first submit) so concurrent
# first-use from multiple sessions can't race two executors into existence.
_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
    max_workers=_default_workers(),
    thread_name_prefix="vsdk-inference",
)


def get_inference_executor() -> ThreadPoolExecutor:
    """Return the process-wide inference executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_default_workers(),
            thread_name_prefix="vsdk-inference",
        )
    return _executor


async def run_inference(fn: Callable[..., Any], *args: Any) -> Any:
    """Run a blocking inference callable on the shared bounded pool.

    `fn` MUST be synchronous — `run_in_executor` would return an unawaited
    coroutine for an async callable, silently breaking the caller.
    """
    if asyncio.iscoroutinefunction(fn):
        raise TypeError(
            f"run_inference requires a synchronous callable, got coroutine function {fn!r}"
        )
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_inference_executor(), fn, *args)


def shutdown_inference_executor(wait: bool = False) -> None:
    """Shut the shared pool down. Call at PROCESS exit only — never per-session."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait)
        _executor = None
