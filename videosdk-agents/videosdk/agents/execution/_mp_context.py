"""Shared multiprocessing context for the SDK.

Avoids bare ``fork()`` on Linux. Plain ``fork()`` inherits any locks held by
background threads in the parent (logging ``Handler.lock``, ``multiprocessing``
``Queue`` feeder thread locks, threads spawned at import time by ML libraries)
and produces an intermittent deadlock where the child blocks forever on the
first acquire of an inherited lock.

``forkserver`` (Linux) and ``spawn`` (other platforms) both isolate the child
from parent thread state. Override via ``VIDEOSDK_MP_START_METHOD`` for rollout
or rollback.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from multiprocessing.context import BaseContext
from typing import Optional

logger = logging.getLogger(__name__)

_VALID_METHODS = ("fork", "spawn", "forkserver")
_cached_ctx: Optional[BaseContext] = None


def _default_start_method() -> str:
    return "spawn" if sys.platform == "win32" else "forkserver"


def get_mp_context() -> BaseContext:
    """Return a process-wide cached multiprocessing context.

    Honors ``VIDEOSDK_MP_START_METHOD`` (``fork``, ``spawn``, ``forkserver``).
    Falls back to ``forkserver`` on Linux, ``spawn`` elsewhere.
    """
    global _cached_ctx
    if _cached_ctx is not None:
        return _cached_ctx

    method = os.environ.get("VIDEOSDK_MP_START_METHOD", "").strip().lower()
    if method and method not in _VALID_METHODS:
        logger.warning(
            "Invalid VIDEOSDK_MP_START_METHOD=%r, falling back to default", method
        )
        method = ""
    if not method:
        method = _default_start_method()

    try:
        ctx = multiprocessing.get_context(method)
    except ValueError:
        fallback = "spawn"
        logger.warning(
            "Start method %r unavailable on this platform, falling back to %r",
            method,
            fallback,
        )
        ctx = multiprocessing.get_context(fallback)
        method = fallback

    logger.info("VideoSDK multiprocessing start method: %s", method)

    preload = os.environ.get("VIDEOSDK_PRELOAD_INFERENCE", "silero,turn_detector").strip()
    if method == "forkserver" and preload and preload.lower() not in ("none", "off", "0", "false"):
        try:
            ctx.set_forkserver_preload(["videosdk.agents._preload_inference"])
            logger.info("forkserver preload enabled for inference models: %s", preload)
        except Exception as e:  # pragma: no cover - platform/version dependent
            logger.warning("forkserver preload setup failed: %s", e)

    _cached_ctx = ctx
    return ctx


def reset_mp_context_cache() -> None:
    """Reset the cached context. Intended for tests only."""
    global _cached_ctx
    _cached_ctx = None
