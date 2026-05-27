from __future__ import annotations

import asyncio
import importlib
import logging
from typing import Iterable, Type

logger = logging.getLogger(__name__)

_REGISTRY: list[tuple[str, str]] = [
    ("videosdk.plugins.silero", "SileroVAD"),
]


async def auto_prewarm_installed_models() -> None:
    """Best-effort: download all registered plugin models that are installed.

    Each ``download_model()`` is a no-op when the cache file already exists,
    so this is cheap to call on every process startup. Downloads run
    concurrently via ``asyncio.gather`` so total time ≈ max(per-model).
    """
    classes = _resolve_registry()
    await _run_download_for_classes(classes)


async def prewarm_classes(classes: Iterable[Type]) -> None:
    """Run ``download_model()`` on each class in ``classes`` concurrently.

    Used when the user supplies an explicit ``prewarm_components`` list on
    :class:`videosdk.agents.Options`, bypassing the auto-discovery registry.
    """
    await _run_download_for_classes(classes)


def _resolve_registry() -> list[Type]:
    """Import each registry entry; return the resolved classes that exist."""
    resolved: list[Type] = []
    for module_path, class_name in _REGISTRY:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            continue
        cls = getattr(mod, class_name, None)
        if cls is None or not hasattr(cls, "download_model"):
            continue
        resolved.append(cls)
    return resolved


async def _run_download_for_classes(classes: Iterable[Type]) -> None:
    tasks = []
    for cls in classes:
        if not hasattr(cls, "download_model"):
            continue
        tasks.append(_safe_download(cls))
    if not tasks:
        return
    await asyncio.gather(*tasks, return_exceptions=True)


async def _safe_download(cls: Type) -> None:
    try:
        await cls.download_model()
    except Exception as e:
        logger.warning(f"Auto-prewarm of {cls.__name__} failed (non-fatal): {e}")
