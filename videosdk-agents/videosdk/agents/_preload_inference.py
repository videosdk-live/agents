from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _warm_silero() -> None:
    from videosdk.plugins.silero import SileroVAD

    SileroVAD()


def _warm_turn_detector() -> None:
    from videosdk.plugins.turn_detector import TurnDetector

    TurnDetector()


_WARMERS = {
    "silero": _warm_silero,
    "turn_detector": _warm_turn_detector,
}


DEFAULT_TARGETS = "silero,turn_detector"


def preload(targets: str | None = None) -> list[str]:
    targets = (targets if targets is not None
               else os.environ.get("VIDEOSDK_PRELOAD_INFERENCE", DEFAULT_TARGETS)).strip()
    if not targets or targets.lower() in ("none", "off", "0", "false"):
        return []
    names = [t.strip() for t in targets.split(",") if t.strip()]
    if "all" in names:
        names = list(_WARMERS)
    warmed: list[str] = []
    for name in names:
        warm = _WARMERS.get(name)
        if warm is None:
            logger.warning("preload: unknown inference target %r", name)
            continue
        try:
            warm()
            warmed.append(name)
            logger.info("preload: warmed %s", name)
        except Exception as e:
            logger.warning("preload: failed to warm %s: %s", name, e)

    if warmed:
        try:
            import gc

            gc.freeze()
            logger.info("preload: gc.freeze() applied (%d objects frozen)", gc.get_freeze_count())
        except Exception as e:
            logger.debug("preload: gc.freeze() skipped: %s", e)
    return warmed


preload()
