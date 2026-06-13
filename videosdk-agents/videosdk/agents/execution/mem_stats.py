from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.environ.get("VIDEOSDK_LOG_MEMORY", "1").strip().lower() not in ("", "0", "false", "no", "off")


def _read_linux_smaps(pid: int):
    keys = ("Rss:", "Pss:", "Private_Clean:", "Private_Dirty:",
            "Shared_Clean:", "Shared_Dirty:")
    vals = {}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                for k in keys:
                    if line.startswith(k):
                        vals[k] = int(line.split()[1])
                        break
    except (FileNotFoundError, PermissionError, ValueError, ProcessLookupError):
        return None
    mb = 1024.0
    rss = vals.get("Rss:", 0) / mb
    pss = vals.get("Pss:", 0) / mb
    private = (vals.get("Private_Clean:", 0) + vals.get("Private_Dirty:", 0)) / mb
    shared = (vals.get("Shared_Clean:", 0) + vals.get("Shared_Dirty:", 0)) / mb
    return rss, pss, private, shared


def _read_psutil(pid: int):
    try:
        import psutil

        p = psutil.Process(pid)
        try:
            mi = p.memory_full_info()
            return mi.rss / 1048576.0, getattr(mi, "uss", 0) / 1048576.0
        except Exception:
            return p.memory_info().rss / 1048576.0, None
    except Exception:
        return None


def log_resource_memory(resources, log=logger) -> None:
    pids = []
    for r in resources:
        proc = getattr(r, "process", None)
        pid = getattr(proc, "pid", None)
        if pid and getattr(proc, "is_alive", lambda: True)():
            pids.append((str(getattr(r, "resource_id", "?")), int(pid)))
    if not pids:
        return

    is_linux = sys.platform.startswith("linux")
    tot_rss = tot_pss = tot_priv = 0.0
    lines = []
    for rid, pid in pids:
        if is_linux:
            res = _read_linux_smaps(pid)
            if res is not None:
                rss, pss, priv, shared = res
                tot_rss += rss
                tot_pss += pss
                tot_priv += priv
                lines.append(f"    {rid[:18]:<18} pid={pid} rss={rss:.0f}M "
                             f"pss={pss:.0f}M private={priv:.0f}M shared={shared:.0f}M")
                continue
        pu = _read_psutil(pid)
        if pu is not None:
            rss, uss = pu
            tot_rss += rss
            if uss is not None:
                tot_priv += uss
            lines.append(f"    {rid[:18]:<18} pid={pid} rss={rss:.0f}M"
                         + (f" uss={uss:.0f}M" if uss is not None else " uss=n/a"))
    if not lines:
        return

    if is_linux:
        log.info("[mem] %d job proc(s): total RSS=%.0fM  PSS=%.0fM  unique=%.0fM  shared=%.0fM",
                 len(lines), tot_rss, tot_pss, tot_priv, tot_rss - tot_priv)
    else:
        log.info("[mem] %d job proc(s): total RSS=%.0fM  unique=%.0fM",
                 len(lines), tot_rss, tot_priv)
    for ln in lines:
        log.info(ln)
