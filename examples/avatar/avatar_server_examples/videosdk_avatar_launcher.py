"""
VideoSDK Avatar Dispatcher

A lightweight FastAPI service that receives launch requests from AvatarAudioOut
and spawns a videosdk_avatar_service.py process for each meeting room.

Start before running the agent:
    python examples/avatar/videosdk_avatar_launcher.py

Options:
    --host  (default 0.0.0.0)
    --port  (default 8089)
"""

import asyncio
import logging
import subprocess
import sys
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("avatar-dispatcher")
logging.basicConfig(level=logging.INFO)

THIS_DIR = Path(__file__).parent.absolute()
SERVICE_SCRIPT = THIS_DIR / "videosdk_avatar_service.py"


# ── request model (matches AvatarJoinInfo from avatar_auth.py) ────────────────

class LaunchRequest(BaseModel):
    room_name: str
    token: str
    participant_id: Optional[str] = "avatar_service"
    signaling_base_url: Optional[str] = None


# ── service lifecycle management ───────────────────────────────────────────────

@dataclass
class _ServiceInfo:
    room_name: str
    process: subprocess.Popen


class ServiceManager:
    def __init__(self) -> None:
        self._services: dict[str, _ServiceInfo] = {}
        self._monitor_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._monitor_task = asyncio.create_task(self._monitor())

    def close(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()
        for info in list(self._services.values()):
            info.process.terminate()
            try:
                info.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                info.process.kill()

    async def launch(self, req: LaunchRequest) -> None:
        # Replace any existing service for this room
        existing = self._services.pop(req.room_name, None)
        if existing:
            existing.process.terminate()
            try:
                existing.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                existing.process.kill()

        cmd = [
            sys.executable,
            str(SERVICE_SCRIPT),
            "--token", req.token,
            "--room-id", req.room_name,
            "--name", "AI Avatar",
            "--participant-id", req.participant_id or "avatar_service",
        ]
        if req.signaling_base_url:
            cmd += ["--signaling-url", req.signaling_base_url]

        try:
            process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
            self._services[req.room_name] = _ServiceInfo(
                room_name=req.room_name, process=process
            )
            logger.info("Launched avatar service for room: %s (pid=%d)", req.room_name, process.pid)
        except Exception as exc:
            logger.error("Failed to launch service: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    async def _monitor(self) -> None:
        while True:
            await asyncio.sleep(1)
            for room_name in list(self._services):
                info = self._services[room_name]
                if info.process.poll() is not None:
                    logger.info(
                        "Service for room %s exited (code=%d)",
                        room_name,
                        info.process.returncode,
                    )
                    self._services.pop(room_name, None)


# ── FastAPI app ────────────────────────────────────────────────────────────────

_manager = ServiceManager()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    await _manager.start()
    yield
    _manager.close()


app = FastAPI(title="VideoSDK Avatar Dispatcher", lifespan=_lifespan)


@app.post("/launch")
async def handle_launch(req: LaunchRequest) -> dict:
    await _manager.launch(req)
    return {"status": "success", "room": req.room_name}


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser(description="VideoSDK Avatar Dispatcher")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8089)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
