import asyncio
import json
import logging
import os
import uuid
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from aiohttp import ClientWebSocketResponse

from .protocol import (
    AvailabilityRequest,
    JobAssignment,
    JobTermination,
    WorkerMessage,
    WorkerPong,
)

logger = logging.getLogger(__name__)


class BackendConnection:
    """Manages WebSocket connection to the backend registry server."""

    def __init__(
        self,
        auth_token: str,
        agent_id: str = "",
        worker_type: str = "room",
        version: str = "1.0.0",
        max_retry: int = 16,
        http_proxy: Optional[str] = None,
        backend_url: str = None,
        load_threshold: float = 0.75,
        max_processes: int = 10,
    ):
        self.auth_token = auth_token
        self.agent_id = agent_id
        self.worker_type = worker_type
        self.version = version
        self.max_retry = max_retry
        self.http_proxy = http_proxy
        self.backend_url = backend_url
        self.load_threshold = load_threshold
        self.max_processes = max_processes

        # Connection state
        self._closed = True
        self._connecting = False
        self._ws: Optional[ClientWebSocketResponse] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._worker_id = "unregistered"
        self._retry_count = 0

        # Message handling
        self._msg_queue: asyncio.Queue[WorkerMessage] = asyncio.Queue()
        self._pending_assignments: Dict[str, asyncio.Future[JobAssignment]] = {}

        # Callbacks
        self._on_availability: Optional[Callable[[AvailabilityRequest], None]] = None
        self._on_assignment: Optional[Callable[[JobAssignment], None]] = None
        self._on_termination: Optional[Callable[[JobTermination], None]] = None
        self._on_register: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._on_pong: Optional[Callable[[WorkerPong], None]] = None

        # Tasks
        self._connection_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._status_task: Optional[asyncio.Task] = None

    def _get_worker_id_file_path(self) -> str:
        """Get the path to the worker ID file."""
        # Use agent ID to create a unique file path
        safe_agent_id = "".join(
            c for c in self.agent_id if c.isalnum() or c in ("-", "_")
        ).rstrip()
        if not safe_agent_id:
            safe_agent_id = "default"

        # Create a directory for worker IDs if it doesn't exist
        worker_id_dir = os.path.expanduser("~/.videosdk-agents/worker-ids")
        os.makedirs(worker_id_dir, exist_ok=True)

        return os.path.join(worker_id_dir, f"{safe_agent_id}.worker_id")

    def _get_worker_id_env_key(self) -> str:
        """Get the environment variable key for worker ID."""
        safe_agent_id = "".join(
            c for c in self.agent_id if c.isalnum() or c in ("-", "_")
        ).rstrip()
        if not safe_agent_id:
            safe_agent_id = "default"
        return f"VIDEOSDK_WORKER_ID_{safe_agent_id.upper()}"

    def _load_memory_worker_id(self) -> Optional[str]:
        """Load worker ID from memory (environment variable only)."""
        env_key = self._get_worker_id_env_key()
        env_worker_id = os.environ.get(env_key)
        if env_worker_id and len(env_worker_id.strip()) > 0:
            logger.info(f"Loaded worker ID from memory: {env_worker_id}")
            return env_worker_id.strip()
        return None

    def _save_memory_worker_id(self, worker_id: str) -> None:
        """Save worker ID to memory only (environment variable)."""
        env_key = self._get_worker_id_env_key()
        os.environ[env_key] = worker_id
        logger.info(f"Saved worker ID to memory: {worker_id}")

    def _load_persistent_worker_id(self) -> Optional[str]:
        """Load worker ID from persistent storage (alias for memory-based method)."""
        return self._load_memory_worker_id()

    def _save_persistent_worker_id(self, worker_id: str) -> None:
        """Save worker ID to persistent storage (alias for memory-based method)."""
        self._save_memory_worker_id(worker_id)

    def _get_registry_assigned_worker_id(self) -> Optional[str]:
        """Get the worker ID that was previously assigned by the registry."""
        return self._load_memory_worker_id()

    def _generate_or_recover_worker_id(self) -> str:
        """Generate a new worker ID or recover existing one."""
        # Try to load existing worker ID
        existing_worker_id = self._load_persistent_worker_id()
        if existing_worker_id:
            logger.info(f"Using existing worker ID: {existing_worker_id}")
            return existing_worker_id

        # Generate new worker ID
        new_worker_id = str(uuid.uuid4())
        logger.info(f"Generated new worker ID: {new_worker_id}")

        # Save the new worker ID
        self._save_persistent_worker_id(new_worker_id)

        return new_worker_id

    @property
    def worker_id(self) -> str:
        """Get the worker ID assigned by the server."""
        return self._worker_id

    @property
    def is_connected(self) -> bool:
        """Check if connected to the backend."""
        return not self._closed and not self._connecting

    def on_availability(self, callback: Callable[[AvailabilityRequest], None]):
        """Set callback for availability requests."""
        self._on_availability = callback

    def on_assignment(self, callback: Callable[[JobAssignment], None]):
        """Set callback for job assignments."""
        self._on_assignment = callback

    def on_termination(self, callback: Callable[[JobTermination], None]):
        """Set callback for job terminations."""
        self._on_termination = callback

    def on_register(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Set callback for registration responses."""
        self._on_register = callback

    def on_pong(self, callback: Callable[[WorkerPong], None]):
        """Set callback for pong responses."""
        self._on_pong = callback

    async def connect(self):
        """Connect to the backend server."""
        if self._closed:
            self._closed = False
            self._connection_task = asyncio.create_task(self._connection_loop())

    async def disconnect(self):
        """Disconnect from backend server."""
        logger.info("Disconnecting from backend server")
        self._closed = True

        # Cancel connection task FIRST to prevent reconnection
        if self._connection_task and not self._connection_task.done():
            logger.info("Cancelling connection task to prevent reconnection")
            self._connection_task.cancel()
            try:
                await self._connection_task
                logger.info("Connection task cancelled successfully")
            except asyncio.CancelledError:
                logger.info("Connection task was cancelled as expected")
            except Exception as e:
                logger.error(f"Error cancelling connection task: {e}")
        else:
            logger.info("Connection task was already done or doesn't exist")

        # Send final status update to inform registry of shutdown
        if self._ws and not self._ws.closed:
            try:
                shutdown_msg = WorkerMessage(
                    type="status_update",
                    worker_id=self._worker_id,
                    status="offline",
                    load=0.0,
                    job_count=0,
                )
                await self._ws.send_str(json.dumps(shutdown_msg.dict()))
                logger.info(
                    f"Sent shutdown notification to registry for worker: {self._worker_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to send shutdown notification: {e}")

        # Close WebSocket connection properly
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")

        # Cancel other tasks
        for task in [
            self._send_task,
            self._recv_task,
            self._status_task,
        ]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close HTTP session if it exists
        if self._http_session:
            await self._http_session.close()

        logger.info("Backend disconnection complete")

    async def send_message(self, message: WorkerMessage):
        """Send a message to the backend."""
        if not self.is_connected:
            raise RuntimeError("Not connected to backend")

        await self._msg_queue.put(message)

    async def _connection_loop(self):
        """Main connection loop with retry logic."""
        logger.info("Connection loop started")
        while not self._closed:
            try:
                logger.debug("Attempting to establish connection")
                self._connecting = True
                await self._establish_connection()
                self._connecting = False
                self._retry_count = 0

                # Start message handling tasks
                self._send_task = asyncio.create_task(self._send_loop())
                self._recv_task = asyncio.create_task(self._recv_loop())
                self._status_task = asyncio.create_task(self._status_loop())

                # Wait for any task to complete (or fail)
                done, pending = await asyncio.wait(
                    [self._send_task, self._recv_task, self._status_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check if we should exit the loop
                if self._closed:
                    logger.info("Connection loop exiting due to shutdown")
                    break

            except asyncio.CancelledError:
                logger.info("Connection loop cancelled")
                break
            except Exception as e:
                if self._closed:
                    logger.info(
                        "Connection loop exiting due to shutdown during exception"
                    )
                    break

                if self._retry_count >= self.max_retry:
                    logger.error(
                        f"Failed to connect after {self._retry_count} attempts"
                    )
                    raise RuntimeError(
                        f"Failed to connect to backend after {self._retry_count} attempts"
                    ) from e

                retry_delay = min(self._retry_count * 2, 10)
                self._retry_count += 1

                logger.warning(f"Connection failed, retrying in {retry_delay}s: {e}")
                await asyncio.sleep(retry_delay)

    async def _establish_connection(self):
        """Establish connection to the backend registry server."""
        logger.debug("Establishing connection to backend")

        if not self._http_session:
            self._http_session = aiohttp.ClientSession()

        # Parse backend URL
        parse = urlparse(self.backend_url)
        scheme = parse.scheme or "wss"
        if scheme.startswith("http"):
            scheme = scheme.replace("http", "wss")

        base = f"{scheme}://{parse.netloc}{parse.path}".rstrip("/") + "/"
        agent_url = urljoin(base, "agent")

        # Connect to WebSocket
        headers = {"Authorization": f"Bearer {self.auth_token}"}

        logger.debug(f"Connecting to WebSocket: {agent_url}")
        self._ws = await self._http_session.ws_connect(
            agent_url,
            headers=headers,
            autoping=True,
            proxy=self.http_proxy or None,
        )
        logger.debug("WebSocket connection established")

        # Get previously assigned worker ID from registry (if any)
        worker_id = self._get_registry_assigned_worker_id()
        if worker_id:
            logger.info(f"Using previously assigned worker ID: {worker_id}")
        else:
            logger.info(
                "No previously assigned worker ID found, requesting new assignment from registry"
            )
            worker_id = ""  # Empty string tells registry to assign a new ID

        register_msg = WorkerMessage(
            type="register",
            worker_id=worker_id,  # Empty string for new assignment, existing ID for reconnection
            agent_name=self.agent_id,
            namespace="default",
            version=self.version,
            capabilities=["room", "voice", "stt", "tts"],
            registry_uuid="default",
            token=self.auth_token,
            # Add workload configuration
            load_threshold=self.load_threshold,
            max_processes=self.max_processes,
        )

        logger.debug(
            f"Sending registration message for worker: {worker_id or 'NEW_ASSIGNMENT'}"
        )
        logger.debug(f"Registration message: {register_msg.dict()}")
        logger.debug(f"Agent ID: '{self.agent_id}', Worker type: '{self.worker_type}'")

        await self._ws.send_str(json.dumps(register_msg.dict()))

        # Wait for registration response
        msg = await self._ws.receive()
        if msg.type == aiohttp.WSMsgType.TEXT:
            data = json.loads(msg.data)
            if data.get("type") == "register" and data.get("success"):
                assigned_worker_id = data.get("worker_id")
                self._worker_id = assigned_worker_id
                logger.info(f"Worker registered with backend: {self._worker_id}")

                # Store the assigned worker ID in memory for future use
                if assigned_worker_id and assigned_worker_id != worker_id:
                    logger.info(
                        f"Registry assigned new worker ID: {assigned_worker_id}"
                    )
                    self._save_memory_worker_id(assigned_worker_id)
                elif assigned_worker_id == worker_id:
                    logger.info(
                        f"Registry confirmed existing worker ID: {assigned_worker_id}"
                    )
                else:
                    logger.warning("Registry did not provide a worker ID")

                if self._on_register:
                    self._on_register(self._worker_id, data.get("payload", {}))
            else:
                raise RuntimeError(
                    f"Registration failed: {data.get('message', 'Unknown error')}"
                )
        else:
            raise RuntimeError("Unexpected message type during registration")

    async def _send_loop(self):
        """Send messages to the backend."""
        while not self._closed and self._ws:
            try:
                msg = await asyncio.wait_for(self._msg_queue.get(), timeout=1.0)
                await self._ws.send_str(json.dumps(msg.dict()))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                break

    async def _recv_loop(self):
        """Receive messages from the backend."""
        while not self._closed and self._ws:
            try:
                msg = await self._ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    logger.info("WebSocket connection closed")
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning(f"Unexpected message type: {msg.type}")
                    continue

                data = json.loads(msg.data)
                await self._handle_server_message(data)

            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break

    async def _handle_server_message(self, data: Dict[str, Any]):
        """Handle messages from the server."""
        msg_type = data.get("type")

        if msg_type == "availability_request":
            if self._on_availability:
                request = AvailabilityRequest(**data)
                self._on_availability(request)

        elif msg_type == "job_assignment":
            if self._on_assignment:
                assignment = JobAssignment(**data)
                self._on_assignment(assignment)

        elif msg_type == "job_termination":
            if self._on_termination:
                termination = JobTermination(**data)
                self._on_termination(termination)

        elif msg_type == "pong":
            if self._on_pong:
                pong = WorkerPong(**data)
                self._on_pong(pong)

        else:
            logger.warning(f"Unknown message type: {msg_type}")

    async def _status_loop(self):
        """Send periodic status updates."""
        # This method is deprecated - status updates are now handled by the worker
        # The worker sends status updates through the message queue
        while not self._closed:
            try:
                await asyncio.sleep(30)  # Just keep the loop alive
                # No status updates sent from here - worker handles them
            except Exception as e:
                logger.error(f"Error in status loop: {e}")
                break

    async def wait_for_assignment(
        self, job_id: str, timeout: float = 7.5
    ) -> JobAssignment:
        """Wait for a job assignment with timeout."""
        future = asyncio.Future()
        self._pending_assignments[job_id] = future

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_assignments.pop(job_id, None)
