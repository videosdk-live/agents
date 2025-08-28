import asyncio
import os
import sys
import time
from typing import Any, Callable, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import signal

from .init_config import fetch_agent_init_config

# Updated imports to use new execution module
from .execution import (
    ExecutorType,
    ResourceType,
    ResourceConfig,
    TaskType,
    TaskExecutor,
)
from .job import JobContext, RoomOptions, JobAcceptArguments, RunningJobInfo
from .backend import (
    BackendConnection,
    WorkerMessage,
    AvailabilityRequest,
    JobAssignment,
    JobTermination,
    AvailabilityResponse,
    JobUpdate,
)
from .debug import HttpServer, Tracing

logger = logging.getLogger(__name__)


# Automatic platform-based defaults
if sys.platform.startswith("win"):
    # Some python versions on Windows gets a BrokenPipeError when creating a new process
    _default_executor_type = ExecutorType.THREAD
else:
    _default_executor_type = ExecutorType.PROCESS


class WorkerType(Enum):
    ROOM = "room"


@dataclass
class WorkerPermissions:
    """Permissions for the agent participant."""

    can_publish: bool = True
    can_subscribe: bool = True
    can_publish_data: bool = True
    can_update_metadata: bool = True
    hidden: bool = False


@dataclass
class WorkerOptions:
    """Configuration options for the VideoSDK worker."""

    entrypoint_fnc: Callable[[JobContext], Any]
    """Entrypoint function that will be called when a job is assigned to this worker."""

    request_fnc: Optional[Callable[["JobRequest"], Any]] = None
    """Function to handle job requests and decide whether to accept them."""

    initialize_process_fnc: Optional[Callable[[Any], Any]] = None
    """A function to perform any necessary initialization before the job starts."""

    executor_type: ExecutorType = _default_executor_type
    """Which executor to use to run jobs. Automatically selected based on platform."""

    num_idle_processes: int = 2
    """Number of idle processes/threads to keep warm."""

    initialize_timeout: float = 10.0
    """Maximum amount of time to wait for a process/thread to initialize/prewarm"""

    close_timeout: float = 60.0
    """Maximum amount of time to wait for a job to shut down gracefully"""

    memory_warn_mb: float = 500.0
    """Memory warning threshold in MB."""

    memory_limit_mb: float = 0.0
    """Maximum memory usage for a job in MB. Defaults to 0 (disabled)."""

    ping_interval: float = 30.0
    """Interval between health check pings."""

    max_processes: int = 10
    """Maximum number of processes/threads."""

    agent_id: str = "VideoSDKAgent"
    """ID of the agent."""

    auth_token: Optional[str] = None
    """VideoSDK authentication token. Uses VIDEOSDK_AUTH_TOKEN env var if not provided. 
    This token is used for both VideoSDK services and registry authentication."""

    worker_type: WorkerType = WorkerType.ROOM
    """Type of worker (room or publisher)."""

    permissions: WorkerPermissions = field(default_factory=WorkerPermissions)
    """Permissions for the agent participant."""

    max_retry: int = 16
    """Maximum number of times to retry connecting to VideoSDK."""

    load_threshold: float = 0.75
    """Load threshold above which worker is marked as unavailable."""

    register: bool = False
    """Whether to register with the backend. Defaults to False for local development."""

    signaling_base_url: str = "api.videosdk.live"
    """Signaling base URL for VideoSDK services. Defaults to api.videosdk.live."""

    host: str = "0.0.0.0"
    """Host for the debug HTTP server."""

    port: int = 8081
    """Port for the debug HTTP server."""

    log_level: str = "INFO"
    """Log level for SDK logging. Options: DEBUG, INFO, WARNING, ERROR. Defaults to INFO."""

    def __post_init__(self):
        """Post-initialization setup."""
        if not self.auth_token:
            self.auth_token = os.getenv("VIDEOSDK_AUTH_TOKEN")

        # Log the selected executor type
        logger.info(f"Worker configured with {self.executor_type.value} executor")


@dataclass
class JobRequest:
    """Job request from the backend."""

    job: Any
    on_reject: Callable[[], Any]
    on_accept: Callable[[JobAcceptArguments], Any]


@dataclass
class DirectRoomOptions:
    """Information for direct room joining without backend registration."""

    room_id: str
    participant_identity: Optional[str] = None
    room_name: Optional[str] = None
    auth_token: Optional[str] = None


class Worker:
    """
    VideoSDK worker that manages job execution and backend registration.

    def run(self):
        job_context = functools.partial(self.job.jobctx)
        entrypoint = functools.partial(self.job.entrypoint)
        p = multiprocessing.Process(
            target=_job_runner, args=(entrypoint, job_context)
    Automatically selects the appropriate executor type based on platform.
    """

    def __init__(self, options: WorkerOptions):
        """Initialize the worker."""
        self.options = options

        self._shutdown = False
        self._draining = False
        self._worker_load = 0.0
        self._current_jobs: Dict[str, RunningJobInfo] = {}
        self._tasks: Set[asyncio.Task] = set()
        self.backend_connection: Optional[BackendConnection] = None
        self.process_manager: Optional[TaskExecutor] = (
            None  # Changed from ProcessManager
        )
        self._http_server: Optional[HttpServer] = None

        # Add debounce mechanism for status updates
        self._last_status_update = 0.0
        self._status_update_debounce_seconds = (
            2.0  # Minimum 2 seconds between status updates
        )
        # Initialize tracing
        self._tracing = Tracing.with_handle("worker")
        self._worker_load_graph = Tracing.add_graph(
            title="worker_load",
            x_label="time",
            y_label="load",
            x_type="time",
            y_range=(0, 1),
            max_data_points=1000,
        )

        # Validate configuration
        if not self.options.auth_token:
            raise ValueError(
                "auth_token is required, or add VIDEOSDK_AUTH_TOKEN in your environment"
            )

    @staticmethod
    def run_worker(
        options: WorkerOptions, simulate_job: Optional[DirectRoomOptions | str] = None
    ):
        """
        Run a VideoSDK worker with the given options.

        This is the main entry point for running a VideoSDK worker,
        providing a high-level interface for worker initialization, job management, and lifecycle control.

        Args:
            options: Worker configuration options
            simulate_job: Optional job simulation for direct room joining

        Example:
            ```python
            from videosdk.agents import Worker, WorkerOptions

            def my_agent(job_ctx):
                # Your agent code here
                pass

            # Configure worker with custom log level - logging is automatically configured!
            options = WorkerOptions(
                entrypoint_fnc=my_agent,
                log_level="DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR
            )

            # Run the worker - no manual logging setup needed!
            Worker.run_worker(options)
            ```
        """
        worker = Worker(options)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def main_task():
            try:
                await worker.initialize()

                if simulate_job:
                    # Direct room joining mode - force register=False for simulation
                    logger.info("Simulation mode: forcing register=False")
                    worker.options.register = False
                    await worker.simulate_job(simulate_job)
                elif options.register:
                    # Backend registration mode
                    await worker._run_backend_mode()
                else:
                    # Default mode - just keep alive
                    while not worker._shutdown:
                        await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("Main task cancelled")
            except Exception as e:
                logger.error(f"Worker error: {e}")
                raise
            finally:
                await worker.shutdown()

        main_future = loop.create_task(main_task())
        shutting_down = False

        def signal_handler(signum, frame):
            nonlocal shutting_down
            if shutting_down:
                # If already shutting down, cancel all tasks more aggressively
                for task in asyncio.all_tasks(loop):
                    task.cancel()
                return
            shutting_down = True
            logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
            # Cancel the main task
            loop.call_soon_threadsafe(main_future.cancel)
            # Set a timeout for graceful shutdown
            loop.call_later(3.0, lambda: [task.cancel() for task in asyncio.all_tasks(loop)])

        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            loop.run_until_complete(main_future)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            if not shutting_down:
                shutting_down = True
                if not main_future.done():
                    main_future.cancel()
                loop.run_until_complete(worker.shutdown())
        finally:
            try:
                loop.close()
            except Exception as e:
                logger.error(f"Error closing event loop: {e}")
            
        if loop.is_closed():
            logger.info("Event loop closed successfully")

    async def initialize(self):
        """Initialize the worker."""
        logger.info("Initializing VideoSDK worker")

        # Initialize task executor with new execution architecture
        # Convert ExecutorType to ResourceType
        resource_type = (
            ResourceType.THREAD
            if self.options.executor_type == ExecutorType.THREAD
            else ResourceType.PROCESS
        )

        config = ResourceConfig(
            resource_type=resource_type,
            num_idle_resources=self.options.num_idle_processes,
            max_resources=self.options.max_processes,
            initialize_timeout=self.options.initialize_timeout,
            close_timeout=self.options.close_timeout,
            memory_warn_mb=self.options.memory_warn_mb,
            memory_limit_mb=self.options.memory_limit_mb,
            ping_interval=self.options.ping_interval,
            load_threshold=self.options.load_threshold,
            max_concurrent_tasks=1,  # Each resource handles one task at a time
            executor_type=self.options.executor_type,
            # Legacy IPC compatibility - dedicated inference process
            use_dedicated_inference_process=False,  # Disable dedicated inference process for now
            inference_process_timeout=30.0,  # Longer timeout for AI model loading
            inference_memory_warn_mb=1000.0,  # Higher threshold for AI models
        )

        self.process_manager = TaskExecutor(config)
        await self.process_manager.start()

        # Initialize backend connection if registering
        if self.options.register:
            await self._initialize_backend_connection()

        # Initialize and start debug HTTP server
        self._http_server = HttpServer(
            host=self.options.host,
            port=self.options.port,
        )
        self._http_server.set_worker(self)
        await self._http_server.start()

        logger.info("VideoSDK worker initialized successfully")

    async def _initialize_backend_connection(self):
        """Initialize connection to the backend registry."""
        if not self.options.register:
            return

        # Fetch agent init config to get registry URL
        try:
            logger.info("Fetching agent init config...")
            registry_url = await fetch_agent_init_config(
                auth_token=self.options.auth_token,
                api_base_url=f"https://{self.options.signaling_base_url}",
            )
            logger.info(f"Using registry URL: {registry_url}")
        except Exception as e:
            logger.error(f"Failed to fetch agent init config: {e}")
            raise RuntimeError(f"Agent init config is mandatory. Error: {e}")

        self.backend_connection = BackendConnection(
            auth_token=self.options.auth_token,
            agent_id=self.options.agent_id,
            worker_type=self.options.worker_type.value,
            version="1.0.0",
            max_retry=self.options.max_retry,
            backend_url=registry_url,
            load_threshold=self.options.load_threshold,
            max_processes=self.options.max_processes,
        )

        # Set up message handlers
        self.backend_connection.on_register(self._handle_register)
        self.backend_connection.on_availability(self._handle_availability)
        self.backend_connection.on_assignment(self._handle_assignment)
        self.backend_connection.on_termination(self._handle_termination)

        # Connect to backend
        await self.backend_connection.connect()

    async def _run_backend_mode(self):
        """Run the worker in backend registration mode."""
        logger.info("Running in backend registration mode")

        # Start status update loop
        status_task = asyncio.create_task(self._status_update_loop())
        self._tasks.add(status_task)

        try:
            # Keep the worker running
            while not self._shutdown:
                await asyncio.sleep(1)
        finally:
            status_task.cancel()
            self._tasks.discard(status_task)

    def _handle_register(self, worker_id: str, server_info: Dict[str, Any]):
        """Handle registration response from backend."""
        logger.info(f"Registered with backend: {worker_id}")
        logger.info(f"Server info: {server_info}")

    def _handle_availability(self, request: AvailabilityRequest):
        """Handle availability request from backend."""
        logger.info(f"Received availability request for job {request.job_id}")
        asyncio.create_task(self._answer_availability(request))

    async def _answer_availability(self, request: AvailabilityRequest):
        """Answer availability request."""
        try:
            # Check if we can accept the job
            can_accept = (
                not self._draining
                and self._worker_load < self.options.load_threshold
                and len(self._current_jobs) < self.options.max_processes
            )

            if can_accept:
                # Accept the job and provide our auth token
                response = AvailabilityResponse(
                    job_id=request.job_id,
                    available=True,
                    token=self.options.auth_token,  # Provide worker's auth token
                )
                logger.info(f"Accepting job {request.job_id}")
            else:
                # Reject the job
                response = AvailabilityResponse(
                    job_id=request.job_id,
                    available=False,
                    error="Worker at capacity or draining",
                )
                logger.info(f"Rejecting job {request.job_id}")

            # Send response
            await self.backend_connection.send_message(response)

        except Exception as e:
            logger.error(f"Error handling availability request: {e}")
            # Send rejection on error
            response = AvailabilityResponse(
                job_id=request.job_id,
                available=False,
                error=str(e),
            )
            await self.backend_connection.send_message(response)

    def _handle_assignment(self, assignment: JobAssignment):
        """Handle job assignment from backend."""
        logger.info(f"Received job assignment: {assignment.job_id}")
        asyncio.create_task(self._handle_job_assignment(assignment))

    async def _handle_job_assignment(self, assignment: JobAssignment):
        """Handle job assignment."""
        try:
            # Create job accept arguments
            args = JobAcceptArguments(
                identity=f"agent_{assignment.job_id}",
                name=self.options.agent_id,
                metadata="",
            )

            # Launch the job
            await self._launch_job_from_assignment(assignment, args)

        except Exception as e:
            logger.error(f"Error handling job assignment: {e}")
            # Send job update with error
            job_update = JobUpdate(
                job_id=assignment.job_id,
                status="failed",
                error=str(e),
            )
            await self.backend_connection.send_message(job_update)

    async def _handle_termination(self, termination: JobTermination):
        """Handle job termination request."""
        logger.info(f"Received job termination: {termination.job_id}")

        if termination.job_id in self._current_jobs:
            job_info = self._current_jobs[termination.job_id]

            try:
                await job_info.job.shutdown()
                logger.info(f"Successfully terminated job {termination.job_id}")
            except Exception as e:
                logger.error(f"Error terminating job {termination.job_id}: {e}")

            # Remove job from current jobs
            del self._current_jobs[termination.job_id]
            logger.info(
                f"Removed job {termination.job_id} from current jobs. Remaining jobs: {len(self._current_jobs)}"
            )

            # Notify registry about job completion
            if self.backend_connection and self.backend_connection.is_connected:
                try:
                    job_update = JobUpdate(
                        job_id=termination.job_id,
                        status="completed",
                        error="Job terminated by registry",
                    )
                    await self.backend_connection.send_message(job_update)
                    logger.info(
                        f"Sent job completion update for terminated job {termination.job_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to send job completion update for terminated job {termination.job_id}: {e}"
                    )

            # IMMEDIATELY send status update to reflect reduced job count
            # This bypasses the debounce mechanism to ensure registry gets correct info
            await self._send_immediate_status_update()
        else:
            logger.warning(
                f"Job {termination.job_id} not found in current jobs for termination"
            )

    async def _handle_meeting_end(self, job_id: str, reason: str = "meeting_ended"):
        """Handle meeting end/leave events and inform registry."""
        logger.info(f"Meeting ended for job {job_id}, reason: {reason}")
        logger.info(
            f"Checking if job {job_id} is in current_jobs: {job_id in self._current_jobs}"
        )
        logger.info(f"Current jobs: {list(self._current_jobs.keys())}")

        if job_id in self._current_jobs:
            # Remove job from worker's current jobs
            job_info = self._current_jobs.pop(job_id, None)
            if job_info:
                logger.info(
                    f"Removed job {job_id} from worker's current jobs. Remaining jobs: {len(self._current_jobs)}"
                )

            # Inform registry about job completion
            if self.backend_connection and self.backend_connection.is_connected:
                try:
                    job_update = JobUpdate(
                        job_id=job_id,
                        status="completed",
                        error=f"Meeting ended: {reason}",
                    )
                    await self.backend_connection.send_message(job_update)
                    logger.info(
                        f"Sent job completion update to registry for job {job_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to send job completion update to registry: {e}"
                    )

            # IMMEDIATELY send status update to reflect reduced job count
            # This bypasses the debounce mechanism to ensure registry gets correct info
            await self._send_immediate_status_update()
        else:
            logger.warning(f"Job {job_id} not found in current jobs when meeting ended")

    async def _send_immediate_status_update(self):
        """Send an immediate status update, bypassing debounce mechanism."""
        if not self.backend_connection or not self.backend_connection.is_connected:
            return

        try:
            # Calculate current load
            job_count = len(self._current_jobs)
            load = min(job_count / self.options.max_processes, 1.0)
            self._worker_load = load

            logger.info(
                f"Sending immediate status update - job_count: {job_count}, load: {load}, max_processes: {self.options.max_processes}"
            )

            # Log the actual job IDs for debugging
            if job_count > 0:
                job_ids = list(self._current_jobs.keys())
                logger.info(f"Active job IDs: {job_ids}")
            else:
                logger.info("No active jobs")

            # Send status update
            status_msg = WorkerMessage(
                type="status_update",
                worker_id=self.backend_connection.worker_id,
                agent_name=self.options.agent_id,
                status="available" if not self._draining else "draining",
                load=load,
                job_count=job_count,
            )

            await self.backend_connection.send_message(status_msg)
            logger.info("Immediate status update sent successfully")

        except Exception as e:
            logger.error(f"Error sending immediate status update: {e}")

    def setup_meeting_event_handlers(self, job_context, job_id: str):
        """Set up meeting event handlers for a specific job."""
        if not job_context.room:
            logger.warning(
                f"Cannot set up meeting handlers for job {job_id}: room not available"
            )
            # Set up a delayed handler setup that will be called when room becomes available
            original_connect = job_context.connect

            def delayed_handler_setup():
                if job_context.room:
                    self._setup_meeting_event_handlers_impl(job_context, job_id)
                else:
                    logger.warning(
                        f"Room still not available for job {job_id} after connect"
                    )

            # Override connect method to set up handlers after room is created
            async def connect_with_handlers():
                result = await original_connect()
                delayed_handler_setup()
                return result

            job_context.connect = connect_with_handlers
            logger.info(f"Set up delayed meeting event handlers for job {job_id}")
            return

        # Room is available, set up handlers immediately
        self._setup_meeting_event_handlers_impl(job_context, job_id)

    def _setup_meeting_event_handlers_impl(self, job_context, job_id: str):
        """Internal method to set up the actual meeting event handlers."""
        if not job_context.room:
            logger.warning(f"Room not available for job {job_id} in handler setup")
            return

        # Store original event handlers
        original_on_meeting_left = job_context.room.on_meeting_left

        # Override the on_meeting_left handler
        def on_meeting_left_wrapper(data=None):
            # Call original handler with proper parameter handling
            if original_on_meeting_left:
                try:
                    # Check if original handler expects data parameter
                    import inspect

                    sig = inspect.signature(original_on_meeting_left)
                    if len(sig.parameters) > 0:
                        original_on_meeting_left(data)
                    else:
                        original_on_meeting_left()
                except Exception as e:
                    logger.warning(
                        f"Error calling original on_meeting_left handler: {e}"
                    )

            # Handle meeting end for this specific job
            asyncio.create_task(self._handle_meeting_end(job_id, "meeting_left"))

        # Set the new handler
        job_context.room.on_meeting_left = on_meeting_left_wrapper

        logger.info(f"Set up meeting end handler for job {job_id}")

    async def _launch_job_from_assignment(
        self, assignment: JobAssignment, args: JobAcceptArguments
    ):
        """Launch a job from backend assignment."""
        try:
            # Use assignment token if available, otherwise fall back to worker's auth token
            auth_token = (
                assignment.token if assignment.token else self.options.auth_token
            )

            # Create room options from assignment (this was already done in _handle_job_assignment)
            room_options = RoomOptions(
                room_id=assignment.room_id,
                name=assignment.room_name,  # Use 'name' instead of 'room_name'
                auth_token=auth_token,
                signaling_base_url=self.options.signaling_base_url,
            )

            # Apply RoomOptions from assignment if provided
            if assignment.room_options:
                logger.info(
                    f"Received room_options from assignment: {assignment.room_options}"
                )
                if "auto_end_session" in assignment.room_options:
                    room_options.auto_end_session = assignment.room_options[
                        "auto_end_session"
                    ]
                    logger.info(
                        f"Set auto_end_session: {room_options.auto_end_session}"
                    )
                if "session_timeout_seconds" in assignment.room_options:
                    room_options.session_timeout_seconds = assignment.room_options[
                        "session_timeout_seconds"
                    ]
                    logger.info(
                        f"Set session_timeout_seconds: {room_options.session_timeout_seconds}"
                    )
                if "playground" in assignment.room_options:
                    room_options.playground = assignment.room_options["playground"]
                    logger.info(f"Set playground: {room_options.playground}")
                if "vision" in assignment.room_options:
                    room_options.vision = assignment.room_options["vision"]
                    logger.info(f"Set vision: {room_options.vision}")
                if "join_meeting" in assignment.room_options:
                    room_options.join_meeting = assignment.room_options["join_meeting"]
                    logger.info(f"Set join_meeting: {room_options.join_meeting}")
            else:
                logger.warning("No room_options received from assignment")

            # Create job context
            job_context = JobContext(
                room_options=room_options,
            )

            # Create running job info with correct parameters
            job_info = RunningJobInfo(
                accept_arguments=args,
                job=job_context,
                url=assignment.url,
                token=auth_token,
                worker_id=self.backend_connection.worker_id,
            )

            # Store job info BEFORE executing entrypoint
            self._current_jobs[assignment.job_id] = job_info
            logger.info(
                f"Added job {assignment.job_id} to worker's current jobs. Total jobs: {len(self._current_jobs)}"
            )

            # Send job update to registry
            job_update = JobUpdate(
                job_id=assignment.job_id,
                status="running",
            )
            await self.backend_connection.send_message(job_update)

            # Set up session end callback BEFORE executing entrypoint
            # This ensures the callback is set up even if entrypoint fails
            self.setup_session_end_callback(job_context, assignment.job_id)
            logger.info(f"Session end callback set up for job {assignment.job_id}")

            # Set up meeting event handlers to ensure proper event handling
            self.setup_meeting_event_handlers(job_context, assignment.job_id)
            logger.info(f"Meeting event handlers set up for job {assignment.job_id}")

            # Execute the job using the worker's entrypoint function
            logger.info(f"Executing job {assignment.job_id} with entrypoint function")

            try:
                # Set the current job context so pipeline auto-registration works
                from .job import _set_current_job_context, _reset_current_job_context

                token = _set_current_job_context(job_context)
                try:
                    # Execute the entrypoint function
                    await self.options.entrypoint_fnc(job_context)
                    logger.info(
                        f"Entrypoint function completed for job {assignment.job_id}"
                    )
                finally:
                    pass
            except Exception as entrypoint_error:
                logger.error(
                    f"Entrypoint function failed for job {assignment.job_id}: {entrypoint_error}"
                )
                # Don't remove the job from _current_jobs here - let the session end callback handle it
                # The job should remain active until the session actually ends

                # Send error update but keep job active
                error_update = JobUpdate(
                    job_id=assignment.job_id,
                    status="error",
                    error=f"Entrypoint failed: {entrypoint_error}",
                )
                await self.backend_connection.send_message(error_update)

            # The job should remain in _current_jobs until the session ends
            # This ensures the registry sees the correct load and job count
            logger.info(
                f"Job {assignment.job_id} remains active in worker's current jobs: {len(self._current_jobs)} total jobs"
            )

        except Exception as e:
            logger.error(f"Error launching job {assignment.job_id}: {e}")
            # Send error update
            job_update = JobUpdate(
                job_id=assignment.job_id,
                status="failed",
                error=str(e),
            )
            await self.backend_connection.send_message(job_update)
            # Remove job from current jobs since it failed to launch
            self._current_jobs.pop(assignment.job_id, None)
            logger.info(f"Removed failed job {assignment.job_id} from current jobs")

            # Send immediate status update to reflect reduced job count
            await self._send_immediate_status_update()

    def setup_session_end_callback(self, job_context, job_id: str):
        """Set up session end callback for automatic session ending."""
        if not job_context.room:
            logger.warning(
                f"Cannot set up session end callback for job {job_id}: room not available"
            )
            # Set up a delayed callback setup that will be called when room becomes available
            original_connect = job_context.connect

            def delayed_callback_setup():
                if job_context.room:
                    self._setup_session_end_callback_impl(job_context, job_id)
                else:
                    logger.warning(
                        f"Room still not available for job {job_id} after connect"
                    )

            # Override connect method to set up callback after room is created
            async def connect_with_callback():
                result = await original_connect()
                delayed_callback_setup()
                return result

            job_context.connect = connect_with_callback
            logger.info(f"Set up delayed session end callback for job {job_id}")
            return

        # Room is available, set up callback immediately
        self._setup_session_end_callback_impl(job_context, job_id)

    def _setup_session_end_callback_impl(self, job_context, job_id: str):
        """Internal method to set up the actual session end callback."""
        if not job_context.room:
            logger.warning(f"Room not available for job {job_id} in callback setup")
            return

        def on_session_end(reason: str):
            logger.info(f"Session ended for job {job_id}, reason: {reason}")
            logger.info(f"Calling _handle_meeting_end for job {job_id}")
            # Handle session end asynchronously
            asyncio.create_task(
                self._handle_meeting_end(job_id, f"session_ended: {reason}")
            )

        # Set the session end callback
        job_context.room.on_session_end = on_session_end
        logger.info(f"Session end callback set up for job {job_id}")

    async def _status_update_loop(self):
        """Periodic status update loop."""
        while not self._shutdown:
            try:
                await self._update_worker_status()
                await asyncio.sleep(self.options.ping_interval)
            except Exception as e:
                logger.error(f"Error in status update loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _update_worker_status(self):
        """Update worker status with backend."""
        if not self.backend_connection or not self.backend_connection.is_connected:
            return

        # Check debounce - don't send status updates too frequently
        current_time = time.time()
        if (
            current_time - self._last_status_update
            < self._status_update_debounce_seconds
        ):
            logger.debug("Skipping status update due to debounce")
            return

        try:
            # Calculate current load
            job_count = len(self._current_jobs)
            load = min(job_count / self.options.max_processes, 1.0)
            self._worker_load = load

            # Add detailed logging to track job count changes
            logger.info(
                f"Updating worker status - job_count: {job_count}, load: {load}, max_processes: {self.options.max_processes}"
            )

            # Log the actual job IDs for debugging
            if job_count > 0:
                job_ids = list(self._current_jobs.keys())
                logger.info(f"Active job IDs: {job_ids}")
            else:
                logger.info("No active jobs")

            # Send status update
            status_msg = WorkerMessage(
                type="status_update",
                worker_id=self.backend_connection.worker_id,
                agent_name=self.options.agent_id,  # Include agent_id
                status="available" if not self._draining else "draining",
                load=load,
                job_count=job_count,
            )

            await self.backend_connection.send_message(status_msg)

            # Update last status update time
            self._last_status_update = current_time

            # Update tracing
            self._worker_load_graph.add_point(load)

        except Exception as e:
            logger.error(f"Error updating worker status: {e}")

    async def execute_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a job using the task executor."""
        if not self.process_manager:
            raise RuntimeError("Task executor not initialized")

        # Extract entrypoint function from job data
        entrypoint = job_data.get("entrypoint", self.options.entrypoint_fnc)

        # Execute using new task executor
        result = await self.process_manager.execute(
            entrypoint=entrypoint,
            task_type=TaskType.JOB,
            timeout=job_data.get("timeout", 300.0),
            retry_count=job_data.get("retry_count", 3),
            priority=job_data.get("priority", 0),
            *job_data.get("args", ()),
            **job_data.get("kwargs", {}),
        )

        # Convert TaskResult to expected format
        return {
            "status": result.status.value,
            "result": result.result,
            "error": result.error,
            "execution_time": result.execution_time,
            "task_id": result.task_id,
        }

    async def execute_inference(self, inference_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an inference using the task executor."""
        if not self.process_manager:
            raise RuntimeError("Task executor not initialized")

        # Extract entrypoint function from inference data
        entrypoint = inference_data.get("entrypoint", self.options.entrypoint_fnc)

        # Execute using new task executor
        result = await self.process_manager.execute(
            entrypoint=entrypoint,
            task_type=TaskType.INFERENCE,
            timeout=inference_data.get("timeout", 300.0),
            retry_count=inference_data.get("retry_count", 3),
            priority=inference_data.get("priority", 0),
            *inference_data.get("args", ()),
            **inference_data.get("kwargs", {}),
        )

        # Convert TaskResult to expected format
        return {
            "status": result.status.value,
            "result": result.result,
            "error": result.error,
            "execution_time": result.execution_time,
            "task_id": result.task_id,
        }

    async def launch_direct_job(self, room_options: "RoomOptions") -> None:
        """Launch a job directly without backend registration."""
        logger.info("Launching direct job")

        # Create job context
        job_context = JobContext(
            room_options=room_options,
        )

        # Set up session end callback if room is created
        if job_context.room:

            def on_session_end(reason: str):
                logger.info(f"Direct job session ended, reason: {reason}")
                # For direct jobs, we can just log the session end
                logger.info(f"Session ended: {reason}")

            job_context.room.on_session_end = on_session_end
            logger.info("Set up session end callback for direct job")

        # Execute the job using the worker's entrypoint function
        await self.options.entrypoint_fnc(job_context)

    async def simulate_job(self, info: DirectRoomOptions | str) -> None:
        """Simulate a job for testing purposes."""
        if isinstance(info, str):
            # Simple string format: "room_id"
            room_id = info
            info = DirectRoomOptions(room_id=room_id)

        logger.info(f"Simulating job for room: {info.room_id}")

        # Create room options
        room_options = RoomOptions(
            room_id=info.room_id,
            name=info.room_name or f"test_room_{info.room_id}",
            auth_token=info.auth_token or self.options.auth_token,
            signaling_base_url=self.options.signaling_base_url,
            auto_end_session=True,
        )

        # Launch the job
        await self.launch_direct_job(room_options)

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        # Calculate current load dynamically
        job_count = len(self._current_jobs)
        current_load = min(job_count / self.options.max_processes, 1.0)

        stats = {
            "worker_load": current_load,
            "draining": self._draining,
            "current_jobs": job_count,
            "max_processes": self.options.max_processes,
            "agent_id": self.options.agent_id,
            "register": self.options.register,
        }

        if self.backend_connection:
            stats.update(
                {
                    "backend_connected": self.backend_connection.is_connected,
                    "worker_id": self.backend_connection.worker_id,
                }
            )

        if self.process_manager:
            try:
                process_stats = self.process_manager.get_stats()
                logger.debug(f"Process manager stats: {process_stats}")

                # Get current resource stats and dedicated inference status
                if "resource_stats" in process_stats:
                    stats["resource_stats"] = process_stats["resource_stats"]
                    logger.debug(f"Resource stats: {process_stats['resource_stats']}")
                if "dedicated_inference" in process_stats:
                    stats["dedicated_inference"] = process_stats["dedicated_inference"]

                # Also get current resource info for more detailed stats
                try:
                    resource_info = self.process_manager.get_resource_info()
                    logger.debug(
                        f"Resource info count: {len(resource_info) if resource_info else 0}"
                    )

                    if resource_info:
                        stats["resource_info"] = [
                            {
                                "resource_id": info.resource_id,
                                "resource_type": info.resource_type.value,
                                "status": info.status.value,
                                "current_load": info.current_load,
                                "memory_usage_mb": info.memory_usage_mb,
                                "cpu_usage_percent": info.cpu_usage_percent,
                                "active_tasks": info.active_tasks,
                                "total_tasks_processed": info.total_tasks_processed,
                                "last_heartbeat": info.last_heartbeat,
                                "metadata": info.metadata,
                            }
                            for info in resource_info
                        ]

                        # Add summary of resource status
                        resource_summary = {
                            "total_resources": len(resource_info),
                            "available_resources": len(
                                [r for r in resource_info if r.status == "IDLE"]
                            ),
                            "active_resources": len(
                                [r for r in resource_info if r.status != "IDLE"]
                            ),
                            "dedicated_inference_active": any(
                                r.resource_type == "DEDICATED_INFERENCE"
                                and r.status != "IDLE"
                                for r in resource_info
                            ),
                        }
                        stats["resource_summary"] = resource_summary
                        logger.debug(f"Resource summary: {resource_summary}")
                except Exception as e:
                    logger.debug(f"Could not get detailed resource info: {e}")
            except Exception as e:
                logger.error(f"Error getting process manager stats: {e}")
                stats["resource_stats"] = {"error": str(e)}
                stats["dedicated_inference"] = None

        return stats

    async def drain(self, timeout: Optional[float] = None) -> None:
        """Drain the worker - wait for current jobs to finish before shutting down."""
        if self._draining:
            return

        logger.info("Draining VideoSDK worker")
        self._draining = True
        await self._update_worker_status()

        # Wait for current jobs to complete
        if self._current_jobs:
            logger.info(
                f"Waiting for {len(self._current_jobs)} active jobs to complete"
            )

            if timeout:
                try:
                    await asyncio.wait_for(self._wait_for_jobs(), timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout waiting for jobs to complete after {timeout}s"
                    )
            else:
                await self._wait_for_jobs()

    async def _wait_for_jobs(self) -> None:
        """Wait for all current jobs to complete."""
        while self._current_jobs:
            # Wait a bit and check again
            await asyncio.sleep(1)
            logger.info(f"Still waiting for {len(self._current_jobs)} jobs to complete")

    async def _cleanup_all_jobs(self):
        """Clean up all current jobs and notify registry."""
        if not self._current_jobs:
            return

        logger.info(f"Cleaning up {len(self._current_jobs)} jobs during shutdown")

        # Create a copy of jobs to iterate over, as they will be modified
        jobs_to_clean = list(self._current_jobs.items())
        
        for job_id, job_info in jobs_to_clean:
            try:
                logger.info(f"Terminating job {job_id}...")
                await job_info.job.shutdown()  # This calls job.shutdown()
                logger.info(f"Job {job_id} terminated successfully.")
            except Exception as e:
                logger.error(f"Error terminating job {job_id}: {e}")

            try:
                if self.backend_connection and self.backend_connection.is_connected:
                    job_update = JobUpdate(
                        job_id=job_id,
                        status="completed",
                        error="Worker shutdown",
                    )
                    await self.backend_connection.send_message(job_update)
                    logger.info(
                        f"Sent job completion update for job {job_id} during shutdown"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to send job completion update for {job_id}: {e}"
                )
        
        # Clear all jobs from the worker's state
        self._current_jobs.clear()
        logger.info("All jobs cleared from worker")

        # Send a final status update reflecting zero jobs
        if self.backend_connection and self.backend_connection.is_connected:
            await self._send_immediate_status_update()
            
    async def shutdown(self):
        """Shutdown the worker."""
        logger.info("Shutting down VideoSDK worker")
        self._shutdown = True
        self._draining = True

        try:
            # Clean up all jobs first to ensure proper room cleanup
            await self._cleanup_all_jobs()
        except Exception as e:
            logger.error(f"Error during job cleanup: {e}")

        try:
            # Send final status update to registry
            if self.backend_connection and self.backend_connection.is_connected:
                try:
                    await self._update_worker_status()
                    logger.info("Sent final status update to registry")
                except Exception as e:
                    logger.warning(f"Failed to send final status update: {e}")

            # Disconnect from backend
            if self.backend_connection:
                logger.info("Disconnecting from backend")
                await self.backend_connection.disconnect()
        except Exception as e:
            logger.error(f"Error during backend cleanup: {e}")

        try:
            # Cancel all tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()

            # Wait briefly for tasks to complete
            if self._tasks:
                done, pending = await asyncio.wait(self._tasks, timeout=2.0)
                for task in pending:
                    task.cancel()
        except Exception as e:
            logger.error(f"Error during task cleanup: {e}")

        try:
            # Shutdown task executor
            if self.process_manager:
                await self.process_manager.stop()
        except Exception as e:
            logger.error(f"Error stopping process manager: {e}")

        try:
            # Stop debug HTTP server
            if self._http_server:
                await self._http_server.aclose()
        except Exception as e:
            logger.error(f"Error stopping HTTP server: {e}")

        logger.info("VideoSDK worker shutdown complete")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
