"""
Protocol definitions for VideoSDK Agent backend communication.

This module defines the message types and structures used for communication
between VideoSDK agents and the backend registry server.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkerStatus(str, Enum):
    """Worker status enumeration."""

    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class UpdateWorkerStatus:
    """Update worker status message."""

    status: WorkerStatus
    load: Optional[float] = None
    job_count: Optional[int] = None
    error: Optional[str] = None


@dataclass
class UpdateJobStatus:
    """Update job status message."""

    job_id: str
    status: JobStatus
    error: Optional[str] = None
    participant_identity: Optional[str] = None
    participant_name: Optional[str] = None
    participant_metadata: Optional[str] = None


@dataclass
class WorkerMessage:
    """Base message from worker to server."""

    type: str
    worker_id: Optional[str] = None
    agent_name: Optional[str] = None
    namespace: Optional[str] = None
    version: Optional[str] = None
    capabilities: Optional[List[str]] = None
    status: Optional[str] = None
    load: Optional[float] = None
    job_count: Optional[int] = None
    job_id: Optional[str] = None
    available: Optional[bool] = None
    participant_identity: Optional[str] = None
    participant_name: Optional[str] = None
    participant_metadata: Optional[str] = None
    error: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    registry_uuid: Optional[str] = None
    max_capacity: Optional[int] = None
    current_load: Optional[float] = None
    token: Optional[str] = None  # Authentication token for registry

    # Workload configuration from agent
    load_threshold: Optional[float] = None  # Agent's load threshold (e.g., 0.8)
    max_processes: Optional[int] = None  # Agent's max processes (e.g., 3)

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result


@dataclass
class ServerMessage:
    """Base message from server to worker."""

    type: str
    worker_id: Optional[str] = None
    success: Optional[bool] = None
    message: Optional[str] = None
    job_id: Optional[str] = None
    job_type: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    agent_name: Optional[str] = None
    namespace: Optional[str] = None
    token: Optional[str] = None
    url: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result


@dataclass
class AvailabilityRequest:
    """Request from server asking if worker is available for a job."""

    type: str = "availability_request"
    job_id: str = ""
    job_type: str = ""
    room_id: str = ""
    room_name: str = ""
    agent_name: str = ""
    namespace: str = ""
    payload: Optional[Dict[str, Any]] = None


@dataclass
class AvailabilityResponse:
    """Response from worker indicating availability."""

    type: str = "availability_response"
    job_id: str = ""
    available: bool = False
    error: Optional[str] = None
    token: Optional[str] = None  # Worker's auth token when accepting job

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result


@dataclass
class JobAssignment:
    """Job assignment from server to worker."""

    type: str = "job_assignment"
    job_id: str = ""
    job_type: str = ""
    room_id: str = ""
    room_name: str = ""
    agent_name: str = ""
    namespace: str = ""
    token: str = ""
    url: str = ""
    payload: Optional[Dict[str, Any]] = None
    room_options: Optional[Dict[str, Any]] = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result


@dataclass
class JobUpdate:
    """Update from worker about job status."""

    type: str = "job_update"
    job_id: str = ""
    status: str = ""
    error: Optional[str] = None
    participant_identity: Optional[str] = None
    participant_name: Optional[str] = None
    participant_metadata: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result


@dataclass
class JobTermination:
    """Job termination request from server."""

    type: str = "job_termination"
    job_id: str = ""
    reason: Optional[str] = None


@dataclass
class WorkerPong:
    """Pong response from server."""

    type: str = "pong"
    timestamp: Optional[int] = None


@dataclass
class WorkerPing:
    """Ping message from worker."""

    type: str = "ping"
    timestamp: Optional[int] = None


# Message type constants
MSG_TYPE_REGISTER = "register"
MSG_TYPE_STATUS_UPDATE = "status_update"
MSG_TYPE_AVAILABILITY_REQUEST = "availability_request"
MSG_TYPE_AVAILABILITY_RESPONSE = "availability_response"
MSG_TYPE_JOB_ASSIGNMENT = "job_assignment"
MSG_TYPE_JOB_UPDATE = "job_update"
MSG_TYPE_JOB_TERMINATION = "job_termination"
MSG_TYPE_PING = "ping"
MSG_TYPE_PONG = "pong"
