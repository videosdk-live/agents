"""
Backend communication module for VideoSDK Agents.

This module provides WebSocket connection and protocol handling for
communicating with the VideoSDK backend server.
"""

from .connection import BackendConnection
from .protocol import (
    AvailabilityRequest,
    AvailabilityResponse,
    JobAssignment,
    JobTermination,
    JobUpdate,
    ServerMessage,
    WorkerMessage,
    WorkerPing,
    WorkerPong,
    WorkerStatus,
    JobStatus,
    UpdateWorkerStatus,
    UpdateJobStatus,
)

__all__ = [
    "BackendConnection",
    "AvailabilityRequest",
    "AvailabilityResponse",
    "JobAssignment",
    "JobTermination",
    "JobUpdate",
    "ServerMessage",
    "WorkerMessage",
    "WorkerPing",
    "WorkerPong",
    "WorkerStatus",
    "JobStatus",
    "UpdateWorkerStatus",
    "UpdateJobStatus",
]
