"""
Inter-Process Communication (IPC) module for VideoSDK Agents.

This module provides optimized process and thread management, resource allocation,
and communication between different agent processes/threads.
"""

from .proc_pool import ProcPool, ProcPoolConfig, ExecutorType
from .job_executor import JobExecutor
from .job_thread_executor import ThreadJobExecutor
from .inference_executor import InferenceExecutor
from .process_manager import ProcessManager
from multiprocessing.connection import Connection

__all__ = [
    # Core IPC components
    "ProcPool",
    "ProcPoolConfig",
    "ExecutorType",
    "JobExecutor",
    "ThreadJobExecutor",
    "InferenceExecutor",
    "ProcessManager",
]
