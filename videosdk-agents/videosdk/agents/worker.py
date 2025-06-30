from __future__ import annotations
import threading
import asyncio
import os
import time
import psutil
import logging
import signal
from dataclasses import dataclass
from typing import Callable, Awaitable, Any, Optional, Union
from enum import Enum
from .job import JobContext, _set_current_job_context, _reset_current_job_context
import inspect

class LoadCalculator:
    def __init__(self, window_size: int = 5) -> None:
        # Moving average state
        self._hist: list[float] = [0] * window_size
        self._sum: float = 0
        self._count: int = 0
        
        # CPU monitoring setup
        self._is_containerized = os.path.exists("/sys/fs/cgroup/cpu.stat")
        if self._is_containerized:
            self._cpu_quota, self._cpu_period = self._read_cpu_max()
        
        # Background monitoring
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._monitor_cpu, daemon=True, name="load_calculator"
        )
        self._thread.start()

    def _monitor_cpu(self) -> None:
        while True:
            cpu_percent = self._get_cpu_usage()
            with self._lock:
                self._add_sample(cpu_percent)
            time.sleep(0.5)

    def _get_cpu_usage(self) -> float:
        if self._is_containerized:
            return self._get_container_cpu_usage()
        else:
            return psutil.cpu_percent(interval=0.5) / 100.0

    def _get_container_cpu_usage(self) -> float:
        cpu_usage_start = self._read_cpu_usage()
        time.sleep(0.5)
        cpu_usage_end = self._read_cpu_usage()
        
        cpu_usage_diff = cpu_usage_end - cpu_usage_start
        cpu_usage_seconds = cpu_usage_diff / 1_000_000
        num_cpus = self._get_container_cpu_count()
        
        return min(cpu_usage_seconds / (0.5 * num_cpus), 1.0)

    def _get_container_cpu_count(self) -> float:
        if self._cpu_quota == "max":
            return os.cpu_count() or 1
        return 1.0 * int(self._cpu_quota) / self._cpu_period

    def _read_cpu_max(self) -> tuple[str, int]:
        try:
            with open("/sys/fs/cgroup/cpu.max") as f:
                data = f.read().strip().split()
            return data[0], int(data[1])
        except FileNotFoundError:
            return "max", 100000

    def _read_cpu_usage(self) -> int:
        with open("/sys/fs/cgroup/cpu.stat") as f:
            for line in f:
                if line.startswith("usage_usec"):
                    return int(line.split()[1])
        raise RuntimeError("Failed to read CPU usage")

    def _add_sample(self, sample: float) -> None:
        self._count += 1
        index = self._count % len(self._hist)
        if self._count > len(self._hist):
            self._sum -= self._hist[index]
        self._sum += sample
        self._hist[index] = sample

    def get_load(self) -> float:
        with self._lock:
            if self._count == 0:
                return 0
            size = min(self._count, len(self._hist))
            return self._sum / size

def get_cpu_load() -> float:
    if not hasattr(get_cpu_load, '_calculator'):
        get_cpu_load._calculator = LoadCalculator()
    return get_cpu_load._calculator.get_load()

class PrewarmStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class WorkerOptions:
    entrypoint: Callable[[JobContext], Awaitable[None]]
    prewarm: Optional[Callable[[Any], None]] = None
    job_context: Optional[Union[
        Callable[[], JobContext],  
        Callable[[asyncio.AbstractEventLoop], JobContext],  
        JobContext  
    ]] = None
    load: Callable[[], float] = lambda: get_cpu_load()
    drain_timeout: float = 10.0

class Worker:
    def __init__(self, options: WorkerOptions, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self.options = options
        self.loop = loop or asyncio.get_event_loop()
        
        self._prewarm_status = PrewarmStatus.NOT_STARTED
        self._prewarm_error: Optional[Exception] = None
        self._prewarm_lock = threading.Lock()
        
        self._shutdown_requested = threading.Event()
        self._job_running = threading.Event()
        self._current_job_context: Optional[JobContext] = None
        
        self._setup_signal_handlers()
        
        self._start_prewarming()
    
    def _setup_signal_handlers(self) -> None:
        """Setup SIGTERM and SIGINT handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
            logging.info(f"Received {signal_name}, initiating graceful shutdown...")
            
            self._shutdown_requested.set()
            
            threading.Thread(
                target=self._drain_and_shutdown,
                daemon=False,
                name="worker_drain"
            ).start()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def _drain_and_shutdown(self) -> None:
        """Wait for job to complete, call shutdown hooks, then exit"""
        logging.info(f"Draining worker - waiting up to {self.options.drain_timeout}s for job to complete")
        
        if self._current_job_context:
            try:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                loop.run_until_complete(self._current_job_context.shutdown())
            except Exception as e:
                logging.error(f"Error during JobContext shutdown: {e}")
        
        if self._job_running.wait(timeout=self.options.drain_timeout):
            logging.info("Job completed, shutting down gracefully")
        else:
            logging.warning(f"Drain timeout reached, forcing shutdown")
        
        os._exit(0)
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested"""
        return self._shutdown_requested.is_set()
    
    def run(self) -> None:
        """Main worker run method"""
        logging.info("Worker starting...")
        
        if not self.is_prewarmed():
            if not self.wait_for_prewarm(timeout=60):
                logging.warning("Prewarming did not complete, continuing anyway")
        
        logging.info("Worker ready - executing entrypoint")
        
        if self.is_shutdown_requested():
            logging.info("Shutdown requested before job start")
            return
        
        try:
            self.loop.run_until_complete(self._execute_entrypoint())
            self.loop.run_forever()
        except KeyboardInterrupt:
            logging.info("Received KeyboardInterrupt")
        
        logging.info("Worker finished")
    
    async def _execute_entrypoint(self) -> None:
        """Execute the entrypoint with JobContext"""
        self._job_running.clear()
        
        try:
            if self.options.job_context:
                job_context = self._create_job_context()
                self._current_job_context = job_context
                
                token = _set_current_job_context(job_context)              
                try:
                    await self.options.entrypoint(job_context)
                finally:
                    _reset_current_job_context(token)
            else:
                await self.options.entrypoint(None)
        except Exception as e:
            logging.error(f"Entrypoint failed: {e}")
            raise
        finally:
            self._job_running.set()
            self._current_job_context = None
    
    def _create_job_context(self) -> JobContext:
        """Create JobContext from factory, handling different factory types"""
        factory = self.options.job_context
        
        if isinstance(factory, JobContext):
            if factory._loop != self.loop:
                return JobContext(
                    room_options=factory.room_options,
                    loop=self.loop
                )
            return factory
        
        if callable(factory):
            sig = inspect.signature(factory)
            params = list(sig.parameters.keys())
            
            if len(params) == 0:
                return factory()
            
            elif len(params) == 1:
                return factory(self.loop)
            
            else:
                raise ValueError(f"job_context must accept 0 or 1 parameters, got {len(params)}")
        
        else:
            raise ValueError(f"job_context must be a callable or JobContext instance, got {type(factory)}")
    
    def _start_prewarming(self) -> None:
        """Start prewarming in background thread"""
        if self.options.prewarm is None:
            self._prewarm_status = PrewarmStatus.COMPLETED
            return
            
        def _prewarm_task():
            try:
                with self._prewarm_lock:
                    self._prewarm_status = PrewarmStatus.IN_PROGRESS
                
                self.options.prewarm(self)
                
                with self._prewarm_lock:
                    self._prewarm_status = PrewarmStatus.COMPLETED
                    
            except Exception as e:
                with self._prewarm_lock:
                    self._prewarm_status = PrewarmStatus.FAILED
                    self._prewarm_error = e
                logging.error(f"Prewarming failed: {e}")
        
        threading.Thread(
            target=_prewarm_task, 
            daemon=True, 
            name="worker_prewarm"
        ).start()
    
    def get_prewarm_status(self) -> PrewarmStatus:
        """Get current prewarming status"""
        with self._prewarm_lock:
            return self._prewarm_status
    
    def is_prewarmed(self) -> bool:
        """Check if prewarming is complete"""
        return self.get_prewarm_status() == PrewarmStatus.COMPLETED
    
    def wait_for_prewarm(self, timeout: float = 30.0) -> bool:
        """Wait for prewarming to complete, returns True if successful"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_prewarm_status()
            if status == PrewarmStatus.COMPLETED:
                return True
            elif status == PrewarmStatus.FAILED:
                return False
            time.sleep(0.1)
        return False
    
    def get_prewarm_error(self) -> Optional[Exception]:
        """Get prewarming error if any"""
        with self._prewarm_lock:
            return self._prewarm_error
    