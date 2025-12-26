"""
Custom logging utility for evaluation pipeline.
All logs are prefixed with 'Eval:' to distinguish from regular application logs.
"""
import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EvalLogger:
    """Custom logger for evaluation with 'Eval:' prefix and component tracking."""
    
    def __init__(self):
        self.component_start_times = {}
    
    def log(self, message: str):
        """Log a message with 'Eval:' prefix."""
        logger.info(f"Eval: {message}")
    
    def debug(self, message: str):
        """Log a debug message (without Eval prefix for detailed logs)."""
        logger.debug(message)
    
    def component_start(self, component: str):
        """Log component start and track start time."""
        self.component_start_times[component] = time.perf_counter()
        self.log(f"[{component}] Starting...")
    
    def component_end(self, component: str, latency_ms: Optional[float] = None):
        """Log component completion with latency."""
        if latency_ms is None and component in self.component_start_times:
            # Calculate latency from tracked start time
            elapsed = time.perf_counter() - self.component_start_times[component]
            latency_ms = elapsed * 1000
            del self.component_start_times[component]
        
        if latency_ms is not None:
            self.log(f"[{component}] Completed in {latency_ms:.2f}ms")
        else:
            self.log(f"[{component}] Completed")
    
    def turn_start(self, turn_num: int, total: int, turn_id: str):
        """Log turn start."""
        self.log(f"{'=' * 50}")
        self.log(f"Turn {turn_num}/{total} (ID: {turn_id})")
        self.log(f"{'=' * 50}")
    
    def turn_end(self, turn_num: int, total_latency_ms: Optional[float] = None):
        """Log turn completion."""
        if total_latency_ms is not None:
            self.log(f"Turn {turn_num} completed - Total latency: {total_latency_ms:.2f}ms")
        else:
            self.log(f"Turn {turn_num} completed")
        self.log("")
    
    def evaluation_start(self, name: str, turn_count: int):
        """Log evaluation start."""
        self.log(f"Starting evaluation: {name} with {turn_count} turns")
        self.log("")
    
    def evaluation_end(self):
        """Log evaluation completion."""
        self.log("Evaluation completed!")
    
    def display_turn_logs(self, turn_index: int, turn_id: str, stt_transcript: str, llm_response: str, tts_audio_file: str):
        """Display formatted logs for a single turn."""
        self.log(f"\n{'=' * 60}")
        self.log(f"TURN {turn_index + 1} RESULTS (ID: {turn_id})")
        self.log(f"{'=' * 60}")
        
        if stt_transcript:
            self.log(f"\nSTT Transcript:\n  {stt_transcript}")
        else:
            self.log(f"\nSTT Transcript:\n  N/A")
        
        if llm_response:
            self.log(f"\nLLM Response:\n  {llm_response}")
        else:
            self.log(f"\nLLM Response:\n  N/A")
        
        if tts_audio_file:
            self.log(f"\nTTS Audio File:\n  → {tts_audio_file}")
        else:
            self.log(f"\nTTS Audio File:\n  N/A")
    
    def display_latency_table(self, results: list):
        """Display all latencies in a formatted table."""
        self.log(f"\n{'=' * 60}")
        self.log("LATENCY SUMMARY")
        self.log(f"{'=' * 60}\n")
        
        # Table header
        header = f"{'Turn':<8} {'STT (ms)':<15} {'LLM (ms)':<15} {'TTS (ms)':<15} {'Total (ms)':<15}"
        self.log(header)
        self.log("-" * 68)
        
        # Table rows
        for res in results:
            turn_idx = res.get("turn_index", 0)
            metrics = res.get("metrics", {})
            
            stt_latency = metrics.get("stt_latency", 0)
            llm_latency = metrics.get("llm_latency", 0)
            tts_latency = metrics.get("tts_latency", 0)
            
            # Calculate total
            total_latency = 0
            if stt_latency: total_latency += stt_latency
            if llm_latency: total_latency += llm_latency
            if tts_latency: total_latency += tts_latency
            
            # Format values (show N/A for 0 or None)
            stt_str = f"{stt_latency:.2f}" if stt_latency else "N/A"
            llm_str = f"{llm_latency:.2f}" if llm_latency else "N/A"
            tts_str = f"{tts_latency:.2f}" if tts_latency else "N/A"
            total_str = f"{total_latency:.2f}" if total_latency else "N/A"
            
            row = f"{turn_idx + 1:<8} {stt_str:<15} {llm_str:<15} {tts_str:<15} {total_str:<15}"
            self.log(row)
        
        self.log("")


# Global eval logger instance
eval_logger = EvalLogger()
