from __future__ import annotations
import time
from typing import Any, Dict, List

class PerformanceMatrix:
    def __init__(self):
        self.marks: Dict[str, float] = {}
        self.measures: List[Dict[str, Any]] = []

    def mark(self, name: str):
        """Mark a specific point in time for performance measurement"""
        self.marks[name] = time.perf_counter()

    def measure(self, measure_name: str, start_or_mark_name: str, end_or_mark_name: str | None = None):
        """Measure duration between two marks or from a mark to current time"""
        end_time = time.perf_counter()
        if end_or_mark_name:
            if end_or_mark_name in self.marks:
                end_time = self.marks.get(end_or_mark_name)
            else:
                pass

        start_time = self.marks.get(start_or_mark_name)
        if start_time is None:
            start_time = end_time

        duration = (end_time - start_time) * 1000  # in ms
        self.measures.append({
            "name": measure_name, 
            "duration": duration, 
            "startTime": start_time * 1000,  
            "endTime": end_time * 1000       
        })

    def get_entries(self):
        """Get all performance measures"""
        return self.measures
    
    def clear(self):
        """Clear all marks and measures"""
        self.marks.clear()
        self.measures.clear()

    def has_mark(self, name: str) -> bool:
        """Check if a mark exists"""
        return name in self.marks 