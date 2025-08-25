import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import defaultdict


@dataclass
class TracingPoint:
    """A single tracing point with timestamp and data."""

    timestamp: float
    data: Dict[str, Any]
    label: str = ""


@dataclass
class TracingEvent:
    """A tracing event with timestamp, name, and optional data."""

    timestamp: float
    name: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class TracingGraph:
    """A tracing graph for collecting time-series data."""

    title: str
    x_label: str
    y_label: str
    x_type: str = "time"
    y_range: tuple = (0, 100)
    max_data_points: int = 1000
    data: List[TracingPoint] = field(default_factory=list)

    def add_point(self, value: float, label: str = "") -> None:
        """Add a data point to the graph."""
        point = TracingPoint(timestamp=time.time(), data={"value": value}, label=label)
        self.data.append(point)

        # Trim old data points if exceeding max
        if len(self.data) > self.max_data_points:
            self.data = self.data[-self.max_data_points :]

    def export(self) -> Dict[str, Any]:
        """Export graph data for JSON serialization."""
        return {
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_type": self.x_type,
            "y_range": self.y_range,
            "data": [
                [point.timestamp, point.data.get("value", 0)] for point in self.data
            ],
        }


class Tracing:
    """
    Tracing system for VideoSDK agents.

    Provides performance monitoring, debugging, and metrics collection.
    """

    _graphs: Dict[str, TracingGraph] = {}
    _handles: Dict[str, "Tracing"] = {}
    _events: Dict[str, List[TracingEvent]] = defaultdict(list)
    _kv_data: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def __init__(self, name: str):
        self.name = name
        self._graphs = {}
        self._events = defaultdict(list)
        self._kv_data = defaultdict(dict)

    @classmethod
    def with_handle(cls, name: str) -> "Tracing":
        """Get or create a tracing handle."""
        if name not in cls._handles:
            cls._handles[name] = cls(name)
        return cls._handles[name]

    @classmethod
    def add_graph(
        cls,
        title: str,
        x_label: str = "time",
        y_label: str = "value",
        x_type: str = "time",
        y_range: tuple = (0, 100),
        max_data_points: int = 1000,
    ) -> TracingGraph:
        """Add a new tracing graph."""
        graph = TracingGraph(
            title=title,
            x_label=x_label,
            y_label=y_label,
            x_type=x_type,
            y_range=y_range,
            max_data_points=max_data_points,
        )
        cls._graphs[title] = graph
        return graph

    @classmethod
    def get_graph(cls, title: str) -> Optional[TracingGraph]:
        """Get a tracing graph by title."""
        return cls._graphs.get(title)

    def add_point(self, graph_title: str, value: float, label: str = "") -> None:
        """Add a point to a specific graph."""
        graph = self._graphs.get(graph_title)
        if graph:
            graph.add_point(value, label)

    def add_event(self, name: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the tracing system."""
        event = TracingEvent(timestamp=time.time(), name=name, data=data)
        self._events[self.name].append(event)

    def set_kv(self, key: str, value: Any) -> None:
        """Set a key-value pair in the tracing data."""
        self._kv_data[self.name][key] = value

    def get_kv(self, key: str) -> Any:
        """Get a key-value pair from the tracing data."""
        return self._kv_data[self.name].get(key)

    def _export(self) -> Dict[str, Any]:
        """Export all data for this tracing instance."""
        return {
            "name": self.name,
            "kv": dict(self._kv_data[self.name]),
            "events": [
                {"timestamp": event.timestamp, "name": event.name, "data": event.data}
                for event in self._events[self.name]
            ],
            "graphs": {title: graph.export() for title, graph in self._graphs.items()},
        }

    @classmethod
    def export_all(cls) -> Dict[str, Any]:
        """Export all tracing data."""
        return {
            "global_graphs": {
                title: graph.export() for title, graph in cls._graphs.items()
            },
            "handles": {
                name: handle._export() for name, handle in cls._handles.items()
            },
        }

    @classmethod
    def export_for_handle(cls, handle_name: str) -> Dict[str, Any]:
        """Export tracing data for a specific handle."""
        handle = cls._handles.get(handle_name)
        if handle:
            return handle._export()
        return {"kv": {}, "events": [], "graphs": []}

    @classmethod
    def create_debug_app(cls, worker: Any) -> Any:
        """Create a debug web application for tracing."""
        from aiohttp import web

        async def tracing_index(request: web.Request) -> web.Response:
            """Serve the tracing dashboard HTML."""
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>VideoSDK Agents Debug Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                    .graph { margin: 10px 0; padding: 10px; background: #f9f9f9; }
                    .endpoint { margin: 5px 0; }
                    .endpoint a { color: #0066cc; text-decoration: none; }
                    .endpoint a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <h1>VideoSDK Agents Debug Dashboard</h1>
                
                <div class="section">
                    <h2>Endpoints</h2>
                    <div class="endpoint"><a href="/health">Health Check</a></div>
                    <div class="endpoint"><a href="/worker">Worker Status</a></div>
                    <div class="endpoint"><a href="/stats">Worker Statistics</a></div>
                    <div class="endpoint"><a href="/tracing">Tracing Data</a></div>
                </div>
                
                <div class="section">
                    <h2>Worker Information</h2>
                    <div id="worker-info">Loading...</div>
                </div>
                
                <div class="section">
                    <h2>Tracing Graphs</h2>
                    <div id="tracing-graphs">Loading...</div>
                </div>
                
                <script>
                    // Load worker info
                    fetch('/worker')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('worker-info').innerHTML = 
                                '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                        });
                    
                    // Load tracing data
                    fetch('/tracing')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('tracing-graphs').innerHTML = 
                                '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                        });
                </script>
            </body>
            </html>
            """
            return web.Response(text=html_content, content_type="text/html")

        async def tracing_data(request: web.Request) -> web.Response:
            """Serve tracing data as JSON."""
            return web.json_response(cls.export_all())

        app = web.Application()
        app.add_routes([web.get("", tracing_index)])
        app.add_routes([web.get("/", tracing_index)])
        app.add_routes([web.get("/tracing", tracing_data)])

        return app
