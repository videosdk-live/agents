import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)


class HttpServer:
    """
    HTTP server for VideoSDK agents debugging and monitoring.

    Provides endpoints for health checks, worker status, and debugging information.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 0,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._host = host
        self._port = port
        self._app = web.Application(loop=self._loop)
        self._lock = asyncio.Lock()
        self._server = None
        self._worker = None

    @property
    def app(self) -> web.Application:
        """Get the aiohttp application."""
        return self._app

    @property
    def port(self) -> int:
        """Get the port the server is listening on."""
        return self._port

    def set_worker(self, worker: Any) -> None:
        """Set the worker instance for status endpoints."""
        self._worker = worker

    async def start(self) -> None:
        """Start the HTTP server."""
        async with self._lock:
            # Add routes - matching structure
            self._app.add_routes([web.get("/", self._handle_dashboard)])
            self._app.add_routes([web.get("/debug/worker/", self._worker_debug)])
            self._app.add_routes([web.get("/debug/runners/", self._runners_list)])
            self._app.add_routes([web.get("/debug/runner/", self._runner_details)])
            self._app.add_routes([web.get("/health", self._health_check)])
            self._app.add_routes([web.get("/worker", self._worker_status)])
            self._app.add_routes([web.get("/stats", self._worker_stats)])
            self._app.add_routes([web.get("/debug", self._debug_info)])
            self._app.add_routes([web.get("/api/status", self._api_status)])

            # Create server
            handler = self._app.make_handler()
            self._server = await self._loop.create_server(
                handler, self._host, self._port
            )

            # Get actual port if using port 0
            if self._port == 0:
                self._port = self._server.sockets[0].getsockname()[1]

            await self._server.start_serving()

    async def aclose(self) -> None:
        """Close the HTTP server."""
        async with self._lock:
            if self._server:
                self._server.close()
                await self._server.wait_closed()

    async def _handle_dashboard(self, request: web.Request) -> web.Response:
        """Serve the main dashboard HTML page - matching style."""
        html_content = """<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8" />
  <title>videosdk-agents - tracing</title>
  <style>
    body {
      font-family: sans-serif;
      margin: 8px;
      padding: 0;
    }

    .section {
      padding: 8px;
      font-size: 0.9em;
      margin-top: 8px;
    }

    .collapsible-title {
      display: block;
      cursor: pointer;
      user-select: none;
    }

    .collapsible-title::before {
      content: "▶ ";
    }

    .collapsible-title.expanded::before {
      content: "▼ ";
    }

    .collapsible-content {
      display: none;
      margin-left: 20px;
      /* optional indent for nested content */
    }

    .nested-collapsible-title {}

    .nested-collapsible-content {}

    .horizontal-group {
      display: flex;
      align-items: center;
      margin-bottom: 8px;
    }

    .refresh-icon {
      font-size: 16px;
      font-weight: bold;
      margin-right: 4px;
    }

    canvas {
      border: 1px solid #ccc;
    }

    .graph-title {
      font-weight: bold;
      margin-top: 8px;
    }
  </style>
</head>

<body>
  <!-- Worker Section -->
  <div class="section">
    <div class="horizontal-group">
      <h2 style="margin: 0 8px 0 0">Worker</h2>
      <button onclick="refreshWorker()">
        <span class="refresh-icon">⟳</span>Refresh
      </button>
    </div>
    <div id="workerSection"></div>
  </div>

  <!-- Runners List -->
  <div class="section">
    <div class="horizontal-group">
      <h2 style="margin: 0 8px 0 0">Runners</h2>
      <button onclick="refreshRunners()">
        <span class="refresh-icon">⟳</span>Refresh
      </button>
    </div>
    <div id="runnersList"></div>
  </div>

  <script>
    // Global state to remember which collapsibles are open
    // runnerOpenState[runnerId] = { open: true/false, sub: { "Key/Value": bool, "Events": bool }, ... }
    // We'll also store 'Worker' as a special ID => runnerOpenState["__WORKER__"] for worker KV / Events
    const runnerOpenState = {};

    const $ = (id) => document.getElementById(id);

    // ------------------------------
    // HTTP Utility
    // ------------------------------
    async function fetchJSON(url) {
      const r = await fetch(url);
      if (!r.ok) throw new Error("Network error");
      return r.json();
    }

    // ------------------------------
    // Collapsible toggle logic
    // ------------------------------
    function toggleCollapsible(titleEl, contentEl) {
      const isOpen = contentEl.style.display === "block";
      contentEl.style.display = isOpen ? "none" : "block";
      titleEl.classList.toggle("expanded", !isOpen);
    }

    // Re-apply state if we know something should be open
    function applyOpenState(titleEl, contentEl, open) {
      if (open) {
        contentEl.style.display = "block";
        titleEl.classList.add("expanded");
      } else {
        contentEl.style.display = "none";
        titleEl.classList.remove("expanded");
      }
    }

    // ------------------------------
    // Time label
    // ------------------------------
    function timeLabel(val) {
      const d = new Date(val * 1000);
      let hh = String(d.getHours()).padStart(2, "0");
      let mm = String(d.getMinutes()).padStart(2, "0");
      let ss = String(d.getSeconds()).padStart(2, "0");
      return `${hh}:${mm}:${ss}`;
    }

    // ------------------------------
    // Export Utility
    // ------------------------------
    function exportEventsToJSON(events) {
      const dataStr = JSON.stringify(events, null, 2);
      const blob = new Blob([dataStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);

      // Create a temporary link and auto-click to download
      const link = document.createElement("a");
      link.href = url;
      link.download = "events.json";
      document.body.appendChild(link);
      link.click();

      // Cleanup
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }

    // ------------------------------
    // Rendering Tracing Data
    // ------------------------------
    function renderKeyValue(container, kv) {
      const ul = document.createElement("ul");
      Object.entries(kv).forEach(([k, v]) => {
        const li = document.createElement("li");
        li.textContent = `${k}: ${JSON.stringify(v)}`;
        ul.appendChild(li);
      });
      container.appendChild(ul);
    }

    //
    // Keep each event on a single line. Don't show "click to expand" if data is null.
    //
    function renderEvents(container, events) {
      const ul = document.createElement("ul");
      events.forEach((e) => {
        // Each event => list item
        const li = document.createElement("li");

        // Create a wrapper span for the event name/time
        const titleLine = document.createElement("span");
        titleLine.textContent = `${new Date(
          e.timestamp * 1000
        ).toLocaleTimeString()} - ${e.name}`;
        li.appendChild(titleLine);

        // Only show the collapsible "Data" button if e.data is not null
        if (e.data != null) {
          const dataTitle = document.createElement("span");
          dataTitle.style.fontSize = "0.8em";
          dataTitle.style.marginLeft = "10px";
          dataTitle.style.cursor = "pointer";
          dataTitle.textContent = "[Data (click to expand)]";

          // Collapsible content block (hidden by default)
          const dataContent = document.createElement("div");
          dataContent.className =
            "collapsible-content nested-collapsible-content";
          dataContent.style.display = "none";

          // Pretty-print JSON with 2-space indentation
          const pre = document.createElement("pre");
          pre.textContent = JSON.stringify(e.data, null, 2);
          dataContent.appendChild(pre);

          li.appendChild(dataTitle);
          li.appendChild(dataContent);

          // Wire up the click event to toggle the data display
          dataTitle.addEventListener("click", () => {
            toggleCollapsible(dataTitle, dataContent);
          });
        }

        ul.appendChild(li);
      });
      container.appendChild(ul);
    }

    function drawGraph(canvas, g) {
      const ctx = canvas.getContext("2d");
      const w = canvas.width,
        h = canvas.height,
        pad = 40;
      ctx.clearRect(0, 0, w, h);

      if (!g.data?.length) {
        ctx.fillText("No data", w / 2 - 20, h / 2);
        return;
      }
      const xs = g.data.map((d) => d[0]);
      const ys = g.data.map((d) => d[1]);
      let [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
      if (minX === maxX) [minX, maxX] = [0, 1];
      let [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
      if (g.y_range) [minY, maxY] = g.y_range;
      else if (minY === maxY) [minY, maxY] = [0, 1];

      // Axes
      ctx.strokeStyle = "#000";
      ctx.beginPath();
      ctx.moveTo(pad, h - pad);
      ctx.lineTo(w - pad, h - pad);
      ctx.moveTo(pad, pad);
      ctx.lineTo(pad, h - pad);
      ctx.stroke();

      const pw = w - 2 * pad,
        ph = h - 2 * pad;
      const toCX = (x) => pad + (x - minX) * (pw / (maxX - minX));
      const toCY = (y) => h - pad - (y - minY) * (ph / (maxY - minY));

      // Graph line
      ctx.strokeStyle = "red";
      ctx.beginPath();
      ctx.moveTo(toCX(xs[0]), toCY(ys[0]));
      for (let i = 1; i < xs.length; i++) {
        ctx.lineTo(toCX(xs[i]), toCY(ys[i]));
      }
      ctx.stroke();

      // Ticks
      ctx.strokeStyle = "#000";
      ctx.fillStyle = "#000";
      ctx.font = "10px sans-serif";

      // X
      for (let i = 0; i <= 5; i++) {
        let vx = minX + (i * (maxX - minX)) / 5;
        let cx = toCX(vx),
          cy = h - pad;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx, cy + 5);
        ctx.stroke();
        let label = g.x_type === "time" ? timeLabel(vx) : vx.toFixed(2);
        let tw = ctx.measureText(label).width;
        ctx.fillText(label, cx - tw / 2, cy + 15);
      }
      // Y
      for (let i = 0; i <= 5; i++) {
        let vy = minY + (i * (maxY - minY)) / 5;
        let cx = pad,
          cy = toCY(vy);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx - 5, cy);
        ctx.stroke();
        let lbl = vy.toFixed(2),
          tw = ctx.measureText(lbl).width;
        ctx.fillText(lbl, cx - tw - 6, cy + 3);
      }

      // Labels
      if (g.x_label) {
        let tw = ctx.measureText(g.x_label).width;
        ctx.fillText(g.x_label, w / 2 - tw / 2, h - 5);
      }
      if (g.y_label) {
        ctx.save();
        ctx.translate(10, h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillText(g.y_label, 0, 0);
        ctx.restore();
      }
    }

    function renderGraphs(container, graphs) {
      graphs.forEach((g) => {
        const gt = document.createElement("div");
        gt.className = "graph-title";
        gt.innerText = g.title;
        container.appendChild(gt);

        const c = document.createElement("canvas");
        c.width = 400;
        c.height = 200;
        container.appendChild(c);

        drawGraph(c, g);
      });
    }

    // Render top-level Key/Value, Events, Graphs
    function renderTracing(container, tracing, runnerId = "__WORKER__") {
      if (!tracing) {
        container.textContent = "No tracing data";
        return;
      }

      // Key/Value
      if (tracing.kv) {
        const kvTitle = document.createElement("div");
        kvTitle.className = "collapsible-title nested-collapsible-title";
        kvTitle.innerText = "Key/Value";
        container.appendChild(kvTitle);

        const kvContent = document.createElement("div");
        kvContent.className =
          "collapsible-content nested-collapsible-content";
        container.appendChild(kvContent);

        // Ensure the open state matches what we have in runnerOpenState
        let subKey = "Key/Value";
        applyOpenState(
          kvTitle,
          kvContent,
          getSubSectionOpen(runnerId, subKey)
        );

        kvTitle.onclick = () => {
          toggleCollapsible(kvTitle, kvContent);
          setSubSectionOpen(
            runnerId,
            subKey,
            kvContent.style.display === "block"
          );
        };
        renderKeyValue(kvContent, tracing.kv);
      }

      // Events
      if (tracing.events) {
        const eTitle = document.createElement("div");
        eTitle.className = "collapsible-title nested-collapsible-title";
        eTitle.innerText = "Events";
        container.appendChild(eTitle);

        const eContent = document.createElement("div");
        eContent.className = "collapsible-content nested-collapsible-content";
        container.appendChild(eContent);

        let subKey = "Events";
        applyOpenState(eTitle, eContent, getSubSectionOpen(runnerId, subKey));

        eTitle.onclick = () => {
          toggleCollapsible(eTitle, eContent);
          setSubSectionOpen(
            runnerId,
            subKey,
            eContent.style.display === "block"
          );
        };

        // Create a button to export the events to JSON
        const exportBtn = document.createElement("button");
        exportBtn.textContent = "Export Events to JSON";
        exportBtn.style.marginBottom = "8px";
        exportBtn.onclick = () => exportEventsToJSON(tracing.events);
        eContent.appendChild(exportBtn);

        // Render the events
        renderEvents(eContent, tracing.events);
      }

      // Graphs
      if (tracing.graph) {
        renderGraphs(container, tracing.graph);
      }
    }

    // ------------------------------
    // Global State Accessors
    // ------------------------------
    function getRunnerState(id) {
      if (!runnerOpenState[id]) {
        runnerOpenState[id] = { open: false, sub: {} };
      }
      return runnerOpenState[id];
    }

    function isRunnerOpen(id) {
      return getRunnerState(id).open;
    }
    function setRunnerOpen(id, open) {
      getRunnerState(id).open = open;
    }

    function getSubSectionOpen(runnerId, subsection) {
      return getRunnerState(runnerId).sub[subsection] === true;
    }
    function setSubSectionOpen(runnerId, subsection, open) {
      getRunnerState(runnerId).sub[subsection] = open;
    }

    // ------------------------------
    // Worker
    // ------------------------------
    async function refreshWorker() {
      const sec = $("workerSection");
      sec.textContent = "Loading...";
      try {
        const data = await fetchJSON("/debug/worker/");
        sec.innerHTML = "";
        renderTracing(sec, data.tracing, "__WORKER__"); // use a special ID
      } catch (e) {
        sec.textContent = "Error: " + e;
      }
    }

    // ------------------------------
    // Runners
    // ------------------------------
    async function refreshRunners() {
      const rl = $("runnersList");
      rl.textContent = "Loading...";
      try {
        const data = await fetchJSON("/debug/runners/");
        rl.innerHTML = "";

        data.runners.forEach((r) => {
          const runnerId = String(r.id);

          const wrap = document.createElement("div");
          wrap.style.marginBottom = "16px";

          // Collapsible runner title
          const title = document.createElement("div");
          title.className = "collapsible-title";
          title.innerText = `room: ${r.room} — status: ${r.status}, task_id: ${r.task_id}  ${r.id}`;
          wrap.appendChild(title);

          // Collapsible content
          const content = document.createElement("div");
          content.className = "collapsible-content";
          wrap.appendChild(content);

          // Apply saved open state from runnerOpenState
          applyOpenState(title, content, isRunnerOpen(runnerId));

          // On title click => toggle + fetch details (only if we open)
          title.onclick = async () => {
            if (content.style.display !== "block") {
              // about to open
              content.textContent = "Loading...";
              toggleCollapsible(title, content);
              setRunnerOpen(runnerId, true);
              await fetchRunnerDetails(runnerId, content);
            } else {
              // about to close
              toggleCollapsible(title, content);
              setRunnerOpen(runnerId, false);
            }
          };

          rl.appendChild(wrap);
          // If runner is open from before, we fetch details right away
          if (isRunnerOpen(runnerId)) {
            fetchRunnerDetails(runnerId, content);
          }
        });
      } catch (e) {
        rl.textContent = "Error: " + e;
      }
    }

    async function fetchRunnerDetails(id, container) {
      try {
        const data = await fetchJSON(
          `/debug/runner/?id=${encodeURIComponent(id)}`
        );
        container.innerHTML = "";

        const dataDiv = document.createElement("div");
        container.appendChild(dataDiv);

        await loadRunnerTracing(id, dataDiv);
      } catch (e) {
        container.textContent = "Error: " + e;
      }
    }

    async function loadRunnerTracing(id, container) {
      try {
        const d = await fetchJSON(
          `/debug/runner/?id=${encodeURIComponent(id)}`
        );
        container.innerHTML = "";
        renderTracing(container, d.tracing, id);
      } catch (e) {
        container.textContent = "Error: " + e;
      }
    }

    // Initial calls
    refreshWorker();
    refreshRunners();
  </script>
</body>

</html>"""
        return web.Response(text=html_content, content_type="text/html")

    async def _worker_debug(self, request: web.Request) -> web.Response:
        """Worker debug endpoint - matching structure."""
        try:
            if not self._worker:
                return web.json_response({"tracing": None})

            # Get tracing data from the tracing system
            from .tracing import Tracing

            tracing_data = Tracing.export_for_handle("worker")

            # Add worker stats as key-value data if not already present
            if not tracing_data.get("kv"):
                try:
                    stats = self._worker.get_stats()
                    tracing_data["kv"] = {
                        "agent_id": stats.get("agent_id", "Unknown"),
                        "executor_type": (
                            getattr(
                                self._worker.options, "executor_type", "Unknown"
                            ).value
                            if hasattr(self._worker.options, "executor_type")
                            else "Unknown"
                        ),
                        "worker_load": stats.get("worker_load", 0.0),
                        "current_jobs": stats.get("current_jobs", 0),
                        "max_processes": stats.get("max_processes", 0),
                        "backend_connected": stats.get("backend_connected", False),
                        "worker_id": stats.get("worker_id", "unregistered"),
                        "draining": stats.get("draining", False),
                        "register": stats.get("register", False),
                    }
                except Exception as stats_error:
                    logger.error(f"Error getting worker stats: {stats_error}")
                    tracing_data["kv"] = {
                        "agent_id": getattr(
                            self._worker.options, "agent_id", "Unknown"
                        ),
                        "executor_type": (
                            getattr(
                                self._worker.options, "executor_type", "Unknown"
                            ).value
                            if hasattr(self._worker.options, "executor_type")
                            else "Unknown"
                        ),
                        "worker_load": 0.0,
                        "current_jobs": 0,
                        "max_processes": 0,
                        "backend_connected": False,
                        "worker_id": "unregistered",
                        "draining": False,
                        "register": getattr(self._worker.options, "register", False),
                        "error": f"Stats error: {str(stats_error)}",
                    }

            # Add some default events if none exist
            if not tracing_data.get("events"):
                tracing_data["events"] = [
                    {
                        "timestamp": time.time(),
                        "name": "worker_started",
                        "data": {
                            "agent_id": getattr(
                                self._worker.options, "agent_id", "Unknown"
                            )
                        },
                    }
                ]

            # Add graphs if available
            try:
                if (
                    hasattr(self._worker, "_worker_load_graph")
                    and self._worker._worker_load_graph
                ):
                    graph_data = self._worker._worker_load_graph.export()
                    if graph_data.get("data"):
                        tracing_data["graph"] = [graph_data]
            except Exception as graph_error:
                logger.error(f"Error getting worker load graph: {graph_error}")

            return web.json_response({"tracing": tracing_data})
        except Exception as e:
            logger.error(f"Error in worker debug endpoint: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _runners_list(self, request: web.Request) -> web.Response:
        """Runners list endpoint - matching structure."""
        try:
            if not self._worker:
                return web.json_response({"runners": []})

            runners = []

            # Get current jobs as runners
            if hasattr(self._worker, "_current_jobs"):
                for job_id, job_info in self._worker._current_jobs.items():
                    try:
                        # Extract room information safely
                        room_options = {}
                        if hasattr(job_info, "job") and job_info.job:
                            if hasattr(job_info.job, "room_options"):
                                room_options = job_info.job.room_options
                            elif isinstance(job_info.job, dict):
                                room_options = job_info.job.get("room_options", {})

                        runners.append(
                            {
                                "id": job_id,
                                "room": getattr(room_options, "room_id", "unknown"),
                                "status": "running",  # Default status
                                "task_id": job_id,  # Changed from job_id to task_id
                            }
                        )
                    except Exception as e:
                        # Log error but continue with other jobs
                        logger.warning(f"Error processing job {job_id}: {e}")
                        runners.append(
                            {
                                "id": job_id,
                                "room": "error",
                                "status": "error",
                                "task_id": job_id,  # Changed from job_id to task_id
                            }
                        )

            # If no runners found and we're in direct mode, create a placeholder
            if not runners and not getattr(self._worker.options, "register", False):
                # Check if we have any active processes/threads
                try:
                    if (
                        hasattr(self._worker, "process_manager")
                        and self._worker.process_manager
                    ):
                        stats = self._worker.process_manager.get_stats()
                        # New execution module returns different stats format
                        executor_stats = stats.get("executor_stats", {})
                        active_tasks = executor_stats.get("pending_tasks", 0)
                        running_tasks = executor_stats.get("running_tasks", 0)
                        total_active = active_tasks + running_tasks

                        for i in range(total_active):
                            runners.append(
                                {
                                    "id": f"direct_job_{i}",
                                    "room": "direct_mode",
                                    "status": "running",
                                    "task_id": f"direct_job_{i}",  # Changed from job_id to task_id
                                }
                            )
                except Exception as e:
                    logger.warning(f"Error getting process manager stats: {e}")
                    # Add a default runner if we can't get stats
                    runners.append(
                        {
                            "id": "direct_job_0",
                            "room": "direct_mode",
                            "status": "unknown",
                            "task_id": "direct_job_0",  # Changed from job_id to task_id
                        }
                    )

            # If still no runners, add a placeholder for the current worker
            if not runners:
                runners.append(
                    {
                        "id": "worker_main",
                        "room": "main_worker",
                        "status": "idle",
                        "task_id": "worker_main",  # Changed from job_id to task_id
                    }
                )

            return web.json_response({"runners": runners})
        except Exception as e:
            logger.error(f"Error in _runners_list: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return web.json_response({"error": str(e)}, status=500)

    async def _runner_details(self, request: web.Request) -> web.Response:
        """Runner details endpoint - matching structure."""
        try:
            runner_id = request.query.get("id")
            if not runner_id:
                return web.json_response({"error": "Missing runner ID"}, status=400)

            if not self._worker:
                return web.json_response({"tracing": None})

            # Handle placeholder runners
            if runner_id == "worker_main":
                # Return worker-level tracing data
                from .tracing import Tracing

                tracing_data = Tracing.export_for_handle("worker")

                # Add worker-specific key-value data
                try:
                    stats = self._worker.get_stats()
                    tracing_data["kv"] = {
                        "runner_id": runner_id,
                        "type": "main_worker",
                        "status": "idle",
                        "agent_id": stats.get("agent_id", "Unknown"),
                        "executor_type": (
                            getattr(
                                self._worker.options, "executor_type", "Unknown"
                            ).value
                            if hasattr(self._worker.options, "executor_type")
                            else "Unknown"
                        ),
                        "register": stats.get("register", False),
                    }
                except Exception as e:
                    logger.warning(
                        f"Error getting worker stats for runner details: {e}"
                    )
                    tracing_data["kv"] = {
                        "runner_id": runner_id,
                        "type": "main_worker",
                        "status": "idle",
                        "agent_id": getattr(
                            self._worker.options, "agent_id", "Unknown"
                        ),
                        "executor_type": (
                            getattr(
                                self._worker.options, "executor_type", "Unknown"
                            ).value
                            if hasattr(self._worker.options, "executor_type")
                            else "Unknown"
                        ),
                        "register": getattr(self._worker.options, "register", False),
                        "error": f"Stats error: {str(e)}",
                    }

                return web.json_response({"tracing": tracing_data})

            # Handle direct job runners
            if runner_id.startswith("direct_job_"):
                from .tracing import Tracing

                tracing_data = Tracing.export_for_handle(f"runner_{runner_id}")

                # Add direct job key-value data
                tracing_data["kv"] = {
                    "runner_id": runner_id,
                    "type": "direct_job",
                    "status": "running",
                    "mode": "direct",
                }

                return web.json_response({"tracing": tracing_data})

            # Handle regular backend jobs
            if not hasattr(self._worker, "_current_jobs"):
                return web.json_response({"tracing": None})

            # Find the runner
            job_info = self._worker._current_jobs.get(runner_id)
            if not job_info:
                return web.json_response({"tracing": None})

            # Get tracing data for this specific runner
            from .tracing import Tracing

            tracing_data = Tracing.export_for_handle(f"runner_{runner_id}")

            # Add job-specific key-value data
            room_options = {}
            if hasattr(job_info, "job") and job_info.job:
                if hasattr(job_info.job, "room_options"):
                    room_options = job_info.job.room_options
                elif isinstance(job_info.job, dict):
                    room_options = job_info.job.get("room_options", {})

            tracing_data["kv"] = {
                "task_id": runner_id,  # Changed from job_id to task_id
                "room_id": getattr(room_options, "room_id", "unknown"),
                "room_name": getattr(room_options, "name", "unknown"),
                "status": "running",
                "worker_id": getattr(job_info, "worker_id", "unknown"),
                "url": getattr(job_info, "url", "unknown"),
            }

            # Add job-specific events if none exist
            if not tracing_data.get("events"):
                tracing_data["events"] = [
                    {
                        "timestamp": time.time(),
                        "name": "job_started",
                        "data": {
                            "job_id": runner_id,
                            "room_id": getattr(room_options, "room_id", "unknown"),
                        },
                    }
                ]

            return web.json_response({"tracing": tracing_data})
        except Exception as e:
            logger.error(f"Error in _runner_details: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return web.json_response({"error": str(e)}, status=500)

    async def _api_status(self, request: web.Request) -> web.Response:
        """API endpoint for dashboard data."""
        try:
            worker_status = None
            stats = None

            if self._worker:
                try:
                    worker_status = {
                        "available": True,
                        "connected": (
                            self._worker.backend_connection.is_connected
                            if self._worker.backend_connection
                            else False
                        ),
                        "worker_id": (
                            self._worker.backend_connection.worker_id
                            if self._worker.backend_connection
                            else "unregistered"
                        ),
                        "options": {
                            "agent_id": getattr(
                                self._worker.options, "agent_id", "Unknown"
                            ),
                            "executor_type": (
                                getattr(
                                    self._worker.options, "executor_type", "Unknown"
                                ).value
                                if hasattr(self._worker.options, "executor_type")
                                else "Unknown"
                            ),
                            "register": getattr(
                                self._worker.options, "register", False
                            ),
                            "max_processes": getattr(
                                self._worker.options, "max_processes", 0
                            ),
                            "log_level": getattr(
                                self._worker.options, "log_level", "INFO"
                            ),
                        },
                    }
                except Exception as e:
                    worker_status = {"available": False, "error": str(e)}

                try:
                    stats = self._worker.get_stats()
                except Exception as e:
                    stats = {"error": str(e)}

            server_info = {
                "host": self._host,
                "port": self._port,
                "endpoints": [
                    "/",
                    "/health",
                    "/worker",
                    "/stats",
                    "/debug",
                    "/api/status",
                    "/debug/worker/",
                    "/debug/runners/",
                    "/debug/runner/",
                ],
            }

            return web.json_response(
                {
                    "worker": worker_status,
                    "stats": stats,
                    "server": server_info,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.Response(text="OK", content_type="text/plain")

    async def _worker_status(self, request: web.Request) -> web.Response:
        """Worker status endpoint."""
        if not self._worker:
            return web.json_response({"error": "Worker not available"})

        try:
            # Get current jobs count from the worker
            current_jobs = (
                len(self._worker._current_jobs)
                if hasattr(self._worker, "_current_jobs")
                else 0
            )

            status = {
                "agent_id": getattr(self._worker.options, "agent_id", "Unknown"),
                "executor_type": (
                    getattr(self._worker.options, "executor_type", "Unknown").value
                    if hasattr(self._worker.options, "executor_type")
                    else "Unknown"
                ),
                "active_jobs": current_jobs,
                "connected": (
                    self._worker.backend_connection.is_connected
                    if self._worker.backend_connection
                    else False
                ),
                "worker_id": (
                    self._worker.backend_connection.worker_id
                    if self._worker.backend_connection
                    else "unregistered"
                ),
                "register": getattr(self._worker.options, "register", False),
                "draining": getattr(self._worker, "_draining", False),
                "worker_load": getattr(self._worker, "_worker_load", 0.0),
            }
        except Exception as e:
            logger.error(f"Error getting worker status: {e}")
            status = {
                "agent_id": getattr(self._worker.options, "agent_id", "Unknown"),
                "executor_type": "Unknown",
                "active_jobs": 0,
                "connected": False,
                "worker_id": "unregistered",
                "register": getattr(self._worker.options, "register", False),
                "draining": False,
                "worker_load": 0.0,
                "error": str(e),
            }

        return web.json_response(status)

    async def _worker_stats(self, request: web.Request) -> web.Response:
        """Worker statistics endpoint."""
        if not self._worker:
            return web.json_response({"error": "Worker not available"})

        try:
            stats = self._worker.get_stats()
            return web.json_response(stats)
        except Exception as e:
            logger.error(f"Error getting worker stats: {e}")
            return web.json_response(
                {
                    "error": str(e),
                    "agent_id": getattr(self._worker.options, "agent_id", "Unknown"),
                    "executor_type": "Unknown",
                    "current_jobs": 0,
                    "max_processes": getattr(self._worker.options, "max_processes", 0),
                    "register": getattr(self._worker.options, "register", False),
                }
            )

    async def _debug_info(self, request: web.Request) -> web.Response:
        """Debug information endpoint."""
        try:
            debug_info = {
                "server": {
                    "host": self._host,
                    "port": self._port,
                    "endpoints": ["/", "/health", "/worker", "/stats", "/debug"],
                },
                "worker": {
                    "available": self._worker is not None,
                    "options": (
                        {
                            "agent_id": (
                                getattr(self._worker.options, "agent_id", "Unknown")
                                if self._worker
                                else None
                            ),
                            "executor_type": (
                                getattr(
                                    self._worker.options, "executor_type", "Unknown"
                                ).value
                                if self._worker
                                and hasattr(self._worker.options, "executor_type")
                                else "Unknown"
                            ),
                            "register": (
                                getattr(self._worker.options, "register", False)
                                if self._worker
                                else None
                            ),
                            "max_processes": (
                                getattr(self._worker.options, "max_processes", 0)
                                if self._worker
                                else 0
                            ),
                        }
                        if self._worker
                        else None
                    ),
                },
            }

            return web.json_response(debug_info)
        except Exception as e:
            logger.error(f"Error in debug info endpoint: {e}")
            return web.json_response(
                {
                    "error": str(e),
                    "server": {
                        "host": self._host,
                        "port": self._port,
                        "endpoints": ["/", "/health", "/worker", "/stats", "/debug"],
                    },
                    "worker": {
                        "available": self._worker is not None,
                        "options": None,
                    },
                }
            )
