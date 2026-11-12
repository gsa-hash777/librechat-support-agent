"""Small, process-local Prometheus metrics without storing request contents.

Only registered route templates and operation IDs become labels. Counters reset
on restart; run one worker per container and scrape each container separately.
HTTP/tool success means transport success, not a correct Agent answer. The
evaluation suite, rather than these counters, measures business-task success.
"""

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Callable

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send


BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


def _labels(**values: str) -> str:
    def escape(value):
        return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    return "{" + ",".join(f'{name}="{escape(str(value))}"' for name, value in values.items()) + "}"


@dataclass
class _Histogram:
    count: int = 0
    total: float = 0.0
    buckets: list[int] = field(default_factory=lambda: [0] * len(BUCKETS))

    def observe(self, seconds: float):
        self.count += 1
        self.total += seconds
        for index, bound in enumerate(BUCKETS):
            if seconds <= bound:
                self.buckets[index] += 1

    def render(self, name: str, labels: dict) -> list[str]:
        lines = [f'{name}_bucket{_labels(**labels, le=format(bound, "g"))} {count}'
                 for bound, count in zip(BUCKETS, self.buckets)]
        lines.extend((f'{name}_bucket{_labels(**labels, le="+Inf")} {self.count}',
                      f'{name}_sum{_labels(**labels)} {self.total:.9f}',
                      f'{name}_count{_labels(**labels)} {self.count}'))
        return lines


class MetricsRegistry:
    """Bounded, thread-safe counters; accepts no user-supplied label values."""

    def __init__(self, routes, *, clock: Callable[[], float] = time.perf_counter):
        self.clock = clock
        self.started = clock()
        self._lock = threading.Lock()
        self._routes = {}
        for route in routes:
            path = getattr(route, "path", None)
            if path is None:
                continue
            operation = getattr(route, "operation_id", None) if path.startswith("/api/") else None
            self._routes[id(route)] = (path, operation)
        self._requests = Counter()
        self._auth_failures = Counter()
        self._tools = Counter()
        self._http_latency = {}
        self._tool_latency = {}
        self._readiness_checks = Counter()
        self._ready = None
        self._in_flight = 0

    def enter(self):
        with self._lock:
            self._in_flight += 1

    def complete(self, scope: Scope, status: int | None, seconds: float, *, exception=False, aborted=False):
        route, operation = self._routes.get(id(scope.get("route")), ("unmatched", None))
        method = scope.get("method", "OTHER")
        method = method if method in METHODS else "OTHER"
        status_class = f"{status // 100}xx" if isinstance(status, int) and 100 <= status <= 599 else "unknown"
        if aborted:
            status_class, outcome = "aborted", "aborted"
        elif exception:
            status_class, outcome = "exception", "error"
        elif status is not None and 200 <= status < 300:
            outcome = "success"
        elif status is not None and 400 <= status < 500:
            outcome = "rejected"
        else:
            outcome = "error"
        duration = seconds if math.isfinite(seconds) and seconds >= 0 else 0.0
        with self._lock:
            self._in_flight -= 1
            self._requests[route, method, status_class] += 1
            self._http_latency.setdefault((route, method), _Histogram()).observe(duration)
            if status in (401, 403):
                self._auth_failures[route, "unauthenticated" if status == 401 else "forbidden"] += 1
            if operation:
                self._tools[operation, outcome] += 1
                self._tool_latency.setdefault(operation, _Histogram()).observe(duration)

    def readiness(self, ready: bool):
        with self._lock:
            self._ready = ready
            self._readiness_checks["ready" if ready else "not_ready"] += 1

    def render(self) -> str:
        # Formatting under the same lock produces one consistent scrape; its
        # size depends on routes/methods, never on traffic volume or query text.
        with self._lock:
            lines = [
                "# HELP support_process_uptime_seconds Seconds since this process registered metrics.",
                "# TYPE support_process_uptime_seconds gauge",
                f"support_process_uptime_seconds {max(0.0, self.clock() - self.started):.6f}",
                "# HELP support_http_in_flight Active HTTP requests in this process.",
                "# TYPE support_http_in_flight gauge",
                f"support_http_in_flight {self._in_flight}",
                "# HELP support_http_requests_total Completed HTTP requests; exception/aborted are not HTTP response codes.",
                "# TYPE support_http_requests_total counter",
            ]
            for (route, method, status), count in sorted(self._requests.items()):
                lines.append(f'support_http_requests_total{_labels(route=route, method=method, status_class=status)} {count}')
            lines.extend(("# HELP support_http_auth_failures_total HTTP 401 or 403 responses; intentionally opaque 404 denials remain in the audit log.",
                          "# TYPE support_http_auth_failures_total counter"))
            for (route, reason), count in sorted(self._auth_failures.items()):
                lines.append(f'support_http_auth_failures_total{_labels(route=route, reason=reason)} {count}')
            lines.extend(("# HELP support_tool_requests_total Tool HTTP outcomes, not Agent task correctness. Rejected includes HTTP 4xx.",
                          "# TYPE support_tool_requests_total counter"))
            for (operation, outcome), count in sorted(self._tools.items()):
                lines.append(f'support_tool_requests_total{_labels(operation=operation, outcome=outcome)} {count}')
            lines.extend(("# HELP support_http_request_duration_seconds Full HTTP application duration including response streaming.",
                          "# TYPE support_http_request_duration_seconds histogram"))
            for (route, method), histogram in sorted(self._http_latency.items()):
                lines.extend(histogram.render("support_http_request_duration_seconds", {"route": route, "method": method}))
            lines.extend(("# HELP support_tool_request_duration_seconds Tool HTTP application duration; excludes upstream model generation.",
                          "# TYPE support_tool_request_duration_seconds histogram"))
            for operation, histogram in sorted(self._tool_latency.items()):
                lines.extend(histogram.render("support_tool_request_duration_seconds", {"operation": operation}))
            lines.extend(("# HELP support_readiness_checks_total Explicit dependency readiness probe results.",
                          "# TYPE support_readiness_checks_total counter"))
            for outcome, count in sorted(self._readiness_checks.items()):
                lines.append(f'support_readiness_checks_total{_labels(outcome=outcome)} {count}')
            if self._ready is not None:
                lines.extend(("# HELP support_ready Last observed readiness result; absent before first readiness probe.",
                              "# TYPE support_ready gauge", f"support_ready {int(self._ready)}"))
        return "\n".join(lines) + "\n"


class MetricsMiddleware:
    """Pure ASGI wrapper, so streaming is observed without buffering its body."""

    def __init__(self, app: ASGIApp, registry: MetricsRegistry):
        self.app, self.registry = app, registry

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = self.registry.clock()
        self.registry.enter()
        status, failed, aborted = None, False, False

        async def observe_send(message: Message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, observe_send)
        except Exception:
            failed = True
            raise
        except BaseException:
            # Cancellation must propagate, while releasing the inflight gauge.
            aborted = True
            raise
        finally:
            self.registry.complete(scope, status, self.registry.clock() - started, exception=failed, aborted=aborted)


def sqlite_readiness(db_path: Path | str) -> Callable[[], bool]:
    """Read-only, bounded SQLite check. Never creates a missing database.

    No inference, corpus hashing, vector cache warming, or database writes occur
    on this probe. It checks accessibility plus essential application tables.
    """
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"

    def check() -> bool:
        connection = sqlite3.connect(uri, uri=True, timeout=0.25)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("SELECT id FROM users LIMIT 0")
            connection.execute("SELECT id FROM tickets LIMIT 0")
            return True
        finally:
            connection.close()
    return check


def register_observability(app: FastAPI, *, authorize_metrics: Callable, readiness_check: Callable[[], bool]) -> MetricsRegistry:
    """Install once after all application routes; authorization is mandatory.

    authorize_metrics is a FastAPI dependency which must raise 401/403 unless
    the current caller is an administrator or presents the internal scrape
    credential. A false return is not an authorization rejection in FastAPI.
    readiness_check is synchronous and runs in FastAPI's worker threadpool.
    """
    if not callable(authorize_metrics) or not callable(readiness_check):
        raise ValueError("Metrics authorization and readiness callbacks are required.")
    if getattr(app.state, "metrics", None) is not None:
        raise ValueError("Observability is already registered.")
    if any(getattr(route, "path", None) in {"/metrics", "/ready"} for route in app.routes):
        raise ValueError("Observability routes are already registered.")

    @app.get("/ready", include_in_schema=False)
    def ready():
        try:
            available = readiness_check() is True
        except Exception:
            available = False
        registry.readiness(available)
        return JSONResponse({"status": "ready" if available else "not_ready", "service": "support-agent"},
                            status_code=200 if available else 503, headers={"Cache-Control": "no-store"})

    @app.get("/metrics", include_in_schema=False, dependencies=[Depends(authorize_metrics)])
    def metrics():
        return Response(registry.render(), media_type="text/plain; version=0.0.4; charset=utf-8",
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    registry = MetricsRegistry(app.routes)
    app.state.metrics = registry
    app.add_middleware(MetricsMiddleware, registry=registry)
    return registry
