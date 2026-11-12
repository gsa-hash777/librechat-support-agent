import asyncio
from concurrent.futures import ThreadPoolExecutor
import hmac
import sqlite3
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.testclient import TestClient
import pytest

from app.observability import MetricsMiddleware, MetricsRegistry, register_observability, sqlite_readiness


SCRAPE_KEY = "unit-test-internal-scrape-token"


def protected_metrics(request: Request):
    actual = request.headers.get("authorization", "")
    if not hmac.compare_digest(actual.encode(), ("Bearer " + SCRAPE_KEY).encode()):
        raise HTTPException(401, "Metrics authentication required.")


def make_app(check=lambda: True):
    app = FastAPI()

    @app.get("/api/tickets/{ticket_id}", operation_id="getMyTicket")
    def ticket(ticket_id: str, code: int = 200):
        if code != 200:
            raise HTTPException(code, "unit-test-private-error")
        return {"ticket_id": ticket_id}

    @app.post("/api/crash", operation_id="crashTool")
    def crash():
        raise RuntimeError("unit-test-password-from-exception")

    @app.get("/console", include_in_schema=False)
    def console():
        return Response("console")

    registry = register_observability(app, authorize_metrics=protected_metrics, readiness_check=check)
    return app, registry


def test_http_tool_statuses_auth_failures_and_histogram_are_distinct():
    app, registry = make_app()
    with TestClient(app) as client:
        for code in (200, 401, 403, 404, 422, 500, 503):
            assert client.get(f"/api/tickets/T-SECRET?code={code}").status_code == code
        assert client.get("/console").status_code == 200
    text = registry.render()
    assert 'support_tool_requests_total{operation="getMyTicket",outcome="success"} 1' in text
    assert 'support_tool_requests_total{operation="getMyTicket",outcome="rejected"} 4' in text
    assert 'support_tool_requests_total{operation="getMyTicket",outcome="error"} 2' in text
    assert 'support_http_requests_total{route="/api/tickets/{ticket_id}",method="GET",status_class="4xx"} 4' in text
    assert 'support_http_auth_failures_total{route="/api/tickets/{ticket_id}",reason="unauthenticated"} 1' in text
    assert 'support_http_auth_failures_total{route="/api/tickets/{ticket_id}",reason="forbidden"} 1' in text
    assert 'support_tool_request_duration_seconds_count{operation="getMyTicket"} 7' in text
    assert 'support_tool_request_duration_seconds_bucket{operation="getMyTicket",le="+Inf"} 7' in text
    assert "support_http_in_flight 0\n" in text
    assert "unit-test-private-error" not in text
    assert "T-SECRET" not in text


def test_metrics_protected_not_in_openapi_and_not_cacheable():
    app, registry = make_app()
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"authorization": "Bearer wrong"}).status_code == 401
        response = client.get("/metrics", headers={"authorization": "Bearer " + SCRAPE_KEY})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
        assert response.headers["cache-control"] == "no-store"
        assert "/metrics" not in client.get("/openapi.json").json()["paths"]
        assert "/ready" not in client.get("/openapi.json").json()["paths"]
    assert SCRAPE_KEY not in registry.render()


def test_unknown_paths_methods_queries_headers_cannot_expand_labels():
    app, registry = make_app()
    with TestClient(app) as client:
        for number in range(30):
            client.request(f"SECRET-METHOD-{number}", f"/missing/private-{number}?q=private-prompt",
                           headers={"Authorization": "Bearer private-key", "Cookie": "email=private@example.invalid"})
            client.get(f"/api/tickets/private-user-{number}?q=private-prompt")
    text = registry.render()
    assert 'support_http_requests_total{route="unmatched",method="OTHER",status_class="4xx"} 30' in text
    assert 'support_http_requests_total{route="/api/tickets/{ticket_id}",method="GET",status_class="2xx"} 30' in text
    lines = [line for line in text.splitlines() if line.startswith("support_http_requests_total{")]
    assert len(lines) == 2
    for private in ("private-", "private@", "SECRET-METHOD", "authorization", "cookie"):
        assert private.lower() not in text.lower()


def test_unhandled_error_is_counted_once_and_exception_is_not_swallowed():
    app, registry = make_app()
    with TestClient(app) as client, pytest.raises(RuntimeError, match="unit-test-password"):
        client.post("/api/crash")
    text = registry.render()
    assert 'support_http_requests_total{route="/api/crash",method="POST",status_class="exception"} 1' in text
    assert 'support_tool_requests_total{operation="crashTool",outcome="error"} 1' in text
    assert "support_http_in_flight 0\n" in text
    assert "unit-test-password" not in text


def test_streaming_passes_through_without_buffering_and_measures_complete_duration():
    route = SimpleNamespace(path="/api/stream", operation_id="streamTool")
    now = [10.0]
    registry = MetricsRegistry([route], clock=lambda: now[0])
    sent = []

    async def downstream(scope, receive, send):
        scope["route"] = route
        assert "support_http_in_flight 1\n" in registry.render()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        assert sent[-1]["body"] == b"first"
        now[0] += 0.5
        await send({"type": "http.response.body", "body": b"last", "more_body": False})

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    asyncio.run(MetricsMiddleware(downstream, registry)({"type": "http", "method": "GET"}, receive, send))
    text = registry.render()
    assert 'support_tool_request_duration_seconds_sum{operation="streamTool"} 0.500000000' in text
    assert 'support_tool_request_duration_seconds_bucket{operation="streamTool",le="0.25"} 0' in text
    assert 'support_tool_request_duration_seconds_bucket{operation="streamTool",le="0.5"} 1' in text
    assert len(sent) == 3


def test_cancelled_request_releases_gauge_and_propagates():
    registry = MetricsRegistry([])

    async def cancelled(scope, receive, send):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(MetricsMiddleware(cancelled, registry)({"type": "http", "method": "GET"}, None, None))
    text = registry.render()
    assert "support_http_in_flight 0\n" in text
    assert 'status_class="aborted"} 1' in text


def test_non_http_scope_is_not_observed():
    registry = MetricsRegistry([])
    observed = []

    async def passthrough(scope, receive, send):
        observed.append(scope["type"])

    asyncio.run(MetricsMiddleware(passthrough, registry)({"type": "lifespan"}, None, None))
    assert observed == ["lifespan"]
    assert "support_http_requests_total{" not in registry.render()


@pytest.mark.parametrize("outcome", [True, False, "truthy", None, "exception"])
def test_readiness_status_and_errors_are_safe(outcome):
    def check():
        if outcome == "exception":
            raise OSError("secret-password /private/database/path")
        return outcome

    app, registry = make_app(check)
    assert "\nsupport_ready " not in registry.render()
    with TestClient(app) as client:
        response = client.get("/ready")
    expected = outcome is True
    assert response.status_code == (200 if expected else 503)
    assert response.json() == {"status": "ready" if expected else "not_ready", "service": "support-agent"}
    assert f"support_ready {int(expected)}\n" in registry.render()
    assert "secret-password" not in response.text + registry.render()


def test_sqlite_readiness_is_readonly_and_does_not_create_missing_db(tmp_path):
    db_path = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        sqlite_readiness(db_path)()
    assert not db_path.exists()
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE users (id TEXT PRIMARY KEY)")
    with pytest.raises(sqlite3.OperationalError):
        sqlite_readiness(db_path)()
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE tickets (id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO users VALUES('private-user')")
    before = db_path.read_bytes()
    assert sqlite_readiness(db_path)() is True
    assert db_path.read_bytes() == before


def test_concurrent_requests_do_not_lose_counts():
    route = SimpleNamespace(path="/api/check", operation_id="checkTool")
    registry = MetricsRegistry([route])

    def one(_):
        registry.enter()
        registry.complete({"route": route, "method": "GET"}, 200, 0.25)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(one, range(400)))
    text = registry.render()
    assert 'support_tool_requests_total{operation="checkTool",outcome="success"} 400' in text
    assert 'support_tool_request_duration_seconds_sum{operation="checkTool"} 100.000000000' in text
    assert "support_http_in_flight 0\n" in text


def test_registration_requires_callbacks_and_is_single_use():
    app = FastAPI()
    with pytest.raises(ValueError, match="required"):
        register_observability(app, authorize_metrics=None, readiness_check=lambda: True)
    register_observability(app, authorize_metrics=protected_metrics, readiness_check=lambda: True)
    with pytest.raises(ValueError, match="already registered"):
        register_observability(app, authorize_metrics=protected_metrics, readiness_check=lambda: True)
