"""SSE first-token waiting and absolute budgets, without real delays or model calls."""
from http.client import HTTPResponse
import io
import json
import socket
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from evaluation import agent


START = {"streamId": "existing-stream", "generationCreatedAt": 123, "conversationId": "existing-conversation"}


class Clock:
    now = 100.0

    def monotonic(self):
        return self.now


class FakeSocket:
    def __init__(self):
        self.timeout = None
        self.timeouts = []

    def settimeout(self, seconds):
        assert seconds > 0
        self.timeout = seconds
        self.timeouts.append(seconds)


class Response:
    def __init__(self, clock, schedule):
        self.clock = clock
        self.schedule = list(schedule)
        self.sock = FakeSocket()
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=self.sock))
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def readline(self):
        if not self.schedule:
            return b""
        delay, line = self.schedule.pop(0)
        if delay > self.sock.timeout:
            self.clock.now += self.sock.timeout
            raise TimeoutError("simulated blocking read reached its socket timeout")
        self.clock.now += delay
        return line


class Opener:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        if hasattr(self.response, "sock"):
            self.response.sock.settimeout(timeout)
        return self.response


def api_for(response):
    return SimpleNamespace(base="http://localhost:3080", token="fixture-token", opener=Opener(response))


def sse(value):
    return ("data: " + json.dumps(value) + "\n").encode()


def test_first_token_after_20_seconds_can_finish_same_get(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(agent.time, "monotonic", clock.monotonic)
    usage = {"input_tokens": 1000, "output_tokens": 2}
    response = Response(clock, [(35, sse({"event": "on_token_usage", "data": usage})),
                                (0, b"\n"), (2, sse({"final": True})), (0, b"\n")])
    api = api_for(response)
    events = []
    agent.stream(api, START, 220, events)
    assert clock.now == 137
    assert events == [usage] and response.closed
    assert len(api.opener.requests) == 1
    request, timeout = api.opener.requests[0]
    assert request.get_method() == "GET" and timeout == 120
    assert "existing-stream" in request.full_url and "resume=true" in request.full_url
    assert response.sock.timeouts[-1] == 83


def test_heartbeat_does_not_reset_total_deadline(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(agent.time, "monotonic", clock.monotonic)
    response = Response(clock, [(15, b": heartbeat\n"), (19, b": heartbeat\n"), (19, sse({"final": True}))])
    api = api_for(response)
    with pytest.raises(agent.BootstrapError, match="timed out"):
        agent.stream(api, START, 140, [])
    assert clock.now == 140
    assert response.sock.timeouts[-1] == 6
    assert len(api.opener.requests) == 1 and response.closed


def test_expired_budget_never_opens_stream(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(agent.time, "monotonic", clock.monotonic)
    api = api_for(Response(clock, []))
    with pytest.raises(agent.BootstrapError, match="timed out"):
        agent.stream(api, START, 100, [])
    assert api.opener.requests == []


def test_connection_time_also_consumes_the_stream_budget(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(agent.time, "monotonic", clock.monotonic)
    response = Response(clock, [(30, sse({"final": True})), (0, b"\n")])
    api = api_for(response)
    original = api.opener.open
    def delayed_open(request, timeout):
        clock.now += 15
        return original(request, timeout)
    api.opener.open = delayed_open
    with pytest.raises(agent.BootstrapError, match="timed out"):
        agent.stream(api, START, 140, [])
    assert response.sock.timeouts[-1] == 25 and clock.now == 140


def test_socketless_fake_is_rejected_instead_of_claiming_deadline_support(monkeypatch):
    monkeypatch.setattr(agent.time, "monotonic", lambda: 100)
    api = api_for(io.BytesIO(b'data: {"final": true}\n\n'))
    with pytest.raises(agent.BootstrapError, match="deadline-aware socket"):
        agent.stream(api, START, 220, [])
    assert len(api.opener.requests) == 1


def test_real_httpresponse_exposes_supported_socket_path_without_waiting():
    # Socketpair is local and fully preloaded; no server, network service or sleep.
    left, right = socket.socketpair()
    try:
        left.settimeout(1)
        payload = b'data: {"final": true}\n\n'
        right.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
                      + str(len(payload)).encode() + b"\r\n\r\n" + payload)
        right.shutdown(socket.SHUT_WR)
        response = HTTPResponse(left)
        response.begin()
        assert response.fp.raw._sock is left
        agent.stream(api_for(response), START, agent.time.monotonic() + 1, [])
        assert response.isclosed()
    finally:
        left.close()
        right.close()


def test_approval_event_is_still_rejected_and_no_reconnect(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(agent.time, "monotonic", clock.monotonic)
    response = Response(clock, [(0, sse({"event": "on_pending_action"})), (0, b"\n")])
    api = api_for(response)
    with pytest.raises(agent.BootstrapError, match="never grants"):
        agent.stream(api, START, 220, [])
    assert len(api.opener.requests) == 1


def test_http404_existing_stream_is_not_recreated(monkeypatch):
    monkeypatch.setattr(agent.time, "monotonic", lambda: 100)
    calls = []
    def missing(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, 404, "gone", {}, None)
    api = SimpleNamespace(base="http://localhost:3080", token="fixture", opener=SimpleNamespace(open=missing))
    agent.stream(api, START, 220, [])
    assert len(calls) == 1 and calls[0].get_method() == "GET"


def test_deadline_failure_aborts_once_without_resubmitting_paid_post(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(agent.time, "monotonic", clock.monotonic)
    response = Response(clock, [(999, b"unreachable\n")])
    api = api_for(response)
    calls = []
    def request(method, path, payload=None):
        calls.append((method, path))
        if path == "/api/agents/chat":
            return START
        if path == "/api/agents/chat/abort":
            return {"ok": True}
        if path.startswith("/api/agents/chat/status/"):
            return {"active": False}
        if path.startswith("/api/messages/"):
            return []
        raise AssertionError("Unexpected request")
    api.request = request
    result = agent.turn(api, "fixture", "prompt", None, None, set(), 120, "fixture-model")
    assert result["execution_error"] == "Agent evaluation timed out."
    assert result["drained"] is True and result["usage_events"] == []
    assert calls.count(("POST", "/api/agents/chat")) == 1
    assert calls.count(("POST", "/api/agents/chat/abort")) == 1
    assert len(api.opener.requests) == 1
