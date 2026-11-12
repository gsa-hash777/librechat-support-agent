from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from scripts import monitor


TOKEN = "private-internal-token-do-not-display-123456"
METRICS = '''# HELP ignored ignored
support_process_uptime_seconds 120
support_http_in_flight 1
support_tool_requests_total{operation="private-operation",outcome="success"} 8
support_tool_requests_total{operation="private-operation",outcome="rejected"} 1
support_tool_requests_total{operation="private-operation",outcome="error"} 1
support_http_auth_failures_total{route="private-route",reason="forbidden"} 1
support_tool_request_duration_seconds_sum{operation="private-operation"} 3
support_tool_request_duration_seconds_count{operation="private-operation"} 10
support_tool_request_duration_seconds_bucket{operation="private-operation",le="0.1"} 2
support_tool_request_duration_seconds_bucket{operation="private-operation",le="0.5"} 10
support_tool_request_duration_seconds_bucket{operation="private-operation",le="+Inf"} 10
'''


def test_metrics_are_aggregated_without_identifiers_and_have_defined_rates():
    result = monitor.parse_metrics(METRICS)
    assert result["tool_requests"] == 10
    assert result["tool_http_success_rate"] == .8
    assert result["tool_http_error_rate"] == .1
    assert result["tool_mean_latency_ms"] == 300
    assert result["tool_p95_latency_bucket_upper_ms"] == 500
    assert result["tool_p95_over_max_bucket"] is False
    assert result["auth_failures"] == 1
    assert "private-" not in json.dumps(result)


def test_sampling_uses_deltas_and_detects_process_restart():
    previous = monitor.parse_metrics(METRICS)
    current = monitor.parse_metrics(METRICS.replace("120", "150").replace('success"} 8', 'success"} 18'))
    window = monitor.metric_window(current, previous)
    assert window["tool_requests"] == 10
    assert window["tool_http_error_rate"] == 0
    assert window["window"] == "since_previous_sample"
    restarted = {**current, "process_uptime_seconds": 1}
    assert monitor.metric_window(restarted, previous)["window"] == "process_lifetime"


def test_empty_service_metrics_do_not_claim_success_rate():
    result = monitor.parse_metrics("support_process_uptime_seconds 1\n")
    assert result["tool_http_success_rate"] is None
    assert result["tool_http_error_rate"] is None
    assert result["tool_p95_latency_bucket_upper_ms"] is None
    with pytest.raises(ValueError):
        monitor.parse_metrics("unrelated_metric 1\n")


class Reply:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return self.payload[:limit]


def test_http_monitor_uses_bearer_but_never_emits_response_body_or_token(monkeypatch):
    seen = []

    class Opener:
        def open(self, request, timeout):
            seen.append((request, timeout))
            return Reply(METRICS.encode())

    monkeypatch.setattr(monitor, "build_opener", lambda *handlers: Opener())
    check, aggregate = monitor.http_check("http://localhost:8090/metrics", token=TOKEN, kind="metrics")
    assert check["status"] == "ok"
    assert seen[0][0].get_header("Authorization") == "Bearer " + TOKEN
    result = json.dumps((check, aggregate))
    assert TOKEN not in result and "private-operation" not in result
    assert monitor.NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.invalid") is None


def test_slow_successful_request_warns_without_becoming_unreachable(monkeypatch):
    ticks = iter((10, 11.5))
    monkeypatch.setattr(monitor.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(monitor, "build_opener", lambda *handlers: SimpleNamespace(open=lambda *args, **kwargs: Reply(b"")))
    check, _ = monitor.http_check("http://localhost:8090/health", warning_ms=1000)
    assert check == {"status": "warning", "code": "slow_response", "http_status": 200, "latency_ms": 1500}


@pytest.mark.parametrize("kind,payload", [("ready", b'{"status":"unknown","secret":"private"}'),
                                           ("ready", b'[]'), ("metrics", b'private-password')])
def test_bad_service_responses_become_safe_errors(monkeypatch, kind, payload):
    monkeypatch.setattr(monitor, "build_opener", lambda *handlers: SimpleNamespace(open=lambda *args, **kwargs: Reply(payload)))
    check, aggregate = monitor.http_check("http://localhost:8090/ready", kind=kind)
    assert check["status"] == "error" and check["code"] == "invalid_response"
    assert aggregate is None
    assert "private" not in json.dumps(check)


def test_connection_and_http_errors_do_not_expose_exception_messages(monkeypatch):
    for error in (OSError("private-password"), HTTPError("https://private.invalid", 401, "private-password", {}, None)):
        def fail(*args, **kwargs):
            raise error
        monkeypatch.setattr(monitor, "build_opener", lambda *handlers: SimpleNamespace(open=fail))
        check, aggregate = monitor.http_check("http://localhost:8090/metrics", token=TOKEN, kind="metrics")
        assert check["status"] == "error" and aggregate is None
        assert "private" not in json.dumps(check)


def write_manifest(folder, completed, **overrides):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps({"format": "support-agent-backup-v1", "completed_at": completed.isoformat(),
                                                   "private-config": TOKEN, **overrides}))


def test_backup_age_uses_completed_manifest_and_excludes_partial_backups(tmp_path):
    now = datetime(2026, 11, 10, tzinfo=timezone.utc)
    write_manifest(tmp_path / "backup-old", now - timedelta(hours=26))
    write_manifest(tmp_path / "backup-partial", now)
    (tmp_path / "backup-partial/.incomplete").touch()
    check = monitor.backup_check(tmp_path, now=now, max_age_hours=24)
    assert check["status"] == "warning" and check["code"] == "backup_stale"
    assert check["valid_backups"] == 1 and check["age_hours"] == 26
    write_manifest(tmp_path / "backup-current", now - timedelta(hours=1))
    check = monitor.backup_check(tmp_path, now=now, max_age_hours=24)
    assert check["status"] == "ok" and check["age_hours"] == 1
    assert check["integrity_checked"] is False
    assert TOKEN not in json.dumps(check)


def test_backup_missing_invalid_future_and_naive_timestamps_are_not_healthy(tmp_path):
    now = datetime.now(timezone.utc)
    assert monitor.backup_check(tmp_path, now=now, max_age_hours=24)["code"] == "backup_missing"
    write_manifest(tmp_path / "backup-invalid", now, format="other")
    write_manifest(tmp_path / "backup-future", now + timedelta(days=1))
    write_manifest(tmp_path / "backup-naive", now.replace(tzinfo=None))
    check = monitor.backup_check(tmp_path, now=now, max_age_hours=24)
    assert check["status"] == "warning" and check["code"] == "backup_metadata_invalid"
    assert check["valid_backups"] == 0


def test_docker_output_only_uses_safe_aggregate_fields(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "ps" in command:
            result = "a" * 64
        elif "inspect" in command:
            result = json.dumps({"running": True, "restarts": 1, "oom_killed": False, "health": "healthy"})
        else:
            result = json.dumps({"CPUPerc": "1.2%", "MemPerc": "20%", "Name": "private-user", "extra": TOKEN})
        return SimpleNamespace(stdout=result)

    monkeypatch.setattr(monitor.subprocess, "run", run)
    check = monitor.docker_check(tmp_path)
    assert check["status"] == "warning"
    assert len(check["services"]) == 3
    assert all(state["restart_count"] == 1 for state in check["services"].values())
    assert "private" not in json.dumps(check)
    assert all(".Config" not in str(command) for command in calls)
    assert all(".State.Health.Log" not in str(command) for command in calls)


def test_docker_errors_are_safe(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise OSError(TOKEN)
    monkeypatch.setattr(monitor.subprocess, "run", fail)
    check = monitor.docker_check(tmp_path)
    assert check["status"] == "error"
    assert TOKEN not in json.dumps(check)


def test_rotating_jsonl_is_bounded_and_preserves_valid_records(tmp_path):
    path = tmp_path / "status.jsonl"
    for count in range(30):
        monitor.write_report(path, {"number": count, "status": "ok", "data": "x" * 100}, 1024)
    assert path.stat().st_size <= 1024
    assert path.with_name(path.name + ".1").stat().st_size <= 1024
    assert len(list(tmp_path.iterdir())) == 2
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert entries[-1]["number"] == 29


def test_finite_watch_counts_samples_and_returns_worst_status(monkeypatch, tmp_path, capsys):
    reports = iter([{"status": "warning"}, {"status": "ok"}, {"status": "error"}])
    sleeps = []
    monkeypatch.setattr(monitor, "collect", lambda args, previous=None: (next(reports), None))
    monkeypatch.setattr(monitor.time, "sleep", lambda delay: sleeps.append(delay))
    output = tmp_path / "status.jsonl"
    assert monitor.main(["--watch", "--samples", "3", "--interval", "1", "--output", str(output)]) == 2
    assert sleeps == [1, 1]
    assert len(output.read_text().splitlines()) == 3
    assert len(capsys.readouterr().out.splitlines()) == 3


@pytest.mark.parametrize("url", ["http://example.com", "http://user:password@localhost:8090", "http://localhost:8090/?token=private", "file:///secret"])
def test_insecure_or_credential_urls_are_rejected(url):
    with pytest.raises(monitor.MonitorError):
        monitor.validated_base(url)


def test_config_validation_and_main_errors_are_redacted(tmp_path, capsys):
    path = tmp_path / "operations.json"
    path.write_text(json.dumps({"monitor_token": TOKEN, "oauth": {"client_secret": "private-oauth"}}))
    assert monitor.load_token(path) == TOKEN
    path.write_text("private-invalid-config")
    with pytest.raises(monitor.MonitorError) as error:
        monitor.load_token(path)
    assert "private" not in str(error.value)
    assert monitor.main(["--support-url", "http://private-user:private-password@localhost"]) == 2
    assert "private" not in capsys.readouterr().out


def test_collect_reports_missing_credential_without_reading_any_model(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor, "tls_check", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(monitor, "http_check", lambda *args, **kwargs: ({"status": "ok", "http_status": 200}, None))
    args = SimpleNamespace(support_url="http://localhost:8090", librechat_url="http://localhost:3080", timeout=3,
                           latency_warning_ms=1000, operations=tmp_path / "missing.json", backup_dir=tmp_path,
                           backup_max_age_hours=24, docker=False, error_rate_warning=.05)
    report, previous = monitor.collect(args)
    assert report["status"] == "error" and previous is None
    assert report["checks"]["metrics"]["code"] == "monitor_configuration_invalid"


def test_error_rate_warning_uses_interval_and_expected_denials_are_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor, "tls_check", lambda *args, **kwargs: {"status": "ok"})
    operations = tmp_path / "operations.json"
    operations.write_text(json.dumps({"monitor_token": TOKEN}))
    aggregate = monitor.parse_metrics(METRICS)
    monkeypatch.setattr(monitor, "http_check", lambda *args, **kwargs: ({"status": "ok"}, aggregate if kwargs["kind"] == "metrics" else None))
    monkeypatch.setattr(monitor, "backup_check", lambda *args, **kwargs: {"status": "ok"})
    args = SimpleNamespace(support_url="http://localhost:8090", librechat_url="http://localhost:3080", timeout=3,
                           latency_warning_ms=1000, operations=operations, backup_dir=tmp_path,
                           backup_max_age_hours=24, docker=False, error_rate_warning=.05)
    first, previous = monitor.collect(args)
    assert first["status"] == "warning"
    assert first["checks"]["tool_errors"]["error_rate"] == .1
    second, _ = monitor.collect(args, previous=previous)
    assert second["status"] == "ok"
    assert second["checks"]["tool_errors"]["error_rate"] is None
    assert second["checks"]["tool_errors"]["window"] == "since_previous_sample"
    assert TOKEN not in json.dumps((first, second))


@pytest.mark.parametrize("url", ["https://evil.invalid:8443/metrics", "http://localhost:9000/metrics",
                                  "http://127.0.0.1:8090/metrics?redirect=private", "http://localhost:8090/api/me"])
def test_monitor_credential_only_reaches_fixed_local_metrics_targets(monkeypatch, url):
    monkeypatch.setattr(monitor, "build_opener", lambda *args: pytest.fail("Credential must not reach the network"))
    with pytest.raises(monitor.MonitorError):
        monitor.http_check(url, token=TOKEN, kind="metrics")


def test_tls_missing_ca_is_explicit_and_does_not_use_unverified_ssl(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor.socket, "create_connection", lambda *args, **kwargs: pytest.fail("No CA means no TLS probe"))
    assert monitor.tls_check(tmp_path / "missing.crt") == {"status": "warning", "code": "tls_not_configured", "certificate_verified": False}


@pytest.mark.parametrize("days,status,code", [(90, "ok", "tls_healthy"), (5, "warning", "tls_certificate_expiring"),
                                               (-1, "error", "tls_certificate_expired")])
def test_tls_probe_verifies_hostname_and_warns_before_certificate_expiry(monkeypatch, tmp_path, days, status, code):
    ca = tmp_path / "server.crt"
    ca.touch()
    observed = {}
    now = datetime(2026, 11, 10, tzinfo=timezone.utc)
    monkeypatch.setattr(monitor.time, "time", lambda: now.timestamp())

    class Secure:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def getpeercert(self):
            return {"notAfter": (now + timedelta(days=days)).strftime("%b %d %H:%M:%S %Y GMT"), "private-subject": TOKEN}

    class Context:
        def wrap_socket(self, raw, server_hostname):
            observed["hostname"] = server_hostname
            return Secure()

    def create_context(*, cafile):
        observed["ca"] = cafile
        return Context()

    def connect(address, timeout):
        observed["address"] = address
        return Secure()

    class HTTP:
        def __init__(self, host, port, timeout):
            assert (host, port) == ("localhost", 8443)

        def request(self, method, path, headers):
            assert (method, path) == ("GET", "/ready")
            assert "Authorization" not in headers

        def getresponse(self):
            return Reply(b'{"status":"ready"}')

        def close(self):
            pass

    monkeypatch.setattr(monitor.ssl, "create_default_context", create_context)
    monkeypatch.setattr(monitor.socket, "create_connection", connect)
    monkeypatch.setattr(monitor, "HTTPConnection", HTTP)
    result = monitor.tls_check(ca)
    assert result["status"] == status and result["code"] == code
    assert result["certificate_verified"] is True
    assert result["certificate_days_remaining"] == days
    assert observed == {"hostname": "localhost", "ca": str(ca), "address": ("127.0.0.1", 8443)}
    assert TOKEN not in json.dumps(result)


def test_tls_certificate_failure_is_safe_and_never_falls_back(monkeypatch, tmp_path):
    ca = tmp_path / "server.crt"
    ca.touch()

    def fail(*args, **kwargs):
        raise monitor.ssl.SSLCertVerificationError(TOKEN)

    monkeypatch.setattr(monitor.ssl, "create_default_context", fail)
    result = monitor.tls_check(ca)
    assert result == {"status": "error", "code": "tls_verification_failed", "certificate_verified": False}
    assert TOKEN not in json.dumps(result)


def test_operations_container_probe_includes_tls_with_override(monkeypatch, tmp_path):
    (tmp_path / "compose.operations.yml").touch()
    queried = []

    def run(command, **kwargs):
        if "ps" in command:
            assert command.count("-f") == 2 and "compose.operations.yml" in command
            queried.append(command[-1])
            result = "a" * 64
        elif "inspect" in command:
            result = json.dumps({"running": True, "restarts": 0, "oom_killed": False, "health": "healthy"})
        else:
            result = json.dumps({"CPUPerc": "1%", "MemPerc": "20%"})
        return SimpleNamespace(stdout=result)

    monkeypatch.setattr(monitor.subprocess, "run", run)
    result = monitor.docker_check(tmp_path)
    assert queried == ["api", "mongodb", "support-agent", "support-tls"]
    assert result["services"]["support-tls"]["status"] == "ok"
