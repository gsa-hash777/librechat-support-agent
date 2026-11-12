"""Local health/metrics/backup checks; no model calls or request-body logging.

Exit codes: 0 healthy, 1 warning, 2 error. --watch keeps at most two bounded
JSONL files. With --samples N it exits after N checks, using the worst status
observed. Metrics are transport counters, not Agent answer-quality scores.
"""

import argparse
from datetime import datetime, timezone
from http.client import HTTPConnection, HTTPException
import json
import math
from pathlib import Path
import re
import socket
import ssl
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[3]
LEVEL = {"ok": 0, "warning": 1, "error": 2}
METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([-+0-9.eE]+)$")


class MonitorError(ValueError):
    pass


class NoRedirect(HTTPRedirectHandler):
    # A redirect must never forward the internal scrape token elsewhere.
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def validated_base(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise MonitorError("Invalid monitor service URL.")
    if parts.path not in {"", "/"} or parts.query or parts.fragment:
        raise MonitorError("Monitor URLs must contain only a scheme, host and port.")
    if parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise MonitorError("Monitoring endpoints must be on this machine.")
    return value.rstrip("/")


def load_token(path: Path) -> str:
    try:
        if path.stat().st_size > 65536:
            raise MonitorError("Invalid monitoring configuration.")
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        token = document.get("monitor_token") if isinstance(document, dict) else None
        if not isinstance(token, str) or len(token) < 32 or len(token) > 512 or not token.isascii() or any(c.isspace() for c in token):
            raise MonitorError("Invalid monitoring configuration.")
        return token
    except (OSError, ValueError):
        raise MonitorError("Monitoring configuration is missing or invalid.") from None


def http_check(url: str, *, token: str | None = None, timeout: float = 3, kind: str = "health", warning_ms: float = 1000):
    headers = {"User-Agent": "SupportAgentMonitor/1.0", "Accept": "text/plain" if kind == "metrics" else "application/json"}
    if token:
        target = urlsplit(url)
        if (target.hostname not in {"localhost", "127.0.0.1", "::1"}
                or (target.scheme, target.port) not in {("http", 8090), ("https", 8443)}
                or target.username or target.password or target.path != "/metrics" or target.query or target.fragment):
            raise MonitorError("Monitoring credentials may only reach the fixed local metrics endpoint.")
        headers["Authorization"] = "Bearer " + token
    started = time.perf_counter()
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(Request(url, headers=headers), timeout=timeout) as response:
            status = response.status
            payload = response.read(2_000_001 if kind == "metrics" else 4097) if kind != "health" else b""
    except HTTPError as error:
        status = error.code
        error.close()
        return {"status": "error", "code": "http_status", "http_status": status,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2)}, None
    except (OSError, URLError, TimeoutError, ValueError, HTTPException):
        return {"status": "error", "code": "unreachable",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2)}, None
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    if status != 200:
        return {"status": "error", "code": "http_status", "http_status": status, "latency_ms": elapsed}, None
    try:
        if kind == "ready":
            if len(payload) > 4096 or json.loads(payload).get("status") != "ready":
                raise ValueError()
        elif kind == "metrics":
            if len(payload) > 2_000_000:
                raise ValueError()
            aggregate = parse_metrics(payload.decode("utf-8"))
    except (ValueError, AttributeError, UnicodeError):
        return {"status": "error", "code": "invalid_response", "http_status": status, "latency_ms": elapsed}, None
    return {"status": "warning" if elapsed > warning_ms else "ok", "code": "slow_response" if elapsed > warning_ms else "healthy",
            "http_status": status, "latency_ms": elapsed}, aggregate if kind == "metrics" else None


def tls_check(ca_path: Path, *, timeout: float = 3, warning_days: float = 30, warning_ms: float = 1000) -> dict:
    """Probe the public TLS ingress and peer certificate, without credentials.

    Connect to IPv4 loopback while verifying the public hostname 'localhost'.
    This avoids Windows IPv6 fallback without disabling hostname verification.
    """
    if not ca_path.is_file():
        return {"status": "warning", "code": "tls_not_configured", "certificate_verified": False}
    started = time.perf_counter()
    try:
        context = ssl.create_default_context(cafile=str(ca_path))
        with socket.create_connection(("127.0.0.1", 8443), timeout=timeout) as raw:
            with context.wrap_socket(raw, server_hostname="localhost") as secure:
                certificate = secure.getpeercert()
                expires = ssl.cert_time_to_seconds(certificate["notAfter"])
                connection = HTTPConnection("localhost", 8443, timeout=timeout)
                connection.sock = secure
                try:
                    connection.request("GET", "/ready", headers={"Host": "localhost:8443", "User-Agent": "SupportAgentMonitor/1.0"})
                    response = connection.getresponse()
                    status, body = response.status, response.read(4097)
                finally:
                    connection.close()
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        days = (expires - time.time()) / 86400
        try:
            document = json.loads(body) if len(body) <= 4096 else None
        except ValueError:
            document = None
        if status != 200:
            level, code = "error", "tls_service_not_ready"
        elif not isinstance(document, dict) or document.get("status") != "ready":
            level, code = "error", "tls_response_invalid"
        elif days <= 0:
            level, code = "error", "tls_certificate_expired"
        elif days <= warning_days:
            level, code = "warning", "tls_certificate_expiring"
        elif elapsed > warning_ms:
            level, code = "warning", "slow_response"
        else:
            level, code = "ok", "tls_healthy"
        return {"status": level, "code": code, "http_status": status, "latency_ms": elapsed,
                "certificate_verified": True, "certificate_days_remaining": round(days, 2),
                "certificate_not_after_utc": datetime.fromtimestamp(expires, timezone.utc).isoformat(), "warning_days": warning_days}
    except ssl.SSLCertVerificationError:
        return {"status": "error", "code": "tls_verification_failed", "certificate_verified": False}
    except (OSError, HTTPException, ValueError, KeyError, TypeError):
        return {"status": "error", "code": "tls_unavailable_or_invalid", "certificate_verified": False}


def parse_metrics(text: str) -> dict:
    aggregate = {"tool_success": 0, "tool_rejected": 0, "tool_error": 0, "tool_aborted": 0,
                 "auth_failures": 0, "tool_duration_sum_seconds": 0.0, "tool_duration_count": 0}
    buckets, uptime = {}, False
    for line in text.splitlines():
        match = METRIC_LINE.fullmatch(line)
        if not match:
            continue
        name, labels, raw = match.groups()
        number = float(raw)
        if not math.isfinite(number) or number < 0:
            raise ValueError("Invalid metric number.")
        if name == "support_tool_requests_total":
            outcome = re.search(r'(?:^|,)outcome="(success|rejected|error|aborted)"(?:,|$)', labels or "")
            if outcome:
                aggregate["tool_" + outcome[1]] += number
        elif name == "support_http_auth_failures_total":
            aggregate["auth_failures"] += number
        elif name == "support_tool_request_duration_seconds_sum":
            aggregate["tool_duration_sum_seconds"] += number
        elif name == "support_tool_request_duration_seconds_count":
            aggregate["tool_duration_count"] += number
        elif name == "support_tool_request_duration_seconds_bucket":
            bound = re.search(r'(?:^|,)le="([+0-9.eEInf]+)"(?:,|$)', labels or "")
            if bound:
                upper = float(bound[1])
                buckets[upper] = buckets.get(upper, 0) + number
        elif name == "support_process_uptime_seconds":
            aggregate["process_uptime_seconds"], uptime = number, True
        elif name == "support_http_in_flight":
            aggregate["http_in_flight"] = number
    if not uptime:
        raise ValueError("Support metrics missing.")
    aggregate["tool_requests"] = sum(aggregate["tool_" + label] for label in ("success", "rejected", "error", "aborted"))
    aggregate["tool_http_success_rate"] = aggregate["tool_success"] / aggregate["tool_requests"] if aggregate["tool_requests"] else None
    aggregate["tool_http_error_rate"] = (aggregate["tool_error"] + aggregate["tool_aborted"]) / aggregate["tool_requests"] if aggregate["tool_requests"] else None
    count = aggregate["tool_duration_count"]
    aggregate["tool_mean_latency_ms"] = round(1000 * aggregate["tool_duration_sum_seconds"] / count, 2) if count else None
    upper = next((bound for bound, value in sorted(buckets.items()) if count and value >= count * .95), None)
    aggregate["tool_p95_latency_bucket_upper_ms"] = upper * 1000 if upper is not None and math.isfinite(upper) else None
    aggregate["tool_p95_over_max_bucket"] = upper == math.inf
    aggregate["window"] = "process_lifetime"
    return aggregate


def metric_window(current: dict, previous: dict | None) -> dict:
    keys = ("tool_success", "tool_rejected", "tool_error", "tool_aborted", "auth_failures", "tool_duration_count", "tool_duration_sum_seconds")
    if previous and current["process_uptime_seconds"] >= previous["process_uptime_seconds"] and all(current[key] >= previous[key] for key in keys):
        result = {key: current[key] - previous[key] for key in keys}
        result["tool_requests"] = sum(result["tool_" + label] for label in ("success", "rejected", "error", "aborted"))
        result["tool_http_error_rate"] = (result["tool_error"] + result["tool_aborted"]) / result["tool_requests"] if result["tool_requests"] else None
        result["window"] = "since_previous_sample"
        return result
    return current


def backup_check(directory: Path, *, now: datetime, max_age_hours: float) -> dict:
    candidates = [directory] if directory.name == "manifest.json" else list(directory.glob("backup-*/manifest.json"))
    dates, invalid = [], 0
    for path in candidates:
        if (path.parent / ".incomplete").exists():
            continue
        try:
            if path.stat().st_size > 16_000_000:
                raise ValueError()
            manifest = json.loads(path.read_text(encoding="utf-8"))
            stamp = datetime.fromisoformat(manifest["completed_at"].replace("Z", "+00:00"))
            if manifest.get("format") != "support-agent-backup-v1" or stamp.tzinfo is None or (stamp - now).total_seconds() > 300:
                raise ValueError()
            dates.append(stamp)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            invalid += 1
    if not dates:
        return {"status": "warning", "code": "backup_metadata_invalid" if invalid else "backup_missing", "valid_backups": 0}
    latest = max(dates)
    hours = max(0, (now - latest).total_seconds() / 3600)
    return {"status": "warning" if hours > max_age_hours or invalid else "ok",
            "code": "backup_stale" if hours > max_age_hours else "backup_metadata_invalid" if invalid else "recent_backup",
            "valid_backups": len(dates), "age_hours": round(hours, 3), "latest_completed_at": latest.isoformat(),
            "max_age_hours": max_age_hours, "integrity_checked": False}


def docker_check(root: Path, *, timeout: float = 5) -> dict:
    services = {}
    compose = ["compose", "-f", "compose.support.yml"]
    names = ["api", "mongodb", "support-agent"]
    if (root / "compose.operations.yml").is_file():
        compose.extend(["-f", "compose.operations.yml"])
        names.append("support-tls")
    state_format = '{"running":{{.State.Running}},"restarts":{{.RestartCount}},"oom_killed":{{.State.OOMKilled}},"health":"{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}"}'

    def run(*arguments):
        return subprocess.run(["docker", *arguments], cwd=root, capture_output=True, text=True, check=True, timeout=timeout).stdout.strip()

    for service in names:
        try:
            identifier = run(*compose, "ps", "-q", service)
            if not re.fullmatch(r"[a-f0-9]{12,64}", identifier):
                services[service] = {"status": "error", "code": "container_missing"}
                continue
            state = json.loads(run("inspect", "--format", state_format, identifier))
            stats = json.loads(run("stats", "--no-stream", "--format", "{{json .}}", identifier))
            cpu = float(stats["CPUPerc"].rstrip("%"))
            memory = float(stats["MemPerc"].rstrip("%"))
            if not all(math.isfinite(value) and value >= 0 for value in (cpu, memory)):
                raise ValueError()
            health = state.get("health")
            if health not in {"healthy", "unhealthy", "starting", "none"}:
                raise ValueError()
            if type(state.get("running")) is not bool or type(state.get("oom_killed")) is not bool or type(state.get("restarts")) is not int:
                raise ValueError()
            level = "error" if not state["running"] or state["oom_killed"] or health == "unhealthy" else "warning" if state["restarts"] or health == "starting" or cpu > 85 or memory > 85 else "ok"
            services[service] = {"status": level, "running": state["running"], "health": health,
                                 "restart_count": state["restarts"], "oom_killed": state["oom_killed"],
                                 "cpu_percent": cpu, "memory_percent": memory}
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
            services[service] = {"status": "error", "code": "docker_unavailable_or_invalid"}
    return {"status": max((state["status"] for state in services.values()), key=LEVEL.get), "services": services}


def collect(args, *, previous: dict | None = None) -> tuple[dict, dict | None]:
    now = datetime.now(timezone.utc)
    checks, aggregate = {}, None
    for label, base, path, kind in (("support_health", args.support_url, "/health", "health"),
                                    ("support_ready", args.support_url, "/ready", "ready"),
                                    ("librechat_health", args.librechat_url, "/health", "health")):
        checks[label], _ = http_check(base + path, kind=kind, timeout=args.timeout, warning_ms=args.latency_warning_ms)
    try:
        token = load_token(args.operations)
        checks["metrics"], aggregate = http_check(args.support_url + "/metrics", token=token, kind="metrics", timeout=args.timeout, warning_ms=args.latency_warning_ms)
    except MonitorError:
        checks["metrics"] = {"status": "error", "code": "monitor_configuration_invalid"}
    if aggregate is not None:
        window = metric_window(aggregate, previous)
        rate = window["tool_http_error_rate"]
        checks["tool_errors"] = {"status": "warning" if rate is not None and rate > args.error_rate_warning else "ok",
                                  "error_rate": rate, "threshold": args.error_rate_warning,
                                  "requests": window["tool_requests"], "window": window["window"]}
    checks["backup"] = backup_check(args.backup_dir, now=now, max_age_hours=args.backup_max_age_hours)
    checks["support_tls"] = tls_check(getattr(args, "tls_ca", ROOT / ".local/support-tls/server.crt"), timeout=args.timeout,
                                       warning_days=getattr(args, "tls_warning_days", 30), warning_ms=args.latency_warning_ms)
    if args.docker:
        checks["docker"] = docker_check(ROOT, timeout=max(5, args.timeout))
    report = {"schema": "support-agent-monitor-v1", "timestamp_utc": now.isoformat(),
              "status": max((check["status"] for check in checks.values()), key=LEVEL.get), "checks": checks}
    if aggregate is not None:
        report["metrics"] = aggregate
    return report, aggregate


def write_report(path: Path, report: dict, max_bytes: int):
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    size = len(payload.encode("utf-8"))
    if size > max_bytes:
        raise MonitorError("Monitor report exceeds the configured log size.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size + size > max_bytes:
        path.replace(path.with_name(path.name + ".1"))
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Explicit IPv4 avoids slow IPv6->IPv4 fallback on some Windows hosts.
    parser.add_argument("--support-url", default="http://127.0.0.1:8090")
    parser.add_argument("--librechat-url", default="http://127.0.0.1:3080")
    parser.add_argument("--operations", type=Path, default=ROOT / ".local/operations.json")
    parser.add_argument("--backup-dir", type=Path, default=ROOT / ".local/backups")
    parser.add_argument("--backup-max-age-hours", type=float, default=24)
    parser.add_argument("--timeout", type=float, default=3)
    parser.add_argument("--latency-warning-ms", type=float, default=1000)
    parser.add_argument("--error-rate-warning", type=float, default=.05)
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--tls-ca", type=Path, default=ROOT / ".local/support-tls/server.crt")
    parser.add_argument("--tls-warning-days", type=float, default=30)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--samples", type=int, default=None, help="Default: 1; with --watch: unlimited.")
    parser.add_argument("--output", type=Path, help="JSONL; --watch defaults to .local/monitor/status.jsonl")
    parser.add_argument("--max-log-bytes", type=int, default=1_048_576)
    args = parser.parse_args(argv)
    try:
        args.support_url, args.librechat_url = validated_base(args.support_url), validated_base(args.librechat_url)
        if any(not math.isfinite(value) or value <= 0 for value in (args.timeout, args.backup_max_age_hours, args.interval, args.latency_warning_ms, args.tls_warning_days)):
            raise MonitorError("Monitor timing and thresholds must be finite and positive.")
        if args.interval < 1 or not math.isfinite(args.error_rate_warning) or not 0 <= args.error_rate_warning <= 1 or args.max_log_bytes < 1024:
            raise MonitorError("Invalid monitor thresholds or log size.")
        samples = args.samples if args.samples is not None else (0 if args.watch else 1)
        if samples < 0 or (samples == 0 and not args.watch):
            raise MonitorError("Unlimited samples require --watch.")
        output = args.output or (ROOT / ".local/monitor/status.jsonl" if args.watch else None)
        count, worst, previous = 0, 0, None
        while samples == 0 or count < samples:
            report, previous = collect(args, previous=previous)
            print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
            if output:
                write_report(output, report, args.max_log_bytes)
            worst = max(worst, LEVEL[report["status"]])
            count += 1
            if samples == 0 or count < samples:
                time.sleep(args.interval)
        return worst
    except (MonitorError, OSError, ValueError):
        print(json.dumps({"schema": "support-agent-monitor-v1", "status": "error", "code": "monitor_configuration_or_output_error"}), flush=True)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
