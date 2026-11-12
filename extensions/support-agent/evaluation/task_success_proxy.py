"""Request-level, fail-closed CNY 5 guard for the isolated TSR experiment.

Only this process writes the ledger. Bodies and credentials are never logged.
The reservation counts every UTF-8 request byte as an input token, plus 4096
framing tokens, at peak cache-miss prices. Output is provider-limited. Missing
usage retains the reservation and closes the gate, including after disconnects.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

MODEL = "deepseek-flash"
LIMIT_CNY = 5.0
AMENDED_LIMIT_CNY = 5.5
INPUT_CNY_PER_MILLION = 2.0
OUTPUT_CNY_PER_MILLION = 8.0
MAX_OUTPUT_TOKENS = 1500
FRAMING_TOKENS = 4096
MAX_BODY_BYTES = 1_000_000
UPSTREAM = "https://api.deepseek.com/v1/chat/completions"


class GuardError(Exception):
    def __init__(self, code, status=409):
        self.code, self.status = code, status
        super().__init__(code)


def authorized_limit(state):
    """Only the separately recorded user-authorized 0.50 CNY extension is valid."""
    if state.get("limit_cny") == LIMIT_CNY:
        return LIMIT_CNY
    amendments = state.get("amendments", [])
    if state.get("limit_cny") != AMENDED_LIMIT_CNY or len(amendments) != 1:
        raise GuardError("incompatible_budget_ledger")
    amendment = amendments[0]
    if (amendment.get("from_limit_cny") != LIMIT_CNY or amendment.get("to_limit_cny") != AMENDED_LIMIT_CNY
            or amendment.get("additional_cny") != .5 or amendment.get("authorized_by") != "user"
            or amendment.get("only_unattempted_case_ids") != ["P06", "P07", "P08", "P09", "P10"]
            or amendment.get("do_not_retry_case_ids") != ["P05"]
            or not isinstance(amendment.get("original_ledger_sha256"), str)
            or len(amendment["original_ledger_sha256"]) != 64
            or not amendment.get("authorized_at") or not amendment.get("authorization_text")):
        raise GuardError("incompatible_budget_amendment")
    return AMENDED_LIMIT_CNY


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def request_payload(raw):
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError):
        raise GuardError("invalid_json", 400) from None
    if not isinstance(payload, dict) or payload.get("model") != MODEL:
        raise GuardError("only_frozen_model_allowed", 400)
    tokens = payload.get("max_tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 1 <= tokens <= MAX_OUTPUT_TOKENS:
        raise GuardError("output_limit_required", 400)
    if payload.get("thinking") != {"type": "disabled"}:
        raise GuardError("thinking_must_be_disabled", 400)
    allowed = {"model", "messages", "temperature", "top_p", "max_tokens", "stream", "stream_options", "tools", "tool_choice", "parallel_tool_calls", "response_format", "frequency_penalty", "presence_penalty", "stop", "seed", "logprobs", "top_logprobs", "thinking", "user"}
    if set(payload) - allowed or not isinstance(payload.get("messages"), list):
        raise GuardError("unsupported_request_fields", 400)
    for message in payload["messages"]:
        if not isinstance(message, dict) or set(message) - {"role", "content", "name", "tool_call_id", "tool_calls"} or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise GuardError("only_text_messages_allowed", 400)
        content = message.get("content")
        if isinstance(content, list):
            if not all(isinstance(part, dict) and set(part) == {"type", "text"} and part.get("type") == "text" and isinstance(part.get("text"), str) for part in content):
                raise GuardError("only_text_messages_allowed", 400)
        elif content is not None and not isinstance(content, str):
            raise GuardError("only_text_messages_allowed", 400)
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list) or not all(isinstance(call, dict) and call.get("type") == "function" and isinstance(call.get("function"), dict) and isinstance(call["function"].get("name"), str) and isinstance(call["function"].get("arguments"), str) for call in calls):
            raise GuardError("only_function_calls_allowed", 400)
    if "tools" in payload and (not isinstance(payload["tools"], list) or not all(isinstance(tool, dict) and tool.get("type") == "function" for tool in payload["tools"])):
        raise GuardError("only_function_tools_allowed", 400)
    if payload.get("stream", False) not in (True, False):
        raise GuardError("invalid_stream", 400)
    if payload.get("stream"):
        payload["stream_options"] = {"include_usage": True}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return payload, encoded


def normalized_usage(value):
    if not isinstance(value, dict):
        return None
    counts = []
    for key in ("prompt_tokens", "completion_tokens"):
        count = value.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        counts.append(count)
    return {"prompt_tokens": counts[0], "completion_tokens": counts[1], "total_tokens": sum(counts)}


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save({"schema_version": "task-success-request-guard-v1", "limit_cny": LIMIT_CNY,
                        "created_at": utc_now(), "pricing": {"input_cny_per_million": 2, "output_cny_per_million": 8, "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/", "checked_date": "2026-11-12", "cache_assumption": "all_miss_peak"}, "attempts": []})
        state = self._read()
        authorized_limit(state)
        if state.get("schema_version") != "task-success-request-guard-v1":
            raise GuardError("incompatible_budget_ledger")

    def _read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, state):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    @staticmethod
    def committed(state):
        return sum(x["cost_cny"] if x["status"] == "settled" else x["reserved_cny"] for x in state["attempts"])

    def snapshot(self):
        with self.lock:
            state = self._read()
            return {**state, "committed_cny": round(self.committed(state), 8), "remaining_cny": round(authorized_limit(state)-self.committed(state), 8), "blocked": any(x["status"] != "settled" for x in state["attempts"])}

    def reserve(self, body, output_tokens, tags=None):
        input_bound = len(body) + FRAMING_TOKENS
        reserve = (input_bound * INPUT_CNY_PER_MILLION + output_tokens * OUTPUT_CNY_PER_MILLION) / 1_000_000
        with self.lock:
            state = self._read()
            if any(x["status"] != "settled" for x in state["attempts"]):
                raise GuardError("pending_or_unknown_usage_blocks_budget")
            if self.committed(state) + reserve > authorized_limit(state):
                raise GuardError("budget_exhausted", 402)
            ident = "request-" + uuid.uuid4().hex
            entry = {"attempt_id": ident, "started_at": utc_now(), "status": "pending", "reserved_cny": reserve, "input_token_bound": input_bound, "output_token_bound": output_tokens, "request_sha256": hashlib.sha256(body).hexdigest(), "request_bytes": len(body)}
            for key, value in (tags or {}).items():
                if key in {"run_id", "case_id", "stage"} and isinstance(value, str) and len(value) <= 100 and all(c.isalnum() or c in "-_:." for c in value):
                    entry[key] = value
            state["attempts"].append(entry)
            self._save(state)
            return ident

    def finish(self, ident, usage, reason=None):
        with self.lock:
            state = self._read()
            entry = next(x for x in state["attempts"] if x["attempt_id"] == ident)
            if entry["status"] != "pending":
                raise GuardError("request_already_finalized")
            entry["finished_at"] = utc_now()
            parsed = normalized_usage(usage)
            if parsed is None:
                entry.update(status="unknown_usage", reason=reason or "provider_usage_missing")
            else:
                cost = (parsed["prompt_tokens"] * INPUT_CNY_PER_MILLION + parsed["completion_tokens"] * OUTPUT_CNY_PER_MILLION) / 1_000_000
                if parsed["prompt_tokens"] > entry["input_token_bound"] or parsed["completion_tokens"] > entry["output_token_bound"]:
                    # Preserve the observed cost and stop; never silently call an estimate a guarantee.
                    entry.update(status="bound_violation", reason="provider_token_bound_exceeded", cost_cny=cost, usage=parsed, reserved_cny=max(cost, entry["reserved_cny"]))
                else:
                    entry.update(status="settled", usage=parsed, cost_cny=cost)
            self._save(state)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def handler_type(ledger, api_key, opener=None):
    upstream = opener or build_opener(ProxyHandler({}), NoRedirect())

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def authenticated(self):
            return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + api_key)

        def reply(self, code, payload):
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, {"ready": True, "guard": "task-success-request-guard-v1"})
            if self.path == "/budget" and self.authenticated():
                return self.reply(200, ledger.snapshot())
            return self.reply(401 if self.path == "/budget" else 404, {"error": "unauthorized" if self.path == "/budget" else "not_found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.reply(404, {"error": "not_found"})
            if not self.authenticated():
                return self.reply(401, {"error": "unauthorized"})
            ident = None
            headers_sent = False
            usage = None
            try:
                raw_length = self.headers.get("Content-Length", "")
                if not raw_length.isdigit() or not 0 < int(raw_length) <= MAX_BODY_BYTES:
                    raise GuardError("request_size_rejected", 413)
                payload, body = request_payload(self.rfile.read(int(raw_length)))
                tags = {"run_id": self.headers.get("X-Eval-Run"), "case_id": self.headers.get("X-Eval-Case"), "stage": self.headers.get("X-Eval-Stage", "agent")}
                ident = ledger.reserve(body, payload["max_tokens"], tags)
                req = Request(UPSTREAM, data=body, method="POST", headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json", "Accept": "text/event-stream" if payload.get("stream") else "application/json"})
                with upstream.open(req, timeout=180) as response:
                    if not payload.get("stream"):
                        raw = response.read(MAX_BODY_BYTES + 1)
                        value = json.loads(raw)
                        usage = value.get("usage")
                        ledger.finish(ident, usage)
                        ident = None
                        return self.reply(200, value)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    headers_sent = True
                    disconnected = False
                    saw_done = False
                    for line in response:
                        if line.startswith(b"data:"):
                            event_data = line[5:].strip()
                            if event_data == b"[DONE]":
                                saw_done = True
                                continue
                            try:
                                event = json.loads(event_data)
                                if normalized_usage(event.get("usage")) is not None:
                                    usage = event["usage"]
                            except (ValueError, UnicodeError, AttributeError):
                                pass
                        if not disconnected:
                            try:
                                self.wfile.write(line)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError, OSError):
                                disconnected = True
                    ledger.finish(ident, usage, None if saw_done else "stream_ended_without_done")
                    ident = None
                    if not disconnected and saw_done:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    self.close_connection = True
            except GuardError as error:
                if not headers_sent:
                    self.reply(error.status, {"error": {"type": "evaluation_budget_guard", "code": error.code}})
            except Exception:
                if not headers_sent:
                    self.reply(502, {"error": {"type": "evaluation_budget_guard", "code": "upstream_failed_no_retry"}})
            finally:
                if ident is not None:
                    ledger.finish(ident, usage, "request_incomplete_no_retry")
                self.close_connection = True

    return Handler


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("DeepSeek key is required; no requests started.")
    ledger = Ledger(os.environ.get("TSR_BUDGET_PATH", "/state/budget.json"))
    server = ThreadingHTTPServer(("0.0.0.0", 8093), handler_type(ledger, api_key))
    server.daemon_threads = False
    server.serve_forever()


if __name__ == "__main__":
    main()
