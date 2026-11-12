"""Two real LibreChat OAuth identity checks; dry-run unless --run is explicit.

Uses the running 3080 chat/SSE/bind/callback endpoints and the real TLS consent
page. Never fabricates JWTs, writes Mongo tokens, retries a model submission,
or changes historical evaluation budgets. The CNY 0.20 threshold is an estimate
and cannot stop an already submitted provider request at an exact cost.
"""

from datetime import datetime, timezone
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import ssl
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPCookieProcessor, HTTPSHandler, ProxyHandler, Request, build_opener
import uuid

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE / "integration"))
from bootstrap import BootstrapError, LOCAL_USER_AGENT, LocalApi, NoRedirect, read_json, save_state
from smoke import message_evidence
from evaluation.agent import turn_usage
from evaluation.usage import aggregate_usage, estimate_cost


BUDGET_CNY = .2
RESERVATION_CNY = .1
REPAIR_RESERVATION_CNY = .05
PROMPT = "这是一次只读身份验收。请仅调用一次 getCurrentSupportUser，报告该工具返回的当前账号 id 和 role。不要检索知识库，不调用其他工具，不创建草稿或工单，不请求修改权限。若工具要求连接账号，等待我完成授权后继续原来的工具调用。"
ACTORS = ("alice", "bob")


class SmokeError(BootstrapError):
    pass


class HiddenFields(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.fields = {}
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "input" and attributes.get("type") == "hidden" and attributes.get("name"):
            self.fields[attributes["name"]] = attributes.get("value", "")


def checked_callback(value, action_id, state, *, base):
    parsed = urlsplit(value)
    expected = urlsplit(base)
    query = parse_qs(parsed.query, keep_blank_values=True)
    if ((parsed.scheme, parsed.netloc) != (expected.scheme, expected.netloc)
            or parsed.path != f"/api/actions/{action_id}/oauth/callback" or parsed.fragment
            or set(query) != {"code", "state"} or len(query["code"]) != 1 or not query["code"][0]
            or query["state"] != [state]):
        raise SmokeError("callback_target_or_state_mismatch")
    return parsed.path + "?" + parsed.query


def web_request(opener, url, *, method="GET", form=None, origin=None, timeout=25):
    headers = {"User-Agent": LOCAL_USER_AGENT}
    if origin:
        headers["Origin"] = origin
    data = urlencode(form).encode() if form is not None else None
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        response = opener.open(request, timeout=timeout)
    except HTTPError as error:
        response = error
    with response:
        body = response.read(1_000_001)
        if len(body) > 1_000_000:
            raise SmokeError("oversized_http_response")
        return response.status, response.headers, body


def complete_oauth(api, auth_url, *, support_user, action_id, client_id, tls_context):
    parsed = urlsplit(auth_url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    expected_callback = api.base + f"/api/actions/{action_id}/oauth/callback"
    if (parsed.scheme != "https" or parsed.netloc != "localhost:8443"
            or parsed.path != "/console/oauth/authorize" or parsed.fragment
            or query.get("client_id") != [client_id] or query.get("redirect_uri") != [expected_callback]
            or query.get("response_type") != ["code"] or len(query.get("state", [])) != 1):
        raise SmokeError("authorization_target_invalid")
    state = query["state"][0]
    # This is the real running server's authenticated binding route; the cookie
    # jar is retained for its own callback. Nothing is signed in this script.
    binding = api.request("POST", f"/api/actions/{action_id}/oauth/bind", {})
    if binding != {"success": True}:
        raise SmokeError("native_binding_failed")
    support = build_opener(ProxyHandler({}), NoRedirect(), HTTPSHandler(context=tls_context), HTTPCookieProcessor(CookieJar()))
    origin = "https://localhost:8443"
    status, _, body = web_request(support, origin + "/console/login")
    fields = HiddenFields(body.decode("utf-8")).fields
    if status != 200 or not fields.get("csrf"):
        raise SmokeError("support_login_form_failed")
    status, _, _ = web_request(support, origin + "/console/login", method="POST", origin=origin,
                               form={"username": support_user["id"], "password": support_user["password"],
                                     "csrf": fields["csrf"], "next": "/console"})
    if status != 303:
        raise SmokeError("support_login_failed")
    status, _, body = web_request(support, auth_url)
    fields = HiddenFields(body.decode("utf-8")).fields
    if status != 200 or not all(fields.get(key) for key in ("csrf", "request_id")):
        raise SmokeError("consent_form_failed")
    status, headers, _ = web_request(support, origin + "/console/oauth/authorize", method="POST", origin=origin,
                                     form={"csrf": fields["csrf"], "request_id": fields["request_id"], "decision": "approve"})
    if status != 303:
        raise SmokeError("consent_failed")
    callback = checked_callback(headers.get("Location", ""), action_id, state, base=api.base)
    # The original callback is sent to 3080, never another process or port.
    status, headers, _ = web_request(api.opener, api.base + callback)
    location = urlsplit(headers.get("Location", ""))
    if status != 302 or location.path != "/oauth/success" or location.netloc not in {"", urlsplit(api.base).netloc}:
        raise SmokeError("native_callback_failed")
    return {"bind_http": 200, "support_login_http": 303, "consent_http": 303, "native_callback_http": status,
            "tls_certificate_verified": True, "native_callback_completed": True}


class StreamConsumer:
    """Consume initial events and native resume snapshots without saving URLs."""

    def __init__(self, on_auth, usage_events, observed_tools, event_counts):
        self.on_auth, self.usage_events = on_auth, usage_events
        self.observed_tools, self.event_counts = observed_tools, event_counts
        self.auth_fingerprint = None

    def tools(self, tools):
        for tool in tools if isinstance(tools, list) else []:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name") or tool.get("function", {}).get("name")
            if name:
                self.observed_tools.add(name if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name) else "invalid_tool_name")
                if "getCurrentSupportUser" not in name:
                    raise SmokeError("unexpected_tool_observed_abort_requested")

    def auth(self, value):
        if not isinstance(value, str) or not value:
            return
        fingerprint = hashlib.sha256(value.encode()).hexdigest()
        if fingerprint == self.auth_fingerprint:
            return
        if self.auth_fingerprint:
            raise SmokeError("multiple_oauth_prompts")
        self.on_auth(value)
        self.auth_fingerprint = fingerprint

    def consume(self, event, depth=0):
        if not isinstance(event, dict) or depth > 4:
            raise SmokeError("invalid_sse_event")
        kind = "sync" if event.get("sync") else event.get("event", "other")
        known = {"sync", "on_token_usage", "on_run_step", "on_run_step_delta", "on_run_step_completed", "on_pending_action"}
        label = kind if kind in known else "final" if event.get("final") else "other"
        self.event_counts[label] = self.event_counts.get(label, 0) + 1
        if event.get("sync"):
            state = event.get("resumeState") or {}
            for step in state.get("runSteps", []):
                self.tools(step.get("stepDetails", {}).get("tool_calls", []))
            for part in state.get("aggregatedContent", []):
                if isinstance(part, dict) and part.get("type") == "tool_call":
                    tool = part.get("tool_call", {})
                    self.tools([tool])
                    self.auth(part.get("auth") or tool.get("auth"))
            for replay in state.get("replayEvents", []) or []:
                self.consume(replay, depth + 1)
            for pending in event.get("pendingEvents", []) or []:
                self.consume(pending, depth + 1)
        if kind == "on_token_usage":
            self.usage_events.append(event.get("data", {}))
        data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}
        self.tools(data.get("stepDetails", {}).get("tool_calls", []))
        delta = data.get("delta", {})
        self.tools(delta.get("tool_calls", []))
        self.auth(delta.get("auth"))
        if kind == "on_pending_action":
            raise SmokeError("unexpected_agent_approval")


def stream(api, started, deadline, *, on_auth, usage_events, observed_tools, event_counts):
    # This is the FIRST attachment. Native resume=true skips early-event replay
    # in favour of a sync snapshot; Action OAuth deltas are not durable MCP
    # replay events, so an early authorization prompt could otherwise be lost.
    query = urlencode({"resume": "false", "generationCreatedAt": started["generationCreatedAt"]})
    request = Request(api.base + f"/api/agents/chat/stream/{started['streamId']}?{query}", headers={
        "Accept": "text/event-stream", "Origin": api.base, "Authorization": "Bearer " + api.token, "User-Agent": LOCAL_USER_AGENT})
    consumer = StreamConsumer(on_auth, usage_events, observed_tools, event_counts)
    try:
        response = api.opener.open(request, timeout=max(1, deadline - time.monotonic()))
    except HTTPError as error:
        if error.code == 404:
            return {"oauth_observed": False, "tool_names": []}
        raise SmokeError("sse_http_failed") from None
    with response:
        sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        if not callable(getattr(sock, "settimeout", None)):
            raise SmokeError("sse_deadline_unavailable")
        data = []
        while response.fp is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SmokeError("sse_timeout")
            sock.settimeout(remaining)
            raw = response.readline(1_000_001)
            if len(raw) > 1_000_000:
                raise SmokeError("sse_line_too_large")
            if not raw:
                break
            line = raw.decode().rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif not line and data:
                event = json.loads("\n".join(data))
                data = []
                consumer.consume(event)
                if event.get("final"):
                    break
    return {"oauth_observed": bool(consumer.auth_fingerprint), "tool_names": sorted(observed_tools)}


def storage_evidence(root, accounts, action_id):
    mapping = [{"account": account["support_user_id"], "uid": account["librechat_user_id"]} for account in accounts]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", action_id) or any(not re.fullmatch(r"[0-9a-f]{24}", item["uid"]) for item in mapping):
        raise SmokeError("invalid_storage_query_identity")
    # Native decryption is only used in memory for a boolean check. No token or
    # user ID is emitted, and this process never writes to the token collection.
    source = """
const write = process.stdout.write.bind(process.stdout);
process.stdout.write = () => true; process.stderr.write = () => true;
(async () => {
  const mongoose = require('mongoose');
  const { decryptV2, logger } = require('@librechat/data-schemas');
  logger.silent = true;
  const mapping = __MAPPING__, actionId = __ACTION__;
  await mongoose.connect(process.env.MONGO_URI, { maxPoolSize: 1 });
  const tokens = mongoose.connection.db.collection('tokens');
  const results = [], ciphertexts = [], refreshCiphertexts = [];
  for (const item of mapping) {
    const identifier = `${item.uid}:${actionId}`;
    const rows = await tokens.find({ identifier: { $in: [identifier, `${identifier}:refresh`] }, type: { $in: ['oauth', 'oauth_refresh'] } }).toArray();
    let encrypted = rows.length > 0;
    const encryptedByType = {};
    for (const row of rows) {
      try { const plain = await decryptV2(row.token); const valid = plain !== row.token && plain.length >= 24; encrypted = encrypted && valid; encryptedByType[row.type] = valid; }
      catch { encrypted = false; encryptedByType[row.type] = false; }
    }
    const access = rows.filter(row => row.type === 'oauth' && row.identifier === identifier);
    const refresh = rows.filter(row => row.type === 'oauth_refresh' && row.identifier === `${identifier}:refresh`);
    if (access.length === 1) ciphertexts.push(access[0].token);
    if (refresh.length === 1) refreshCiphertexts.push(refresh[0].token);
    results.push({ account: item.account, access_present: access.length === 1, refresh_present: refresh.length === 1,
      bound_to_lc_user: refresh.length === 1 && access.length <= 1 && rows.every(row => String(row.userId) === item.uid), encrypted_at_rest: encrypted,
      access_encrypted: access.length === 1 && encryptedByType.oauth === true,
      refresh_encrypted: refresh.length === 1 && encryptedByType.oauth_refresh === true });
  }
  await mongoose.disconnect();
  write(JSON.stringify({accounts:results, distinct_user_ciphertexts: ciphertexts.length === mapping.length && new Set(ciphertexts).size === ciphertexts.length,
    distinct_refresh_ciphertexts: refreshCiphertexts.length === mapping.length && new Set(refreshCiphertexts).size === refreshCiphertexts.length})+'\\n');
})().catch(() => { write(JSON.stringify({error:'storage_verification_failed'})+'\\n'); process.exitCode=2; });
""".replace("__MAPPING__", json.dumps(mapping)).replace("__ACTION__", json.dumps(action_id))
    result = subprocess.run(["docker", "compose", "-f", "compose.support.yml", "exec", "-T", "api", "node", "-"],
                            input=source, text=True, encoding="utf-8", capture_output=True, timeout=40, cwd=root)
    if result.returncode:
        raise SmokeError("storage_verification_failed")
    evidence = json.loads(result.stdout)
    if not isinstance(evidence, dict) or set(evidence) != {"accounts", "distinct_user_ciphertexts", "distinct_refresh_ciphertexts"}:
        raise SmokeError("storage_verification_invalid")
    return evidence


def one_turn(api, account, support_user, state, client_id, context, timeout):
    events, oauth, started, error, messages = [], {}, None, None, []
    observed_tools = set()
    event_counts = {}
    began = time.monotonic()
    try:
        started = api.request("POST", "/api/agents/chat", {"endpoint": "agents", "agent_id": state["agent_id"], "text": PROMPT,
            "conversationId": "new", "parentMessageId": "00000000-0000-0000-0000-000000000000", "messageId": str(uuid.uuid4()),
            "clientRequestId": "operations-oauth-" + uuid.uuid4().hex})
        if not started.get("streamId"):
            raise SmokeError("chat_submission_not_acknowledged")

        def connect(auth_url):
            oauth.update(complete_oauth(api, auth_url, support_user=support_user, action_id=state["action_id"],
                                        client_id=client_id, tls_context=context))
        stream(api, started, began + timeout, on_auth=connect, usage_events=events, observed_tools=observed_tools, event_counts=event_counts)
        if api.request("GET", "/api/agents/chat/status/" + started["conversationId"]).get("active"):
            raise SmokeError("chat_still_active")
    except Exception as exception:
        error = str(exception) if isinstance(exception, SmokeError) else "sse_timeout" if isinstance(exception, TimeoutError) or time.monotonic() >= began + timeout else "chat_or_oauth_transport_failed"
        if started and started.get("streamId"):
            try:
                api.request("POST", "/api/agents/chat/abort", {"streamId": started["streamId"], "generationCreatedAt": started["generationCreatedAt"]})
            except Exception:
                pass
    if started and started.get("conversationId"):
        try:
            messages = api.request("GET", "/api/messages/" + started["conversationId"])
        except Exception:
            error = error or "persisted_messages_unavailable"
    assistants = [message for message in messages if not message.get("isCreatedByUser")]
    _, calls = message_evidence(messages)
    identities = []
    names = []
    for call in calls:
        name = call.get("name", "")
        names.append(name if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name) else "invalid_tool_name")
        if "getCurrentSupportUser" in name:
            try:
                output = json.loads(call["output"]) if isinstance(call.get("output"), str) else call.get("output")
                if isinstance(output, dict):
                    identities.append({"account_matches": output.get("id") == support_user["id"],
                                       "role": output.get("role") if output.get("role") in {"requester", "support", "admin"} else "unknown"})
            except ValueError:
                pass
    checks = {"oauth_completed": bool(oauth.get("native_callback_completed")),
              "only_identity_tool": bool(names) and all("getCurrentSupportUser" in name for name in names + list(observed_tools)),
              "identity_matches": bool(identities) and all(item["account_matches"] and item["role"] == support_user.get("role", "requester") for item in identities),
              "stream_completed": error is None}
    usage = turn_usage(assistants, events, state["model"], error)
    return {"account": account["support_user_id"], "status": "passed" if all(checks.values()) else "failed",
            "checks": checks, "oauth_http": oauth, "tool_names": names, "identities": identities,
            "sse_tool_names": sorted(observed_tools),
            "sse_event_counts": event_counts,
            "conversation_id": started.get("conversationId") if started else None,
            "stream_id": started.get("streamId") if started else None,
            "usage": usage, "elapsed_seconds": round(time.monotonic() - began, 3), "error_code": error}


def preflight_accounts(selected, state, prices):
    """One login per account; viewers only exercise the native VIEW endpoint."""
    apis = []
    owner_checked = False
    for account in selected:
        api = LocalApi("http://localhost:3080")
        session = api.request("POST", "/api/auth/login", {"email": account["email"], "password": account["password"]})
        if not session.get("token") or str(session.get("user", {}).get("id") or session.get("user", {}).get("_id")) != account["librechat_user_id"]:
            raise SmokeError("librechat_login_identity_mismatch")
        api.token = session["token"]
        if account["support_user_id"] == "alice":
            agent = api.request("GET", f"/api/agents/{state['agent_id']}/expanded")
            if agent.get("provider") != "DeepSeek" or agent.get("model") != prices["model"]:
                raise SmokeError("unexpected_provider_or_unpriced_model")
            state["model"] = agent["model"]
            if not any("getCurrentSupportUser" in tool for tool in agent.get("tools", []) if isinstance(tool, str)):
                raise SmokeError("identity_tool_missing")
            owner_checked = True
        else:
            visible = api.request("GET", f"/api/agents/{state['agent_id']}")
            if not isinstance(visible, dict) or visible.get("id") != state["agent_id"]:
                raise SmokeError("shared_agent_not_visible")
        apis.append(api)
    if not owner_checked:
        raise SmokeError("owner_configuration_not_verified")
    return apis


def recovery_context(folder):
    raw = (folder / "oauth-report.json").read_bytes()
    baseline = json.loads(raw)
    ledger = read_json(folder / "oauth-ledger.json")
    rows = {row["account"]: row for row in baseline.get("results", [])}
    attempts = ledger.get("attempts", [])
    if (baseline.get("status") != "failed" or rows.get("alice", {}).get("status") != "passed"
            or rows.get("bob", {}).get("status") != "failed" or len(attempts) != 2
            or {row.get("account") for row in attempts} != set(ACTORS)
            or ledger.get("budget_cny") != BUDGET_CNY or ledger.get("repair_attempt")):
        raise SmokeError("recovery_requires_untouched_failed_baseline_and_unused_repair")
    committed = 0.0
    for attempt in attempts:
        amount = attempt.get("estimated_cny") if attempt.get("status") == "settled" else attempt.get("reserved_cny")
        if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
            raise SmokeError("invalid_original_budget_record")
        committed += amount
    if committed + REPAIR_RESERVATION_CNY > BUDGET_CNY + 1e-9:
        raise SmokeError("insufficient_budget_for_one_recovery_attempt")
    return baseline, ledger, hashlib.sha256(raw).hexdigest(), committed


def recovery_storage_pass(storage):
    rows = {row["account"]: row for row in storage["accounts"]}
    return (set(rows) == set(ACTORS) and storage["distinct_refresh_ciphertexts"]
            and all(row.get("refresh_present") and row.get("bound_to_lc_user") and row.get("refresh_encrypted") for row in rows.values())
            and rows["bob"].get("access_present") and rows["bob"].get("access_encrypted"))


def run_bob_recovery(args, folder, accounts, users, state, prices, client_id, context):
    lock, handle = folder / "oauth-smoke.lock", None
    try:
        try:
            handle = lock.open("x", encoding="utf-8")
        except FileExistsError:
            raise SmokeError("another_oauth_smoke_or_stale_lock") from None
        baseline, ledger, baseline_hash, committed = recovery_context(folder)
        report_path = folder / "oauth-repair-bob-report.json"
        if report_path.exists():
            raise SmokeError("repair_report_already_exists_no_automatic_retry")
        alice = next(row for row in baseline["results"] if row["account"] == "alice")
        if alice.get("usage", {}).get("models") != [prices["model"]]:
            raise SmokeError("baseline_model_does_not_match_price_snapshot")
        state["model"] = prices["model"]
        account = next(item for item in accounts if item["support_user_id"] == "bob")
        tls_probe = build_opener(ProxyHandler({}), NoRedirect(), HTTPSHandler(context=context))
        tls_status, _, _ = web_request(tls_probe, "https://localhost:8443/health")
        if tls_status != 200:
            raise SmokeError("tls_service_preflight_failed")
        before = storage_evidence(args.root, [account], state["action_id"])
        if any(row["access_present"] or row["refresh_present"] for row in before["accounts"]):
            raise SmokeError("bob_already_connected_fresh_repair_not_needed")
        api = LocalApi("http://localhost:3080")
        session = api.request("POST", "/api/auth/login", {"email": account["email"], "password": account["password"]})
        if not session.get("token") or str(session.get("user", {}).get("id") or session.get("user", {}).get("_id")) != account["librechat_user_id"]:
            raise SmokeError("librechat_login_identity_mismatch")
        api.token = session["token"]
        visible = api.request("GET", f"/api/agents/{state['agent_id']}")
        if not isinstance(visible, dict) or visible.get("id") != state["agent_id"]:
            raise SmokeError("shared_agent_not_visible")
        # No Alice login, expanded read, permission edit, token mutation or
        # renewed model request is needed for the already successful account.
        report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "pending", "scope": "bob_native_oauth_repair",
                  "baseline_report_sha256": baseline_hash, "baseline_failure_preserved": True,
                  "original_committed_or_reserved_cny": committed, "repair_reserved_cny": REPAIR_RESERVATION_CNY,
                  "total_budget_cny": BUDGET_CNY, "tls_preflight_http": tls_status,
                  "fix": "initial_subscription_resume_false_with_early_event_replay", "results": []}
        save_state(report_path, report)
        attempt = {"account": "bob", "status": "pending", "reserved_cny": REPAIR_RESERVATION_CNY,
                   "baseline_report_sha256": baseline_hash}
        ledger["repair_attempt"] = attempt
        save_state(folder / "oauth-ledger.json", ledger)
        row = one_turn(api, account, users["bob"], state, client_id, context, args.timeout)
        row["cost_estimate"] = estimate_cost(row["usage"], prices)
        cost = row["cost_estimate"]["conservative_budget"]
        attempt.update(status="unknown_usage" if cost is None else "reservation_exceeded" if cost > REPAIR_RESERVATION_CNY else "settled", estimated_cny=cost)
        save_state(folder / "oauth-ledger.json", ledger)
        report.update(results=[row], status="failed", usage=row["usage"], cost_estimate=row["cost_estimate"])
        save_state(report_path, report)
        try:
            storage = storage_evidence(args.root, accounts, state["action_id"])
            report["token_storage"] = storage
            report["refresh_isolation_verified"] = bool(recovery_storage_pass(storage))
        except Exception:
            report["refresh_isolation_verified"] = False
            report["storage_error_code"] = "storage_verification_failed"
        report["original_report_hash_unchanged"] = hashlib.sha256((folder / "oauth-report.json").read_bytes()).hexdigest() == baseline_hash
        report["status"] = "passed" if row["status"] == "passed" and attempt["status"] == "settled" and report["refresh_isolation_verified"] and report["original_report_hash_unchanged"] else "failed"
        save_state(report_path, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["status"] == "passed" else 2
    finally:
        if handle:
            handle.close()
            lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SERVICE.parents[1])
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--recover-bob", action="store_true", help="One separate repair attempt; preserve baseline and its unknown-usage reservation.")
    parser.add_argument("--timeout", type=int, default=150)
    args = parser.parse_args(argv)
    folder = args.root / ".local/operations-acceptance"
    folder.mkdir(parents=True, exist_ok=True)
    lock, handle = folder / "oauth-smoke.lock", None
    try:
        state = read_json(args.root / ".local/support-agent-bootstrap.json")
        accounts = read_json(args.root / ".local/librechat-users.json")["accounts"]
        selected = [next(account for account in accounts if account["support_user_id"] == name) for name in ACTORS]
        users = {user["id"]: user for user in read_json(args.root / ".local/users.json")["users"]}
        operations = read_json(args.root / ".local/operations.json")
        client_id = operations["oauth"]["client_id"]
        context = ssl.create_default_context(cafile=str(args.root / ".local/support-tls/server.crt"))
        raw_prices = read_json(SERVICE / "evaluation/prices.json")
        prices = {"currency": "CNY", "model": raw_prices["model"], "rates": {name: raw_prices[name] for name in ("peak", "off_peak")}}
        plan = {"status": "dry_run", "actors": list(ACTORS), "max_user_messages": 2, "automatic_retries": 0,
                "budget_cny": BUDGET_CNY, "per_turn_reservation_cny": RESERVATION_CNY,
                "price_snapshot_checked_at": raw_prices["checked_at"], "price_snapshot_is_invoice": False,
                "inflight_request_may_exceed_estimate": True, "tls_verification_required": context.verify_mode == ssl.CERT_REQUIRED,
                "local_ca_loaded": True,
                "live_model_called": False, "scope": "real_running_LibreChat_native_OAuth_identity_smoke"}
        if args.recover_bob:
            _, _, baseline_hash, committed = recovery_context(folder)
            plan.update(actors=["bob"], max_user_messages=1, per_turn_reservation_cny=REPAIR_RESERVATION_CNY,
                        baseline_report_sha256=baseline_hash, original_committed_or_reserved_cny=committed,
                        maximum_reserved_with_repair_cny=round(committed + REPAIR_RESERVATION_CNY, 8),
                        scope="bob_native_oauth_repair")
        save_state(folder / ("oauth-repair-bob-plan.json" if args.recover_bob else "oauth-plan.json"), plan)
        if not args.run:
            print(json.dumps(plan, ensure_ascii=False))
            return 0
        if args.timeout < 30 or args.timeout > 240:
            raise SmokeError("timeout_out_of_bounds")
        if args.recover_bob:
            return run_bob_recovery(args, folder, selected, users, state, prices, client_id, context)
        try:
            handle = lock.open("x", encoding="utf-8")
        except FileExistsError:
            raise SmokeError("another_oauth_smoke_or_stale_lock") from None
        ledger_path = folder / "oauth-ledger.json"
        ledger = read_json(ledger_path) if ledger_path.exists() else {"schema": "operations-oauth-budget-v1", "budget_cny": BUDGET_CNY, "attempts": []}
        if ledger.get("budget_cny") != BUDGET_CNY or ledger.get("attempts"):
            raise SmokeError("ledger_already_contains_attempts_no_automatic_retry")
        tls_probe = build_opener(ProxyHandler({}), NoRedirect(), HTTPSHandler(context=context))
        tls_status, _, _ = web_request(tls_probe, "https://localhost:8443/health")
        if tls_status != 200:
            raise SmokeError("tls_service_preflight_failed")
        before = storage_evidence(args.root, selected, state["action_id"])
        if any(row["access_present"] or row["refresh_present"] for row in before["accounts"]):
            raise SmokeError("existing_oauth_tokens_would_skip_fresh_consent")
        report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "scope": plan["scope"], "results": [],
                  "budget_cny": BUDGET_CNY, "tls_preflight_http": tls_status}
        apis = preflight_accounts(selected, state, prices)
        spent = 0.0
        for account, api in zip(selected, apis):
            if spent + RESERVATION_CNY > BUDGET_CNY + 1e-9:
                report["stop_reason"] = "estimated_budget_reached"
                break
            attempt = {"account": account["support_user_id"], "status": "pending", "reserved_cny": RESERVATION_CNY}
            ledger["attempts"].append(attempt)
            save_state(ledger_path, ledger)
            row = one_turn(api, account, users[account["support_user_id"]], state, client_id, context, args.timeout)
            row["cost_estimate"] = estimate_cost(row["usage"], prices)
            cost = row["cost_estimate"]["conservative_budget"]
            attempt.update(status="unknown_usage" if cost is None else "reservation_exceeded" if cost > RESERVATION_CNY else "settled", estimated_cny=cost)
            save_state(ledger_path, ledger)
            report["results"].append(row)
            report["usage"] = aggregate_usage([item["usage"] for item in report["results"]])
            report["cost_estimate"] = estimate_cost(report["usage"], prices)
            report["status"] = "running"
            save_state(folder / "oauth-report.json", report)
            if cost is None or cost > RESERVATION_CNY or row["status"] != "passed":
                report["stop_reason"] = attempt["status"] if row["status"] == "passed" else "smoke_failed"
                break
            spent += cost
        report["token_storage"] = storage_evidence(args.root, selected, state["action_id"])
        storage = report["token_storage"]
        storage_pass = storage["distinct_user_ciphertexts"] and all(all(row[key] for key in ("access_present", "refresh_present", "bound_to_lc_user", "encrypted_at_rest")) for row in storage["accounts"])
        report["status"] = "passed" if len(report["results"]) == 2 and all(row["status"] == "passed" for row in report["results"]) and storage_pass and not report.get("stop_reason") else "failed"
        save_state(folder / "oauth-report.json", report)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["status"] == "passed" else 2
    except Exception as error:
        code = str(error) if isinstance(error, SmokeError) else "local_configuration_or_preflight_failed"
        print(json.dumps({"status": "error", "code": code}, ensure_ascii=False))
        return 2
    finally:
        if handle:
            handle.close()
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
