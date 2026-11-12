"""Run one paid, read-only Agent smoke request through LibreChat.

Run explicitly after bootstrap.py. Does not retry model submissions, approve tools,
or create tickets. Saves a redacted answer, tool evidence and service audit delta.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request

from bootstrap import ApiError, BootstrapError, LOCAL_USER_AGENT, LocalApi, authenticate, local_base, read_json, save_state


PROMPT = "先检索知识库说明 SSO 登录失败如何排查，请引用文档编号，不创建工单或草稿。"
SECRET_FIELD = re.compile(r"password|secret|api.?key|authorization|access.?token|refresh.?token", re.I)


def redact(value, secrets):
    if isinstance(value, dict):
        return {key: "[REDACTED]" if SECRET_FIELD.search(key) else redact(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret and len(secret) >= 4:
                value = value.replace(secret, "[REDACTED]")
        return re.sub(r"(?i)Bearer\s+[A-Za-z0-9_.~-]+", "Bearer [REDACTED]", value)
    return value


def stream_events(api: LocalApi, start: dict, deadline: float):
    query = urlencode({"resume": "true", "generationCreatedAt": start["generationCreatedAt"]})
    path = f"/api/agents/chat/stream/{start['streamId']}?{query}"
    request = Request(api.base + path, headers={
        "Accept": "text/event-stream", "Origin": api.base, "Authorization": f"Bearer {api.token}",
        "User-Agent": LOCAL_USER_AGENT,
    })
    count = 0
    final = None
    try:
        with api.opener.open(request, timeout=30) as response:
            data = []
            for raw_line in response:
                if time.monotonic() >= deadline:
                    raise BootstrapError("Agent smoke timed out.")
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif not line and data:
                    payload = json.loads("\n".join(data))
                    data = []
                    count += 1
                    if payload.get("final"):
                        final = payload
                        break
                    if payload.get("event") == "on_pending_action":
                        raise BootstrapError("Agent paused for user input; no action was approved by smoke.")
    except HTTPError as error:
        if error.code != 404:
            raise ApiError(error.code, "/api/agents/chat/stream") from None
        # A fast completed run can already have left the job store. Read saved messages.
    except (URLError, TimeoutError, OSError, UnicodeError, ValueError):
        raise BootstrapError("Agent stream failed; inspect the saved conversation in LibreChat.") from None
    return count, final


def message_evidence(messages: list) -> tuple[list, list]:
    answers = []
    calls = []
    for message in messages:
        if message.get("isCreatedByUser"):
            continue
        texts = []
        for part in message.get("content", []):
            if part.get("type") == "text":
                text = part.get("text", "")
                texts.append(text.get("value", "") if isinstance(text, dict) else text)
            elif part.get("type") == "tool_call":
                call = part.get("tool_call", {})
                calls.append({key: call[key] for key in ("id", "name", "args", "output", "type") if key in call})
        answer = "\n".join(text for text in texts if isinstance(text, str) and text)
        if not answer:
            answer = message.get("text", "")
        if answer:
            answers.append({"message_id": message.get("messageId"), "text": answer, "error": bool(message.get("error"))})
    return answers, calls


def run(root: Path, base: str, support_url: str, timeout: int) -> dict:
    demo = read_json(root / ".local/demo.json")
    users = read_json(root / ".local/users.json")["users"]
    support_user = next(user for user in users if user["id"] == demo["support_user_id"])
    state = read_json(root / ".local/support-agent-bootstrap.json")
    api = LocalApi(base)
    authenticate(api, demo)
    support = LocalApi(support_url)
    support.token = support_user["api_key"]
    before = support.request("GET", "/api/audit?limit=100")["events"]
    prior_ids = {entry["id"] for entry in before}
    payload = {
        "endpoint": "agents", "agent_id": state["agent_id"], "text": PROMPT,
        "conversationId": "new", "parentMessageId": "00000000-0000-0000-0000-000000000000",
        "messageId": str(uuid.uuid4()), "clientRequestId": "support-smoke-" + uuid.uuid4().hex,
    }
    began = time.monotonic()
    start = api.request("POST", "/api/agents/chat", payload)
    if not isinstance(start, dict) or not start.get("streamId"):
        raise BootstrapError("LibreChat did not acknowledge the Agent run.")
    stream_error = None
    event_count = 0
    try:
        event_count, _ = stream_events(api, start, began + timeout)
    except BootstrapError as error:
        stream_error = str(error)
        try:
            api.request("POST", "/api/agents/chat/abort", {
                "streamId": start["streamId"], "generationCreatedAt": start["generationCreatedAt"],
            })
        except BootstrapError:
            pass
    final_status = api.request("GET", f"/api/agents/chat/status/{start['conversationId']}")
    messages = api.request("GET", f"/api/messages/{start['conversationId']}")
    answers, calls = message_evidence(messages)
    after = support.request("GET", "/api/audit?limit=100")["events"]
    delta = [entry for entry in after if entry["id"] not in prior_ids]
    answer_text = "\n".join(item["text"] for item in answers)
    checks = {
        "search_tool_observed": any("searchSupportKnowledge" in call.get("name", "") for call in calls),
        "search_executed_in_service": any(entry["event"] == "knowledge.search" and entry["outcome"] == "ok" for entry in delta),
        "document_id_cited": "KB-002" in answer_text,
        "no_business_write_observed": not any(entry["event"] in {"draft.create", "ticket.create", "draft.confirm", "draft.cancel", "update.propose", "update.confirm", "update.cancel"} for entry in delta),
        "answer_returned": bool(answers) and not any(item["error"] for item in answers),
        "stream_completed": stream_error is None and final_status.get("active") is False,
    }
    report = {
        "status": "passed" if all(checks.values()) else "needs_review",
        "scope": "One smoke example, not an accuracy benchmark or manual answer-quality review.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": PROMPT, "conversation_id": start["conversationId"],
        "conversation_url": base + "/c/" + start["conversationId"],
        "elapsed_seconds": round(time.monotonic() - began, 3), "sse_event_count": event_count,
        "checks": checks, "answers": answers, "tool_calls": calls, "support_audit_delta": delta,
        "stream_error": stream_error,
    }
    secrets = [api.token, demo["password"]]
    for user in users:
        secrets.extend([user.get("password"), user.get("api_key")])
    if (root / ".env").exists():
        for line in (root / ".env").read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.partition("=")
            if separator and SECRET_FIELD.search(key) and not key.startswith("#"):
                secrets.append(value.strip().strip("\"'"))
    return redact(report, secrets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--base-url", default="http://localhost:3080")
    parser.add_argument("--support-url", default="http://localhost:8090")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = run(args.root, local_base(args.base_url), local_base(args.support_url), args.timeout)
        output = args.output or args.root / ".local/smoke-latest.json"
        save_state(output, report)
        print(json.dumps({key: report[key] for key in ("status", "checks", "elapsed_seconds", "conversation_url")}, indent=2))
        print("Report saved: " + str(output))
        return 0 if report["status"] == "passed" else 2
    except (BootstrapError, KeyError, TypeError, ValueError, StopIteration) as error:
        message = str(error) if isinstance(error, BootstrapError) else "Local smoke configuration is incomplete."
        print(json.dumps({"status": "error", "message": message}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
