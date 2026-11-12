"""Exercise the running local lifecycle using fictional data and real HTTP sessions.

Creates one new demo ticket per run. --with-model invokes DeepSeek once to prepare
the owner's closing proposal. --leave-pending leaves that final proposal for UI QA.
Never edits existing tickets or displays credentials. This is not a task benchmark.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
import uuid

import httpx

from bootstrap import BootstrapError, LocalApi, authenticate, local_base, read_json, save_state
from smoke import message_evidence, redact, stream_events


class Inputs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("name"):
            self.values[attrs["name"]] = attrs.get("value", "")


def checked(response, expected=200):
    if response.status_code != expected:
        raise BootstrapError(f"Lifecycle HTTP check expected {expected}, received {response.status_code}.")
    return response


def csrf(response):
    parser = Inputs()
    parser.feed(checked(response).text)
    if not parser.values.get("csrf"):
        raise BootstrapError("Lifecycle form is missing CSRF protection.")
    return parser.values["csrf"]


def model_proposal(root, base, ticket_id):
    api = LocalApi(base)
    demo = read_json(root / ".local/demo.json")
    authenticate(api, demo)
    state = read_json(root / ".local/support-agent-bootstrap.json")
    prompt = f"这是本地虚构演示验收。请查询工单 {ticket_id} 的最新状态和处理历史。我是提交人，已经验证登录恢复正常，请基于最新版本准备关闭工单的变更提案，说明为‘演示验收：已验证 SSO 登录恢复正常’。返回独立确认链接，等待我打开页面确认，不要声称已经关闭，也不要新建工单。"
    start_time = time.monotonic()
    started = api.request("POST", "/api/agents/chat", {
        "endpoint": "agents", "agent_id": state["agent_id"], "text": prompt,
        "conversationId": "new", "parentMessageId": "00000000-0000-0000-0000-000000000000",
        "messageId": str(uuid.uuid4()), "clientRequestId": "lifecycle-" + uuid.uuid4().hex,
    })
    try:
        stream_events(api, started, start_time + 180)
    except BootstrapError:
        api.request("POST", "/api/agents/chat/abort", {"streamId": started["streamId"], "generationCreatedAt": started["generationCreatedAt"]})
        raise
    messages = api.request("GET", "/api/messages/" + started["conversationId"])
    answers, calls = message_evidence(messages)
    names = [call.get("name", "") for call in calls]
    for expected in ("getMyTicket", "getTicketHistory", "proposeTicketUpdate"):
        if not any(expected in name for name in names):
            raise BootstrapError("Model did not execute all required lifecycle tools.")
    proposal_calls = [call for call in calls if "proposeTicketUpdate" in call.get("name", "")]
    ids = set(re.findall(r"U-[A-Fa-f0-9]{16}", json.dumps(proposal_calls)))
    if len(ids) != 1:
        raise BootstrapError("Model result does not identify one closing proposal.")
    return next(iter(ids)), {
        "conversation_url": base + "/c/" + started["conversationId"],
        "elapsed_seconds": round(time.monotonic() - start_time, 3),
        "prompt": prompt, "answers": answers, "tool_calls": calls,
    }


def run(root, support_url, base, with_model, leave_pending):
    users = {item["id"]: item for item in read_json(root / ".local/users.json")["users"]}
    run_id = uuid.uuid4().hex
    steps = []
    with ExitStack() as stack:
        browsers, apis = {}, {}
        for ident in ("alice", "bob", "support1", "support2", "admin"):
            user = users[ident]
            browser = stack.enter_context(httpx.Client(base_url=support_url, trust_env=False, timeout=20))
            token = csrf(browser.get("/console/login"))
            checked(browser.post("/console/login", data={"username": ident, "password": user["password"], "csrf": token}, headers={"Origin": support_url}), 303)
            browsers[ident] = browser
            apis[ident] = stack.enter_context(httpx.Client(base_url=support_url, trust_env=False, timeout=20, headers={"Authorization": "Bearer " + user["api_key"]}))

        def read_ticket(ident="alice"):
            return checked(apis[ident].get("/api/tickets/" + ticket_id)).json()

        def approve(ident, path):
            browser = browsers[ident]
            token = csrf(browser.get(path))
            checked(browser.post(path + "/confirm", data={"csrf": token}, headers={"Origin": support_url}), 303)
            # Replay the same authorized confirmation; it must not add another event.
            checked(browser.post(path + "/confirm", data={"csrf": token}, headers={"Origin": support_url}), 303)

        def change(ident, action, note, **extra):
            before = read_ticket(ident)
            response = checked(apis[ident].post(f"/api/tickets/{ticket_id}/updates", json={
                "action": action, "note": note, "request_id": run_id + "-" + action,
                "expected_version": before["version"], **extra,
            }), 201).json()
            if read_ticket(ident)["version"] != before["version"]:
                raise BootstrapError("A proposal changed the ticket before confirmation.")
            approve(ident, "/console/updates/" + response["id"])
            after = read_ticket(ident)
            if after["version"] != before["version"] + 1:
                raise BootstrapError("Confirmation/replay violated the ticket version invariant.")
            steps.append({"action": action, "actor": ident, "status": after["status"], "version": after["version"]})

        draft = checked(apis["alice"].post("/api/drafts", json={
            "title": "演示：SSO 登录故障全流程验收", "description": "虚构演示：测试员工无法登录 SSO，已核对回调地址，需要客服排查并记录解决过程。",
            "priority": "normal", "request_id": run_id + "-create",
        }), 201).json()
        approve("alice", "/console/drafts/" + draft["id"])
        ticket_id = checked(apis["alice"].get("/api/drafts/" + draft["id"])).json()["ticket_id"]
        steps.append({"action": "created", "actor": "alice", "status": "open", "version": 1})
        checked(apis["bob"].get("/api/tickets/" + ticket_id), 404)
        checked(apis["support2"].get("/api/tickets/" + ticket_id), 404)
        checked(apis["alice"].post(f"/api/tickets/{ticket_id}/updates", json={"action": "assign", "assignee_id": "support1", "note": "演示越权校验", "request_id": run_id + "-denied", "expected_version": 1}), 403)
        checked(apis["admin"].post(f"/api/tickets/{ticket_id}/updates", json={"action": "resolve", "note": "演示非法状态校验", "request_id": run_id + "-invalid", "expected_version": 1}), 409)
        change("admin", "assign", "演示验收：分配给王工排查 SSO 配置。", assignee_id="support1")
        change("support1", "start", "演示验收：开始核对 ACS 回调地址和身份提供商配置。")
        change("support1", "comment", "演示验收：已定位回调地址不匹配，准备修正测试配置。")
        change("support1", "resolve", "演示验收：修正模拟回调地址后，三个测试账号均可正常登录。")
        model = None
        if with_model:
            update_id, model = model_proposal(root, base, ticket_id)
            proposal = checked(apis["alice"].get("/api/updates/" + update_id)).json()
            if proposal["status"] != "pending" or proposal["action"] != "close" or proposal["ticket_id"] != ticket_id or read_ticket()["status"] != "resolved":
                raise BootstrapError("Model proposal did not preserve the human confirmation boundary.")
            if not leave_pending:
                approve("alice", "/console/updates/" + update_id)
                steps.append({"action": "close", "actor": "alice", "status": read_ticket()["status"], "version": read_ticket()["version"]})
        else:
            change("alice", "close", "演示验收：提交人已验证 SSO 登录恢复正常。")
            update_id = None
        final = read_ticket()
        events = checked(apis["alice"].get(f"/api/tickets/{ticket_id}/history")).json()["events"]
        expected = "resolved" if with_model and leave_pending else "closed"
        if final["status"] != expected or len(events) != final["version"]:
            raise BootstrapError("Lifecycle final state or event count is incorrect.")
        report = {
            "status": "awaiting_browser_confirmation" if expected == "resolved" else "passed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "One fictional lifecycle through live local HTTP, not a success-rate or load benchmark.",
            "ticket_id": ticket_id, "ticket_url": support_url + "/console/tickets/" + ticket_id,
            "closing_update_id": update_id, "steps": steps, "final_ticket": final, "history": events,
            "checks": {"unrelated_requester_denied": True, "unassigned_support_denied": True, "requester_assignment_denied": True, "illegal_transition_denied": True, "proposal_does_not_mutate_ticket": True, "confirmation_replay_idempotent": True},
            "model": model,
        }
        return redact(report, [secret for user in users.values() for secret in (user["password"], user["api_key"])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--support-url", default="http://localhost:8090")
    parser.add_argument("--base-url", default="http://localhost:3080")
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--leave-pending", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = run(args.root, local_base(args.support_url), local_base(args.base_url), args.with_model, args.leave_pending)
        save_state(args.output or args.root / ".local/lifecycle-latest.json", report)
        print(json.dumps({key: report[key] for key in ("status", "ticket_id", "closing_update_id", "checks")}, ensure_ascii=False, indent=2))
        return 0
    except (BootstrapError, httpx.HTTPError, KeyError, ValueError) as error:
        message = str(error) if isinstance(error, BootstrapError) else "Local lifecycle verification failed; inspect configuration and service health."
        print(json.dumps({"status": "failed", "message": message}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
