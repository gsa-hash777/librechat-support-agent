from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import sqlite3
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


class HiddenInputs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if tag == "input" and values.get("name"):
            self.values[values["name"]] = values.get("value", "")


def csrf_from(response):
    assert response.status_code == 200, response.text
    parser = HiddenInputs()
    parser.feed(response.text)
    assert parser.values.get("csrf"), "Browser form must contain a CSRF token."
    return parser.values["csrf"]


def login(client, users, username="alice"):
    user = next(item for item in users if item["id"] == username)
    token = csrf_from(client.get("/console/login"))
    response = client.post("/console/login", data={
        "username": username, "password": user["password"], "csrf": token, "next": "/console",
    }, follow_redirects=False)
    assert response.status_code == 303, response.text
    return response


def payload(request_id="test-request-0001", **changes):
    return {"title": "VPN 连接失败", "description": "更新客户端后仍无法连接公司 VPN，需要支持人员协助。", "priority": "normal", "request_id": request_id, **changes}


def draft(client, headers, username="alice", request_id="test-request-0001"):
    response = client.post("/api/drafts", headers=headers[username], json=payload(request_id))
    assert response.status_code == 201, response.text
    return response.json()


def tickets(client, headers, username="alice"):
    response = client.get("/api/tickets", headers=headers[username])
    assert response.status_code == 200, response.text
    return response.json()["tickets"]


def confirm(client, users, created):
    login(client, users)
    path = f"/console/drafts/{created['id']}"
    csrf = csrf_from(client.get(path))
    response = client.post(path + "/confirm", data={"csrf": csrf}, follow_redirects=False)
    assert response.status_code == 303, response.text
    return path, csrf


@pytest.mark.parametrize("path", ["/api/me", "/api/tickets", "/api/audit", "/api/knowledge/search?q=VPN"])
@pytest.mark.parametrize("authorization", [None, "Bearer invalid-key", "Basic arbitrary"])
def test_tool_endpoints_require_valid_api_key(client, path, authorization):
    request_headers = {} if authorization is None else {"Authorization": authorization}
    assert client.get(path, headers=request_headers).status_code == 401


def test_api_identity_comes_from_key(client, headers):
    for username in headers:
        response = client.get("/api/me", headers=headers[username])
        assert response.status_code == 200
        assert response.json()["id"] == username


def test_draft_only_does_not_create_ticket(client, headers):
    created = draft(client, headers)
    assert created["status"] == "pending"
    assert created["ticket_id"] is None
    assert created["replayed"] is False
    assert created["confirmation_url"] == f"http://testserver/console/drafts/{created['id']}"
    assert tickets(client, headers) == []


@pytest.mark.parametrize("extra", [{"approved": True}, {"user_id": "bob"}, {"org_id": "other"}, {"priority": "critical"}])
def test_model_cannot_supply_approval_or_identity(client, headers, extra):
    response = client.post("/api/drafts", headers=headers["alice"], json=payload(**extra))
    assert response.status_code == 422
    assert tickets(client, headers) == []


def test_draft_replay_and_conflict(client, headers):
    created = draft(client, headers)
    repeated = client.post("/api/drafts", headers=headers["alice"], json=payload())
    assert repeated.status_code == 200
    assert repeated.json()["id"] == created["id"]
    assert repeated.json()["replayed"] is True
    conflict = client.post("/api/drafts", headers=headers["alice"], json=payload(title="更换工单标题"))
    assert conflict.status_code == 409
    original = client.get(f"/api/drafts/{created['id']}", headers=headers["alice"]).json()
    assert original["title"] == payload()["title"]


def test_idempotency_key_is_scoped_to_user(client, headers):
    alice = draft(client, headers)
    bob = draft(client, headers, "bob")
    assert alice["id"] != bob["id"]


def test_concurrent_draft_retries_create_exactly_one_draft(client, headers, settings):
    count = 8
    ready = Barrier(count)

    def request(_):
        ready.wait(timeout=10)
        return client.post("/api/drafts", headers=headers["alice"], json=payload())

    with ThreadPoolExecutor(max_workers=count) as pool:
        responses = list(pool.map(request, range(count)))
    assert sorted(response.status_code for response in responses) == [200] * (count - 1) + [201]
    assert len({response.json()["id"] for response in responses}) == 1
    with sqlite3.connect(settings.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0


def test_cross_user_draft_and_audit_are_isolated(client, headers):
    alice = draft(client, headers)
    bob = draft(client, headers, "bob", "bob-request-0001")
    assert client.get(f"/api/drafts/{alice['id']}", headers=headers["bob"]).status_code == 404
    for username, own, other in [("alice", alice, bob), ("bob", bob, alice)]:
        events = client.get("/api/audit", headers=headers[username]).json()["events"]
        assert any(event["object_id"] == own["id"] for event in events)
        assert all(event["object_id"] != other["id"] for event in events if event["outcome"] == "ok")


def test_api_key_alone_cannot_confirm(client, headers):
    created = draft(client, headers)
    response = client.post(f"/console/drafts/{created['id']}/confirm", headers=headers["alice"], data={"csrf": "fabricated"}, follow_redirects=False)
    assert response.status_code == 401
    assert tickets(client, headers) == []


@pytest.mark.parametrize("csrf", [None, "incorrect-csrf", "错误令牌"])
def test_login_rejects_missing_or_wrong_csrf(client, users, csrf):
    client.get("/console/login")
    data = {"username": "alice", "password": users[0]["password"]}
    if csrf is not None:
        data["csrf"] = csrf
    assert client.post("/console/login", data=data, follow_redirects=False).status_code in (403, 422)
    assert client.cookies.get("support_session") is None


def test_login_rejects_wrong_password(client):
    token = csrf_from(client.get("/console/login"))
    response = client.post("/console/login", data={"username": "alice", "password": "not-correct-password", "csrf": token}, follow_redirects=False)
    assert response.status_code == 401
    assert client.cookies.get("support_session") is None


def test_browser_session_cannot_replace_api_key(client, users):
    login_response = login(client, users)
    session_cookie = next(cookie for cookie in client.cookies.jar if cookie.name == "support_session")
    assert session_cookie.path == "/console"
    assert "httponly" in login_response.headers.get("set-cookie", "").lower()
    assert client.get("/api/tickets").status_code == 401


@pytest.mark.parametrize("action", ["confirm", "cancel"])
@pytest.mark.parametrize("csrf", [None, "wrong-token", "错误令牌"])
def test_browser_mutation_requires_session_csrf(client, users, headers, action, csrf):
    created = draft(client, headers)
    login(client, users)
    data = {} if csrf is None else {"csrf": csrf}
    response = client.post(f"/console/drafts/{created['id']}/{action}", data=data, follow_redirects=False)
    assert response.status_code in (403, 422)
    assert client.get(f"/api/drafts/{created['id']}", headers=headers["alice"]).json()["status"] == "pending"
    assert tickets(client, headers) == []


def test_other_user_browser_cannot_confirm_draft(client, users, headers):
    alice = draft(client, headers)
    bob = draft(client, headers, "bob", "bob-request-0001")
    login(client, users, "bob")
    bob_token = csrf_from(client.get(f"/console/drafts/{bob['id']}"))
    assert client.get(f"/console/drafts/{alice['id']}", follow_redirects=False).status_code == 404
    response = client.post(f"/console/drafts/{alice['id']}/confirm", data={"csrf": bob_token}, follow_redirects=False)
    assert response.status_code == 404
    assert tickets(client, headers, "alice") == []
    assert tickets(client, headers, "bob") == []


def test_confirm_checks_final_state_and_ticket_visibility(client, users, headers):
    created = draft(client, headers)
    path, _ = confirm(client, users, created)
    owned = tickets(client, headers)
    assert len(owned) == 1
    ticket = owned[0]
    assert ticket["title"] == payload()["title"]
    assert ticket["description"] == payload()["description"]
    assert ticket["priority"] == "normal"
    assert ticket["status"] == "open"
    assert client.get(f"/api/tickets/{ticket['id']}", headers=headers["alice"]).status_code == 200
    assert client.get(f"/api/tickets/{ticket['id']}", headers=headers["bob"]).status_code == 404
    assert tickets(client, headers, "bob") == []
    updated = client.get(f"/api/drafts/{created['id']}", headers=headers["alice"]).json()
    assert updated["status"] == "submitted"
    assert updated["ticket_id"] == ticket["id"]
    assert client.get(path).status_code == 200


def test_concurrent_confirmation_creates_exactly_one_ticket(client, users, headers, settings):
    created = draft(client, headers)
    login(client, users)
    path = f"/console/drafts/{created['id']}"
    token = csrf_from(client.get(path))
    count = 8
    ready = Barrier(count)

    def request(_):
        ready.wait(timeout=10)
        return client.post(path + "/confirm", data={"csrf": token}, follow_redirects=False)

    with ThreadPoolExecutor(max_workers=count) as pool:
        responses = list(pool.map(request, range(count)))
    assert all(response.status_code == 303 for response in responses)
    assert len(tickets(client, headers)) == 1
    with sqlite3.connect(settings.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tickets WHERE draft_id=?", (created["id"],)).fetchone()[0] == 1


def test_cancel_prevents_later_submission(client, users, headers):
    created = draft(client, headers)
    login(client, users)
    path = f"/console/drafts/{created['id']}"
    token = csrf_from(client.get(path))
    response = client.post(path + "/cancel", data={"csrf": token}, follow_redirects=False)
    assert response.status_code == 303
    assert client.get(f"/api/drafts/{created['id']}", headers=headers["alice"]).json()["status"] == "cancelled"
    assert client.post(path + "/confirm", data={"csrf": token}, follow_redirects=False).status_code in (409, 410)
    assert tickets(client, headers) == []


def test_expired_draft_cannot_create_ticket(client, users, headers, settings):
    created = draft(client, headers)
    login(client, users)
    path = f"/console/drafts/{created['id']}"
    token = csrf_from(client.get(path))
    with sqlite3.connect(settings.db_path) as connection:
        connection.execute("UPDATE drafts SET expires_at=0 WHERE id=?", (created["id"],))
    assert client.get(f"/api/drafts/{created['id']}", headers=headers["alice"]).json()["status"] == "expired"
    response = client.post(path + "/confirm", data={"csrf": token}, follow_redirects=False)
    assert response.status_code == 410
    assert tickets(client, headers) == []


def test_expired_session_cannot_confirm(client, users, headers, settings):
    created = draft(client, headers)
    login(client, users)
    path = f"/console/drafts/{created['id']}"
    token = csrf_from(client.get(path))
    with sqlite3.connect(settings.db_path) as connection:
        connection.execute("UPDATE sessions SET expires_at=0")
    assert client.post(path + "/confirm", data={"csrf": token}, follow_redirects=False).status_code == 401
    assert tickets(client, headers) == []


def test_knowledge_search_returns_sources_and_limits(client, headers):
    response = client.get("/api/knowledge/search", headers=headers["alice"], params={"q": "忘记密码如何重置", "limit": 1})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == "reset-password"
    assert results[0]["source"] == "fixture://password"
    assert results[0]["score"] > 0
    assert results[0]["content"]


def test_knowledge_search_respects_organization(client, headers):
    alice = client.get("/api/knowledge/search?q=sharedmarker", headers=headers["alice"]).json()["results"]
    eve = client.get("/api/knowledge/search?q=sharedmarker", headers=headers["eve"]).json()["results"]
    assert {item["id"] for item in alice} == {"reset-password", "vpn-connect"}
    assert {item["id"] for item in eve} == {"other-secret"}


def test_unknown_query_returns_empty_results(client, headers):
    response = client.get("/api/knowledge/search?q=unfindableuniquetoken", headers=headers["alice"])
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_credentials_are_hashed_and_not_returned(client, users, headers, settings):
    draft(client, headers)
    responses = [client.get(path, headers=headers["alice"]).text for path in ["/api/me", "/api/audit", "/api/tickets"]]
    with sqlite3.connect(settings.db_path) as connection:
        stored = connection.execute("SELECT password_hash,api_key_hash FROM users WHERE id='alice'").fetchone()
    for secret in [users[0]["password"], users[0]["api_key"]]:
        assert all(secret not in value for value in responses)
        assert all(secret != value for value in stored)


def test_confirmed_ticket_survives_app_restart(client, users, headers, settings):
    created = draft(client, headers)
    confirm(client, users, created)
    original = tickets(client, headers)[0]["id"]
    with TestClient(create_app(settings)) as restarted:
        assert [item["id"] for item in tickets(restarted, headers)] == [original]
        replay = restarted.post("/api/drafts", headers=headers["alice"], json=payload())
        assert replay.status_code == 200
        assert replay.json()["ticket_id"] == original


def test_openapi_exposes_tools_but_no_browser_confirmation(client):
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {
        "/api/me", "/api/knowledge/search", "/api/tickets", "/api/tickets/{ticket_id}",
        "/api/drafts", "/api/drafts/{draft_id}", "/api/audit",
        "/api/staff", "/api/tickets/{ticket_id}/history",
        "/api/tickets/{ticket_id}/updates", "/api/updates/{update_id}",
        "/api/knowledge/{doc_id}/chunks/{chunk_id}",
    }
    draft_schema = schema["components"]["schemas"]["DraftInput"]
    assert draft_schema["additionalProperties"] is False
    assert set(draft_schema["properties"]) == {"title", "description", "priority", "request_id"}
    for path in schema["paths"].values():
        for operation in path.values():
            assert operation["security"] == [{"SupportBearer": []}]


@pytest.mark.parametrize("origin", ["https://attacker.invalid", "null"])
def test_cross_origin_confirm_is_rejected_even_with_valid_csrf(client, users, headers, origin):
    created = draft(client, headers)
    login(client, users)
    path = f"/console/drafts/{created['id']}"
    token = csrf_from(client.get(path))
    response = client.post(path + "/confirm", data={"csrf": token}, headers={"Origin": origin}, follow_redirects=False)
    assert response.status_code == 403
    assert tickets(client, headers) == []


@pytest.mark.parametrize("origin", ["https://attacker.invalid", "null"])
def test_cross_origin_login_is_rejected(client, users, origin):
    token = csrf_from(client.get("/console/login"))
    response = client.post("/console/login", data={"username": "alice", "password": users[0]["password"], "csrf": token}, headers={"Origin": origin}, follow_redirects=False)
    assert response.status_code == 403
    assert client.cookies.get("support_session") is None


def test_login_page_preserves_same_origin_and_blocks_embedding(client):
    response = client.get("/console/login")
    assert response.status_code == 200
    assert response.headers["referrer-policy"] == "same-origin"
    assert response.headers["x-frame-options"] == "DENY"
    assert "form-action 'self'" in response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_browser_escapes_draft_text_and_disallows_framing(client, users, headers):
    response = client.post("/api/drafts", headers=headers["alice"], json=payload(title="<script>alert(1)</script>", description='<img src=x onerror="alert(1)"> 注入演示内容'))
    assert response.status_code == 201
    login(client, users)
    page = client.get(f"/console/drafts/{response.json()['id']}")
    assert page.status_code == 200
    assert "<script>alert(1)</script>" not in page.text
    assert '<img src=x onerror="alert(1)">' not in page.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page.text
    assert page.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]


def test_validation_response_does_not_echo_submitted_values(client, headers):
    secret = "SECRET-MARKER-DO-NOT-ECHO"
    response = client.post("/api/drafts", headers=headers["alice"], json=payload(priority=secret))
    assert response.status_code == 422
    assert secret not in response.text


def test_audit_is_linked_to_response_trace(client, headers):
    response = client.post("/api/drafts", headers=headers["alice"], json=payload())
    assert response.status_code == 201
    trace = response.headers["x-trace-id"]
    events = client.get("/api/audit", headers=headers["alice"]).json()["events"]
    matching = [event for event in events if event["trace_id"] == trace]
    assert len(matching) == 1
    assert matching[0]["object_id"] == response.json()["id"]
    assert matching[0]["event"] == "draft.create"
    assert matching[0]["outcome"] == "ok"
