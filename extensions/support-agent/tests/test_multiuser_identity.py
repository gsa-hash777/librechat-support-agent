"""Multi-user acceptance through real login, OAuth, scoped tools and human approval."""
from contextlib import contextmanager
from dataclasses import replace
import json
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from test_auth_races import mutate_before_write_lock
from test_oauth import AUTH, CALLBACK, CLIENT_ID, CLIENT_SECRET, Inputs, authorization_url
from test_service import csrf_from, login, payload


MONITOR_TOKEN = "monitor-test-token-" + "m" * 32

@pytest.fixture
def users(users):
    return [*users, *[
        {"id": identifier, "display_name": identifier, "org_id": org, "role": role,
         "password": identifier + "-test-password", "api_key": identifier + "-test-key-" + "x" * 32}
        for identifier, org, role in [("staff_a", "demo", "support"), ("staff_b", "demo", "support"),
                                       ("admin", "demo", "admin"), ("other_admin", "other", "admin")]
    ]]


@pytest.fixture
def settings(settings, tmp_path):
    operations = tmp_path / "operations.json"
    operations.write_text(json.dumps({"monitor_token": MONITOR_TOKEN, "oauth": {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
                                               "redirect_uris": [CALLBACK]}}), encoding="utf-8")
    return replace(settings, operations_path=operations, allow_legacy_tool_keys=False)


def connect(client, users, user="alice", scope="support:read support:propose"):
    login(client, users, user)
    response = client.get(authorization_url(scope=scope))
    assert response.status_code == 200, response.text
    form = Inputs(response.text).values
    consent = client.post("/console/oauth/authorize", data={**form, "decision": "approve"}, follow_redirects=False)
    assert consent.status_code == 303, consent.text
    code = parse_qs(urlsplit(consent.headers["location"]).query)["code"][0]
    response = client.post("/oauth/token", headers=AUTH, data={"grant_type": "authorization_code", "code": code, "redirect_uri": CALLBACK})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def actors(app, users):
    sessions = {}
    def get(user="alice", scope="support:read support:propose"):
        key = (user, scope)
        if key not in sessions:
            client = TestClient(app)
            client.__enter__()
            tokens = connect(client, users, user, scope)
            sessions[key] = (client, {"Authorization": "Bearer " + tokens["access_token"]}, tokens)
        return sessions[key]
    yield get
    for client, _, _ in sessions.values():
        client.__exit__(None, None, None)


def create_ticket(actors, owner="alice"):
    client, headers, _ = actors(owner)
    response = client.post("/api/drafts", headers=headers, json=payload("oauth-create-ticket-0001"))
    assert response.status_code == 201, response.text
    draft_id = response.json()["id"]
    path = "/console/drafts/" + draft_id
    response = client.post(path + "/confirm", data={"csrf": csrf_from(client.get(path))}, follow_redirects=False)
    assert response.status_code == 303, response.text
    ticket_id = client.get("/api/drafts/" + draft_id, headers=headers).json()["ticket_id"]
    return client.get("/api/tickets/" + ticket_id, headers=headers).json()


def update(actors, ticket, user, action, **changes):
    client, headers, _ = actors(user)
    current = client.get("/api/tickets/" + ticket["id"], headers=headers).json()
    response = client.post(f"/api/tickets/{ticket['id']}/updates", headers=headers, json={
        "action": action, "expected_version": current["version"], "request_id": f"oauth-{action}-{current['version']:04}",
        "note": "经本人核对并确认本次服务处理步骤。", **changes,
    })
    assert response.status_code == 201, response.text
    path = "/console/updates/" + response.json()["id"]
    approved = client.post(path + "/confirm", data={"csrf": csrf_from(client.get(path))}, follow_redirects=False)
    assert approved.status_code == 303, approved.text
    return client.get("/api/tickets/" + ticket["id"], headers=headers).json()


def test_oauth_accounts_are_distinct_and_legacy_keys_are_disabled(actors, users):
    for username in ("alice", "bob", "admin"):
        client, headers, _ = actors(username)
        response = client.get("/api/me", headers={**headers, "X-User-Id": "admin", "X-User-Role": "admin"})
        assert response.status_code == 200
        assert response.json()["id"] == username
        old_key = next(item["api_key"] for item in users if item["id"] == username)
        assert client.get("/api/me", headers={"Authorization": "Bearer " + old_key}).status_code == 401
    alice, alice_headers, _ = actors("alice")
    bob, bob_headers, _ = actors("bob")
    created = alice.post("/api/drafts", headers=alice_headers, json=payload()).json()
    assert bob.get("/api/drafts/" + created["id"], headers=bob_headers).status_code == 404


def test_oauth_role_matrix_and_complete_human_confirmed_lifecycle(actors):
    ticket = create_ticket(actors)
    for user in ("bob", "staff_a", "staff_b", "eve", "other_admin"):
        client, headers, _ = actors(user)
        assert client.get("/api/tickets/" + ticket["id"], headers=headers).status_code == 404
    admin, admin_headers, _ = actors("admin")
    assert admin.get("/api/tickets/" + ticket["id"], headers=admin_headers).status_code == 200
    assert {item["id"] for item in admin.get("/api/staff", headers=admin_headers).json()["staff"]} == {"staff_a", "staff_b"}
    alice, alice_headers, _ = actors("alice")
    assert alice.get("/api/staff", headers=alice_headers).status_code == 403
    ticket = update(actors, ticket, "admin", "assign", assignee_id="staff_a")
    assert ticket["assignee_id"] == "staff_a"
    ticket = update(actors, ticket, "staff_a", "start")
    assert ticket["status"] == "in_progress"
    ticket = update(actors, ticket, "staff_a", "resolve")
    assert ticket["status"] == "resolved"
    response = admin.post(f"/api/tickets/{ticket['id']}/updates", headers=admin_headers, json={
        "action": "close", "expected_version": ticket["version"], "request_id": "oauth-denied-close-0001", "note": "管理员试图代关他人工单。",
    })
    assert response.status_code == 403
    ticket = update(actors, ticket, "alice", "close")
    assert ticket["status"] == "closed" and ticket["version"] == 5


def test_tool_bearer_never_substitutes_for_browser_or_creator_confirmation(actors, app):
    owner, headers, _ = actors("alice")
    draft = owner.post("/api/drafts", headers=headers, json=payload()).json()
    path = "/console/drafts/" + draft["id"]
    with TestClient(app) as tool_only:
        assert tool_only.post(path + "/confirm", headers=headers, data={"csrf": "not-a-session"}, follow_redirects=False).status_code == 401
        assert tool_only.post("/api/drafts/" + draft["id"] + "/confirm", headers=headers).status_code == 404
    admin, admin_headers, _ = actors("admin")
    assert admin.get("/api/drafts/" + draft["id"], headers=admin_headers).status_code == 404
    assert admin.get(path).status_code == 404
    with app.state.database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0


def test_read_scope_cannot_create_draft_or_update_proposal(actors):
    ticket = create_ticket(actors)
    client, headers, _ = actors("alice", "support:read")
    assert client.get("/api/me", headers=headers).status_code == 200
    assert client.get("/api/knowledge/search?q=VPN", headers=headers).status_code == 200
    assert client.post("/api/drafts", headers=headers, json=payload("read-scope-write-0001")).status_code == 401
    assert client.post(f"/api/tickets/{ticket['id']}/updates", headers=headers, json={
        "action": "comment", "expected_version": ticket["version"], "request_id": "read-scope-update-0001", "note": "读权限不能创建更新提案。",
    }).status_code == 401


def test_logout_revokes_only_this_login_session_and_refresh(actors, app, users):
    first, headers, tokens = actors("alice")
    with TestClient(app) as other_device:
        other = connect(other_device, users, "alice")
        response = first.post("/console/logout", data={"csrf": csrf_from(first.get("/console"))}, follow_redirects=False)
        assert response.status_code == 303
        assert first.get("/api/me", headers=headers).status_code == 401
        assert first.post("/oauth/token", headers=AUTH, data={"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]}).status_code == 400
        assert other_device.get("/api/me", headers={"Authorization": "Bearer " + other["access_token"]}).status_code == 200


def test_browser_account_change_does_not_transfer_existing_tool_identity(actors, users):
    client, headers, _ = actors("alice")
    login(client, users, "bob")
    assert client.get("/api/me", headers=headers).status_code == 401
    assert "bob" in client.get("/console/oauth/authorize", params={
        "client_id": CLIENT_ID, "redirect_uri": CALLBACK, "response_type": "code", "scope": "support:read", "state": "opaque",
    }).text


def mutate_before_read_transaction(monkeypatch, database, statement, parameters):
    original = database.connection
    state = {"triggered": False}
    class Proxy:
        def __init__(self, connection):
            self.connection = connection
        def __getattr__(self, name):
            return getattr(self.connection, name)
        def execute(self, sql, *arguments):
            if sql.strip().upper() == "BEGIN" and not state["triggered"]:
                state["triggered"] = True
                with original() as concurrent:
                    concurrent.execute(statement, parameters)
            return self.connection.execute(sql, *arguments)
    @contextmanager
    def connection():
        with original() as current:
            yield Proxy(current)
    monkeypatch.setattr(database, "connection", connection)
    return state


@pytest.mark.parametrize("operation", ["draft", "update", "knowledge"])
@pytest.mark.parametrize("change", ["revoke", "disable", "role", "organization"])
def test_identity_changes_between_tool_auth_and_transaction_are_rejected(actors, app, monkeypatch, operation, change):
    ticket = create_ticket(actors) if operation == "update" else None
    client, headers, tokens = actors("alice")
    actor = app.state.oauth.authenticate(tokens["access_token"])
    statement, parameters = {
        "revoke": ("UPDATE oauth_grants SET revoked_at=1 WHERE id=?", (actor["_oauth_grant_id"],)),
        "disable": ("UPDATE users SET active=0 WHERE id=?", ("alice",)),
        "role": ("UPDATE users SET role='admin' WHERE id=?", ("alice",)),
        "organization": ("UPDATE users SET org_id='other' WHERE id=?", ("alice",)),
    }[change]
    inject = mutate_before_read_transaction if operation == "knowledge" else mutate_before_write_lock
    mutation = inject(monkeypatch, app.state.database, statement, parameters)
    if operation == "draft":
        response = client.post("/api/drafts", headers=headers, json=payload("oauth-race-draft-0001"))
    elif operation == "update":
        response = client.post(f"/api/tickets/{ticket['id']}/updates", headers=headers, json={
            "action": "comment", "expected_version": ticket["version"], "request_id": "oauth-race-update-0001", "note": "撤销与操作竞争时不得写入。",
        })
    else:
        response = client.get("/api/knowledge/search?q=sharedmarker", headers=headers)
    assert mutation["triggered"]
    assert response.status_code in {401, 403}, response.text
    with app.state.database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM drafts WHERE request_id='oauth-race-draft-0001'").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM ticket_updates WHERE request_id='oauth-race-update-0001'").fetchone()[0] == 0


def test_reinitializing_removed_identity_revokes_existing_grants(actors, app, settings):
    client, headers, _ = actors("alice")
    config = json.loads(settings.credentials_path.read_text(encoding="utf-8"))
    config["users"] = [user for user in config["users"] if user["id"] != "alice"]
    settings.credentials_path.write_text(json.dumps(config), encoding="utf-8")
    app.state.database.initialize()
    assert client.get("/api/me", headers=headers).status_code == 401
    with app.state.database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM oauth_grants WHERE user_id='alice' AND revoked_at IS NULL").fetchone()[0] == 0


def test_https_console_canonical_redirect_and_unknown_routes(settings):
    application = create_app(replace(settings, public_base_url="https://support.example:9443", secure_cookies=True))
    with TestClient(application) as plain:
        response = plain.get("/console/not-a-page?value=opaque", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "https://support.example:9443/console/not-a-page?value=opaque"
        assert plain.get("/api/me").status_code == 401
    with TestClient(application, base_url="https://support.example:9443") as secure:
        response = secure.get("/console/not-a-page", follow_redirects=False)
        assert response.status_code == 404 and "location" not in response.headers
        assert response.headers["x-content-type-options"] == "nosniff"
        response = secure.get("/console/login")
        assert response.status_code == 200 and "Secure" in response.headers["set-cookie"]


def test_monitoring_credential_is_separate_and_metrics_do_not_expose_paths_or_queries(actors, app):
    admin, headers, _ = actors("admin")
    assert admin.get("/metrics", headers=headers).status_code == 401
    with TestClient(app) as monitor:
        authorization = {"Authorization": "Bearer " + MONITOR_TOKEN}
        assert monitor.get("/api/me", headers=authorization).status_code == 401
        marker = "sensitive-ticket-raw-value-not-a-label"
        assert monitor.get("/missing/" + marker + "?query=" + marker).status_code == 404
        response = monitor.get("/metrics", headers=authorization)
        assert response.status_code == 200
        assert marker not in response.text and MONITOR_TOKEN not in response.text
        assert 'route="unmatched"' in response.text
        assert "support_tool_requests_total" in response.text


def test_invalid_utf8_authorization_fails_without_server_error(client):
    response = client.get("/metrics", headers=[(b"authorization", b"Bearer \xff\xfe")])
    assert response.status_code == 401
    response = client.get("/api/me", headers=[(b"authorization", b"Bearer \xff\xfe")])
    assert response.status_code == 401
    response = client.post("/oauth/token", headers={"Authorization": "Basic //4="}, data={"grant_type": "refresh_token"})
    assert response.status_code == 401 and response.json()["error"] == "invalid_client"
