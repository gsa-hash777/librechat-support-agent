"""Local OAuth protocol and isolation tests; no model or paid API requests."""
from concurrent.futures import ThreadPoolExecutor
import base64
from html.parser import HTMLParser
import secrets
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient

from app.db import Database
from app.oauth import OAuthConfig, register_oauth, revalidate_oauth_actor, revoke_oauth_user_grants, revoke_oauth_session_grants
from app.security import token_hash


CLIENT_ID = "local-librechat-test"
CLIENT_SECRET = "test-client-secret-" + "s" * 32
CALLBACK = "http://localhost:3080/api/actions/action-test/oauth/callback"
AUTH = {"Authorization": "Basic " + base64.b64encode((CLIENT_ID + ":" + CLIENT_SECRET).encode()).decode()}


class Inputs(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.values = {}
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and "name" in attrs:
            self.values[attrs["name"]] = attrs.get("value", "")


@pytest.fixture
def oauth_app(settings):
    database = Database(settings)
    database.initialize()
    app = FastAPI()

    def browser_session(request):
        value = request.cookies.get("support_session")
        if not value:
            return None
        with database.connection() as connection:
            row = connection.execute("""SELECT u.*,s.csrf,s.token_hash AS session_hash FROM sessions s
                JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires_at>? AND u.active=1""",
                                     (token_hash(value), time.time())).fetchone()
        return dict(row) if row else None

    def require_session(request):
        session = browser_session(request)
        if not session:
            raise HTTPException(401, "Login required")
        return session

    def check_csrf(request, session, csrf):
        if csrf != session["csrf"] or request.headers.get("origin", "http://testserver") != "http://testserver":
            raise HTTPException(403, "CSRF invalid")

    service = register_oauth(app, database, OAuthConfig(CLIENT_ID, CLIENT_SECRET, (CALLBACK,)),
                            browser_session=browser_session, require_session=require_session,
                            check_csrf=check_csrf, page=lambda title, body, user=None: HTMLResponse(body))
    app.state.database = database
    app.state.oauth = service
    return app


@pytest.fixture
def oauth_client(oauth_app):
    with TestClient(oauth_app) as client:
        yield client


def sign_in(client, app, username="alice"):
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    with app.state.database.connection() as connection:
        connection.execute("INSERT INTO sessions VALUES(?,?,?,?)", (token_hash(token), username, csrf, time.time() + 3600))
    client.cookies.set("support_session", token, domain="testserver.local", path="/console")
    return token, csrf


def authorization_url(**changes):
    values = {"client_id": CLIENT_ID, "redirect_uri": CALLBACK, "response_type": "code",
              "scope": "support:read support:propose", "state": "opaque-state-not-a-trusted-user"}
    values.update(changes)
    return "/console/oauth/authorize?" + urlencode(values)


def consent(client, app, username="alice", **changes):
    sign_in(client, app, username)
    page = client.get(authorization_url(**changes))
    assert page.status_code == 200, page.text
    values = Inputs(page.text).values
    response = client.post("/console/oauth/authorize", data={**values, "decision": "approve"}, follow_redirects=False)
    assert response.status_code == 303, response.text
    return parse_qs(urlsplit(response.headers["location"]).query)["code"][0]


def exchange(client, code, **changes):
    data = {"grant_type": "authorization_code", "code": code, "redirect_uri": CALLBACK}
    data.update(changes)
    return client.post("/oauth/token", data=data, headers=AUTH)


def issue(client, app, username="alice", **changes):
    response = exchange(client, consent(client, app, username, **changes))
    assert response.status_code == 200, response.text
    return response.json()


def refresh(client, token, **changes):
    return client.post("/oauth/token", data={"grant_type": "refresh_token", "refresh_token": token, **changes}, headers=AUTH)


def test_complete_flow_binds_authenticated_business_subject_and_keeps_secrets_hashed(oauth_client, oauth_app):
    code = consent(oauth_client, oauth_app, state='{"user":"bob","role":"admin"}')
    response = exchange(oauth_client, code)
    assert response.status_code == 200
    tokens = response.json()
    actor = oauth_app.state.oauth.authenticate(tokens["access_token"])
    assert actor["id"] == "alice" and actor["role"] == "requester"
    assert response.headers["cache-control"] == "no-store"
    assert tokens["token_type"] == "Bearer"
    with oauth_app.state.database.connection() as connection:
        stored = " ".join(str(tuple(row)) for table in ("oauth_codes", "oauth_tokens", "audit_events")
                          for row in connection.execute("SELECT * FROM " + table))
        assert code not in stored and tokens["access_token"] not in stored and tokens["refresh_token"] not in stored
        assert token_hash(code) in stored and token_hash(tokens["access_token"]) in stored
        assert revalidate_oauth_actor(connection, actor)["id"] == "alice"


@pytest.mark.parametrize("changes", [
    {"redirect_uri": CALLBACK + "/evil"}, {"redirect_uri": "https://evil.example/callback"},
    {"client_id": "another-client"}, {"scope": "support:admin"}, {"response_type": "token"}, {"state": ""},
])
def test_authorization_rejects_untrusted_redirect_client_scopes_and_response(oauth_client, changes):
    response = oauth_client.get(authorization_url(**changes), follow_redirects=False)
    assert response.status_code == 400 and "location" not in response.headers


def test_unauthenticated_authorization_returns_only_to_local_login(oauth_client):
    response = oauth_client.get(authorization_url(), follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/console/login?next=")
    assert oauth_client.get(authorization_url() + "&client_id=duplicate").status_code == 400


def test_consent_displays_identity_requires_post_csrf_and_does_not_issue_on_get(oauth_client, oauth_app):
    sign_in(oauth_client, oauth_app)
    response = oauth_client.get(authorization_url())
    assert "alice" in response.text and "不是自动单点登录" in response.text
    form = Inputs(response.text).values
    with oauth_app.state.database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM oauth_codes").fetchone()[0] == 0
    rejected = oauth_client.post("/console/oauth/authorize", data={**form, "csrf": "wrong", "decision": "approve"})
    assert rejected.status_code == 403
    assert oauth_client.post("/console/oauth/authorize", data={**form, "decision": "approve"}, headers={"Origin": "https://evil.example"}).status_code == 403


def test_consent_form_cannot_be_reused_after_account_switch(oauth_client, oauth_app):
    sign_in(oauth_client, oauth_app, "alice")
    form = Inputs(oauth_client.get(authorization_url()).text).values
    _, bob_csrf = sign_in(oauth_client, oauth_app, "bob")
    response = oauth_client.post("/console/oauth/authorize", data={**form, "csrf": bob_csrf, "decision": "approve"})
    assert response.status_code == 403
    with oauth_app.state.database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM oauth_codes").fetchone()[0] == 0


def test_switch_action_revokes_grants_and_drops_session(oauth_client, oauth_app):
    other_device = issue(oauth_client, oauth_app)
    tokens = issue(oauth_client, oauth_app)
    form = Inputs(oauth_client.get(authorization_url()).text).values
    response = oauth_client.post("/console/oauth/authorize", data={**form, "decision": "switch"}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"].startswith("/console/login?")
    assert oauth_app.state.oauth.authenticate(tokens["access_token"]) is None
    assert refresh(oauth_client, tokens["refresh_token"]).status_code == 400
    assert oauth_app.state.oauth.authenticate(other_device["access_token"])["id"] == "alice"


def test_session_logout_does_not_revoke_other_devices(oauth_client, oauth_app):
    first = issue(oauth_client, oauth_app)
    second = issue(oauth_client, oauth_app)
    actor = oauth_app.state.oauth.authenticate(first["access_token"])
    with oauth_app.state.database.connection() as connection:
        grant = connection.execute("SELECT session_hash FROM oauth_grants WHERE id=?", (actor["_oauth_grant_id"],)).fetchone()
        revoke_oauth_session_grants(connection, grant["session_hash"])
    assert oauth_app.state.oauth.authenticate(first["access_token"]) is None
    assert oauth_app.state.oauth.authenticate(second["access_token"])["id"] == "alice"


def test_consent_is_one_time_and_deny_echoes_only_opaque_state(oauth_client, oauth_app):
    sign_in(oauth_client, oauth_app)
    form = Inputs(oauth_client.get(authorization_url(state="fake-subject-bob")).text).values
    response = oauth_client.post("/console/oauth/authorize", data={**form, "decision": "deny"}, follow_redirects=False)
    assert parse_qs(urlsplit(response.headers["location"]).query) == {"error": ["access_denied"], "state": ["fake-subject-bob"]}
    assert oauth_client.post("/console/oauth/authorize", data={**form, "decision": "approve"}).status_code == 403


def test_client_auth_and_exact_redirect_are_required_before_consuming_code(oauth_client, oauth_app):
    code = consent(oauth_client, oauth_app)
    data = {"grant_type": "authorization_code", "code": code, "redirect_uri": CALLBACK}
    assert oauth_client.post("/oauth/token", data=data).status_code == 401
    assert exchange(oauth_client, code, redirect_uri=CALLBACK + "/bad").status_code == 400
    assert exchange(oauth_client, code).status_code == 200


def test_used_authorization_code_revokes_its_grant(oauth_client, oauth_app):
    code = consent(oauth_client, oauth_app)
    tokens = exchange(oauth_client, code).json()
    assert exchange(oauth_client, code).json() == {"error": "invalid_grant"}
    assert oauth_app.state.oauth.authenticate(tokens["access_token"]) is None


def test_expired_code_is_rejected(oauth_client, oauth_app):
    code = consent(oauth_client, oauth_app)
    with oauth_app.state.database.connection() as connection:
        connection.execute("UPDATE oauth_codes SET expires_at=0")
    assert exchange(oauth_client, code).status_code == 400


def test_refresh_rotation_retires_old_access_and_reuse_revokes_entire_family(oauth_client, oauth_app):
    first = issue(oauth_client, oauth_app)
    response = refresh(oauth_client, first["refresh_token"])
    assert response.status_code == 200
    second = response.json()
    assert second["refresh_token"] != first["refresh_token"]
    assert oauth_app.state.oauth.authenticate(first["access_token"]) is None
    assert oauth_app.state.oauth.authenticate(second["access_token"])["id"] == "alice"
    assert refresh(oauth_client, first["refresh_token"]).status_code == 400
    assert oauth_app.state.oauth.authenticate(second["access_token"]) is None
    assert refresh(oauth_client, second["refresh_token"]).status_code == 400


def test_refresh_cannot_escalate_scope(oauth_client, oauth_app):
    tokens = issue(oauth_client, oauth_app, scope="support:read")
    assert oauth_app.state.oauth.authenticate(tokens["access_token"], "support:propose") is None
    assert refresh(oauth_client, tokens["refresh_token"], scope="support:read support:propose").status_code == 400
    assert refresh(oauth_client, tokens["refresh_token"]).status_code == 200


@pytest.mark.parametrize("statement", [
    "UPDATE users SET active=0 WHERE id='alice'",
    "UPDATE users SET role='admin' WHERE id='alice'",
    "UPDATE users SET org_id='other' WHERE id='alice'",
    "UPDATE users SET password_hash='rotated' WHERE id='alice'",
])
def test_user_changes_fail_closed_for_access_refresh_and_write_recheck(oauth_client, oauth_app, statement):
    tokens = issue(oauth_client, oauth_app)
    actor = oauth_app.state.oauth.authenticate(tokens["access_token"])
    with oauth_app.state.database.connection() as connection:
        connection.execute(statement)
    assert oauth_app.state.oauth.authenticate(tokens["access_token"]) is None
    assert refresh(oauth_client, tokens["refresh_token"]).status_code == 400
    with oauth_app.state.database.connection() as connection:
        assert revalidate_oauth_actor(connection, actor) is None


def test_two_business_users_have_independent_principals_and_revocation(oauth_client, oauth_app):
    alice = issue(oauth_client, oauth_app)
    bob = issue(oauth_client, oauth_app, "bob")
    assert oauth_app.state.oauth.authenticate(alice["access_token"])["id"] == "alice"
    assert oauth_app.state.oauth.authenticate(bob["access_token"])["id"] == "bob"
    response = oauth_client.post("/oauth/revoke", data={"token": alice["access_token"]}, headers=AUTH)
    assert response.status_code == 200
    assert oauth_app.state.oauth.authenticate(alice["access_token"]) is None
    assert oauth_app.state.oauth.authenticate(bob["access_token"])["id"] == "bob"
    assert refresh(oauth_client, alice["refresh_token"]).status_code == 400
    assert oauth_client.post("/oauth/revoke", data={"token": "unknown-token"}, headers=AUTH).json() == {}


def test_connections_are_owner_only_and_revoke_with_csrf(oauth_client, oauth_app):
    alice = issue(oauth_client, oauth_app)
    grant = oauth_app.state.oauth.authenticate(alice["access_token"])["_oauth_grant_id"]
    _, csrf = sign_in(oauth_client, oauth_app, "bob")
    assert grant not in oauth_client.get("/console/oauth/connections").text
    assert oauth_client.post("/console/oauth/connections", data={"grant_id": grant, "csrf": csrf}).status_code == 404
    _, csrf = sign_in(oauth_client, oauth_app, "alice")
    assert grant in oauth_client.get("/console/oauth/connections").text
    response = oauth_client.post("/console/oauth/connections", data={"grant_id": grant, "csrf": csrf}, follow_redirects=False)
    assert response.status_code == 303
    assert oauth_app.state.oauth.authenticate(alice["access_token"]) is None


def test_logout_helper_revokes_grants_and_revalidation_checks_subject(oauth_client, oauth_app):
    tokens = issue(oauth_client, oauth_app)
    actor = oauth_app.state.oauth.authenticate(tokens["access_token"])
    with oauth_app.state.database.connection() as connection:
        assert revalidate_oauth_actor(connection, {**actor, "id": "bob"}) is None
        revoke_oauth_user_grants(connection, "alice")
        assert revalidate_oauth_actor(connection, actor) is None
    assert oauth_app.state.oauth.authenticate(tokens["access_token"]) is None


def test_expired_access_and_refresh_are_rejected(oauth_client, oauth_app):
    tokens = issue(oauth_client, oauth_app)
    with oauth_app.state.database.connection() as connection:
        connection.execute("UPDATE oauth_tokens SET expires_at=0")
    assert oauth_app.state.oauth.authenticate(tokens["access_token"]) is None
    assert refresh(oauth_client, tokens["refresh_token"]).status_code == 400
    assert oauth_app.state.oauth.authenticate("legacy-api-key-value") is None


def test_concurrent_refresh_is_atomic_and_replay_revocation_is_committed(oauth_client, oauth_app):
    tokens = issue(oauth_client, oauth_app)
    def run():
        with TestClient(oauth_app) as parallel:
            return refresh(parallel, tokens["refresh_token"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert sorted(response.status_code for response in results) == [200, 400]
    successful = next(response.json() for response in results if response.status_code == 200)
    assert oauth_app.state.oauth.authenticate(successful["access_token"]) is None


def test_oauth_endpoints_are_not_advertised_as_agent_tools(oauth_client):
    assert oauth_client.get("/openapi.json").json()["paths"] == {}


def test_empty_callbacks_are_safe_bootstrap_state():
    OAuthConfig(CLIENT_ID, CLIENT_SECRET, ()).validate()
    with pytest.raises(ValueError):
        OAuthConfig(CLIENT_ID, CLIENT_SECRET, ("https://example.com/api/actions/a/oauth/callback?redirect=evil",)).validate()
