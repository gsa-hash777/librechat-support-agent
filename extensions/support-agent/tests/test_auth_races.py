"""Authentication must still be current at the business write boundary."""
from contextlib import contextmanager

import pytest

import app.main as main
from app.security import password_hash, token_hash
from test_service import csrf_from, draft, login, payload


def mutate_before_write_lock(monkeypatch, database, statement, parameters=()):
    """Run a committed revocation after request auth, before its write lock."""
    original = database.connection
    state = {"triggered": False}

    class ConnectionProxy:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, sql, *arguments):
            if sql.strip().upper() == "BEGIN IMMEDIATE" and not state["triggered"]:
                state["triggered"] = True
                with original() as concurrent:
                    concurrent.execute(statement, parameters)
            return self.connection.execute(sql, *arguments)

    @contextmanager
    def connection():
        with original() as active:
            yield ConnectionProxy(active)

    monkeypatch.setattr(database, "connection", connection)
    return state


@pytest.mark.parametrize("operation", ["confirm", "cancel"])
@pytest.mark.parametrize("revocation", ["session_deleted", "session_expired", "user_disabled"])
def test_draft_write_rechecks_live_session_and_account(client, app, users, headers, monkeypatch, operation, revocation):
    created = draft(client, headers)
    login(client, users)
    path = "/console/drafts/" + created["id"]
    csrf = csrf_from(client.get(path))
    statement = {
        "session_deleted": "DELETE FROM sessions WHERE user_id='alice'",
        "session_expired": "UPDATE sessions SET expires_at=0 WHERE user_id='alice'",
        "user_disabled": "UPDATE users SET active=0 WHERE id='alice'",
    }[revocation]
    mutation = mutate_before_write_lock(monkeypatch, app.state.database, statement)

    response = client.post(path + "/" + operation, data={"csrf": csrf}, follow_redirects=False)

    assert mutation["triggered"], "Test must reach the authenticated business write boundary."
    assert response.status_code in {401, 403}, response.text
    with app.state.database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0
        assert connection.execute("SELECT status FROM drafts WHERE id=?", (created["id"],)).fetchone()[0] == "pending"
        event = connection.execute(
            "SELECT outcome FROM audit_events WHERE event=? ORDER BY id DESC LIMIT 1",
            ("draft." + operation,),
        ).fetchone()
        assert event is not None and event["outcome"] == "denied"


@pytest.mark.parametrize("revocation", ["user_disabled", "password_rotated"])
def test_login_cannot_issue_session_after_verified_credentials_change(client, app, users, monkeypatch, revocation):
    user = next(item for item in users if item["id"] == "alice")
    csrf = csrf_from(client.get("/console/login"))
    original = main.verify_password
    state = {"triggered": False}
    replacement_hash = password_hash("new-password-after-rotation")

    def verify_then_revoke(plaintext, encoded):
        result = original(plaintext, encoded)
        if result and not state["triggered"]:
            state["triggered"] = True
            with app.state.database.connection() as connection:
                if revocation == "user_disabled":
                    connection.execute("UPDATE users SET active=0 WHERE id=?", (user["id"],))
                else:
                    connection.execute("UPDATE users SET password_hash=? WHERE id=?", (replacement_hash, user["id"]))
        return result

    monkeypatch.setattr(main, "verify_password", verify_then_revoke)
    response = client.post("/console/login", data={
        "username": user["id"], "password": user["password"], "csrf": csrf, "next": "/console",
    }, follow_redirects=False)

    assert state["triggered"], "Test must revoke after the original password was verified."
    assert response.status_code in {401, 403}, response.text
    assert "support_session=" not in response.headers.get("set-cookie", "")
    with app.state.database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM sessions WHERE user_id=?", (user["id"],)).fetchone()[0] == 0


@pytest.mark.parametrize("revocation", ["user_disabled", "key_rotated"])
def test_draft_creation_rechecks_tool_key_and_account_before_writing(client, app, headers, monkeypatch, revocation):
    if revocation == "user_disabled":
        statement, parameters = "UPDATE users SET active=0 WHERE id='alice'", ()
    else:
        statement, parameters = "UPDATE users SET api_key_hash=? WHERE id='alice'", (token_hash("new-key-after-rotation"),)
    mutation = mutate_before_write_lock(monkeypatch, app.state.database, statement, parameters)

    response = client.post("/api/drafts", headers=headers["alice"], json=payload())

    assert mutation["triggered"], "Test must revoke after the original tool key was accepted."
    assert response.status_code in {401, 403}, response.text
    with app.state.database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 0
        event = connection.execute("SELECT outcome FROM audit_events WHERE event='draft.create' ORDER BY id DESC LIMIT 1").fetchone()
        assert event is not None and event["outcome"] == "denied"
