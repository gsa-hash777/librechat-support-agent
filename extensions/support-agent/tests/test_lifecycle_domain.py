from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from threading import Barrier
import time

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.db import Database
from app.lifecycle import Lifecycle, LifecycleError, UpdateInput
from app.security import password_hash, token_hash


def credentials():
    return [{"id": ident, "display_name": ident.title(), "org_id": organization,
             "role": role, "password": ident + "-test-password", "api_key": ident + "-key-" + "k" * 40}
            for ident, role, organization in [
                ("owner", "requester", "demo"), ("other", "requester", "demo"),
                ("agent", "support", "demo"), ("agent2", "support", "demo"),
                ("admin", "admin", "demo"), ("foreign", "admin", "other")]]


def make_settings(tmp_path):
    config = tmp_path / "users.json"
    config.write_text(json.dumps({"users": credentials()}), encoding="utf-8")
    seed = tmp_path / "knowledge.json"
    seed.write_text("[]", encoding="utf-8")
    return Settings(db_path=tmp_path / "domain.db", credentials_path=config,
                    public_base_url="http://testserver", seed_path=seed)


@pytest.fixture
def domain(tmp_path):
    settings = make_settings(tmp_path)
    database = Database(settings)
    database.initialize()
    with database.connection() as connection:
        connection.execute("""INSERT INTO tickets
            (id,user_id,title,description,priority,status,created_at,updated_at,org_id,version)
            VALUES('T-test','owner','Upload failed','A reproducible PDF upload failure','normal','open',?,?, 'demo',1)""",
            (time.time(), time.time()))
        connection.execute("""INSERT INTO ticket_events
            (ticket_id,actor_id,actor_name,action,note,from_status,to_status,version,created_at)
            VALUES('T-test','owner','Owner','created','Initial ticket',NULL,'open',1,?)""", (time.time(),))
    return database, Lifecycle(database)


def actor(identity):
    # A caller-supplied role/org is deliberately not authoritative.
    return {"id": identity, "role": "admin", "org_id": "attacker"}


def propose(lifecycle, identity, action, version, request_id=None, **kwargs):
    return lifecycle.propose(actor(identity), "T-test", UpdateInput(
        action=action, expected_version=version, note="Test change with meaningful detail",
        request_id=request_id or f"request-{action}-{version}", **kwargs), "trace-propose")


def apply(lifecycle, identity, action, version, **kwargs):
    update = propose(lifecycle, identity, action, version, **kwargs)
    return lifecycle.confirm(actor(identity), update["id"], "trace-confirm")


def test_complete_lifecycle_history_and_owner_close(domain):
    _, lifecycle = domain
    apply(lifecycle, "admin", "assign", 1, assignee_id="agent")
    apply(lifecycle, "agent", "start", 2)
    apply(lifecycle, "owner", "comment", 3)
    apply(lifecycle, "agent", "resolve", 4)
    with pytest.raises(LifecycleError) as denied:
        propose(lifecycle, "admin", "close", 5)
    assert denied.value.status == 403
    apply(lifecycle, "owner", "close", 5)
    ticket = lifecycle.get_ticket(actor("owner"), "T-test", "read")
    assert (ticket["status"], ticket["version"], ticket["allowed_actions"]) == ("closed", 6, [])
    history = lifecycle.history(actor("owner"), "T-test", "history")
    assert [event["action"] for event in history] == ["created", "assign", "start", "comment", "resolve", "close"]
    assert [event["version"] for event in history] == list(range(1, 7))


def test_scope_and_forged_role_are_not_trusted(domain):
    _, lifecycle = domain
    for identity in ["other", "agent", "foreign"]:
        with pytest.raises(LifecycleError) as denied:
            lifecycle.get_ticket(actor(identity), "T-test", "denied-read")
        assert denied.value.status == 404
        assert lifecycle.list_tickets(actor(identity), None, 20, "list") == []
    with pytest.raises(LifecycleError) as denied:
        propose(lifecycle, "owner", "assign", 1, assignee_id="agent")
    assert denied.value.status == 403
    assert lifecycle.get_ticket(actor("admin"), "T-test", "admin-read")["allowed_actions"] == ["assign", "start", "comment"]


@pytest.mark.parametrize("target", ["owner", "foreign", "missing"])
def test_assign_targets_are_constrained(domain, target):
    database, lifecycle = domain
    with pytest.raises(LifecycleError) as denied:
        propose(lifecycle, "admin", "assign", 1, assignee_id=target)
    assert denied.value.status == 403
    with database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM ticket_updates").fetchone()[0] == 0
        assert connection.execute("SELECT outcome FROM audit_events ORDER BY id DESC").fetchone()[0] == "denied"


def test_proposal_idempotency_hash_and_version_conflicts(domain):
    database, lifecycle = domain
    first = propose(lifecycle, "owner", "comment", 1, request_id="same-request-key")
    assert first["replayed"] is False
    repeated = propose(lifecycle, "owner", "comment", 1, request_id="same-request-key")
    assert repeated["id"] == first["id"] and repeated["replayed"] is True
    changed = UpdateInput(action="comment", request_id="same-request-key", expected_version=1, note="Different note")
    with pytest.raises(LifecycleError) as conflict:
        lifecycle.propose(actor("owner"), "T-test", changed, "hash-conflict")
    assert conflict.value.status == 409
    assert lifecycle.get_ticket(actor("owner"), "T-test", "read")["version"] == 1
    with database.connection() as connection:
        assert connection.execute("SELECT outcome FROM audit_events WHERE trace_id='hash-conflict'").fetchone()[0] == "conflict"


def test_same_proposal_concurrent_confirm_only_changes_once(domain):
    database, lifecycle = domain
    proposal = propose(lifecycle, "owner", "comment", 1)
    barrier = Barrier(2)
    def confirm():
        barrier.wait()
        return lifecycle.confirm(actor("owner"), proposal["id"], "parallel-confirm")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: confirm(), range(2)))
    assert sorted(result["replayed"] for result in results) == [False, True]
    with database.connection() as connection:
        assert connection.execute("SELECT version FROM tickets WHERE id='T-test'").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM ticket_events WHERE action='comment'").fetchone()[0] == 1


def test_competing_proposals_do_not_overwrite_newer_ticket(domain):
    database, lifecycle = domain
    first = propose(lifecycle, "owner", "comment", 1, request_id="first-proposal")
    second = propose(lifecycle, "owner", "comment", 1, request_id="second-proposal")
    lifecycle.confirm(actor("owner"), first["id"], "first-confirm")
    with pytest.raises(LifecycleError) as conflict:
        lifecycle.confirm(actor("owner"), second["id"], "version-conflict")
    assert conflict.value.status == 409
    with database.connection() as connection:
        assert connection.execute("SELECT status FROM ticket_updates WHERE id=?", (second["id"],)).fetchone()[0] == "pending"
        assert connection.execute("SELECT outcome FROM audit_events WHERE trace_id='version-conflict'").fetchone()[0] == "conflict"


@pytest.mark.parametrize("mutation", ["UPDATE users SET role='requester' WHERE id='admin'",
                                      "UPDATE users SET active=0 WHERE id='admin'",
                                      "UPDATE users SET active=0 WHERE id='agent'",
                                      "UPDATE users SET org_id='other' WHERE id='agent'"])
def test_confirmation_reloads_actor_and_target(domain, mutation):
    database, lifecycle = domain
    proposal = propose(lifecycle, "admin", "assign", 1, assignee_id="agent")
    with database.connection() as connection:
        connection.execute(mutation)
    with pytest.raises(LifecycleError) as denied:
        lifecycle.confirm(actor("admin"), proposal["id"], "fresh-permissions")
    assert denied.value.status in {403, 404}
    with database.connection() as connection:
        assert connection.execute("SELECT assignee_id,version FROM tickets WHERE id='T-test'").fetchone()[:] == (None, 1)
        assert connection.execute("SELECT outcome FROM audit_events WHERE trace_id='fresh-permissions'").fetchone()[0] == "denied"


def test_creator_only_cancel_and_expiry(domain):
    database, lifecycle = domain
    first = propose(lifecycle, "owner", "comment", 1)
    with pytest.raises(LifecycleError) as denied:
        lifecycle.confirm(actor("admin"), first["id"], "wrong-owner")
    assert denied.value.status == 404
    lifecycle.cancel(actor("owner"), first["id"], "cancel")
    with pytest.raises(LifecycleError) as cancelled:
        lifecycle.confirm(actor("owner"), first["id"], "cancelled")
    assert cancelled.value.status == 409
    expired = propose(lifecycle, "owner", "comment", 1, request_id="expired-proposal")
    with database.connection() as connection:
        connection.execute("UPDATE ticket_updates SET expires_at=0 WHERE id=?", (expired["id"],))
    assert lifecycle.get_update(actor("owner"), expired["id"], "view")["status"] == "expired"
    with pytest.raises(LifecycleError) as timeout:
        lifecycle.confirm(actor("owner"), expired["id"], "expired")
    assert timeout.value.status == 410


def test_staff_active_and_scope_filter(domain):
    database, lifecycle = domain
    assert [person["id"] for person in lifecycle.staff(actor("admin"))] == ["agent", "agent2"]
    with database.connection() as connection:
        connection.execute("UPDATE users SET active=0 WHERE id='agent2'")
    assert [person["id"] for person in lifecycle.staff(actor("admin"))] == ["agent"]
    with pytest.raises(LifecycleError) as denied:
        lifecycle.staff(actor("owner"))
    assert denied.value.status == 403


def test_key_rotated_after_authentication_rejects_old_snapshot(domain):
    database, lifecycle = domain
    with database.connection() as connection:
        snapshot = dict(connection.execute("SELECT * FROM users WHERE id='owner'").fetchone())
        connection.execute("UPDATE users SET api_key_hash=? WHERE id='owner'", (token_hash("rotated-key"),))
    with pytest.raises(LifecycleError) as denied:
        lifecycle.propose(snapshot, "T-test", UpdateInput(action="comment", request_id="stale-key-request", expected_version=1, note="Should not be applied"), "stale-key")
    assert denied.value.status == 401
    with database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM ticket_updates").fetchone()[0] == 0
        assert connection.execute("SELECT outcome FROM audit_events WHERE trace_id='stale-key'").fetchone()[0] == "denied"


@pytest.mark.parametrize("session_state", ["revoked", "expired", "wrong-user"])
def test_session_revoked_after_authentication_cannot_confirm(domain, session_state):
    database, lifecycle = domain
    proposal = propose(lifecycle, "owner", "comment", 1)
    with database.connection() as connection:
        snapshot = dict(connection.execute("SELECT * FROM users WHERE id='owner'").fetchone())
        snapshot["session_hash"] = "original-session"
        if session_state != "revoked":
            connection.execute("INSERT INTO sessions VALUES(?,?,?,?)", ("original-session", "other" if session_state == "wrong-user" else "owner", "csrf", 0 if session_state == "expired" else time.time() + 1000))
    with pytest.raises(LifecycleError) as denied:
        lifecycle.confirm(snapshot, proposal["id"], "stale-session")
    assert denied.value.status == 401
    with database.connection() as connection:
        assert connection.execute("SELECT version FROM tickets WHERE id='T-test'").fetchone()[0] == 1
        assert connection.execute("SELECT status FROM ticket_updates WHERE id=?", (proposal["id"],)).fetchone()[0] == "pending"
        assert connection.execute("SELECT outcome FROM audit_events WHERE trace_id='stale-session'").fetchone()[0] == "denied"


@pytest.mark.parametrize("changes", [{"approved": True}, {"user_id": "admin"}, {"note": " "},
                                     {"expected_version": 0}, {"expected_version": True},
                                     {"request_id": "bad key"}, {"assignee_id": "agent"}])
def test_update_input_rejects_unsafe_or_invalid_fields(changes):
    payload = {"action": "comment", "request_id": "valid-request-id", "expected_version": 1, "note": "Valid note"}
    with pytest.raises(ValidationError):
        UpdateInput(**{**payload, **changes})


LEGACY = """
CREATE TABLE users(id TEXT PRIMARY KEY,display_name TEXT NOT NULL,org_id TEXT NOT NULL,password_hash TEXT NOT NULL,api_key_hash TEXT NOT NULL UNIQUE);
CREATE TABLE sessions(token_hash TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id),csrf TEXT NOT NULL,expires_at REAL NOT NULL);
CREATE TABLE drafts(id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id),request_id TEXT NOT NULL,payload_hash TEXT NOT NULL,title TEXT NOT NULL,description TEXT NOT NULL,priority TEXT NOT NULL,status TEXT NOT NULL CHECK(status IN ('pending','submitted','cancelled')),created_at REAL NOT NULL,expires_at REAL NOT NULL,UNIQUE(user_id,request_id));
CREATE TABLE tickets(id TEXT PRIMARY KEY,draft_id TEXT UNIQUE REFERENCES drafts(id),user_id TEXT NOT NULL REFERENCES users(id),title TEXT NOT NULL,description TEXT NOT NULL,priority TEXT NOT NULL,status TEXT NOT NULL CHECK(status IN ('open','pending','resolved')),created_at REAL NOT NULL,updated_at REAL NOT NULL);
CREATE TABLE audit_events(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT REFERENCES users(id),event TEXT NOT NULL,object_id TEXT,outcome TEXT NOT NULL,trace_id TEXT NOT NULL,created_at REAL NOT NULL);
"""


def seed_legacy(settings):
    owner = credentials()[0]
    encoded = password_hash(owner["password"])
    with sqlite3.connect(settings.db_path) as connection:
        connection.executescript(LEGACY)
        connection.execute("INSERT INTO users VALUES(?,?,?,?,?)", (owner["id"], owner["display_name"], owner["org_id"], encoded, token_hash(owner["api_key"])))
        connection.execute("INSERT INTO sessions VALUES('session','owner','csrf',?)", (time.time() + 3600,))
        connection.execute("INSERT INTO drafts VALUES('D-old','owner','old-request','digest','Old title','Old description','normal','submitted',1,2)")
        connection.execute("INSERT INTO tickets VALUES('T-old','D-old','owner','Old title','Old description','normal','pending',1,2)")
        connection.execute("INSERT INTO audit_events(user_id,event,outcome,trace_id,created_at) VALUES('owner','old-event','ok','old-trace',1)")
    return encoded


def test_legacy_migration_preserves_records_passwords_and_sessions(tmp_path):
    settings = make_settings(tmp_path)
    original_hash = seed_legacy(settings)
    database = Database(settings)
    database.initialize()
    database.initialize()
    with database.connection() as connection:
        owner = connection.execute("SELECT * FROM users WHERE id='owner'").fetchone()
        assert owner["password_hash"] == original_hash
        assert (owner["role"], owner["active"]) == ("requester", 1)
        assert connection.execute("SELECT COUNT(*) FROM sessions WHERE token_hash='session'").fetchone()[0] == 1
        ticket = connection.execute("SELECT * FROM tickets WHERE id='T-old'").fetchone()
        assert (ticket["draft_id"], ticket["status"], ticket["org_id"], ticket["version"]) == ("D-old", "open", "demo", 1)
        assert connection.execute("SELECT COUNT(*) FROM audit_events WHERE trace_id='old-trace'").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM drafts WHERE id='D-old'").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM ticket_events WHERE ticket_id='T-old'").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_migration_failure_rolls_back_schema_and_data(tmp_path):
    settings = make_settings(tmp_path)
    original_hash = seed_legacy(settings)
    settings.seed_path.write_text('[{"id":"malformed-document"}]', encoding="utf-8")
    with pytest.raises(KeyError):
        Database(settings).initialize()
    with sqlite3.connect(settings.db_path) as connection:
        assert "org_id" not in [row[1] for row in connection.execute("PRAGMA table_info(tickets)")]
        assert "role" not in [row[1] for row in connection.execute("PRAGMA table_info(users)")]
        assert connection.execute("SELECT status FROM tickets WHERE id='T-old'").fetchone()[0] == "pending"
        assert connection.execute("SELECT password_hash FROM users WHERE id='owner'").fetchone()[0] == original_hash
        assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_migration_uses_original_organization_snapshot(tmp_path):
    settings = make_settings(tmp_path)
    seed_legacy(settings)
    users = credentials()
    users[0]["org_id"] = "new-organization"
    settings.credentials_path.write_text(json.dumps({"users": users}), encoding="utf-8")
    database = Database(settings)
    database.initialize()
    with database.connection() as connection:
        assert connection.execute("SELECT org_id FROM tickets WHERE id='T-old'").fetchone()[0] == "demo"
        assert connection.execute("SELECT org_id FROM users WHERE id='owner'").fetchone()[0] == "new-organization"
        assert connection.execute("SELECT COUNT(*) FROM sessions WHERE user_id='owner'").fetchone()[0] == 0


def test_role_change_revokes_session_and_removed_user_keeps_hash(tmp_path):
    settings = make_settings(tmp_path)
    original_hash = seed_legacy(settings)
    database = Database(settings)
    database.initialize()
    users = credentials()
    users[0]["role"] = "support"
    settings.credentials_path.write_text(json.dumps({"users": users}), encoding="utf-8")
    database.initialize()
    with database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM sessions WHERE user_id='owner'").fetchone()[0] == 0
        assert connection.execute("SELECT password_hash FROM users WHERE id='owner'").fetchone()[0] == original_hash
    settings.credentials_path.write_text(json.dumps({"users": users[1:]}), encoding="utf-8")
    database.initialize()
    with database.connection() as connection:
        user = connection.execute("SELECT * FROM users WHERE id='owner'").fetchone()
        assert user["active"] == 0 and user["password_hash"] == original_hash
