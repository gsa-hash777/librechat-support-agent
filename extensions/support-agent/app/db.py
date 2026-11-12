from contextlib import contextmanager
import json
import re
import sqlite3
import time

from .config import Settings
from .security import password_hash, token_hash, verify_password


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, display_name TEXT NOT NULL, org_id TEXT NOT NULL,
    password_hash TEXT NOT NULL, api_key_hash TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL DEFAULT 'requester' CHECK(role IN ('requester','support','admin')),
    active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1))
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
    csrf TEXT NOT NULL, expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS drafts (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
    request_id TEXT NOT NULL, payload_hash TEXT NOT NULL,
    title TEXT NOT NULL, description TEXT NOT NULL, priority TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','submitted','cancelled')),
    created_at REAL NOT NULL, expires_at REAL NOT NULL,
    UNIQUE(user_id, request_id)
);
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY, draft_id TEXT UNIQUE REFERENCES drafts(id),
    user_id TEXT NOT NULL REFERENCES users(id), title TEXT NOT NULL,
    description TEXT NOT NULL, priority TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('open','in_progress','resolved','closed')),
    created_at REAL NOT NULL, updated_at REAL NOT NULL,
    org_id TEXT NOT NULL, assignee_id TEXT REFERENCES users(id),
    version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1)
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT REFERENCES users(id),
    event TEXT NOT NULL, object_id TEXT, outcome TEXT NOT NULL,
    trace_id TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_docs (
    id TEXT PRIMARY KEY, org_id TEXT NOT NULL, title TEXT NOT NULL,
    content TEXT NOT NULL, source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS drafts_user ON drafts(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS audit_user ON audit_events(user_id, id DESC);
"""

LIFECYCLE_SCHEMA = """
CREATE INDEX IF NOT EXISTS tickets_user ON tickets(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS tickets_org ON tickets(org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS tickets_assignee ON tickets(assignee_id, created_at DESC);
CREATE TABLE IF NOT EXISTS ticket_updates (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
    ticket_id TEXT NOT NULL REFERENCES tickets(id), request_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL, action TEXT NOT NULL,
    note TEXT NOT NULL, assignee_id TEXT REFERENCES users(id),
    expected_version INTEGER NOT NULL CHECK(expected_version >= 1),
    status TEXT NOT NULL CHECK(status IN ('pending','applied','cancelled')),
    created_at REAL NOT NULL, expires_at REAL NOT NULL, applied_at REAL,
    UNIQUE(user_id, request_id)
);
CREATE TABLE IF NOT EXISTS ticket_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT NOT NULL REFERENCES tickets(id),
    actor_id TEXT NOT NULL REFERENCES users(id), actor_name TEXT NOT NULL,
    action TEXT NOT NULL, note TEXT NOT NULL, from_status TEXT, to_status TEXT NOT NULL,
    assignee_id TEXT REFERENCES users(id), version INTEGER NOT NULL,
    created_at REAL NOT NULL, UNIQUE(ticket_id, version)
);
CREATE INDEX IF NOT EXISTS updates_user ON ticket_updates(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS events_ticket ON ticket_events(ticket_id, version);
"""


def execute_statements(connection, script: str) -> None:
    """Unlike executescript, individual DDL statements preserve the caller's transaction."""
    for statement in script.split(";"):
        if statement.strip():
            connection.execute(statement)


def migrate(connection) -> None:
    user_columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)")}
    if "role" not in user_columns:
        connection.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'requester' CHECK(role IN ('requester','support','admin'))")
    if "active" not in user_columns:
        connection.execute("ALTER TABLE users ADD COLUMN active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1))")
    ticket_columns = {row["name"] for row in connection.execute("PRAGMA table_info(tickets)")}
    if "org_id" not in ticket_columns:
        connection.execute("""CREATE TABLE tickets_migrating (
            id TEXT PRIMARY KEY, draft_id TEXT UNIQUE REFERENCES drafts(id),
            user_id TEXT NOT NULL REFERENCES users(id), title TEXT NOT NULL,
            description TEXT NOT NULL, priority TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('open','in_progress','resolved','closed')),
            created_at REAL NOT NULL, updated_at REAL NOT NULL,
            org_id TEXT NOT NULL, assignee_id TEXT REFERENCES users(id),
            version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1))""")
        connection.execute("""INSERT INTO tickets_migrating
            (id,draft_id,user_id,title,description,priority,status,created_at,updated_at,org_id,assignee_id,version)
            SELECT t.id,t.draft_id,t.user_id,t.title,t.description,t.priority,
                   CASE WHEN t.status='pending' THEN 'open' ELSE t.status END,
                   t.created_at,t.updated_at,u.org_id,NULL,1
            FROM tickets t JOIN users u ON u.id=t.user_id""")
        old_count = connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        new_count = connection.execute("SELECT COUNT(*) FROM tickets_migrating").fetchone()[0]
        if old_count != new_count:
            raise sqlite3.IntegrityError("Cannot migrate tickets with missing owner records.")
        connection.execute("DROP TABLE tickets")
        connection.execute("ALTER TABLE tickets_migrating RENAME TO tickets")
    execute_statements(connection, LIFECYCLE_SCHEMA)
    # Imported tickets get a transparent migration baseline, not invented action history.
    connection.execute("""INSERT INTO ticket_events
        (ticket_id,actor_id,actor_name,action,note,from_status,to_status,assignee_id,version,created_at)
        SELECT t.id,t.user_id,u.display_name,'imported','迁移前已有工单；更早的流转记录未采集。',
               NULL,t.status,t.assignee_id,t.version,t.created_at
        FROM tickets t JOIN users u ON u.id=t.user_id
        WHERE NOT EXISTS(SELECT 1 FROM ticket_events e WHERE e.ticket_id=t.id)""")
    connection.execute("PRAGMA user_version=2")


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = str(settings.db_path)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        from .oauth import revoke_oauth_user_grants
        settings = self.settings
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not settings.credentials_path.is_file():
            raise RuntimeError("Missing local support credentials. Run the project bootstrap script first.")
        config = json.loads(settings.credentials_path.read_text(encoding="utf-8-sig"))
        users = config.get("users", [])
        if not users:
            raise ValueError("The support credentials file must contain a nonempty users list.")
        seen_ids, seen_keys = set(), set()
        for user in users:
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", user.get("id", "")):
                raise ValueError("Invalid support user ID.")
            if len(user.get("password", "")) < 12 or len(user.get("api_key", "")) < 32:
                raise ValueError("Support passwords need at least 12 characters; API keys need at least 32.")
            if not user.get("display_name") or not user.get("org_id"):
                raise ValueError("Each support user needs display_name and org_id.")
            if user.get("role", "requester") not in {"requester", "support", "admin"}:
                raise ValueError("Support user roles must be requester, support, or admin.")
            if not isinstance(user.get("active", True), bool):
                raise ValueError("Support user active must be a boolean.")
            if user["id"] in seen_ids or user["api_key"] in seen_keys:
                raise ValueError("Support user IDs and API keys must be unique.")
            seen_ids.add(user["id"])
            seen_keys.add(user["api_key"])
        documents = json.loads(settings.seed_path.read_text(encoding="utf-8"))
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            execute_statements(connection, SCHEMA)
            migrate(connection)
            for user in users:
                existing = connection.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
                unchanged = existing and verify_password(user["password"], existing["password_hash"])
                encoded = existing["password_hash"] if unchanged else password_hash(user["password"])
                key_digest = token_hash(user["api_key"])
                role, active = user.get("role", "requester"), int(user.get("active", True))
                if existing and (not unchanged or existing["api_key_hash"] != key_digest or existing["org_id"] != user["org_id"] or existing["role"] != role or existing["active"] != active):
                    connection.execute("DELETE FROM sessions WHERE user_id=?", (user["id"],))
                    revoke_oauth_user_grants(connection, user["id"], "identity_changed")
                connection.execute(
                    "INSERT INTO users(id,display_name,org_id,password_hash,api_key_hash,role,active) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                    "display_name=excluded.display_name, org_id=excluded.org_id, "
                    "password_hash=excluded.password_hash, api_key_hash=excluded.api_key_hash, "
                    "role=excluded.role, active=excluded.active",
                    (user["id"], user["display_name"], user["org_id"], encoded, key_digest, role, active),
                )
            # Config is the identity authority: removed accounts lose all active credentials.
            for existing in connection.execute("SELECT id FROM users").fetchall():
                if existing["id"] not in seen_ids:
                    connection.execute("DELETE FROM sessions WHERE user_id=?", (existing["id"],))
                    revoke_oauth_user_grants(connection, existing["id"], "identity_removed")
                    connection.execute("UPDATE users SET active=0 WHERE id=?", (existing["id"],))
            connection.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
            for document in documents:
                connection.execute(
                    "INSERT INTO knowledge_docs VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                    "org_id=excluded.org_id, title=excluded.title, content=excluded.content, source=excluded.source",
                    (document["id"], document.get("org_id", "demo"), document["title"], document["content"], document["source"]),
                )


def audit(connection, user_id: str | None, event: str, object_id: str | None, outcome: str, trace_id: str) -> None:
    connection.execute(
        "INSERT INTO audit_events(user_id,event,object_id,outcome,trace_id,created_at) VALUES (?,?,?,?,?,?)",
        (user_id, event, object_id, outcome, trace_id, time.time()),
    )
