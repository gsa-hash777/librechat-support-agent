"""Confidential-client OAuth for explicitly connecting a business account.

LibreChat's state is opaque: it is echoed, never decoded as proof of identity.
The immutable subject here is users.id established by the existing browser login.
Authorization grants cannot approve tickets or replace their human confirmation.
"""
import base64
import binascii
from dataclasses import dataclass
from html import escape
import hmac
import secrets
import time
from urllib.parse import urlencode, urlsplit

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from .db import audit, execute_statements
from .security import token_hash


ACCESS_PREFIX = "sa_at_"
REFRESH_PREFIX = "sa_rt_"
SCOPES = {"support:read": "读取本人获准访问的知识与工单", "support:propose": "创建待人工确认的工单草稿和变更提案"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS oauth_grants (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
    client_id TEXT NOT NULL, scopes TEXT NOT NULL, identity_version TEXT NOT NULL,
    session_hash TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL,
    revoked_at REAL, revocation_reason TEXT
);
CREATE TABLE IF NOT EXISTS oauth_requests (
    request_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
    session_hash TEXT NOT NULL, client_id TEXT NOT NULL, redirect_uri TEXT NOT NULL,
    state TEXT NOT NULL, scopes TEXT NOT NULL, expires_at REAL NOT NULL, consumed_at REAL
);
CREATE TABLE IF NOT EXISTS oauth_codes (
    code_hash TEXT PRIMARY KEY, grant_id TEXT NOT NULL REFERENCES oauth_grants(id) ON DELETE CASCADE,
    client_id TEXT NOT NULL, redirect_uri TEXT NOT NULL, expires_at REAL NOT NULL, consumed_at REAL
);
CREATE TABLE IF NOT EXISTS oauth_tokens (
    token_hash TEXT PRIMARY KEY, grant_id TEXT NOT NULL REFERENCES oauth_grants(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK(kind IN ('access','refresh')), expires_at REAL NOT NULL, consumed_at REAL
);
CREATE INDEX IF NOT EXISTS oauth_grants_user ON oauth_grants(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS oauth_tokens_grant ON oauth_tokens(grant_id);
"""


@dataclass(frozen=True)
class OAuthConfig:
    client_id: str
    client_secret: str
    redirect_uris: tuple[str, ...]
    allowed_scopes: tuple[str, ...] = ("support:read", "support:propose")
    code_ttl_seconds: int = 120
    access_ttl_seconds: int = 900
    refresh_ttl_seconds: int = 604800

    def validate(self):
        if not self.client_id or len(self.client_id) > 128 or len(self.client_secret) < 32:
            raise ValueError("OAuth requires a client ID and a secret of at least 32 characters.")
        # Empty callbacks are a safe bootstrap state: every authorization fails closed.
        if not set(self.allowed_scopes) or not set(self.allowed_scopes) <= set(SCOPES):
            raise ValueError("OAuth requires supported scopes.")
        if not (1 <= self.code_ttl_seconds <= 600 and 1 <= self.access_ttl_seconds <= 3600
                and self.access_ttl_seconds <= self.refresh_ttl_seconds <= 2592000):
            raise ValueError("OAuth token lifetimes are outside the supported limits.")
        for value in self.redirect_uris:
            parsed = urlsplit(value)
            local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            if ((parsed.scheme != "https" and not local_http) or not parsed.netloc or parsed.username
                    or parsed.password or parsed.fragment or parsed.query or not parsed.path.startswith("/api/actions/")
                    or not parsed.path.endswith("/oauth/callback") or any(ord(c) < 33 for c in value)):
                raise ValueError("OAuth callback URLs must be explicit LibreChat HTTPS or local development callbacks.")


def initialize_oauth(connection):
    """Additive, idempotent migration; never rewrites existing business records."""
    execute_statements(connection, SCHEMA)


def identity_version(user):
    return token_hash("\0".join(str(user[key]) for key in ("id", "password_hash", "api_key_hash", "org_id", "role")))


def revoke_oauth_user_grants(connection, user_id, reason="logout"):
    """Safe for startup migrations that have not installed OAuth tables yet."""
    exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='oauth_grants'").fetchone()
    if exists:
        connection.execute("UPDATE oauth_grants SET revoked_at=?,revocation_reason=? WHERE user_id=? AND revoked_at IS NULL",
                           (time.time(), reason, user_id))


def revoke_oauth_session_grants(connection, session_hash, reason="logout"):
    """Logging out one browser must not revoke other devices' independent grants."""
    exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='oauth_grants'").fetchone()
    if exists:
        connection.execute("UPDATE oauth_grants SET revoked_at=?,revocation_reason=? WHERE session_hash=? AND revoked_at IS NULL",
                           (time.time(), reason, session_hash))


def revalidate_oauth_actor(connection, actor):
    """Resolve the trusted actor again inside an existing business transaction."""
    access_hash = actor.get("_oauth_token_hash")
    if not access_hash:
        return None
    now = time.time()
    token = connection.execute("""SELECT g.* FROM oauth_tokens t JOIN oauth_grants g ON g.id=t.grant_id
        WHERE t.token_hash=? AND t.kind='access' AND t.consumed_at IS NULL
        AND t.expires_at>? AND g.expires_at>? AND g.revoked_at IS NULL""", (access_hash, now, now)).fetchone()
    scope = actor.get("_oauth_required_scope", "support:read")
    if (not token or token["client_id"] != actor.get("_oauth_client_id")
            or scope not in token["scopes"].split() or token["user_id"] != actor.get("id")):
        return None
    row = connection.execute("SELECT * FROM users WHERE id=? AND active=1", (token["user_id"],)).fetchone()
    if not row or not hmac.compare_digest(identity_version(row), token["identity_version"]):
        return None
    return {**dict(row), "_oauth_token_hash": access_hash, "_oauth_grant_id": token["id"],
            "_oauth_scopes": token["scopes"], "_oauth_required_scope": scope, "_oauth_client_id": token["client_id"]}


class OAuthService:
    def __init__(self, database, config: OAuthConfig):
        config.validate()
        self.database, self.config = database, config
        with database.connection() as connection:
            initialize_oauth(connection)

    @staticmethod
    def revoke_user_grants(connection, user_id, reason="logout"):
        revoke_oauth_user_grants(connection, user_id, reason)

    @staticmethod
    def revoke_session_grants(connection, session_hash, reason="logout"):
        revoke_oauth_session_grants(connection, session_hash, reason)

    def authenticate_in_connection(self, connection, access_hash, required_scope="support:read"):
        """Recheck at the business write lock, so a concurrent revoke takes effect."""
        now = time.time()
        token = connection.execute("""SELECT g.*, t.expires_at AS token_expires_at FROM oauth_tokens t
            JOIN oauth_grants g ON g.id=t.grant_id WHERE t.token_hash=? AND t.kind='access'
            AND t.consumed_at IS NULL AND t.expires_at>? AND g.expires_at>? AND g.revoked_at IS NULL""",
                                   (access_hash, now, now)).fetchone()
        if not token or token["client_id"] != self.config.client_id or required_scope not in token["scopes"].split():
            return None
        row = connection.execute("SELECT * FROM users WHERE id=? AND active=1", (token["user_id"],)).fetchone()
        if not row or not hmac.compare_digest(identity_version(row), token["identity_version"]):
            return None
        return {**dict(row), "_oauth_token_hash": access_hash, "_oauth_grant_id": token["id"],
                "_oauth_scopes": token["scopes"], "_oauth_required_scope": required_scope, "_oauth_client_id": token["client_id"]}

    def authenticate(self, token, required_scope="support:read"):
        if not token.startswith(ACCESS_PREFIX) or len(token) > 200:
            return None
        with self.database.connection() as connection:
            return self.authenticate_in_connection(connection, token_hash(token), required_scope)

    def _grant_is_current(self, connection, grant):
        if not grant or grant["revoked_at"] is not None or grant["expires_at"] <= time.time() or grant["client_id"] != self.config.client_id:
            return False
        row = connection.execute("SELECT * FROM users WHERE id=? AND active=1", (grant["user_id"],)).fetchone()
        return bool(row and hmac.compare_digest(identity_version(row), grant["identity_version"]))

    def _issue_tokens(self, connection, grant):
        now = time.time()
        access, refresh = ACCESS_PREFIX + secrets.token_urlsafe(32), REFRESH_PREFIX + secrets.token_urlsafe(32)
        access_expiry = min(now + self.config.access_ttl_seconds, grant["expires_at"])
        connection.executemany("INSERT INTO oauth_tokens(token_hash,grant_id,kind,expires_at) VALUES(?,?,?,?)", [
            (token_hash(access), grant["id"], "access", access_expiry),
            (token_hash(refresh), grant["id"], "refresh", grant["expires_at"]),
        ])
        return {"access_token": access, "refresh_token": refresh, "token_type": "Bearer",
                "expires_in": max(1, int(access_expiry - now)), "refresh_token_expires_in": max(1, int(grant["expires_at"] - now)),
                "scope": grant["scopes"]}


def register_oauth(app, database, config: OAuthConfig, *, browser_session, require_session, check_csrf, page):
    service = OAuthService(database, config)
    app.state.oauth = service

    def error_response(error, status=400):
        headers = {"Cache-Control": "no-store", "Pragma": "no-cache"}
        if status == 401:
            headers["WWW-Authenticate"] = 'Basic realm="support-oauth"'
        return JSONResponse({"error": error}, status_code=status, headers=headers)

    def trace(request):
        return getattr(request.state, "trace_id", "oauth")

    def validate_authorization(request):
        params = request.query_params
        for key in ("client_id", "redirect_uri", "response_type", "scope", "state"):
            if len(params.getlist(key)) != 1:
                raise HTTPException(400, "授权请求参数不完整或重复，请从 LibreChat 重新发起连接。")
        if params["client_id"] != config.client_id or params["redirect_uri"] not in config.redirect_uris:
            raise HTTPException(400, "授权客户端或回调地址不受信任。")
        scopes = params["scope"].split()
        if (params["response_type"] != "code" or not scopes or not set(scopes) <= set(config.allowed_scopes)
                or not params["state"] or len(params["state"]) > 8192):
            raise HTTPException(400, "授权类型、权限范围或请求状态无效。")
        return {"client_id": params["client_id"], "redirect_uri": params["redirect_uri"],
                "response_type": "code", "scope": " ".join(sorted(set(scopes))), "state": params["state"]}

    def login_redirect(parameters):
        return RedirectResponse("/console/login?" + urlencode({"next": "/console/oauth/authorize?" + urlencode(parameters)}), status_code=303)

    def live_session(connection, session):
        row = connection.execute("""SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id
            WHERE s.token_hash=? AND s.user_id=? AND s.expires_at>? AND u.active=1""",
                                 (session["session_hash"], session["id"], time.time())).fetchone()
        if not row or not hmac.compare_digest(identity_version(row), identity_version(session)):
            raise HTTPException(401, "账号或登录状态已变更，请重新发起连接。")
        return row

    @app.get("/console/oauth/authorize", include_in_schema=False)
    def authorize_page(request: Request):
        params = validate_authorization(request)
        session = browser_session(request)
        if not session:
            return login_redirect(params)
        request_id = secrets.token_urlsafe(32)
        with database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            live_session(connection, session)
            connection.execute("DELETE FROM oauth_requests WHERE expires_at<?", (time.time(),))
            connection.execute("""INSERT INTO oauth_requests
                (request_hash,user_id,session_hash,client_id,redirect_uri,state,scopes,expires_at) VALUES(?,?,?,?,?,?,?,?)""",
                               (token_hash(request_id), session["id"], session["session_hash"], config.client_id,
                                params["redirect_uri"], params["state"], params["scope"], time.time() + 600))
        scope_html = "".join(f"<li>{escape(SCOPES[scope])}</li>" for scope in params["scope"].split())
        body = f'''<section class="card narrow"><h1>连接你的业务账号</h1>
            <p>将当前 LibreChat 账号连接到以下服务台账号：</p>
            <p><strong>{escape(session['display_name'])}</strong> · {escape(session['id'])} · {escape(session['org_id'])}</p>
            <p>这是业务账号授权连接，不是自动单点登录。请核对账号后再授权。</p><ul>{scope_html}</ul>
            <p>角色和可访问范围由服务台实时检查；正式创建或修改工单仍需要本人在独立页面确认。</p>
            <form method="post" action="/console/oauth/authorize">
            <input type="hidden" name="csrf" value="{escape(session['csrf'])}">
            <input type="hidden" name="request_id" value="{request_id}">
            <div class="actions"><button name="decision" value="approve">确认连接此账号</button>
            <button class="secondary" name="decision" value="switch">退出并切换账号</button>
            <button class="secondary" name="decision" value="deny">取消</button></div></form>
            <p class="note">退出或切换账号会撤销本次登录会话创建的 Agent 授权，不影响其他设备独立创建的授权。切换页面账号不会将已有连接变成新身份；请重新授权。</p></section>'''
        return page("连接业务账号", body, session)

    @app.post("/console/oauth/authorize", include_in_schema=False)
    async def authorize(request: Request):
        session = require_session(request)
        form = await request.form()
        check_csrf(request, session, str(form.get("csrf", "")))
        decision = str(form.get("decision", ""))
        if decision not in {"approve", "deny", "switch"}:
            raise HTTPException(400, "请选择连接、切换或取消。")
        with database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            user = live_session(connection, session)
            pending = connection.execute("SELECT * FROM oauth_requests WHERE request_hash=?", (token_hash(str(form.get("request_id", ""))),)).fetchone()
            if (not pending or pending["consumed_at"] is not None or pending["expires_at"] <= time.time()
                    or pending["user_id"] != user["id"] or pending["session_hash"] != session["session_hash"]
                    or pending["client_id"] != config.client_id or pending["redirect_uri"] not in config.redirect_uris):
                raise HTTPException(403, "授权页面已过期、已处理或账号发生变化，请从 LibreChat 重新发起连接。")
            connection.execute("UPDATE oauth_requests SET consumed_at=? WHERE request_hash=?", (time.time(), pending["request_hash"]))
            if decision == "switch":
                service.revoke_session_grants(connection, session["session_hash"], "account_switch")
                connection.execute("DELETE FROM sessions WHERE token_hash=?", (session["session_hash"],))
                response = login_redirect({"client_id": pending["client_id"], "redirect_uri": pending["redirect_uri"],
                                           "response_type": "code", "scope": pending["scopes"], "state": pending["state"]})
                response.delete_cookie("support_session", path="/console")
            elif decision == "deny":
                response = RedirectResponse(pending["redirect_uri"] + "?" + urlencode({"error": "access_denied", "state": pending["state"]}), status_code=303)
            else:
                grant_id, code, now = "OG-" + secrets.token_hex(16), secrets.token_urlsafe(32), time.time()
                connection.execute("""INSERT INTO oauth_grants
                    (id,user_id,client_id,scopes,identity_version,session_hash,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?)""",
                                   (grant_id, user["id"], config.client_id, pending["scopes"], identity_version(user),
                                    session["session_hash"], now, now + config.refresh_ttl_seconds))
                connection.execute("INSERT INTO oauth_codes(code_hash,grant_id,client_id,redirect_uri,expires_at) VALUES(?,?,?,?,?)",
                                   (token_hash(code), grant_id, config.client_id, pending["redirect_uri"], now + config.code_ttl_seconds))
                response = RedirectResponse(pending["redirect_uri"] + "?" + urlencode({"code": code, "state": pending["state"]}), status_code=303)
            audit(connection, user["id"], "oauth.authorize", None, decision, trace(request))
        return response

    async def client_form(request):
        if len(await request.body()) > 16384:
            return None
        form = await request.form()
        if any(len(form.getlist(key)) > 1 for key in ("client_id", "client_secret", "code", "redirect_uri", "grant_type", "refresh_token", "scope", "token")):
            return None
        client_id, secret = str(form.get("client_id", "")), str(form.get("client_secret", ""))
        authorization = request.headers.get("authorization", "")
        if authorization:
            if client_id or secret or not authorization.startswith("Basic "):
                return None
            try:
                decoded = base64.b64decode(authorization[6:], validate=True).decode("utf-8")
                client_id, secret = decoded.split(":", 1)
            except (ValueError, UnicodeError, binascii.Error):
                return None
        if not (hmac.compare_digest(client_id.encode(), config.client_id.encode())
                and hmac.compare_digest(secret.encode(), config.client_secret.encode())):
            return None
        return form

    @app.post("/oauth/token", include_in_schema=False)
    async def exchange(request: Request):
        form = await client_form(request)
        if form is None:
            return error_response("invalid_client", 401)
        kind = str(form.get("grant_type", ""))
        if kind not in {"authorization_code", "refresh_token"}:
            return error_response("unsupported_grant_type")
        failure, result, user_id = None, None, None
        with database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if kind == "authorization_code":
                value = str(form.get("code", ""))
                credential = connection.execute("SELECT * FROM oauth_codes WHERE code_hash=?", (token_hash(value),)).fetchone()
            else:
                value = str(form.get("refresh_token", ""))
                credential = connection.execute("SELECT * FROM oauth_tokens WHERE token_hash=? AND kind='refresh'", (token_hash(value),)).fetchone()
            grant = connection.execute("SELECT * FROM oauth_grants WHERE id=?", (credential["grant_id"],)).fetchone() if credential else None
            if grant:
                user_id = grant["user_id"]
            if not credential or not service._grant_is_current(connection, grant) or credential["expires_at"] <= time.time():
                failure = "invalid_grant"
            elif credential["consumed_at"] is not None:
                # Commit revocation before returning the error; raising here would roll it back.
                connection.execute("UPDATE oauth_grants SET revoked_at=?,revocation_reason='credential_reuse' WHERE id=?", (time.time(), grant["id"]))
                failure = "invalid_grant"
            elif kind == "authorization_code" and (credential["client_id"] != config.client_id or credential["redirect_uri"] != str(form.get("redirect_uri", ""))):
                failure = "invalid_grant"
            elif form.get("scope") and str(form["scope"]) != grant["scopes"]:
                failure = "invalid_scope"
            else:
                if kind == "authorization_code":
                    connection.execute("UPDATE oauth_codes SET consumed_at=? WHERE code_hash=?", (time.time(), token_hash(value)))
                else:
                    connection.execute("UPDATE oauth_tokens SET consumed_at=? WHERE token_hash=?", (time.time(), token_hash(value)))
                    # An old access token is retired with its refresh credential.
                    connection.execute("UPDATE oauth_tokens SET consumed_at=? WHERE grant_id=? AND kind='access' AND consumed_at IS NULL", (time.time(), grant["id"]))
                result = service._issue_tokens(connection, grant)
            audit(connection, user_id, "oauth.token", None, failure or "ok", trace(request))
        return error_response(failure) if failure else JSONResponse(result, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})

    @app.post("/oauth/revoke", include_in_schema=False)
    async def revoke(request: Request):
        form = await client_form(request)
        if form is None:
            return error_response("invalid_client", 401)
        with database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""SELECT g.id,g.user_id FROM oauth_tokens t JOIN oauth_grants g ON g.id=t.grant_id
                WHERE t.token_hash=? AND g.client_id=?""", (token_hash(str(form.get("token", ""))), config.client_id)).fetchone()
            if row:
                connection.execute("UPDATE oauth_grants SET revoked_at=?,revocation_reason='client_revoke' WHERE id=?", (time.time(), row["id"]))
                audit(connection, row["user_id"], "oauth.revoke", row["id"], "ok", trace(request))
        # Do not disclose whether a token existed.
        return JSONResponse({}, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})

    @app.get("/console/oauth/connections", include_in_schema=False)
    def connections_page(request: Request):
        session = browser_session(request)
        if not session:
            return RedirectResponse("/console/login?" + urlencode({"next": "/console/oauth/connections"}), status_code=303)
        with database.connection() as connection:
            grants = connection.execute("SELECT * FROM oauth_grants WHERE user_id=? AND revoked_at IS NULL AND expires_at>? ORDER BY created_at DESC",
                                        (session["id"], time.time())).fetchall()
        items = []
        for grant in grants:
            items.append(f'''<li><strong>{escape(grant['client_id'])}</strong><small>业务账号：{escape(session['id'])} · 授权编号：{grant['id']}</small>
                <form method="post" action="/console/oauth/connections"><input type="hidden" name="csrf" value="{escape(session['csrf'])}">
                <input type="hidden" name="grant_id" value="{grant['id']}"><button class="secondary">撤销此连接</button></form></li>''')
        body = '<section class="card"><h1>我的 Agent 授权</h1><p>撤销会立即使此连接的访问和刷新凭证失效。重新连接需从 LibreChat 发起授权。</p><ul class="list">' + ("".join(items) or "<li>没有有效连接。</li>") + "</ul></section>"
        return page("我的 Agent 授权", body, session)

    @app.post("/console/oauth/connections", include_in_schema=False)
    async def revoke_connection(request: Request):
        session = require_session(request)
        form = await request.form()
        check_csrf(request, session, str(form.get("csrf", "")))
        with database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            user = live_session(connection, session)
            row = connection.execute("SELECT id FROM oauth_grants WHERE id=? AND user_id=?", (str(form.get("grant_id", "")), user["id"])).fetchone()
            if not row:
                raise HTTPException(404, "连接不存在或无权访问。")
            connection.execute("UPDATE oauth_grants SET revoked_at=?,revocation_reason='user_revoke' WHERE id=?", (time.time(), row["id"]))
            audit(connection, user["id"], "oauth.revoke", row["id"], "ok", trace(request))
        return RedirectResponse("/console/oauth/connections", status_code=303)

    return service
