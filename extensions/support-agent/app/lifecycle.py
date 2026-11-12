"""Role-aware ticket transitions. Only the trusted browser controller calls confirm."""
from datetime import datetime, timezone
import hmac
import json
import secrets
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .db import Database, audit
from .policy import ASSIGNEE_ROLE, action_error, allowed_actions, can_read_ticket, owns_record, valid_assignee
from .security import token_hash
from .oauth import revalidate_oauth_actor


class UpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    action: Literal["assign", "start", "comment", "resolve", "close"]
    request_id: str = Field(min_length=8, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    expected_version: int = Field(ge=1, strict=True)
    note: str = Field(min_length=2, max_length=2000)
    assignee_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")

    @field_validator("note")
    @classmethod
    def validate_note(cls, value: str) -> str:
        if any(ord(character) < 32 and character not in "\n\r\t" for character in value):
            raise ValueError("Control characters are not allowed.")
        return value

    @model_validator(mode="after")
    def validate_assignee(self):
        if self.action == "assign" and self.assignee_id is None:
            raise ValueError("assign requires assignee_id.")
        if self.action != "assign" and self.assignee_id is not None:
            raise ValueError("assignee_id is only permitted for assign.")
        return self


class LifecycleError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


TICKET_SELECT = """SELECT t.*, owner.display_name AS requester_name,
    assignee.display_name AS assignee_name
    FROM tickets t JOIN users owner ON owner.id=t.user_id
    LEFT JOIN users assignee ON assignee.id=t.assignee_id"""


def timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


class Lifecycle:
    def __init__(self, database: Database):
        self.database = database

    def _run(self, user, event: str, object_id: str | None, trace_id: str, operation):
        """Commit a business denial's audit first, then propagate its safe error."""
        failure = None
        result = None
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM users WHERE id=?", (user.get("id"),)).fetchone()
            actor = dict(row) if row else None
            try:
                if user.get("_oauth_token_hash") and revalidate_oauth_actor(connection, user) is None:
                    raise LifecycleError(401, "工具授权已撤销或过期，请重新连接服务台账号。")
                if actor is None or not actor["active"]:
                    raise LifecycleError(403, "业务账号未启用，请重新登录或联系管理员。")
                if "api_key_hash" in user and not hmac.compare_digest(str(user["api_key_hash"]), actor["api_key_hash"]):
                    raise LifecycleError(401, "身份凭证已更新，请重新认证。")
                if "session_hash" in user:
                    session = connection.execute("SELECT 1 FROM sessions WHERE token_hash=? AND user_id=? AND expires_at>?",
                                                 (user["session_hash"], actor["id"], time.time())).fetchone()
                    if session is None:
                        raise LifecycleError(401, "登录会话已撤销或过期，请重新登录。")
                result = operation(connection, actor)
            except LifecycleError as error:
                failure = error
                outcome = "conflict" if error.status == 409 else "expired" if error.status == 410 else "denied"
            else:
                outcome = "replayed" if isinstance(result, dict) and result.get("replayed") else "ok"
            audit(connection, actor["id"] if actor else None, event, object_id[:100] if object_id else None, outcome, trace_id)
        if failure:
            raise failure
        return result

    @staticmethod
    def _visible(actor: dict, ticket) -> bool:
        return can_read_ticket(actor, ticket)

    def _ticket(self, connection, actor: dict, ticket_id: str):
        ticket = connection.execute(TICKET_SELECT + " WHERE t.id=?", (ticket_id,)).fetchone()
        if ticket is None or not self._visible(actor, ticket):
            raise LifecycleError(404, "工单不存在或无权访问。")
        return ticket

    @staticmethod
    def _allowed_actions(actor: dict, ticket) -> list[str]:
        return allowed_actions(actor, ticket)

    def _ticket_view(self, actor: dict, ticket) -> dict:
        return {"id": ticket["id"], "draft_id": ticket["draft_id"], "title": ticket["title"],
                "description": ticket["description"], "priority": ticket["priority"], "status": ticket["status"],
                "user_id": ticket["user_id"], "org_id": ticket["org_id"], "assignee_id": ticket["assignee_id"],
                "assignee_name": ticket["assignee_name"], "requester_name": ticket["requester_name"],
                "version": ticket["version"], "created_at": timestamp(ticket["created_at"]),
                "updated_at": timestamp(ticket["updated_at"]), "allowed_actions": self._allowed_actions(actor, ticket)}

    def _update_view(self, update) -> dict:
        now = time.time()
        status = update["status"]
        if status == "pending" and update["expires_at"] <= now:
            status = "expired"
        return {"id": update["id"], "ticket_id": update["ticket_id"], "action": update["action"],
                "note": update["note"], "assignee_id": update["assignee_id"], "expected_version": update["expected_version"],
                "status": status, "status_evaluated_at": timestamp(now),
                "status_guidance": "status已按服务端当前时间计算；pending不能被解释为已过期。不要根据聊天日期或时区自行改写状态；正式确认时服务端还会再次校验有效期与权限。",
                "created_at": timestamp(update["created_at"]), "expires_at": timestamp(update["expires_at"]),
                "confirmation_url": f"{self.database.settings.public_base_url.rstrip('/')}/console/updates/{update['id']}"}

    def _update(self, connection, actor: dict, update_id: str):
        update = connection.execute("SELECT * FROM ticket_updates WHERE id=?", (update_id,)).fetchone()
        if update is None or not owns_record(actor, update):
            raise LifecycleError(404, "更新提案不存在或无权访问。")
        ticket = self._ticket(connection, actor, update["ticket_id"])
        return update, ticket

    @staticmethod
    def _authorize(connection, actor: dict, ticket, action: str, assignee_id: str | None):
        error = action_error(actor, ticket, action)
        if error:
            raise LifecycleError(*error)
        if action == "assign":
            assignee = connection.execute("SELECT role,active,org_id FROM users WHERE id=?", (assignee_id,)).fetchone()
            if not valid_assignee(actor, assignee):
                raise LifecycleError(403, "负责人必须是当前组织已启用的支持人员。")

    def get_ticket(self, user, ticket_id: str, trace_id: str) -> dict:
        return self._run(user, "ticket.read", ticket_id, trace_id,
                         lambda connection, actor: self._ticket_view(actor, self._ticket(connection, actor, ticket_id)))

    def list_tickets(self, user, status: str | None, limit: int, trace_id: str) -> list[dict]:
        def operation(connection, actor):
            if status is not None and status not in {"open", "in_progress", "resolved", "closed"}:
                raise LifecycleError(422, "不支持的工单状态。")
            if not 1 <= limit <= 100:
                raise LifecycleError(422, "查询数量必须在1到100之间。")
            rows = connection.execute(TICKET_SELECT + """ WHERE t.org_id=? AND
                (?='admin' OR t.user_id=? OR (?='support' AND t.assignee_id=?))
                AND (? IS NULL OR t.status=?) ORDER BY t.created_at DESC,t.id LIMIT ?""",
                (actor["org_id"], actor["role"], actor["id"], actor["role"], actor["id"], status, status, limit)).fetchall()
            return [self._ticket_view(actor, row) for row in rows]
        return self._run(user, "ticket.list", None, trace_id, operation)

    def history(self, user, ticket_id: str, trace_id: str) -> list[dict]:
        def operation(connection, actor):
            self._ticket(connection, actor, ticket_id)
            rows = connection.execute("SELECT * FROM ticket_events WHERE ticket_id=? ORDER BY version,id", (ticket_id,)).fetchall()
            return [{"id": row["id"], "actor_name": row["actor_name"], "action": row["action"], "note": row["note"],
                     "from_status": row["from_status"], "to_status": row["to_status"], "assignee_id": row["assignee_id"],
                     "status_summary": (f"前状态未记录；记录后状态为 {row['to_status']}" if row["from_status"] is None
                                        else f"{row['from_status']} → {row['to_status']}"),
                     "created_at": timestamp(row["created_at"]), "version": row["version"]} for row in rows]
        return self._run(user, "ticket.history", ticket_id, trace_id, operation)

    def staff(self, user) -> list[dict]:
        def operation(connection, actor):
            if actor["role"] != "admin":
                raise LifecycleError(403, "只有管理员可以查看可分配支持人员。")
            rows = connection.execute("SELECT id,display_name,role FROM users WHERE org_id=? AND active=1 AND role=? ORDER BY display_name,id", (actor["org_id"], ASSIGNEE_ROLE)).fetchall()
            return [dict(row) for row in rows]
        return self._run(user, "staff.list", None, secrets.token_hex(12), operation)

    def propose(self, user, ticket_id: str, payload: UpdateInput, trace_id: str) -> dict:
        def operation(connection, actor):
            ticket = self._ticket(connection, actor, ticket_id)
            contents = {"ticket_id": ticket_id, **payload.model_dump(exclude={"request_id"})}
            digest = token_hash(json.dumps(contents, ensure_ascii=False, sort_keys=True))
            existing = connection.execute("SELECT * FROM ticket_updates WHERE user_id=? AND request_id=?", (actor["id"], payload.request_id)).fetchone()
            if existing:
                if existing["payload_hash"] != digest:
                    raise LifecycleError(409, "此 request_id 已用于不同更新内容；重试必须使用原始内容。")
                return {**self._update_view(existing), "replayed": True}
            self._authorize(connection, actor, ticket, payload.action, payload.assignee_id)
            if ticket["version"] != payload.expected_version:
                raise LifecycleError(409, "工单版本已更新，请重新读取工单后创建新的提案。")
            update_id = "U-" + secrets.token_hex(8).upper()
            now = time.time()
            connection.execute("""INSERT INTO ticket_updates
                (id,user_id,ticket_id,request_id,payload_hash,action,note,assignee_id,expected_version,status,created_at,expires_at)
                VALUES(?,?,?,?,?,?,?,?,?,'pending',?,?)""",
                (update_id, actor["id"], ticket_id, payload.request_id, digest, payload.action, payload.note,
                 payload.assignee_id, payload.expected_version, now, now + self.database.settings.draft_ttl_seconds))
            row = connection.execute("SELECT * FROM ticket_updates WHERE id=?", (update_id,)).fetchone()
            return {**self._update_view(row), "replayed": False}
        return self._run(user, "update.propose", ticket_id, trace_id, operation)

    def get_update(self, user, update_id: str, trace_id: str) -> dict:
        def operation(connection, actor):
            update, _ = self._update(connection, actor, update_id)
            return self._update_view(update)
        return self._run(user, "update.read", update_id, trace_id, operation)

    def confirm(self, user, update_id: str, trace_id: str) -> dict:
        def operation(connection, actor):
            update, ticket = self._update(connection, actor, update_id)
            if update["status"] == "applied":
                return {**self._update_view(update), "replayed": True}
            if update["status"] == "cancelled":
                raise LifecycleError(409, "提案已取消，不能执行。")
            if update["expires_at"] <= time.time():
                raise LifecycleError(410, "提案已过期，请重新创建。")
            self._authorize(connection, actor, ticket, update["action"], update["assignee_id"])
            if ticket["version"] != update["expected_version"]:
                raise LifecycleError(409, "工单版本已更新，此提案不能覆盖新状态；请重新创建。")
            status = {"start": "in_progress", "resolve": "resolved", "close": "closed"}.get(update["action"], ticket["status"])
            assignee_id = update["assignee_id"] if update["action"] == "assign" else ticket["assignee_id"]
            version, now = ticket["version"] + 1, time.time()
            connection.execute("UPDATE tickets SET status=?,assignee_id=?,version=?,updated_at=? WHERE id=?", (status, assignee_id, version, now, ticket["id"]))
            connection.execute("""INSERT INTO ticket_events
                (ticket_id,actor_id,actor_name,action,note,from_status,to_status,assignee_id,version,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (ticket["id"], actor["id"], actor["display_name"], update["action"], update["note"], ticket["status"], status, assignee_id, version, now))
            connection.execute("UPDATE ticket_updates SET status='applied',applied_at=? WHERE id=?", (now, update_id))
            applied = connection.execute("SELECT * FROM ticket_updates WHERE id=?", (update_id,)).fetchone()
            return {**self._update_view(applied), "replayed": False}
        return self._run(user, "update.confirm", update_id, trace_id, operation)

    def cancel(self, user, update_id: str, trace_id: str) -> dict:
        def operation(connection, actor):
            update, _ = self._update(connection, actor, update_id)
            if update["status"] == "applied":
                raise LifecycleError(409, "提案已经执行，不能取消。")
            if update["status"] == "cancelled":
                return {**self._update_view(update), "replayed": True}
            if update["expires_at"] <= time.time():
                raise LifecycleError(410, "提案已过期，请重新创建。")
            connection.execute("UPDATE ticket_updates SET status='cancelled' WHERE id=?", (update_id,))
            cancelled = connection.execute("SELECT * FROM ticket_updates WHERE id=?", (update_id,)).fetchone()
            return {**self._update_view(cancelled), "replayed": False}
        return self._run(user, "update.cancel", update_id, trace_id, operation)

    def list_updates(self, user, limit: int = 20) -> list[dict]:
        def operation(connection, actor):
            if not 1 <= limit <= 100:
                raise LifecycleError(422, "查询数量必须在1到100之间。")
            rows = connection.execute("""SELECT p.* FROM ticket_updates p JOIN tickets t ON t.id=p.ticket_id
                WHERE p.user_id=? AND t.org_id=? AND
                (?='admin' OR t.user_id=? OR (?='support' AND t.assignee_id=?))
                ORDER BY p.created_at DESC,p.id LIMIT ?""",
                (actor["id"], actor["org_id"], actor["role"], actor["id"], actor["role"], actor["id"], limit)).fetchall()
            return [self._update_view(row) for row in rows]
        return self._run(user, "update.list", None, secrets.token_hex(12), operation)
