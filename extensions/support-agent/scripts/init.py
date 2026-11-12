"""Initialize local demo accounts without replacing existing roles or secrets."""

import argparse
import json
import secrets
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEMO_USERS = (
    ("alice", "林晓", "requester"),
    ("bob", "周宁", "requester"),
    ("support1", "王工", "support"),
    ("support2", "陈工", "support"),
    ("admin", "服务台管理员", "admin"),
)


def write_once(path: Path, content: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except FileExistsError:
        pass


def ensure_demo_users(path: Path) -> None:
    """Append missing local identities; configuration, never an LLM, owns roles."""
    if path.exists():
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            raise ValueError("Existing users.json is unreadable; it was left unchanged.") from None
        users = document.get("users") if isinstance(document, dict) else None
        if not isinstance(users, list) or any(
            not isinstance(user, dict) or not isinstance(user.get("id"), str) or not user["id"]
            for user in users
        ):
            raise ValueError("Existing users.json has invalid identities; it was left unchanged.")
        ids = {user["id"] for user in users}
        if len(ids) != len(users):
            raise ValueError("Existing users.json contains duplicate IDs; it was left unchanged.")
    else:
        document = {"users": []}
        users = document["users"]
        ids = set()

    added = False
    for ident, name, role in DEMO_USERS:
        if ident in ids:
            # Preserve existing records, including intentionally changed roles.
            # Legacy records omit role; the service defaults them to requester.
            continue
        users.append({
            "id": ident,
            "display_name": name,
            "org_id": "demo",
            "role": role,
            "password": secrets.token_urlsafe(18),
            "api_key": secrets.token_urlsafe(32),
        })
        added = True
    if not added:
        return
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize(root: Path) -> None:
    local = root / ".local"
    local.mkdir(parents=True, exist_ok=True)
    ensure_demo_users(local / "users.json")
    write_once(local / "demo.json", json.dumps({
        "email": "demo@local.test", "name": "工单演示", "username": "demo",
        "password": secrets.token_urlsafe(18), "support_user_id": "alice",
    }, ensure_ascii=False, indent=2))
    state_path = local / "support-agent-bootstrap.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    callback = "http://localhost:3080/api/actions/" + state["action_id"] + "/oauth/callback" if state.get("action_id") else None
    write_once(local / "operations.json", json.dumps({
        "oauth": {"client_id": "librechat-support", "client_secret": secrets.token_urlsafe(48),
                  "redirect_uris": [callback] if callback else []},
        "monitor_token": secrets.token_urlsafe(48),
    }, ensure_ascii=False, indent=2))
    write_once(root / ".env", "\n".join([
        "# Local only. Fill DEEPSEEK_API_KEY then run the Start task again.",
        "DEEPSEEK_API_KEY=", "HOST=0.0.0.0", "PORT=3080",
        "DOMAIN_CLIENT=http://localhost:3080", "DOMAIN_SERVER=http://localhost:3080",
        "ENDPOINTS=agents,custom", "ALLOW_REGISTRATION=true",
        "ALLOW_EMAIL_LOGIN=true", "ALLOW_UNVERIFIED_EMAIL_LOGIN=true",
        "SESSION_COOKIE_SECURE=false", "APP_TITLE=Support Agent Lab",
        "SEARCH=false", "NO_INDEX=true", "MONGO_MAX_POOL_SIZE=10",
        "CREDS_KEY=" + secrets.token_hex(32), "CREDS_IV=" + secrets.token_hex(16),
        "JWT_SECRET=" + secrets.token_hex(32), "JWT_REFRESH_SECRET=" + secrets.token_hex(32),
        "DEBUG_LOGGING=false", "DEBUG_CONSOLE=false", "SCHEDULES_SINGLE_PROCESS=true", "",
    ]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        initialize(args.root)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    except OSError:
        print("Could not initialize local configuration; check directory access.", file=sys.stderr)
        return 1
    print("Local configuration ready. Existing roles and secrets were preserved in .env and .local/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
