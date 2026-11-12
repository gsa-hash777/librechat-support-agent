"""Provision a private evaluation account, role-specific Agents, and isolated tools."""
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import sys

SERVICE = Path(__file__).resolve().parents[1]
ROOT = SERVICE.parents[1]
sys.path.insert(0, str(SERVICE / "integration"))
from bootstrap import INSTRUCTIONS, ApiError, BootstrapError, LocalApi, action_functions, authenticate, read_json, save_state

ACTORS = {"alice": ("requester", "demo"), "bob": ("requester", "demo"), "support1": ("support", "demo"), "support2": ("support", "demo"), "admin": ("admin", "demo"), "eve": ("requester", "other"), "other_admin": ("admin", "other")}
COMPOSE = ["docker", "compose", "-f", "compose.support.yml", "-f", "compose.evaluation.yml", "--profile", "evaluation"]
MODEL_PARAMETERS = {"temperature": 0.2, "max_tokens": 1200}
RECURSION_LIMIT = 12
LOCAL = ROOT / ".local/evaluation"


def instructions(actor):
    old = "本演示的工具凭证固定绑定 alice。用户在独立服务台页面切换账号不会切换此 Agent 的工具身份，不得据此声称获得客服或管理员权限。"
    return INSTRUCTIONS.replace(old, f"本评测的工具凭证固定绑定 {actor}。用户在独立服务台页面切换账号不会切换此 Agent 的工具身份，不得据此声称获得其他权限。")


def credentials():
    LOCAL.mkdir(parents=True, exist_ok=True)
    if not (LOCAL / "users.json").exists():
        save_state(LOCAL / "users.json", {"users": [{"id": ident, "display_name": "评测-" + ident, "org_id": org, "role": role, "password": secrets.token_urlsafe(20), "api_key": secrets.token_urlsafe(32)} for ident, (role, org) in ACTORS.items()]})
    if not (LOCAL / "demo.json").exists():
        save_state(LOCAL / "demo.json", {"email": "support-evaluation@local.test", "name": "独立评测账号", "username": "support-evaluation", "password": secrets.token_urlsafe(20)})
    users = read_json(LOCAL / "users.json")["users"]
    if {u["id"]: (u["role"], u["org_id"]) for u in users} != ACTORS:
        raise BootstrapError("Evaluation identity configuration does not match the declared fixture roles.")
    return users


def prepare(model="deepseek-flash"):
    users = credentials()
    subprocess.run([*COMPOSE, "up", "-d", "--build", "--wait", "--wait-timeout", "120", "support-eval"], cwd=ROOT, check=True)
    config_hash = hashlib.sha256((SERVICE / "deploy/config.yaml").read_bytes()).hexdigest()
    state_path = LOCAL / "agents.json"
    state = read_json(state_path) if state_path.exists() else {"agents": {}}
    if state.get("platform_config_hash") != config_hash:
        subprocess.run([*COMPOSE, "restart", "api"], cwd=ROOT, check=True)
        subprocess.run([*COMPOSE, "up", "-d", "--wait", "--wait-timeout", "180", "api"], cwd=ROOT, check=True)
    api = LocalApi("http://localhost:3080")
    owner = authenticate(api, read_json(LOCAL / "demo.json"))
    owner_id = str(owner.get("id") or owner.get("_id"))
    schema = LocalApi("http://localhost:8091").request("GET", "/openapi.json")
    domain = schema["servers"][0]["url"]
    if domain != "http://support-eval:8090":
        raise BootstrapError("Evaluation Actions cannot point at the normal demo service.")
    spec = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    functions = action_functions(schema)
    for user in users:
        ident = user["id"]
        entry = state.setdefault("agents", {}).get(ident, {})
        desired = {"name": "评测专用 · " + ident, "description": "[support-agent-evaluation:v1] 独立合成数据评测；非真实用户身份联动。", "instructions": instructions(ident), "provider": "DeepSeek", "model": model, "model_parameters": MODEL_PARAMETERS, "recursion_limit": RECURSION_LIMIT, "category": "general"}
        if entry.get("agent_id"):
            current = api.request("GET", f"/api/agents/{entry['agent_id']}/expanded")
            if str(current.get("author")) != owner_id:
                raise BootstrapError("Evaluation Agent ownership mismatch.")
            changed = {key: value for key, value in desired.items() if current.get(key) != value}
            if changed:
                api.request("PATCH", f"/api/agents/{entry['agent_id']}", changed)
        else:
            created = api.request("POST", "/api/agents", {**desired, "tools": []})
            entry = {"agent_id": created["id"]}
            state["agents"][ident] = entry
            save_state(state_path, state)
        fingerprint = hashlib.sha256((spec + user["api_key"]).encode()).hexdigest()
        if entry.get("action_fingerprint") != fingerprint:
            payload = {"functions": functions, "metadata": {"domain": domain, "raw_spec": spec, "api_key": user["api_key"], "auth": {"type": "service_http", "authorization_type": "bearer"}}}
            if entry.get("action_id"):
                payload["action_id"] = entry["action_id"]
            _, action = api.request("POST", f"/api/agents/actions/{entry['agent_id']}", payload)
            entry.update(action_id=action["action_id"], action_fingerprint=fingerprint)
        saved = api.request("GET", f"/api/agents/{entry['agent_id']}/expanded")
        if len(saved.get("tools", [])) != len(functions):
            raise BootstrapError("Evaluation Agent does not expose the expected tool count.")
        stored_actions = api.request("GET", "/api/agents/actions")
        selected = [item for item in stored_actions if item.get("agent_id") == entry["agent_id"]]
        if len(selected) != 1 or selected[0].get("action_id") != entry["action_id"]:
            raise BootstrapError("Evaluation Agent must have exactly its isolated Action.")
        stored_metadata = selected[0].get("metadata", {})
        if stored_metadata.get("domain") != domain or stored_metadata.get("raw_spec") != spec:
            raise BootstrapError("Evaluation Action metadata differs from the isolated schema.")
        entry.update(tools=saved["tools"], actions=saved.get("actions"),
                     stored_metadata_sha256=hashlib.sha256(json.dumps(stored_metadata, sort_keys=True).encode()).hexdigest())
        save_state(state_path, state)
    state.update(platform_config_hash=config_hash, model=model, parameters=MODEL_PARAMETERS, recursion_limit=RECURSION_LIMIT, tool_count=len(functions), instructions_sha256=hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(), schema_sha256=hashlib.sha256(spec.encode()).hexdigest())
    save_state(state_path, state)
    print(json.dumps({"status": "ready", "evaluation_agents": len(users), "tools_per_agent": len(functions), "model_called": False}))
    return state


if __name__ == "__main__":
    try:
        prepare()
    except (BootstrapError, subprocess.CalledProcessError) as error:
        print(str(error) if isinstance(error, BootstrapError) else "Evaluation container preparation failed.")
        raise SystemExit(1)
