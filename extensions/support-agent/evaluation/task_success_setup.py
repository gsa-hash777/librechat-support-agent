"""Prepare seven private TSR Agents and a request-level spending guard; no LLM calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

SERVICE = Path(__file__).resolve().parents[1]
ROOT = SERVICE.parents[1]
sys.path.insert(0, str(SERVICE / "integration"))
from bootstrap import INSTRUCTIONS, BootstrapError, LocalApi, action_functions, authenticate, read_json, save_state
from evaluation.setup import ACTORS, credentials

LOCAL = ROOT / ".local/task-success-v1"
STATE_PATH = LOCAL / "agents.json"
PROVIDER = "DeepSeekTaskEval"
MODEL = "deepseek-flash"
MODEL_PARAMETERS = {"temperature": 0.2, "max_tokens": 1200, "maxContextTokens": 32768}
RECURSION_LIMIT = 12
DOMAIN = "http://support-eval:8090"
MARKER = "[support-agent-task-success:v1]"
COMPOSE = ["docker", "compose", "-f", "compose.support.yml", "-f", "compose.evaluation.yml", "-f", "compose.task-success.yml", "--profile", "evaluation"]
PROVIDER_CONFIG = """    - name: DeepSeekTaskEval
      apiKey: "${DEEPSEEK_API_KEY}"
      baseURL: http://support-eval-proxy:8093/v1
      models:
        default: [deepseek-flash]
        fetch: false
      titleConvo: false
      modelDisplayLabel: DeepSeek Task Evaluation (budget guarded)
      addParams:
        maxRetries: 0
        thinking:
          type: disabled
"""


def sha(value):
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def instructions(actor):
    if actor not in ACTORS:
        raise BootstrapError("Unknown evaluation identity.")
    lines = INSTRUCTIONS.splitlines()
    matches = [i for i, line in enumerate(lines) if line.startswith("每位 LibreChat 用户须通过 OAuth")]
    if len(matches) != 1:
        raise BootstrapError("Current identity instructions changed; review evaluation adaptation.")
    lines[matches[0]] = f"本次隔离评测的工具凭证固定绑定 {actor}，角色和组织仍须以 getCurrentSupportUser 返回值为准。本评测不测试个人 OAuth 登录授权。用户在独立服务台页面切换账号不会改变此 Agent 的工具身份，不能通过聊天声明获得其他权限。"
    return "\n".join(lines) + "\n"


def desired_agent(actor):
    return {"name": "任务成功率评测 · " + actor, "description": MARKER + " 自建历史回归集，隔离固定身份；非真实用户 OAuth。", "instructions": instructions(actor), "provider": PROVIDER, "model": MODEL, "model_parameters": dict(MODEL_PARAMETERS), "recursion_limit": RECURSION_LIMIT, "category": "general"}


def verify_provider_config():
    text = (SERVICE / "deploy/config.yaml").read_text(encoding="utf-8")
    if text.count(PROVIDER_CONFIG) != 1:
        raise BootstrapError("Dedicated budget-guarded provider configuration differs.")
    return sha(text)


def configure_provider():
    path = SERVICE / "deploy/config.yaml"
    text = path.read_text(encoding="utf-8")
    if PROVIDER_CONFIG in text:
        return False
    if "name: DeepSeekTaskEval" in text or text.count("\nactions:\n") != 1:
        raise BootstrapError("Provider configuration requires manual review.")
    path.write_text(text.replace("\nactions:\n", "\n" + PROVIDER_CONFIG + "actions:\n"), encoding="utf-8")
    return True


def validate_agents(api, state):
    """Read-only gate used immediately before the parent's paid run."""
    if state.get("provider") != PROVIDER or state.get("model") != MODEL or state.get("parameters") != MODEL_PARAMETERS or state.get("recursion_limit") != RECURSION_LIMIT:
        raise BootstrapError("Frozen TSR inference settings differ.")
    if state.get("platform_config_hash") != verify_provider_config():
        raise BootstrapError("Platform provider configuration changed after setup.")
    if state.get("instructions_sha256") != sha(INSTRUCTIONS):
        raise BootstrapError("Live business policy changed after setup.")
    if set(state.get("agents", {})) != set(ACTORS):
        raise BootstrapError("TSR must have all seven declared evaluation identities.")
    actions = api.request("GET", "/api/agents/actions")
    for actor, entry in state["agents"].items():
        saved = api.request("GET", f"/api/agents/{entry['agent_id']}/expanded")
        desired = desired_agent(actor)
        if any(saved.get(key) != value for key, value in desired.items()) or str(saved.get("author")) != state["owner_id"]:
            raise BootstrapError("TSR Agent policy, ownership, or inference settings changed.")
        if saved.get("tools") != entry["tools"] or len(saved.get("tools", [])) != 12 or saved.get("actions") != entry["actions"]:
            raise BootstrapError("TSR Agent tools changed.")
        selected = [item for item in actions if item.get("agent_id") == entry["agent_id"]]
        if len(selected) != 1 or selected[0].get("action_id") != entry["action_id"]:
            raise BootstrapError("TSR Agent must have exactly its isolated Action.")
        metadata = selected[0].get("metadata", {})
        if metadata.get("domain") != DOMAIN or sha(metadata.get("raw_spec", "")) != state["schema_sha256"] or sha(json.dumps(metadata, sort_keys=True)) != entry["stored_metadata_sha256"]:
            raise BootstrapError("TSR Action credentials or isolated schema changed.")
    if LocalApi("http://127.0.0.1:8093").request("GET", "/health").get("guard") != "task-success-request-guard-v1":
        raise BootstrapError("Budget guard is not available.")
    return {"validated_agents": 7, "tool_count": 12, "provider": PROVIDER, "model_called": False}


def prepare():
    users = credentials()
    LOCAL.mkdir(parents=True, exist_ok=True)
    subprocess.run([*COMPOSE, "up", "-d", "--build", "--no-deps", "--wait", "--wait-timeout", "120", "support-eval"], cwd=ROOT, check=True)
    subprocess.run([*COMPOSE, "up", "-d", "--no-deps", "--wait", "--wait-timeout", "120", "support-eval-proxy"], cwd=ROOT, check=True)
    if configure_provider():
        # Restart the existing container, retaining its operations/TLS mounts.
        subprocess.run(["docker", "restart", "librechat-support-api-1"], check=True, capture_output=True)
        for _ in range(90):
            try:
                LocalApi("http://127.0.0.1:3080").request("GET", "/api/config")
                break
            except BootstrapError:
                time.sleep(2)
        else:
            raise BootstrapError("LibreChat did not recover after provider reload.")
    api = LocalApi("http://127.0.0.1:3080")
    owner = authenticate(api, read_json(ROOT / ".local/evaluation/demo.json"))
    owner_id = str(owner.get("id") or owner.get("_id"))
    schema = LocalApi("http://127.0.0.1:8091").request("GET", "/openapi.json")
    if schema["servers"][0]["url"] != DOMAIN:
        raise BootstrapError("Evaluation tools point outside the isolated service.")
    functions = action_functions(schema)
    if len(functions) != 12:
        raise BootstrapError("Expected current twelve support tools.")
    spec = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    state = read_json(STATE_PATH) if STATE_PATH.exists() else {"agents": {}}
    state.update(owner_id=owner_id, platform_config_hash=verify_provider_config(), model=MODEL, provider=PROVIDER, parameters=MODEL_PARAMETERS, recursion_limit=RECURSION_LIMIT, tool_count=12, instructions_sha256=sha(INSTRUCTIONS), schema_sha256=sha(spec), credentials_file=".local/evaluation/demo.json", budget_bounds={"type": "request_level_transport_guard", "budget_path": ".local/task-success-v1/budget.json", "limit_cny": 5, "input_bound": "UTF-8 request bytes + 4096 framing tokens", "max_output_tokens": 1500, "all_requests_guarded": True}, proxy_base_url="http://127.0.0.1:8093/v1")
    for user in users:
        actor = user["id"]
        desired = desired_agent(actor)
        entry = state["agents"].get(actor)
        if entry:
            current = api.request("GET", f"/api/agents/{entry['agent_id']}/expanded")
            if str(current.get("author")) != owner_id or not str(current.get("description", "")).startswith(MARKER):
                raise BootstrapError("Refusing to edit an unrelated Agent.")
            changed = {key: value for key, value in desired.items() if current.get(key) != value}
            if changed:
                api.request("PATCH", f"/api/agents/{entry['agent_id']}", changed)
        else:
            saved = api.request("POST", "/api/agents", {**desired, "tools": []})
            entry = {"agent_id": saved["id"]}
            state["agents"][actor] = entry
            save_state(STATE_PATH, state)
        fingerprint = sha(spec + user["api_key"])
        if entry.get("action_fingerprint") != fingerprint:
            payload = {"functions": functions, "metadata": {"domain": DOMAIN, "raw_spec": spec, "api_key": user["api_key"], "auth": {"type": "service_http", "authorization_type": "bearer"}}}
            if entry.get("action_id"):
                payload["action_id"] = entry["action_id"]
            _, action = api.request("POST", f"/api/agents/actions/{entry['agent_id']}", payload)
            entry.update(action_id=action["action_id"], action_fingerprint=fingerprint)
        saved = api.request("GET", f"/api/agents/{entry['agent_id']}/expanded")
        actions = [item for item in api.request("GET", "/api/agents/actions") if item.get("agent_id") == entry["agent_id"]]
        if len(actions) != 1:
            raise BootstrapError("Unexpected TSR Action count.")
        entry.update(tools=saved["tools"], actions=saved.get("actions"), stored_metadata_sha256=sha(json.dumps(actions[0]["metadata"], sort_keys=True)), instructions_sha256=sha(desired["instructions"]))
        save_state(STATE_PATH, state)
    validate_agents(api, state)
    save_state(STATE_PATH, state)
    print(json.dumps({"status": "ready", "agents": 7, "tools_per_agent": 12, "provider": PROVIDER, "model_called": False}))
    return state


if __name__ == "__main__":
    try:
        prepare()
    except (BootstrapError, subprocess.CalledProcessError) as error:
        print(str(error) if isinstance(error, BootstrapError) else "Evaluation service preparation failed.")
        raise SystemExit(1)
