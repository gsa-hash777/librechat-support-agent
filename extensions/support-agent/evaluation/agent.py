"""Real LibreChat Agent evaluation, isolated fixtures, single concurrency, no auto approval.

Dry-run is the default. --run invokes the configured paid model; never retries a
model POST. A failed or interrupted attempt remains in the denominator.
"""
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request
import uuid

SERVICE = Path(__file__).resolve().parents[1]
ROOT = SERVICE.parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE / "integration"))
from bootstrap import BootstrapError, LOCAL_USER_AGENT, LocalApi, authenticate, read_json, save_state
from smoke import message_evidence, redact
from evaluation.knowledge_judge import JUDGE_VERSION, judge
from evaluation.usage import aggregate_usage, estimate_cost, normalize_usage
from evaluation.setup import COMPOSE, LOCAL, MODEL_PARAMETERS, RECURSION_LIMIT, instructions
from evaluation.report import build_report, render_html


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_digest():
    checksum = hashlib.sha256()
    for directory in (SERVICE / "app", SERVICE / "evaluation"):
        for path in sorted(directory.glob("*.py")):
            checksum.update(path.relative_to(SERVICE).as_posix().encode())
            checksum.update(path.read_bytes())
    for name in ("bootstrap.py", "smoke.py"):
        checksum.update((SERVICE / "integration" / name).read_bytes())
    for path in (SERVICE / "data/knowledge.json", SERVICE / "deploy/config.yaml", SERVICE / "requirements.txt", ROOT / "compose.support.yml", ROOT / "compose.evaluation.yml"):
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


@contextmanager
def exclusive_run():
    """Lock the shared fixture database across runner processes, even after crashes."""
    LOCAL.mkdir(parents=True, exist_ok=True)
    handle = (LOCAL / "run.lock").open("a+b")
    try:
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise BootstrapError("Another evaluation owns the isolated database; wait for it to finish.") from None
        yield
    finally:
        handle.close()


def turn_usage(assistants, events, model, error):
    message = assistants[-1] if assistants else None
    usage = normalize_usage(message=message, usage_events=events, model=model)
    rollup = normalize_usage(message=message, model=model)
    if events and rollup.get("input_tokens") is not None and rollup.get("output_tokens") is not None:
        if any(usage.get(key) != rollup.get(key) for key in ("input_tokens", "output_tokens")):
            # Prefer the persisted total if a replay missed SSE events. Never add both.
            if all(usage.get(key) is not None and rollup[key] >= usage[key] for key in ("input_tokens", "output_tokens")):
                usage = rollup
                usage["notes"].append("sse_and_persisted_totals_differ_using_persisted_rollup")
            else:
                usage = aggregate_usage([usage, normalize_usage(model=model)])
                usage["notes"].append("persisted_rollup_below_observed_usage_total_unknown")
    if error:
        usage = aggregate_usage([usage, normalize_usage(model=model)])
        usage["notes"].append("failed_turn_may_have_unreported_charge_total_unknown")
    return usage


def fixture_command(payload):
    completed = subprocess.run([*COMPOSE, "exec", "-T", "support-eval", "python", "/app/evaluation/lab.py"], cwd=ROOT, input=json.dumps(payload), text=True, capture_output=True, encoding="utf-8", timeout=30)
    if completed.returncode:
        raise BootstrapError("Isolated fixture control failed; normal support data was not targeted.")
    return json.loads(completed.stdout)


def render_case(case, ids):
    def render(value):
        if isinstance(value, str):
            for name, replacement in ids.items():
                value = value.replace("{" + name + "}", replacement)
            return value
        if isinstance(value, list):
            return [render(item) for item in value]
        if isinstance(value, dict):
            return {key: render(item) for key, item in value.items()}
        return value
    return {**render(case), "target_ticket_id": ids["ticket_id"]}


def stream(api, start, deadline, events):
    query = urlencode({"resume": "true", "generationCreatedAt": start["generationCreatedAt"]})
    request = Request(api.base + f"/api/agents/chat/stream/{start['streamId']}?{query}", headers={"Accept": "text/event-stream", "Origin": api.base, "Authorization": "Bearer " + api.token, "User-Agent": LOCAL_USER_AGENT})
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BootstrapError("Agent evaluation timed out.")
        with api.opener.open(request, timeout=remaining) as response:
            # urllib HTTPResponse keeps the connected socket beneath its buffered
            # reader. One timeout at open() would also cap a slow first token and
            # would not shrink after a heartbeat. Do not silently accept a reader
            # whose blocking timeout we cannot control.
            sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
            if not callable(getattr(sock, "settimeout", None)):
                raise BootstrapError("Agent stream does not expose a deadline-aware socket.")
            data = []
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BootstrapError("Agent evaluation timed out.")
                if response.fp is None:
                    break
                sock.settimeout(remaining)
                line = response.readline()
                if time.monotonic() >= deadline:
                    raise BootstrapError("Agent evaluation timed out.")
                if not line:
                    break
                line = line.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif not line and data:
                    item = json.loads("\n".join(data))
                    data = []
                    if item.get("event") == "on_token_usage":
                        events.append(item.get("data", {}))
                    if item.get("event") == "on_pending_action":
                        raise BootstrapError("Agent requested an approval; evaluation never grants model approvals.")
                    if item.get("final"):
                        return
    except HTTPError as error:
        if error.code != 404:
            raise BootstrapError(f"Agent stream HTTP {error.code}.") from None
    except (URLError, TimeoutError, OSError, UnicodeError, ValueError):
        if time.monotonic() >= deadline:
            raise BootstrapError("Agent evaluation timed out.") from None
        raise BootstrapError("Agent stream transport failed.") from None


def turn(api, agent_id, prompt, conversation_id, parent_id, previous_ids, timeout, model):
    started = None
    events = []
    error = None
    began = time.monotonic()
    drained = True
    try:
        started = api.request("POST", "/api/agents/chat", {"endpoint": "agents", "agent_id": agent_id, "text": prompt,
            "conversationId": conversation_id or "new", "parentMessageId": parent_id or "00000000-0000-0000-0000-000000000000", "messageId": str(uuid.uuid4()), "clientRequestId": "eval-" + uuid.uuid4().hex})
        stream(api, started, began + timeout, events)
        status = api.request("GET", "/api/agents/chat/status/" + started["conversationId"])
        if status.get("active"):
            raise BootstrapError("Agent run did not finish within its stream.")
    except (BootstrapError, KeyError, TypeError) as exc:
        error = str(exc) if isinstance(exc, BootstrapError) else "Unexpected Agent run response."
        drained = False
        if started and started.get("streamId"):
            try:
                api.request("POST", "/api/agents/chat/abort", {"streamId": started["streamId"], "generationCreatedAt": started["generationCreatedAt"]})
                for _ in range(20):
                    if not api.request("GET", "/api/agents/chat/status/" + started["conversationId"]).get("active"):
                        drained = True
                        break
                    time.sleep(0.25)
            except BootstrapError:
                pass
    messages = []
    if started and started.get("conversationId"):
        try:
            messages = api.request("GET", "/api/messages/" + started["conversationId"])
        except BootstrapError:
            error = error or "Unable to read persisted Agent evidence."
    fresh = [message for message in messages if message.get("messageId") not in previous_ids]
    answers, calls = message_evidence(fresh)
    assistants = [message for message in fresh if not message.get("isCreatedByUser")]
    content_errors = any(isinstance(block, dict) and block.get("type") == "error"
                         for message in assistants
                         for block in (message.get("content") if isinstance(message.get("content"), list) else []))
    if any(message.get("error") for message in assistants) or content_errors:
        error = error or "Persisted Agent response is marked as failed."
    if not answers:
        error = error or "Agent finished without a final text response."
    usage = turn_usage(assistants, events, model, error)
    return {
        "prompt": prompt, "answers": answers[-1:], "intermediate_answers": answers[:-1], "tool_calls": calls, "usage": usage, "usage_events": events,
        "elapsed_seconds": time.monotonic() - began, "execution_error": error, "drained": drained,
        "conversation_id": started.get("conversationId") if started else None,
        "parent_id": assistants[-1].get("messageId") if assistants else parent_id,
        "message_ids": [message.get("messageId") for message in messages],
    }


def validate_agents(api, state, actors):
    actions = api.request("GET", "/api/agents/actions")
    for actor in actors:
        entry = state['agents'][actor]
        saved = api.request("GET", f"/api/agents/{entry['agent_id']}/expanded")
        if (saved.get("instructions") != instructions(actor) or saved.get("model") != state["model"] or saved.get("provider") != "DeepSeek"
                or saved.get("model_parameters") != MODEL_PARAMETERS or saved.get("recursion_limit") != RECURSION_LIMIT
                or len(saved.get("tools", [])) != state.get("tool_count") or saved.get("tools") != entry.get("tools")
                or saved.get("actions") != entry.get("actions")):
            raise BootstrapError("Evaluation Agent configuration changed; rerun evaluation/setup.py before evaluating.")
        selected = [item for item in actions if item.get("agent_id") == entry["agent_id"]]
        if len(selected) != 1 or selected[0].get("action_id") != entry["action_id"]:
            raise BootstrapError("Evaluation Action binding changed.")
        metadata = selected[0].get("metadata", {})
        fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        if (metadata.get("domain") != "http://support-eval:8090" or fingerprint != entry.get("stored_metadata_sha256")
                or hashlib.sha256(metadata.get("raw_spec", "").encode()).hexdigest() != state["schema_sha256"]):
            raise BootstrapError("Evaluation Action schema, domain, or credential binding changed.")


def renew_session(api, demo, forbidden, timeout):
    # Read expiry only to schedule login; the server still validates identity.
    try:
        payload = api.token.split(".")[1]
        expiry = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["exp"]
    except (ValueError, IndexError, KeyError, TypeError):
        raise BootstrapError("Evaluation session has no readable expiry.") from None
    if expiry - time.time() < 2 * timeout + 60:
        authenticate(api, demo)
        forbidden.append(api.token)


def execute_case(api, state, case, run_id, prices, timeout):
    fixture = fixture_command({"action": "prepare", "case": case, "run_id": run_id})
    rendered = render_case(case, fixture["ids"])
    prompts = rendered.get("turns") or [rendered["prompt"]]
    turns = []
    conversation_id, parent_id, previous_ids = None, None, set()
    began = time.monotonic()
    for prompt in prompts:
        item = turn(api, state["agents"][case["actor"]]["agent_id"], prompt, conversation_id, parent_id, previous_ids, timeout, state["model"])
        turns.append(item)
        conversation_id, parent_id = item["conversation_id"], item["parent_id"]
        previous_ids = set(item["message_ids"])
        if item["execution_error"]:
            break
    after = fixture_command({"action": "snapshot"})
    evidence = {"answers": [answer for item in turns for answer in item["answers"]], "intermediate_answers": [answer for item in turns for answer in item.get("intermediate_answers", [])], "tool_calls": [call for item in turns for call in item["tool_calls"]], "execution_error": next((item["execution_error"] for item in turns if item["execution_error"]), None)}
    usage = aggregate_usage([item["usage"] for item in turns])
    return {"case_id": case["id"], "category": case["category"], "actor": case["actor"], "evidence": evidence,
        "judgement": judge(rendered, evidence, fixture["snapshot"], after, trusted_source_base_url="http://localhost:8091"), "usage": usage, "cost": estimate_cost(usage, prices),
        "elapsed_seconds": round(time.monotonic() - began, 3), "error": evidence["execution_error"],
        "conversation_id": conversation_id, "conversation_url": api.base + "/c/" + conversation_id if conversation_id else None,
        "turns": turns, "case": rendered, "fixture_before": fixture["snapshot"], "fixture_after": after,
        "drained": all(item["drained"] for item in turns)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["dev", "test", "repair", "followup", "verify"], default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-budget-cny", type=float)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.max_budget_cny is None:
        args.max_budget_cny = read_json(args.resume / "manifest.json")["max_budget_cny"] if args.resume else 5.0
    dataset_file = SERVICE / "evaluation" / {"dev": "agent_dev_cases.json", "test": "agent_cases.json", "repair": "agent_repair_cases.json", "followup": "agent_followup_cases.json", "verify": "agent_verify_cases.json"}[args.dataset]
    document = read_json(dataset_file)
    cases = document["cases"] if isinstance(document, dict) else document
    if args.limit:
        cases = cases[:args.limit]
    if not cases or not math.isfinite(args.max_budget_cny) or args.max_budget_cny <= 0 or args.timeout < 10 or (args.limit is not None and args.limit < 1):
        raise BootstrapError("Evaluation count, budget, and timeout must be positive.")
    if not args.run:
        print(json.dumps({"mode": "dry_run", "dataset": args.dataset, "planned_cases": len(cases), "dataset_sha256": digest(dataset_file), "model_calls": 0, "note": "Add --run to invoke the paid model."}))
        return 0
    with exclusive_run():
        return run_paid(args, dataset_file, cases)


def run_paid(args, dataset_file, cases):
    state = read_json(LOCAL / "agents.json")
    raw_prices = read_json(SERVICE / "evaluation/prices.json")
    prices = {**raw_prices, "rates": {"off_peak": raw_prices["off_peak"], "peak": raw_prices["peak"]}}
    api = LocalApi("http://localhost:3080")
    demo = read_json(LOCAL / "demo.json")
    authenticate(api, demo)
    validate_agents(api, state, {case["actor"] for case in cases})
    forbidden = [api.token, demo["password"]]
    for user in read_json(LOCAL / "users.json")["users"]:
        forbidden.extend([user["password"], user["api_key"]])
    # Also redact normal demo secrets defensively, even though the lab never receives them.
    for user in read_json(ROOT / ".local/users.json")["users"]:
        forbidden.extend([user["password"], user["api_key"]])
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        name, separator, value = line.partition("=")
        if separator and any(marker in name.upper() for marker in ("KEY", "SECRET", "TOKEN", "PASSWORD")):
            forbidden.append(value.strip().strip("\"'"))
    reference_file = SERVICE / "evaluation/reference_answers_v2.json"
    manifest = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6], "created_at": datetime.now(timezone.utc).isoformat(), "dataset": args.dataset,
        "planned_cases": len(cases), "case_ids": [case["id"] for case in cases], "judge_version": JUDGE_VERSION, "dataset_sha256": digest(dataset_file), "code_sha256": code_digest(), "prices_sha256": digest(SERVICE / "evaluation/prices.json"),
        "model": state["model"], "provider": "DeepSeek", "parameters": state["parameters"], "recursion_limit": state["recursion_limit"], "instructions_sha256": state["instructions_sha256"], "schema_sha256": state["schema_sha256"],
        "concurrency": 1, "max_budget_cny": args.max_budget_cny, "timeout_seconds": args.timeout, "synthetic": True, "frozen_before_run": True, "status": "running", "stop_reason": None,
        "corpus_sha256": digest(SERVICE / "data/knowledge.json"), "platform_config_sha256": digest(SERVICE / "deploy/config.yaml"), "python_version": sys.version,
        "reference_sha256": digest(reference_file) if reference_file.exists() else None,
        "latency_scope": "Includes model turns, tool calls, and final DB snapshot; excludes fixture reset, login, and initial validation.",
        "budget_note": "Estimated peak/no-cache cost stopping threshold; an in-flight task can cross the threshold. Unknown usage stops the run.", "pricing_source": raw_prices["source"]}
    results = []
    if args.resume:
        folder = args.resume.resolve()
        existing = read_json(folder / "manifest.json")
        for key in ("dataset_sha256", "code_sha256", "prices_sha256", "instructions_sha256", "schema_sha256", "case_ids", "model", "parameters", "max_budget_cny", "recursion_limit", "timeout_seconds", "corpus_sha256", "platform_config_sha256", "reference_sha256"):
            if existing.get(key) != manifest[key]:
                raise BootstrapError("Resume refused because the frozen evaluation configuration changed.")
        if existing.get("pending_case"):
            raise BootstrapError("An interrupted model attempt has unknown outcome; inspect it before starting a separate new run. It will not be silently retried.")
        manifest = existing
        manifest.update(status="running", stop_reason=None)
        results = read_json(folder / "results.json")
    else:
        folder = LOCAL / "results" / manifest["run_id"]
    folder.mkdir(parents=True, exist_ok=True)
    save_state(folder / "manifest.json", manifest)
    completed = {item["case_id"] for item in results}
    for case in cases:
        if case["id"] in completed:
            continue
        reference_hash = digest(reference_file) if reference_file.exists() else None
        if code_digest() != manifest["code_sha256"] or digest(dataset_file) != manifest["dataset_sha256"] or reference_hash != manifest.get("reference_sha256"):
            manifest.update(status="stopped", stop_reason="frozen_sources_changed")
            break
        spent = sum(item["cost"].get("conservative_budget") or 0 for item in results)
        if any(item["cost"].get("conservative_budget") is None for item in results) or any(not item.get("drained", True) for item in results):
            manifest.update(status="stopped", stop_reason="unknown_usage_or_active_request")
            break
        if spent >= args.max_budget_cny:
            manifest.update(status="stopped", stop_reason="estimated_budget_reached")
            break
        renew_session(api, demo, forbidden, args.timeout)
        manifest["pending_case"] = case["id"]
        save_state(folder / "manifest.json", manifest)
        began = time.monotonic()
        try:
            result = execute_case(api, state, case, manifest["run_id"], prices, args.timeout)
        except (BootstrapError, subprocess.TimeoutExpired, OSError, ValueError, KeyError, TypeError):
            # A paid request may already have happened. Keep this attempt failed/unknown,
            # and retain pending_case so recovery cannot silently retry or reset fixtures.
            usage = normalize_usage(model=state["model"])
            evidence = {"answers": [], "tool_calls": [], "execution_error": "Attempt evidence incomplete; no automatic retry."}
            result = {"case_id": case["id"], "category": case["category"], "actor": case["actor"],
                      "evidence": evidence, "judgement": judge(case, evidence, {}, {}), "usage": usage,
                      "cost": estimate_cost(usage, prices), "elapsed_seconds": time.monotonic() - began,
                      "error": evidence["execution_error"], "drained": False, "case": case, "outcome_unknown": True}
        result = redact(result, forbidden)
        results.append(result)
        save_state(folder / (case["id"] + ".json"), result)
        save_state(folder / "results.json", results)
        if not result.get("outcome_unknown"):
            manifest.pop("pending_case", None)
        save_state(folder / "manifest.json", manifest)
        report = build_report(manifest, results)
        save_state(folder / "report.json", report)
        (folder / "report.html").write_text(render_html(report), encoding="utf-8")
        print(json.dumps({"completed": len(results), "planned": len(cases), "case": case["id"], "contract": result["judgement"]["contract_pass"], "strict": result["judgement"]["strict_rule_pass"], "seconds": result["elapsed_seconds"], "cost_ceiling_cny": result["cost"].get("conservative_budget")}), flush=True)
        if result["error"]:
            manifest.update(status="stopped", stop_reason="execution_error_requires_review")
            break
    if len(results) == len(cases):
        manifest.update(status="completed", stop_reason=None)
    save_state(folder / "manifest.json", manifest)
    report = build_report(manifest, results)
    save_state(folder / "report.json", report)
    (folder / "report.html").write_text(render_html(report), encoding="utf-8")
    save_state(LOCAL / "latest.json", {"run_directory": str(folder), "dataset": args.dataset, "status": manifest["status"]})
    print(json.dumps({"status": manifest["status"], "report": str(folder / "report.html"), "attempted": len(results), "planned": len(cases)}))
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BootstrapError, subprocess.TimeoutExpired, KeyError, ValueError) as error:
        print(json.dumps({"status": "stopped", "message": str(error) if isinstance(error, BootstrapError) else "Evaluation encountered a configuration or evidence error; no automatic model retry was made."}))
        raise SystemExit(2)
