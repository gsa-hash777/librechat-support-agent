"""One-attempt support Agent regression, with a separate CNY 5 shared ledger.

The task success oracle is frozen before paid calls. This is a seen synthetic
project regression, not an official tau-bench score. Dry-run is the default.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, build_opener, ProxyHandler
import uuid

SERVICE = Path(__file__).resolve().parents[1]
ROOT = SERVICE.parents[1]
LOCAL = ROOT / ".local/task-success-v1"
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE / "integration"))
from bootstrap import BootstrapError, LocalApi, NoRedirect, authenticate, read_json, save_state
from smoke import redact, SECRET_FIELD
from evaluation.agent import exclusive_run, fixture_command, render_case, renew_session, turn
from evaluation.usage import aggregate_usage, estimate_cost, normalize_usage

LIMIT = Decimal("5")
MODEL = "deepseek-flash"
DATA_SHA = "befcb7171e7d9feb7b2d5766fc750a043ff5a51f6be73055caa48e9921c9830b"
JUDGE_OUTPUT = 1500
JUDGE_INPUT_BYTES = 100000
REVIEW_SEED = "support-task-success-v1-review-20261112"
RUNNER_RELATIVE = "extensions/support-agent/evaluation/task_success_run.py"
ORACLE_RELATIVE = "extensions/support-agent/evaluation/task_success.py"
SOURCE_SCORER_RELATIVE = "extensions/support-agent/evaluation/knowledge_judge.py"
PROXY_RELATIVE = "extensions/support-agent/evaluation/task_success_proxy.py"
SEMANTIC_PROTOCOL = "support-task-success-semantic-v1.1"
FORMAT_SUFFIX_SHA256 = "77b999b25ed4706d5e3be28e68c3e7e3e7388c07e62253e5cbdc0bf746040bfa"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Budget:
    """Read-only view of the proxy-owned durable budget ledger."""
    def __init__(self, path):
        self.path = Path(path)
        self.refresh()

    def refresh(self):
        self.state = read_json(self.path)
        from evaluation.task_success_proxy import authorized_limit, GuardError
        try:
            authorized_limit(self.state)
        except GuardError:
            raise BootstrapError("The proxy budget must match the recorded user-authorized ceiling.") from None
        return self.state

    @property
    def occupied(self):
        self.refresh()
        return sum((Decimal(str(row.get("cost_cny", row["reserved_cny"])))
                    for row in self.state["attempts"]), Decimal(0))

    def settled(self):
        self.refresh()
        return all(row["status"] == "settled" for row in self.state["attempts"])


def price_snapshot():
    return {"schema_version": 1, "model": MODEL, "currency": "CNY",
            "documented_model_version": "DeepSeek-V4.1-Flash-1017", "checked_at": "2026-11-12",
            "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
            "rates": {"off_peak": {"input_per_million": 1, "cache_hit_per_million": .02, "output_per_million": 4},
                      "peak": {"input_per_million": 2, "cache_hit_per_million": .04, "output_per_million": 8}},
            "note": "Peak, all-input-cache-miss reservation and settlement; estimate, not an invoice."}


def reserve_cost(input_tokens, output_tokens, calls=1):
    return float((Decimal(input_tokens) * 2 + Decimal(output_tokens) * 8) * calls / 1000000)


def secrets_and_key():
    values, key = [], None
    for path in (ROOT / ".local/evaluation/demo.json", ROOT / ".local/evaluation/users.json",
                 ROOT / ".local/demo.json", ROOT / ".local/users.json"):
        if path.exists():
            def walk(value):
                if isinstance(value, dict):
                    for name, child in value.items():
                        if SECRET_FIELD.search(name) and isinstance(child, str):
                            values.append(child)
                        else:
                            walk(child)
                elif isinstance(value, list):
                    for child in value:
                        walk(child)
            walk(read_json(path))
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        name, separator, value = line.strip().partition("=")
        if separator and not name.startswith("#"):
            value = value.strip().strip("\"'")
            if SECRET_FIELD.search(name):
                values.append(value)
            if name == "DEEPSEEK_API_KEY":
                key = value
    if not key:
        raise BootstrapError("DeepSeek key is not configured locally.")
    return values, key


def judge_once(messages, key, budget, run_id, case_id, prices):
    # UTF-8 bytes conservatively bound tokenizer content; overhead covers the
    # two-message wrapper. Oversize packets are unscored, never truncated.
    input_bound = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 4096
    if input_bound > JUDGE_INPUT_BYTES:
        return {"status": "unscored", "reasons": ["judge_input_exceeds_frozen_admission"]}, None
    before_requests = len(budget.refresh()["attempts"])
    payload = {"model": MODEL, "messages": messages, "temperature": 0,
               "max_tokens": JUDGE_OUTPUT, "thinking": {"type": "disabled"},
               "response_format": {"type": "json_object"}, "stream": False}
    request = Request("http://127.0.0.1:8093/v1/chat/completions",
                      data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers={"Content-Type": "application/json", "Authorization": "Bearer " + key,
                               "X-Eval-Run": run_id, "X-Eval-Case": case_id, "X-Eval-Stage": "semantic_judge"}, method="POST")
    began = time.monotonic()
    try:
        # No SDK retries and no redirect forwarding of the credential.
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=90) as response:
            data = json.loads(response.read(2_000_000))
        usage = normalize_usage(usage_events=[data.get("usage", {})], model=MODEL)
        cost = estimate_cost(usage, prices)
        settled = budget.settled() and len(budget.state["attempts"]) > before_requests
        record = {"usage": usage, "cost": cost, "elapsed_seconds": time.monotonic() - began,
                  "response_model": data.get("model"), "response_id": data.get("id"), "settled": settled}
        from evaluation.task_success import parse_judge_response, redact as redact_judge
        content = data["choices"][0]["message"]["content"]
        record["raw_response_redacted"] = redact_judge(content)
        record["finish_reason"] = data["choices"][0].get("finish_reason")
        record["response_shape"] = judge_response_shape(content)
        verdict = parse_judge_response(content)
        if data["choices"][0].get("finish_reason") != "stop":
            verdict = {"status": "unscored", "reasons": ["judge_response_not_complete"]}
        return verdict, record
    except Exception:
        # Transport or parse exceptions may follow a billed request. Do not
        # log provider payloads, release a charge, or resubmit automatically.
        settled = budget.settled() and len(budget.state["attempts"]) > before_requests
        return {"status": "unscored", "reasons": ["judge_request_or_evidence_error_no_retry"]}, {
            "settled": settled, "elapsed_seconds": time.monotonic() - began}


def judge_response_shape(content):
    """Structure-only diagnosis; no generated prose, secret or attack text."""
    from evaluation.task_success import DIMENSIONS
    report = {"content_type": type(content).__name__, "content_characters": len(content) if isinstance(content, str) else None}
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        return {**report, "valid_json": False}
    report.update(valid_json=True, root_type=type(value).__name__)
    if not isinstance(value, dict):
        return report
    report["top_level_known_keys"] = sorted(set(value) & {"dimensions", "reasons", "status"})
    report["unexpected_top_level_key_count"] = len(set(value) - {"dimensions", "reasons"})
    dimensions = value.get("dimensions")
    report["dimensions_type"] = type(dimensions).__name__
    report["reasons_type"] = type(value.get("reasons")).__name__
    report["reasons_count"] = len(value["reasons"]) if isinstance(value.get("reasons"), list) else None
    if isinstance(dimensions, dict):
        report["missing_dimensions"] = sorted(set(DIMENSIONS) - dimensions.keys())
        report["unexpected_dimension_count"] = len(dimensions.keys() - set(DIMENSIONS))
        report["dimensions"] = {}
        for name in DIMENSIONS:
            item = dimensions.get(name)
            fields = {"status", "reason", "evidence"}
            shape = {"type": type(item).__name__}
            if isinstance(item, dict):
                shape.update(known_keys=sorted(item.keys() & fields), unexpected_key_count=len(item.keys() - fields),
                             status=item.get("status") if item.get("status") in ("pass", "fail", "uncertain") else "invalid_or_missing",
                             reason_type=type(item.get("reason")).__name__, evidence_type=type(item.get("evidence")).__name__,
                             reason_characters=len(item["reason"]) if isinstance(item.get("reason"), str) else None,
                             evidence_characters=len(item["evidence"]) if isinstance(item.get("evidence"), str) else None)
            report["dimensions"][name] = shape
    return report


def source_paths():
    paths = list((SERVICE / "app").glob("*.py"))
    paths += [SERVICE / "evaluation" / name for name in (
        "agent.py", "agent_cases.json", "task_success.py", "task_success_references.json",
        "task_success_run.py", "task_success_setup.py", "task_success_proxy.py", "usage.py", "lab.py", "judge.py",
        "knowledge_judge.py", "setup.py", "reference_answers_v2.json")]
    paths += [SERVICE / "integration" / name for name in ("bootstrap.py", "smoke.py")]
    paths += [SERVICE / "data/knowledge.json", SERVICE / "deploy/config.yaml", SERVICE / "requirements.txt",
              ROOT / "compose.support.yml", ROOT / "compose.evaluation.yml", ROOT / "compose.task-success.yml", LOCAL / "agents.json"]
    return sorted(set(paths))


def fingerprints():
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in source_paths()}


def validate_result_prefix(results, cases):
    identifiers = [row.get("case_id") for row in results]
    if identifiers != [case["id"] for case in cases[:len(results)]] or len(results) > len(cases):
        raise BootstrapError("Prior results must be an unchanged, unique prefix of the original 100 tasks.")
    if any(not isinstance(row.get("outcome"), dict) or row["outcome"].get("status") not in {"pass", "fail", "unscored"} for row in results):
        raise BootstrapError("Prior task outcomes are incomplete; refusing automatic execution replay.")


def source_change_review(previous, current, frozen_folder, review=None):
    """Only runner repair is implicit; parser-only repair needs exact approval."""
    if set(previous) != set(current):
        raise BootstrapError("Continuation cannot change the frozen source inventory.")
    changes = []
    for name in sorted(previous):
        source = frozen_folder / name
        if digest(source) != previous[name]:
            raise BootstrapError("Original frozen source archive changed.")
        if previous[name] == current[name]:
            continue
        item = {"path": name, "before_sha256": previous[name], "after_sha256": current[name]}
        if name == RUNNER_RELATIVE:
            item["reason"] = "Transparent run continuation/checkpoint support and capture of redacted semantic response structure; no repeated Agent executions or changed task requirements."
        elif name == ORACLE_RELATIVE:
            if (not isinstance(review, dict) or review.get("path") != name or review.get("before_sha256") != previous[name]
                    or review.get("after_sha256") != current[name] or review.get("semantic_rules_unchanged") is not True
                    or not isinstance(review.get("reason"), str) or not review["reason"].strip()):
                raise BootstrapError("Parser changes require an explicit exact-hash format-only review.")
            before_tree = ast.parse(source.read_text(encoding="utf-8"))
            after_tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            if review.get("kind") == "semantic-format-v1.1":
                def without_prompt(tree):
                    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                                   and any(isinstance(target, ast.Name) and target.id == "JUDGE_SYSTEM" for target in node.targets)]
                    if len(assignments) != 1:
                        raise BootstrapError("Expected exactly one semantic system prompt.")
                    value = ast.literal_eval(assignments[0].value)
                    tree.body = [node for node in tree.body if node not in assignments]
                    return value, ast.dump(tree, include_attributes=False)
                before_prompt, before_rest = without_prompt(before_tree)
                after_prompt, after_rest = without_prompt(after_tree)
                suffix = after_prompt[len(before_prompt):] if after_prompt.startswith(before_prompt) else ""
                if before_rest != after_rest or hashlib.sha256(suffix.encode()).hexdigest() != FORMAT_SUFFIX_SHA256:
                    raise BootstrapError("Semantic v1.1 may only append the reviewed exact JSON format suffix; rubric, hard checks and parser must remain identical.")
                item["reason"] = review["reason"]
                item["format_only_review"] = review
                changes.append(item)
                continue
            def without_parser(tree):
                functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_judge_response"]
                if len(functions) != 1:
                    raise BootstrapError("Expected exactly one semantic response parser.")
                tree.body = [node for node in tree.body if node not in functions]
                return ast.dump(tree, include_attributes=False)
            if without_parser(before_tree) != without_parser(after_tree):
                raise BootstrapError("Format repair changed grading, references or judge prompt outside the response parser.")
            item["reason"] = review["reason"]
            item["format_only_review"] = review
        elif name == SOURCE_SCORER_RELATIVE:
            repair = review.get("source_scorer_repair") if isinstance(review, dict) else None
            if (not isinstance(review, dict) or review.get("kind") != "semantic-format-v1.1" or not isinstance(repair, dict)
                    or repair.get("path") != name or repair.get("before_sha256") != previous[name]
                    or repair.get("after_sha256") != current[name]
                    or repair.get("semantic_rules_unchanged") is not True
                    or not isinstance(repair.get("reason"), str) or not repair["reason"].strip()):
                raise BootstrapError("Source-syntax repairs require an explicit exact-hash review and uniform offline regrading.")
            item["reason"] = repair["reason"]
            item["source_scorer_repair"] = repair
        elif name == PROXY_RELATIVE:
            if (not isinstance(review, dict) or review.get("kind") != "budget-amendment-v1"
                    or review.get("path") != name or review.get("before_sha256") != previous[name]
                    or review.get("after_sha256") != current[name] or review.get("to_limit_cny") != 5.5
                    or review.get("from_limit_cny") != 5 or not review.get("reason")):
                raise BootstrapError("Proxy changes require the exact-hash user-authorized budget amendment review.")
            item["reason"] = review["reason"]
            item["budget_amendment_review"] = review
        else:
            raise BootstrapError("Continuation cannot change Agent policy, dataset, references, tool service, model settings or safety rules.")
        changes.append(item)
    return changes


def inspect_continuation(source_folder, cases, frozen, prices, budget, review=None):
    """Read/verify only; the old run is never written or regraded."""
    source_folder = source_folder.resolve()
    if not source_folder.is_relative_to((LOCAL / "results").resolve()):
        raise BootstrapError("Continuation source must be inside this protocol's result directory.")
    prior = read_json(source_folder / "manifest.json")
    results = read_json(source_folder / "results.json")
    if prior.get("pending_case") or prior.get("pending_judge_case"):
        raise BootstrapError("A prior case is unresolved; continuation must not silently retry it.")
    if prior.get("status") not in {"stopped", "checkpoint_stopped"}:
        raise BootstrapError("Continuation requires a stopped prior run.")
    if (prior.get("protocol") != "support-task-success-v1" or prior.get("prices") != prices
            or prior.get("case_ids") != [case["id"] for case in cases] or prior.get("max_budget_cny") != 5
            or prior.get("judge_model") != MODEL or prior.get("judge_temperature") != 0
            or prior.get("judge_max_tokens") != JUDGE_OUTPUT or prior.get("judge_max_input_bytes") != JUDGE_INPUT_BYTES):
        raise BootstrapError("Continuation must preserve the original tasks, judge settings, prices and budget.")
    validate_result_prefix(results, cases)
    changes = source_change_review(prior["source_sha256"], frozen, source_folder / "frozen", review)
    artifacts = {"manifest.json": digest(source_folder / "manifest.json"), "results.json": digest(source_folder / "results.json")}
    for row in results:
        name = row["case_id"] + ".json"
        # An inherited result can point to an immutable artifact in an older
        # lineage; the flattened results themselves must still be unchanged.
        if (source_folder / name).exists():
            if read_json(source_folder / name) != row:
                raise BootstrapError("Prior per-case result differs from its aggregate.")
            artifacts[name] = digest(source_folder / name)
        elif row["case_id"] not in prior.get("continuation", {}).get("inherited_case_ids", []):
            raise BootstrapError("A prior case lacks its preserved execution artifact.")
    prior_report = read_json(source_folder / "report.json")
    old_attempts = prior_report["budget"]["attempts"]
    budget.refresh()
    if budget.state["attempts"][:len(old_attempts)] != old_attempts:
        raise BootstrapError("Shared budget ledger no longer contains the prior run's exact paid-attempt prefix.")
    if len(budget.state["attempts"]) != len(old_attempts):
        raise BootstrapError("New paid calls exist after the prior report; inspect their evidence before continuation.")
    return prior, results, {"source_run_id": prior["run_id"], "source_directory": str(source_folder),
            "source_artifacts_sha256": artifacts, "inherited_case_ids": [row["case_id"] for row in results],
            "inherited_execution_count": len(results), "source_changes": changes,
            "prior_semantic_outcomes_preserved": True, "agent_executions_repeated": 0,
            "budget_ledger_path": str(budget.path.resolve()), "budget_attempts_at_start": len(old_attempts),
            "budget_occupied_at_start_cny": float(budget.occupied), "format_repair_review": review}


def archive_sources(folder):
    for path in source_paths():
        destination = folder / "frozen" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


def make_continuation(source_folder, cases, frozen, prices, budget, timeout, review=None):
    prior, results, lineage = inspect_continuation(source_folder, cases, frozen, prices, budget, review)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-continue-" + uuid.uuid4().hex[:6]
    folder = LOCAL / "results" / run_id
    claim_dir = LOCAL / "continuation-claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    claim_path = claim_dir / (hashlib.sha256(str(source_folder.resolve()).encode()).hexdigest() + ".json")
    # The claim is outside the old run, so old evidence remains byte-for-byte
    # intact. A second branch cannot repeat the same unattempted suffix.
    with claim_path.open("x", encoding="utf-8") as claim:
        json.dump({"source_directory": str(source_folder.resolve()), "destination": str(folder.resolve()),
                   "source_results_sha256": lineage["source_artifacts_sha256"]["results.json"]}, claim, indent=2)
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {**prior, "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_sha256": frozen, "continuation": lineage, "status": "running", "stop_reason": None,
                "timeout_seconds": timeout, "frozen_before_run": True,
                "agent_latency_timing_unchanged": True, "new_executions_this_run": 0}
    manifest.pop("pending_case", None)
    manifest.pop("checkpoint", None)
    if review and review.get("kind") == "budget-amendment-v1":
        amendment = budget.state.get("amendments", [])
        if (len(amendment) != 1 or review.get("amendment") != amendment[0]
                or [row["case_id"] for row in results][-1:] != ["P05"]
                or [case["id"] for case in cases[len(results):]] != ["P06", "P07", "P08", "P09", "P10"]):
            raise BootstrapError("The budget extension may only continue the five unattempted cases after P05.")
        results = deepcopy(results)
        interrupted = results[-1]
        if interrupted.get("stop_reason") != "agent_runtime_or_usage_requires_review" or interrupted.get("judge_record"):
            raise BootstrapError("Expected the preserved incomplete P05 execution without a semantic regrade.")
        interrupted["outcome_before_administrative_classification"] = deepcopy(interrupted["outcome"])
        interrupted["administrative_status"] = "admin_budget_interruption"
        interrupted["outcome"] = {**interrupted["outcome"], "status": "unscored", "success": None,
                                  "administrative_reason": "HTTP 402 evaluation_budget_guard/budget_exhausted; Agent continuation not attempted again."}
        manifest.update(max_budget_cny=5.5, budget_amendment=amendment[0],
                        administrative_interruptions=[{"case_id": "P05", "reason": "admin_budget_interruption", "reexecuted": False}],
                        planned_messages_total=110)
    archive_sources(folder)
    save_state(folder / "manifest.json", manifest)
    save_state(folder / "results.json", results)
    if review and review.get("kind") == "budget-amendment-v1":
        save_state(folder / "P05.json", results[-1])
    save_state(folder / "report.json", report(manifest, results, budget))
    save_state(LOCAL / "latest.json", {"run_directory": str(folder)})
    return folder, manifest, results


def locate_case_artifact(folder, case_id, suffix, depth=0):
    """Resolve inherited evidence without copying or altering earlier runs."""
    folder = Path(folder).resolve()
    if depth > 20 or not folder.is_relative_to((LOCAL / "results").resolve()):
        raise BootstrapError("Invalid evaluation evidence lineage.")
    path = folder / (case_id + suffix)
    if path.is_file():
        return path
    lineage = read_json(folder / "manifest.json").get("continuation", {})
    if case_id not in lineage.get("inherited_case_ids", []):
        raise BootstrapError("Required prior judge input is missing; no Agent replay permitted.")
    return locate_case_artifact(lineage["source_directory"], case_id, suffix, depth + 1)


def format_upgrade_review(source_folder, frozen, scorer_review=None):
    prior = read_json(Path(source_folder) / "manifest.json")
    return {"kind": "semantic-format-v1.1", "path": ORACLE_RELATIVE,
            "before_sha256": prior["source_sha256"][ORACLE_RELATIVE], "after_sha256": frozen[ORACLE_RELATIVE],
            "semantic_rules_unchanged": True, "source_scorer_repair": scorer_review,
            "reason": "Approved format-only v1.1: append a complete neutral JSON template and require evidence strings in all four dimensions. Preserve the complete original rubric, references, deterministic checks and strict parser. Uniformly rejudge every inherited answer, never select the best label or rerun its Agent."}


def make_format_upgrade(source_folder, cases, frozen, prices, budget, timeout, scorer_review=None):
    from evaluation import task_success as oracle
    review = format_upgrade_review(source_folder, frozen, scorer_review)
    # Validate all saved packets BEFORE claiming a new run or making any call.
    prior, original, _ = inspect_continuation(source_folder, cases, frozen, prices, budget, review)
    packet_paths, new_hard, new_packets = {}, {}, {}
    for row in original:
        path = locate_case_artifact(source_folder, row["case_id"], ".judge-input.json")
        messages = read_json(path)
        if (len(messages) != 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user"
                or not oracle.JUDGE_SYSTEM.startswith(messages[0].get("content", ""))
                or hashlib.sha256(oracle.JUDGE_SYSTEM[len(messages[0]["content"]):].encode()).hexdigest() != FORMAT_SUFFIX_SHA256):
            raise BootstrapError("Prior semantic packet differs from the reviewed original prompt; refusing mixed standards.")
        packet_paths[row["case_id"]] = path
        new_hard[row["case_id"]] = oracle.deterministic_checks(row)
        if not new_hard[row["case_id"]]["input_valid"]:
            raise BootstrapError("Inherited execution evidence is incomplete; no Agent replay permitted.")
        new_packets[row["case_id"]] = oracle.build_judge_messages(row)
    folder, manifest, _ = make_continuation(source_folder, cases, frozen, prices, budget, timeout, review)
    results = []
    for old in original:
        result = deepcopy(old)
        result.setdefault("semantic_history", []).append({"semantic_protocol": prior.get("semantic_protocol", "support-task-success-semantic-v1"),
                "source_run_id": prior["run_id"], "source_directory": str(Path(source_folder).resolve()),
                "semantic": deepcopy(old["semantic"]), "judge_record": deepcopy(old.get("judge_record", {})), "outcome": deepcopy(old["outcome"])})
        result.setdefault("hard_history", []).append({"source_run_id": prior["run_id"],
                "source_directory": str(Path(source_folder).resolve()), "hard": deepcopy(old["hard"]),
                "outcome": deepcopy(old["outcome"])})
        result["hard"] = new_hard[old["case_id"]]
        result["semantic"] = {"status": "unscored", "reasons": ["awaiting_v1_1_uniform_regrade"], "protocol": SEMANTIC_PROTOCOL}
        result["judge_record"] = {}
        result["outcome"] = oracle.combine(result["hard"], result["semantic"])
        result["agent_execution_provenance"] = {"source_run_id": prior["run_id"], "source_directory": str(Path(source_folder).resolve()),
                                               "case_id": old["case_id"], "reexecuted": False}
        result["source_judge_input"] = str(packet_paths[old["case_id"]].resolve())
        result["source_judge_input_sha256"] = digest(packet_paths[old["case_id"]])
        result["stop_reason"] = None
        packet_path = folder / (old["case_id"] + ".judge-input.json")
        save_state(packet_path, new_packets[old["case_id"]])
        result["judge_input_sha256"] = digest(packet_path)
        results.append(result)
        save_state(folder / (result["case_id"] + ".json"), result)
    manifest.update(semantic_protocol=SEMANTIC_PROTOCOL,
                    deterministic_regrade={"case_ids": [row["case_id"] for row in original],
                                           "source_scorer_repair": scorer_review,
                                           "policy": "Uniform offline recomputation from unchanged execution evidence; preserve every old hard verdict."},
                    semantic_regrade={"planned_case_ids": [row["case_id"] for row in original], "completed_case_ids": [],
                                      "initial_format_check_count": 2, "format_checkpoint_passed": False,
                                      "selection_policy": "All inherited cases rejudged regardless of prior pass/fail/unscored; new verdict is used without best-of selection."})
    save_state(folder / "manifest.json", manifest)
    save_state(folder / "results.json", results)
    save_state(folder / "report.json", report(manifest, results, budget))
    return folder, manifest, results


def rejudge_inherited(folder, manifest, results, key, secrets, budget, prices):
    """One new judge request for every inherited answer; no Agent execution."""
    from evaluation import task_success as oracle
    plan = manifest.get("semantic_regrade")
    if plan is None:
        return True
    if manifest.get("stop_reason"):
        raise BootstrapError("A stopped semantic regrade requires evidence review; no implicit paid retry.")
    if manifest.get("pending_judge_case"):
        raise BootstrapError("A prior judge request is unresolved; no automatic paid retry.")
    completed = plan["completed_case_ids"]
    if completed != plan["planned_case_ids"][:len(completed)]:
        raise BootstrapError("Uniform regrade order differs from its frozen prefix.")
    lookup = {row["case_id"]: row for row in results}
    for case_id in plan["planned_case_ids"][len(completed):]:
        if fingerprints() != manifest["source_sha256"]:
            manifest.update(status="stopped", stop_reason="frozen_sources_changed")
            break
        if not budget.settled():
            manifest.update(status="stopped", stop_reason="proxy_budget_unresolved")
            break
        result = lookup[case_id]
        source_packet = Path(result["source_judge_input"])
        if digest(source_packet) != result["source_judge_input_sha256"]:
            raise BootstrapError("Original semantic evidence packet changed.")
        current_packet = folder / (case_id + ".judge-input.json")
        if digest(current_packet) != result["judge_input_sha256"]:
            raise BootstrapError("Frozen v1.1 semantic evidence packet changed.")
        messages = read_json(current_packet)
        messages = redact(messages, secrets)
        save_state(folder / (case_id + ".judge-input.json"), messages)
        result["judge_input_sha256"] = digest(current_packet)
        manifest["pending_judge_case"] = case_id
        save_state(folder / "manifest.json", manifest)
        verdict, record = judge_once(messages, key, budget, manifest["run_id"], case_id, prices)
        result["semantic"] = {**verdict, "protocol": SEMANTIC_PROTOCOL}
        result["judge_record"] = record or {}
        result["outcome"] = oracle.combine(result["hard"], result["semantic"])
        result["semantic_format_valid"] = (set(verdict.get("dimensions", {})) == set(oracle.DIMENSIONS)
                                           and bool(record) and record.get("finish_reason") == "stop")
        if record is None or not record.get("settled"):
            result["stop_reason"] = "judge_usage_or_admission_unresolved"
            manifest.update(status="stopped", stop_reason=result["stop_reason"])
        completed.append(case_id)
        manifest.pop("pending_judge_case", None)
        safe = redact(result, secrets)
        result.clear(); result.update(safe)
        save_state(folder / (case_id + ".json"), result)
        save_state(folder / "results.json", results)
        if len(completed) == plan["initial_format_check_count"]:
            plan["format_checkpoint_passed"] = all(lookup[cid].get("semantic_format_valid") for cid in completed)
            if not plan["format_checkpoint_passed"]:
                manifest.update(status="stopped", stop_reason="v1_1_initial_format_check_failed")
            print(json.dumps({"semantic_format_checkpoint": plan["format_checkpoint_passed"], "checked_cases": len(completed)}), flush=True)
        save_state(folder / "manifest.json", manifest)
        save_state(folder / "report.json", report(manifest, results, budget))
        print(json.dumps({"phase": "uniform_v1_1_regrade", "completed": len(completed), "planned": len(plan["planned_case_ids"]),
                          "case": case_id, "status": result["outcome"]["status"], "format_valid": result["semantic_format_valid"],
                          "budget_occupied_cny": round(float(budget.occupied), 6)}), flush=True)
        if manifest.get("stop_reason"):
            break
    save_state(folder / "manifest.json", manifest)
    save_state(folder / "report.json", report(manifest, results, budget))
    return len(completed) == len(plan["planned_case_ids"]) and not manifest.get("stop_reason")


def execute_task(api, state, case, run_id, prices, budget, timeout):
    fixture = fixture_command({"action": "prepare", "case": case, "run_id": run_id})
    rendered = render_case(case, fixture["ids"])
    prompts = rendered.get("turns") or [rendered["prompt"]]
    turns = []
    conversation_id, parent_id, previous_ids = None, None, set()
    began = time.monotonic()
    stop_reason = None
    for index, prompt in enumerate(prompts):
        if not budget.settled():
            stop_reason = "proxy_budget_unresolved"
            break
        before_requests = len(budget.state["attempts"])
        try:
            item = turn(api, state["agents"][case["actor"]]["agent_id"], prompt,
                        conversation_id, parent_id, previous_ids, timeout, MODEL)
        except Exception:
            stop_reason = "agent_request_evidence_unknown_no_retry"
            break
        turns.append(item)
        settled = budget.settled() and len(budget.state["attempts"]) > before_requests
        item["proxy_request_ids"] = [r["attempt_id"] for r in budget.state["attempts"][before_requests:]]
        conversation_id, parent_id = item["conversation_id"], item["parent_id"]
        previous_ids = set(item["message_ids"])
        if not settled or item.get("execution_error") or not item.get("drained"):
            stop_reason = "agent_runtime_or_usage_requires_review"
            break
    after = fixture_command({"action": "snapshot"})
    evidence = {"answers": [a for item in turns for a in item["answers"]],
                "intermediate_answers": [a for item in turns for a in item.get("intermediate_answers", [])],
                "tool_calls": [call for item in turns for call in item["tool_calls"]],
                "execution_error": stop_reason or next((i["execution_error"] for i in turns if i["execution_error"]), None)}
    usage = aggregate_usage([item["usage"] for item in turns])
    return {"case_id": case["id"], "category": case["category"], "actor": case["actor"],
            "case": rendered, "evidence": evidence, "fixture_before": fixture["snapshot"], "fixture_after": after,
            "turns": turns, "usage": usage, "cost": estimate_cost(usage, prices), "error": evidence["execution_error"],
            "elapsed_seconds": time.monotonic() - began, "drained": all(i.get("drained") for i in turns),
            "conversation_id": conversation_id, "conversation_url": api.base + "/c/" + conversation_id if conversation_id else None,
            "planned_messages": len(prompts), "completed_messages": len(turns), "stop_reason": stop_reason}


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * p) - 1)] if ordered else None


def report(manifest, results, budget):
    budget.refresh()
    statuses = Counter(item["outcome"]["status"] for item in results)
    category = {}
    for name, count in manifest["categories"].items():
        rows = [item for item in results if item["category"] == name]
        counts = Counter(item["outcome"]["status"] for item in rows)
        category[name] = {"planned": count, **dict(counts), "success_rate": counts["pass"] / count}
    complete = [i for i in results if not i.get("error") and i["completed_messages"] == i["planned_messages"]]
    latency = {}
    for name, selected in (("all_completed_tasks", complete), ("single_message", [i for i in complete if i["planned_messages"] == 1]),
                           ("two_message", [i for i in complete if i["planned_messages"] == 2])):
        values = [i["elapsed_seconds"] for i in selected]
        latency[name] = {"n": len(values), "p50_seconds": percentile(values, .5), "p95_seconds": percentile(values, .95)}
    successful = [i["case_id"] for i in results if i["outcome"]["status"] == "pass"]
    sampled = sorted(successful, key=lambda cid: hashlib.sha256((REVIEW_SEED + cid).encode()).hexdigest())[:5]
    review = [{"case_id": i["case_id"], "status": i["outcome"]["status"],
               "reason": "predeclared_hash_sample" if i["case_id"] in sampled else "nonpass_or_layer_disagreement"}
              for i in results if i["case_id"] in sampled or i["outcome"]["status"] != "pass"
              or i["hard"]["passed"] != (i["semantic"].get("status") == "pass")]
    return {"manifest": manifest, "planned": 100, "attempted": len(results), "completed": len(complete),
            "success": statuses["pass"], "failure": statuses["fail"], "unscored": statuses["unscored"],
            "not_attempted": 100 - len(results), "task_success_rate": statuses["pass"] / 100,
            "deterministic_pass": sum(bool(i["hard"]["passed"]) for i in results), "categories": category,
            "latency": latency, "agent_usage": aggregate_usage([i["usage"] for i in results]),
            "judge_usage": aggregate_usage([i["judge_record"]["usage"] for i in results if i.get("judge_record", {}).get("usage")]),
            "historical_judge_usage": aggregate_usage([history["judge_record"]["usage"] for item in results
                for history in item.get("semantic_history", []) if history.get("judge_record", {}).get("usage")]),
            "semantic_protocol": manifest.get("semantic_protocol", "support-task-success-semantic-v1"),
            "semantic_regrade_complete": (manifest.get("semantic_regrade", {}).get("completed_case_ids", [])
                == manifest.get("semantic_regrade", {}).get("planned_case_ids", [])),
            "budget": budget.state, "budget_occupied_cny": float(budget.occupied), "review_queue": review,
            "results": [{"case_id": i["case_id"], "category": i["category"], "hard": i["hard"],
                         "semantic": i["semantic"], "outcome": i["outcome"], "seconds": i["elapsed_seconds"]} for i in results]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", type=Path)
    continuation.add_argument("--continue-from", type=Path)
    continuation.add_argument("--upgrade-semantic-from", type=Path,
                              help="Uniform v1.1 format-only regrade of every inherited answer, followed by unattempted tasks.")
    parser.add_argument("--format-repair-review", type=Path,
                        help="Exact-hash, explicitly reviewed parser-format-only change; never authorizes rubric changes.")
    parser.add_argument("--scorer-format-review", type=Path,
                        help="Exact-hash source-syntax repair reviewed for uniform offline recomputation; requires semantic upgrade.")
    parser.add_argument("--budget-amendment-review", type=Path,
                        help="Exact-hash user-authorized extension; only continues P06-P10 and never retries P05.")
    parser.add_argument("--stop-after-new", type=int,
                        help="Stop after this many new tasks; inherited tasks are never rerun or rejudged.")
    parser.add_argument("--timeout", type=int, default=150)
    args = parser.parse_args()
    if args.stop_after_new is not None and args.stop_after_new < 1:
        parser.error("--stop-after-new must be at least 1")
    if args.format_repair_review and not args.continue_from:
        parser.error("--format-repair-review requires --continue-from")
    if args.scorer_format_review and not args.upgrade_semantic_from:
        parser.error("--scorer-format-review requires --upgrade-semantic-from")
    if args.budget_amendment_review and (not args.continue_from or args.format_repair_review):
        parser.error("--budget-amendment-review requires --continue-from without a format repair")
    dataset_path = SERVICE / "evaluation/agent_cases.json"
    cases = read_json(dataset_path)["cases"]
    if len(cases) != 100 or digest(dataset_path) != DATA_SHA:
        raise BootstrapError("Original 100-task dataset changed; refusing this protocol.")
    if not args.run:
        print(json.dumps({"mode": "dry_run", "tasks": 100, "messages": sum(len(c.get("turns") or [c["prompt"]]) for c in cases), "model_calls": 0, "max_budget_cny": 5}))
        return 0
    from evaluation import task_success as oracle
    from evaluation import task_success_setup as setup
    with exclusive_run():
        budget = Budget(LOCAL / "budget.json")
        if not budget.settled():
            raise BootstrapError("A prior provider request is unresolved; no automatic retry.")
        if budget.state["attempts"] and not (args.resume or args.continue_from or args.upgrade_semantic_from):
            raise BootstrapError("The authorized single-attempt run already started; use its verified resume directory.")
        state = read_json(LOCAL / "agents.json")
        api = LocalApi("http://127.0.0.1:3080")
        demo = read_json(ROOT / ".local/evaluation/demo.json")
        authenticate(api, demo)
        setup.validate_agents(api, state)
        secrets, key = secrets_and_key()
        secrets.append(api.token)
        prices = price_snapshot()
        frozen = fingerprints()
        if args.resume:
            folder = args.resume.resolve()
            if not folder.is_relative_to(LOCAL / "results"):
                raise BootstrapError("Resume must reference this protocol's result directory.")
            manifest = read_json(folder / "manifest.json")
            if manifest["source_sha256"] != frozen or manifest.get("pending_case") or manifest.get("pending_judge_case") or manifest["prices"] != prices:
                raise BootstrapError("Frozen inputs changed or a prior case is unresolved.")
            results = read_json(folder / "results.json")
            validate_result_prefix(results, cases)
            if manifest.get("stop_reason") not in {None, "requested_checkpoint"}:
                raise BootstrapError("This stopped run needs an audited continuation, not an implicit retry.")
            manifest.update(status="running", stop_reason=None)
        elif args.continue_from:
            review = read_json(args.budget_amendment_review or args.format_repair_review) if (args.budget_amendment_review or args.format_repair_review) else None
            if review and review.get("kind") == "semantic-format-v1.1":
                raise BootstrapError("Semantic format v1.1 must use uniform --upgrade-semantic-from, not mixed-standard continuation.")
            folder, manifest, results = make_continuation(args.continue_from, cases, frozen, prices, budget, args.timeout, review)
        elif args.upgrade_semantic_from:
            scorer_review = read_json(args.scorer_format_review) if args.scorer_format_review else None
            folder, manifest, results = make_format_upgrade(args.upgrade_semantic_from, cases, frozen, prices, budget, args.timeout, scorer_review)
            print(json.dumps({"phase": "v1_1_upgrade_started", "inherited_agent_executions": len(results),
                              "uniform_regrade_planned": len(results), "remaining_agent_tasks": len(cases) - len(results),
                              "budget_occupied_cny": float(budget.occupied), "result_directory": str(folder)}), flush=True)
        else:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
            folder = LOCAL / "results" / run_id
            folder.mkdir(parents=True, exist_ok=False)
            manifest = {"run_id": run_id, "protocol": "support-task-success-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                        "source_sha256": frozen, "prices": prices, "categories": dict(Counter(c["category"] for c in cases)),
                        "case_ids": [c["id"] for c in cases], "max_budget_cny": 5, "concurrency": 1,
                        "model": MODEL, "provider": state["provider"], "parameters": state["parameters"],
                        "recursion_limit": state["recursion_limit"], "budget_control": "request-level proxy; reserve before forwarding, settle from provider usage",
                        "judge_model": MODEL, "judge_temperature": 0, "judge_max_tokens": JUDGE_OUTPUT,
                        "semantic_protocol": SEMANTIC_PROTOCOL,
                        "judge_max_input_bytes": JUDGE_INPUT_BYTES, "review_seed": REVIEW_SEED, "review_pass_sample": 5,
                        "single_attempt": True, "timeout_seconds": args.timeout, "frozen_before_run": True,
                        "limitations": ["Seen, developer-authored synthetic regression; not an unseen or official benchmark",
                            "Separate prompt, same model semantic judge; not independent human verification",
                            "Static isolated actor bindings; does not retest live per-user OAuth",
                            "Write goals end at pending artifacts; human approval is not counted as autonomous execution",
                            "References incorporate historical reviews; post-hoc provenance disclosed",
                            "Agent latency excludes fixture reset and semantic judging, includes final state capture"],
                        "status": "running", "stop_reason": None}
            archive_sources(folder)
            save_state(folder / "manifest.json", manifest)
            save_state(folder / "results.json", [])
            save_state(LOCAL / "latest.json", {"run_directory": str(folder)})
            results = []
        if not rejudge_inherited(folder, manifest, results, key, secrets, budget, prices):
            print(json.dumps({"status": manifest["status"], "stop_reason": manifest.get("stop_reason"),
                              "phase": "uniform_regrade_stopped", "result_directory": str(folder)}), flush=True)
            return 2
        completed = {i["case_id"] for i in results}
        new_executions = 0
        for case in cases:
            if case["id"] in completed:
                continue
            if fingerprints() != frozen:
                manifest.update(status="stopped", stop_reason="frozen_sources_changed")
                break
            renew_session(api, demo, secrets, args.timeout)
            setup.validate_agents(api, state)
            manifest["pending_case"] = case["id"]
            save_state(folder / "manifest.json", manifest)
            result = execute_task(api, state, case, manifest["run_id"], prices, budget, args.timeout)
            # Preserve the paid execution before scoring, so a local grader
            # failure cannot erase evidence or justify another Agent attempt.
            save_state(folder / (case["id"] + ".execution.json"), redact(result, secrets))
            result["hard"] = oracle.deterministic_checks(result)
            result["semantic"] = {"status": "unscored", "reasons": ["execution_incomplete"]}
            result["judge_record"] = {}
            if not result["stop_reason"] and result["hard"]["input_valid"]:
                messages = redact(oracle.build_judge_messages(result), secrets)
                save_state(folder / (case["id"] + ".judge-input.json"), messages)
                result["semantic"], record = judge_once(messages, key, budget, manifest["run_id"], case["id"], prices)
                result["judge_record"] = record or {}
                if record is not None and not record.get("settled"):
                    result["stop_reason"] = "judge_usage_unknown"
            result["semantic"]["protocol"] = manifest.get("semantic_protocol", "support-task-success-semantic-v1")
            result["semantic_format_valid"] = (set(result["semantic"].get("dimensions", {})) == set(oracle.DIMENSIONS)
                                                 and result["judge_record"].get("finish_reason") == "stop")
            result["outcome"] = oracle.combine(result["hard"], result["semantic"])
            result = redact(result, secrets)
            results.append(result)
            new_executions += 1
            manifest["new_executions_this_invocation"] = new_executions
            manifest["new_executions_this_run"] = len(results) - manifest.get("continuation", {}).get("inherited_execution_count", 0)
            save_state(folder / (case["id"] + ".json"), result)
            save_state(folder / "results.json", results)
            manifest.pop("pending_case", None)
            if result["stop_reason"]:
                manifest.update(status="stopped", stop_reason=result["stop_reason"])
            elif args.stop_after_new is not None and new_executions >= args.stop_after_new and len(results) < 100:
                manifest.update(status="checkpoint_stopped", stop_reason="requested_checkpoint",
                                checkpoint={"new_tasks_this_invocation": new_executions, "last_case_id": case["id"],
                                            "attempted_total": len(results), "next_case_id": cases[len(results)]["id"]})
            save_state(folder / "manifest.json", manifest)
            save_state(folder / "report.json", report(manifest, results, budget))
            print(json.dumps({"attempted": len(results), "planned": 100, "case": case["id"],
                              "status": result["outcome"]["status"], "agent_seconds": round(result["elapsed_seconds"], 2),
                              "budget_occupied_cny": round(float(budget.occupied), 6)}, ensure_ascii=False), flush=True)
            if result["stop_reason"] or manifest.get("stop_reason") == "requested_checkpoint":
                break
        if len(results) == 100 and not manifest.get("stop_reason"):
            manifest["status"] = "completed"
        save_state(folder / "manifest.json", manifest)
        manifest["completed_messages_total"] = sum(item["completed_messages"] for item in results if not item.get("error"))
        manifest["sent_messages_total"] = sum(item["completed_messages"] for item in results)
        save_state(folder / "manifest.json", manifest)
        save_state(folder / "report.json", report(manifest, results, budget))
        print(json.dumps({"status": manifest["status"], "attempted": len(results), "result_directory": str(folder)}), flush=True)
        return 0 if manifest["status"] in {"completed", "checkpoint_stopped"} else 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(main())
    except Exception as error:
        import traceback
        frames = [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name} for f in traceback.extract_tb(error.__traceback__)]
        print(json.dumps({"error_type": type(error).__name__, "frames": frames}), flush=True)
        print(json.dumps({"status": "stopped", "reason": "Configuration or evidence error; no automatic paid retry. Inspect local manifest and ledger."}), flush=True)
        raise SystemExit(2)
