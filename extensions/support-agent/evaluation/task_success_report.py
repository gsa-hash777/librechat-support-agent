"""Render a read-only Chinese review report from the frozen TSR run.

This module never calls a model, edits scoring, resets fixtures, or writes the
proxy ledger. Run it again after the experiment finishes to replace a snapshot.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re

SERVICE = Path(__file__).resolve().parents[1]
ROOT = SERVICE.parents[1]
LOCAL = ROOT / ".local/task-success-v1"
DEFAULT_OUTPUT = LOCAL / "reports"
TITLE = "LibreChat-任务成功率评测"
CATEGORIES = {"knowledge": "有答案知识问答", "missing_info": "缺信息与追问", "no_answer": "无答案与缺证", "create": "创建工单草稿", "idempotency": "重复请求与幂等", "authorization": "角色与组织权限", "lifecycle": "工单处理流程", "injection": "不可信内容与注入"}
STATUSES = {"pass": "成功", "fail": "失败", "unscored": "未评分", "not_attempted": "未尝试"}
DIMENSIONS = {"goal_completion": "目标完成", "factual_support": "回答有据", "policy_and_safety": "权限与安全", "handoff_clarity": "人工确认交接"}
SECRET_KEYS = re.compile(r"^(?:password|passwd|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|authorization|cookie|csrf|token_hash|session_hash|payload_hash)$", re.I)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe(value):
    """Defense in depth for already redacted execution records and string JSON."""
    if isinstance(value, dict):
        return {key: "[已脱敏]" if SECRET_KEYS.fullmatch(key) and isinstance(item, str) else safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe(item) for item in value]
    if isinstance(value, str):
        value = value.replace("EVAL_FORBIDDEN_CANARY", "[测试禁披露标记已隐藏]")
        value = re.sub(r"(?i)Bearer\s+[A-Za-z0-9_.~-]+", "Bearer [已脱敏]", value)
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[密钥已脱敏]", value)
        value = re.sub(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b", "[令牌已脱敏]", value)
        if value.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (dict, list)):
                    return json.dumps(safe(parsed), ensure_ascii=False, indent=2)
            except (ValueError, RecursionError):
                pass
    return value


def percentile(values, p):
    ordered = sorted(value for value in values if isinstance(value, (int, float)))
    return ordered[max(0, math.ceil(len(ordered) * p)-1)] if ordered else None


def quantiles(rows):
    values = [row.get("elapsed_seconds") for row in rows]
    return {"n": len(rows), "p50_seconds": percentile(values, .5), "p95_seconds": percentile(values, .95)}


def execution_counts(row):
    """A saved turn is sent, but is complete only with a drained error-free response."""
    turns = row.get("turns", [])
    sent = len(turns) if "turns" in row else row.get("completed_messages", 0)
    complete = sum(not turn.get("execution_error") and turn.get("drained") is True for turn in turns)
    if "turns" not in row:
        complete = sent if not row.get("error") else 0
    return {"planned_messages": row.get("planned_messages", 0), "sent_messages": sent,
            "completed_messages": complete,
            "execution_complete": not row.get("error") and complete == row.get("planned_messages")}


def state_changes(before, after):
    result = {}
    for name in ("tickets", "drafts", "ticket_updates", "ticket_events", "users"):
        old = {item.get("id"): item for item in before.get("rows", {}).get(name, [])}
        new = {item.get("id"): item for item in after.get("rows", {}).get(name, [])}
        fields = ("id", "user_id", "org_id", "status", "version", "assignee_id", "action", "ticket_id", "expected_version", "priority", "title", "note", "role", "active")
        brief = lambda row: {key: row[key] for key in fields if key in row}
        result[name] = {"before_count": len(old), "after_count": len(new),
                        "added": [brief(new[key]) for key in new.keys()-old.keys()],
                        "removed": [brief(old[key]) for key in old.keys()-new.keys()],
                        "changed": [{"before": brief(old[key]), "after": brief(new[key])} for key in old.keys() & new.keys() if old[key] != new[key]]}
    return result


def budget_summary(budget):
    groups = {}
    labels = {"agent": "Agent 执行", "semantic_judge": "语义评审（含旧版、诊断及重评）"}
    stages = ["agent", "semantic_judge"] + sorted({row.get("stage", "agent") for row in budget.get("attempts", [])}-{ "agent", "semantic_judge"})
    for stage in stages:
        title = labels.get(stage, "其他已计费阶段：" + str(stage))
        rows = [row for row in budget.get("attempts", []) if row.get("stage", "agent") == stage]
        settled = [row for row in rows if row.get("status") == "settled"]
        groups[stage] = {"label": title, "requests": len(rows), "settled_requests": len(settled),
                         "input_tokens": sum(row.get("usage", {}).get("prompt_tokens", 0) for row in settled),
                         "output_tokens": sum(row.get("usage", {}).get("completion_tokens", 0) for row in settled),
                         "settled_estimate_cny": sum(row.get("cost_cny", 0) for row in settled),
                         "unresolved_reserved_cny": sum(max(row.get("cost_cny", 0), row.get("reserved_cny", 0)) for row in rows if row.get("status") != "settled")}
    return {"limit_cny": budget.get("limit_cny", 5), "stages": groups,
            "settled_estimate_cny": sum(v["settled_estimate_cny"] for v in groups.values()),
            "unresolved_reserved_cny": sum(v["unresolved_reserved_cny"] for v in groups.values()),
            "blocked": any(row.get("status") != "settled" for row in budget.get("attempts", [])),
            "pricing": budget.get("pricing", {}), "amendments": budget.get("amendments", [])}


def lineage(folder):
    """Follow immutable parent manifests, without rewriting inherited results."""
    nodes, visited = [], set()
    while folder is not None:
        folder = Path(folder).resolve()
        if folder in visited or len(visited) > 8 or not folder.is_relative_to((LOCAL / "results").resolve()):
            raise ValueError("Invalid or cyclic report lineage.")
        visited.add(folder)
        manifest = read_json(folder / "manifest.json")
        rows = read_json(folder / "results.json")
        stored_report = read_json(folder / "report.json") if (folder / "report.json").exists() else {}
        continuation = manifest.get("continuation", {})
        source = continuation.get("source_directory")
        verified = {}
        if source:
            source = Path(source).resolve()
            if not source.is_relative_to((LOCAL / "results").resolve()):
                raise ValueError("Lineage source is outside the isolated results root.")
            for filename in ("manifest.json", "results.json"):
                expected = continuation.get("source_artifacts_sha256", {}).get(filename)
                if expected:
                    actual = hashlib.sha256((source / filename).read_bytes()).hexdigest()
                    verified[filename] = actual == expected
                    if actual != expected:
                        raise ValueError("Preserved lineage artifact fingerprint differs.")
        inherited = continuation.get("inherited_execution_count", 0)
        nodes.append({"run_id": manifest["run_id"], "run_directory": str(folder), "status": manifest.get("status"),
                      "semantic_protocol": manifest.get("semantic_protocol", "support-task-success-semantic-v1"),
                      "saved_task_count": len(rows), "inherited_execution_count": inherited,
                      "new_execution_count": len(rows)-inherited,
                      "source_run_id": continuation.get("source_run_id"), "source_artifacts_verified": verified,
                      "budget_attempts_at_start": continuation.get("budget_attempts_at_start", 0),
                      "budget_requests_in_saved_report": len(stored_report.get("budget", {}).get("attempts", [])),
                      "semantic_regrade": manifest.get("semantic_regrade"),
                      "source_changes": continuation.get("source_changes", []),
                      "agent_executions_repeated": continuation.get("agent_executions_repeated", 0)})
        folder = source
    return list(reversed(nodes))


def budget_phases(budget, nodes):
    """Disjoint chronological ledger slices; never add inherited report totals."""
    rows = budget.get("attempts", [])
    phases = []
    starts = [node["budget_attempts_at_start"] for node in nodes]
    if any(not isinstance(value, int) or value < 0 for value in starts) or starts != sorted(starts):
        raise ValueError("Invalid chronological budget boundaries.")
    for i, node in enumerate(nodes):
        start = node["budget_attempts_at_start"]
        end = starts[i+1] if i+1 < len(nodes) else len(rows)
        if i+1 == len(nodes) and node["status"] in {"stopped", "checkpoint_stopped", "completed"}:
            end = min(end, node["budget_requests_in_saved_report"])
        part = rows[start:end]
        settled = [r for r in part if r.get("status") == "settled"]
        phases.append({"run_id": node["run_id"], "semantic_protocol": node["semantic_protocol"], "ledger_start_index": start,
                       "ledger_end_exclusive": end, "requests": len(part),
                       "agent_requests": sum(row.get("stage", "agent") == "agent" for row in part),
                       "judge_requests": sum(row.get("stage") == "semantic_judge" for row in part),
                       "settled_estimate_cny": sum(row.get("cost_cny", 0) for row in settled),
                       "unresolved_reserved_cny": sum(max(row.get("cost_cny",0),row.get("reserved_cny",0)) for row in part if row.get("status") != "settled")})
    accounted = sum(row["requests"] for row in phases)
    if accounted < len(rows):
        remaining = rows[accounted:]
        phases.append({"run_id": "after_selected_snapshot", "semantic_protocol": "选中版本快照之后的全局请求", "ledger_start_index": accounted, "ledger_end_exclusive": len(rows),
                       "requests": len(remaining), "agent_requests": sum(r.get("stage", "agent") == "agent" for r in remaining), "judge_requests": sum(r.get("stage") == "semantic_judge" for r in remaining),
                       "settled_estimate_cny": sum(r.get("cost_cny",0) for r in remaining if r.get("status") == "settled"),
                       "unresolved_reserved_cny": sum(max(r.get("cost_cny",0),r.get("reserved_cny",0)) for r in remaining if r.get("status") != "settled")})
    if sum(row["requests"] for row in phases) != len(rows):
        raise ValueError("Budget lineage slices do not account for every request.")
    return phases


def make_payload(folder):
    folder = Path(folder).resolve()
    manifest = read_json(folder / "manifest.json")
    results = read_json(folder / "results.json")
    if len(results) > 100 or len({row["case_id"] for row in results}) != len(results):
        raise ValueError("Invalid run result count or duplicate task ID.")
    frozen_root = folder / "frozen/extensions/support-agent/evaluation"
    references = read_json(frozen_root / "task_success_references.json")
    dataset = read_json(frozen_root / "agent_cases.json")
    reference_map = {item["case_id"]: item for item in references["cases"]}
    task_map = {item["id"]: item for item in dataset["cases"]}
    if len(task_map) != 100 or set(manifest["case_ids"]) != set(task_map) or any(row["case_id"] not in task_map for row in results):
        raise ValueError("Frozen dataset must contain the same 100 tasks.")
    saved_report = read_json(folder / "report.json") if (folder / "report.json").exists() else {}
    budget = read_json(LOCAL / "budget.json")
    nodes = lineage(folder)
    indexed = {row["case_id"]: row for row in results}
    statuses = Counter(row["outcome"]["status"] for row in results)
    counts = {row["case_id"]: execution_counts(row) for row in results}
    completed = [row for row in results if counts[row["case_id"]]["execution_complete"]]
    interruptions = [row["case_id"] for row in results if row.get("administrative_status") == "admin_budget_interruption"]
    if any(row["outcome"]["status"] != "unscored" for row in results if row["case_id"] in interruptions):
        raise ValueError("An administrative budget interruption must remain unscored.")
    regrade = manifest.get("semantic_regrade")
    regrade_complete = not regrade or (set(regrade.get("planned_case_ids", [])) == set(regrade.get("completed_case_ids", [])) and regrade.get("format_checkpoint_passed") is True)
    final = manifest.get("status") == "completed" and len(results) == 100 and not manifest.get("pending_case") and regrade_complete
    categories = []
    for category, planned in manifest["categories"].items():
        rows = [row for row in results if row["category"] == category]
        count = Counter(row["outcome"]["status"] for row in rows)
        categories.append({"category": category, "label": CATEGORIES.get(category, category), "planned": planned, "attempted": len(rows), "success": count["pass"], "failure": count["fail"], "unscored": count["unscored"], "not_attempted": planned-len(rows), "success_rate": count["pass"]/planned})
    cases = []
    for case_id in manifest["case_ids"]:
        row = indexed.get(case_id)
        task = row.get("case", task_map[case_id]) if row else task_map[case_id]
        item = {"case_id": case_id, "category": task["category"], "actor": task["actor"], "status": row["outcome"]["status"] if row else "not_attempted", "prompts": task.get("turns") or [task["prompt"]], "frozen_goal": reference_map[case_id]}
        if row:
            item.update({key: row.get(key) for key in ("evidence", "hard", "semantic", "outcome", "elapsed_seconds", "planned_messages", "completed_messages", "usage", "judge_record", "error", "stop_reason", "conversation_url")})
            item["semantic_history"] = row.get("semantic_history", [])
            item["hard_history"] = row.get("hard_history", [])
            item["execution_counts"] = counts[case_id]
            item["administrative_status"] = row.get("administrative_status")
            item["outcome_before_administrative_classification"] = row.get("outcome_before_administrative_classification")
            item["agent_execution_provenance"] = row.get("agent_execution_provenance")
            item["source_judge_input_sha256"] = row.get("source_judge_input_sha256")
            item["semantic_protocol"] = (row.get("semantic") or {}).get("protocol", manifest.get("semantic_protocol", "support-task-success-semantic-v1"))
            item["business_state_changes"] = state_changes(row.get("fixture_before", {}), row.get("fixture_after", {}))
            evidence_path = next((Path(node["run_directory"]) / (case_id + ".json") for node in reversed(nodes) if (Path(node["run_directory"]) / (case_id + ".json")).exists()), None)
            item["execution_evidence_file"] = str(evidence_path) if evidence_path else None
        cases.append(item)
    data = {"title": "企业客服工单 Agent · 任务成功率评测", "generated_at": datetime.now(timezone.utc).isoformat(), "run_directory": str(folder), "manifest": manifest,
            "is_final": final, "snapshot_note": "最终运行结果，仍为自动判分与模型辅助评审" if final else "进行中快照／尚未完成，不能用于最终简历指标",
            "summary": {"planned": 100, "planned_messages": 110, "attempted": len(results), "completed": len(completed), "success": statuses["pass"], "failure": statuses["fail"], "unscored": statuses["unscored"], "not_attempted": 100-len(results), "task_success_rate": statuses["pass"]/100,
                        "deterministic_pass": sum(row.get("hard", {}).get("passed") is True for row in results),
                        "sent_messages": sum(row["sent_messages"] for row in counts.values()),
                        "completed_messages": sum(row["completed_messages"] for row in counts.values()),
                        "administrative_budget_interruptions": len(interruptions), "administrative_interrupted_case_ids": interruptions},
            "categories": categories, "latency": {"all_completed_tasks": quantiles(completed), "single_message": quantiles([r for r in completed if r["planned_messages"] == 1]), "two_message": quantiles([r for r in completed if r["planned_messages"] == 2])},
            "budget": budget_summary(budget), "budget_ledger": budget, "budget_phases": budget_phases(budget, nodes), "lineage": nodes, "regrade_complete": regrade_complete, "review_queue": saved_report.get("review_queue", []),
            "reference_provenance": references.get("provenance"), "acceptance_policy": references.get("acceptance_policy"), "business_policy": references.get("policy"), "cases": cases,
            "limitations_zh": ["复用已见过的 100 条开发者自建合成任务，是项目历史回归集，不是官方 τ-bench 分数，也不是独立未见测试集。", "语义评审使用独立提示词，但仍为同款 DeepSeek 模型，不是人工验收或独立客观真值。", "标准目标吸收了历史复核意见，已在本轮调用前冻结；没有声称 100 条均经过独立人工验收。", "当前固定评测身份检验角色与组织隔离，不重新验证真实多人 OAuth 登录链路。", "创建与变更任务的授权终点是 pending 草稿／提案及正确的独立确认交接；不把预置状态或人工审批算作 Agent 自主完成。", "格式不合法或无法判断的语义评审保留为未评分，不按成功处理。早期未保存原始评审文本的任务，不能由格式失败推断其答案对错；有脱敏原文时会在逐题证据中展示。", "v1.1 对继承的全部 38 条执行证据统一重评；旧成功、失败与格式无效项一并保留，但均不参与新版成绩选择。Agent 执行不因此重跑。", "本次按实际目标、数据库状态、安全约束与回答依据联合验收；与过去的 95% 严格规则通过率不是同一个指标，不能直接做增减比较。", "费用读取唯一全局账本，包含早期无效评审、诊断与新版重评，不把各版本累计费用相加，也不因舍弃旧判定而舍弃费用。", "金额按已核验峰时价格、输入全部未命中缓存估算；不是服务商账单，也不是真实企业节省成本。"]}
    if not regrade:
        data["limitations_zh"] = [line for line in data["limitations_zh"] if not line.startswith("v1.1 对继承")]
    if interruptions:
        data["limitations_zh"].append("预算中断是行政执行限制，单列未评分并保留中断前原判定；未重跑该任务，也不把它从固定100分母中删除。计划110条消息与实际完整完成的消息数分开报告。")
    return safe(data)


def esc(value):
    return html.escape(str(value), quote=True)


def jsafe(value):
    return esc(json.dumps(value, ensure_ascii=False, indent=2))


def fmt(value, digits=2):
    return "—" if value is None else f"{value:,.{digits}f}"


def details(label, body, opened=False):
    return f'<details{" open" if opened else ""}><summary>{esc(label)}</summary>{body}</details>'


def bullets(values):
    return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in values or []) + "</ul>"


def reason_text(case):
    if case["status"] == "not_attempted":
        return "尚无执行和判分结果。"
    if case.get("administrative_status") == "admin_budget_interruption":
        return "原5元预算在本题执行中触发请求级拦截，未获得完整回答，列为预算中断／未评分。该任务未重跑，不认定为业务失败，也不补为成功；固定100分母不变。"
    outcome = case.get("outcome") or {}
    parts = []
    for name in outcome.get("failed_hard_checks", []):
        check = (case.get("hard") or {}).get("checks", {}).get(name, {})
        parts.append(f"规则 {name}：{check.get('reason', '未通过')}")
    semantic = case.get("semantic") or {}
    for name, row in semantic.get("dimensions", {}).items():
        if row.get("status") != "pass":
            parts.append(f"{DIMENSIONS.get(name,name)}：{row.get('reason','未确定')}")
    if not parts:
        parts.extend(semantic.get("reasons", []))
    if not parts and case.get("error"):
        parts.append(str(case["error"]))
    known = {"Semantic judge output did not match the frozen schema.": "语义评审返回内容不符合冻结的结构要求，因此未评分；这不等于已证明回答失败或成功。", "execution_incomplete": "执行证据不完整，无法完成语义验收。", "judge_response_not_complete": "语义评审输出未完整结束，保留为未评分。", "awaiting_v1_1_regrade": "等待 v1.1 统一重评；旧版判断留存对照，不回退使用旧版通过结果。"}
    return "\n".join(known.get(part, part) for part in parts) or "规则与四项语义目标均通过。"


def render_case(case):
    status = case["status"]
    category = CATEGORIES.get(case["category"], case["category"])
    text = " ".join(case["prompts"])
    label = f'{case["case_id"]} · {category} · {STATUSES.get(status,status)}'
    body = f'<p class="meta">身份：{esc(case["actor"])}　执行耗时：{fmt(case.get("elapsed_seconds"))} 秒</p><h4>用户实际问题</h4>'
    body += "".join(f'<pre class="question">第 {i+1} 次消息\n{esc(prompt)}</pre>' for i, prompt in enumerate(case["prompts"]))
    body += '<h4>本题判定及具体原因</h4><pre>' + esc(reason_text(case)) + '</pre>'
    goal = case["frozen_goal"]
    goal_body = '<h4>必要事实与目标</h4>' + bullets(goal.get("essential_facts")) + '<h4>可接受结果</h4>' + bullets(goal.get("acceptable_outcomes")) + '<h4>禁止作出的结论</h4>' + bullets(goal.get("prohibited_claims")) + '<h4>预期数据库结果</h4><pre>' + jsafe(goal.get("state_target")) + '</pre>'
    body += details("冻结目标与验收依据", goal_body)
    if status != "not_attempted":
        if case.get("administrative_status"):
            body += details("预算中断与行政归类记录", '<p>预算限制导致执行未完成，保留原运行判定供审计；行政归类不是重新评审，也不是 Agent 能力提升。</p><pre>' + jsafe({"administrative_status": case["administrative_status"], "original_outcome": case.get("outcome_before_administrative_classification"), "current_outcome": case.get("outcome"), "execution_counts": case.get("execution_counts")}) + '</pre>')
        history = case.get("semantic_history", [])
        if history:
            history_body = '<p>此题 Agent 只执行一次。下面旧判定全部留存；当前成绩只使用统一的新协议结果，不从多次评审中选择较好结果。旧版格式不合格意味着未评分，不等于任务成功或失败。</p><table><thead><tr><th>版本</th><th>语义结果</th><th>综合结果</th><th>计入当前成绩</th></tr></thead><tbody>'
            for old in history:
                history_body += '<tr><td>' + esc(old.get("semantic_protocol", "v1")) + '</td><td>' + esc(STATUSES.get(old.get("semantic", {}).get("status"), old.get("semantic", {}).get("status", "未知"))) + '</td><td>' + esc(STATUSES.get(old.get("outcome", {}).get("status"),old.get("outcome", {}).get("status", "未知"))) + '</td><td>否，留存对照</td></tr>'
            history_body += '<tr><td>' + esc(case.get("semantic_protocol", "当前协议")) + '</td><td>' + esc(STATUSES.get((case.get("semantic") or {}).get("status"), (case.get("semantic") or {}).get("status", "未知"))) + '</td><td>' + esc(STATUSES.get(status,status)) + '</td><td>是</td></tr></tbody></table>'
            history_body += '<h4>旧版完整判断及费用记录</h4><pre>' + jsafe(history) + '</pre><h4>沿用的 Agent 执行证据</h4><pre>' + jsafe({"agent_execution_provenance":case.get("agent_execution_provenance"),"source_judge_input_sha256":case.get("source_judge_input_sha256")}) + '</pre>'
            body += details("旧版与新版判断对照（只计新版，不择优）", history_body)
        if case.get("hard_history"):
            body += details("硬规则修复前后对照", '<p>旧版硬规则判断完整留存。当前成绩只使用修复后的统一规则核验；来源格式解析修复属于评测器修复，不能当作 Agent 在相同执行中的能力提升。</p><h4>旧版规则结果</h4><pre>' + jsafe(case["hard_history"]) + '</pre><h4>当前规则结果</h4><pre>' + jsafe(case.get("hard")) + '</pre>')
        evidence = case.get("evidence") or {}
        answers = evidence.get("answers", [])
        body += '<h4>Agent 完整回答</h4>' + ("".join('<pre class="answer">' + esc(answer.get("text", answer) if isinstance(answer,dict) else answer) + '</pre>' for answer in answers) or '<p>没有完整最终回答。</p>')
        intermediate = evidence.get("intermediate_answers", [])
        if intermediate:
            body += details("中间回答", '<pre>' + jsafe(intermediate) + '</pre>')
        calls = evidence.get("tool_calls", [])
        calls_body = ""
        for index, call in enumerate(calls):
            name = call.get("name", "未知工具").split("_action_")[0]
            calls_body += details(f'{index+1}. {name}', '<h4>实际参数</h4><pre>' + esc(call.get("args", "")) + '</pre><h4>工具实际返回</h4><pre>' + esc(call.get("output", "")) + '</pre>')
        body += details(f"真实工具调用 · {len(calls)} 次", calls_body or '<p>没有工具调用。</p>')
        dims = (case.get("semantic") or {}).get("dimensions", {})
        semantic_body = "".join(f'<h4>{esc(DIMENSIONS.get(name,name))} · {esc(STATUSES.get(row.get("status"),row.get("status")))}</h4><p>{esc(row.get("reason",""))}</p><pre>{esc(row.get("evidence",""))}</pre>' for name,row in dims.items())
        semantic_body += '<pre>' + jsafe((case.get("semantic") or {}).get("reasons", [])) + '</pre>'
        record = case.get("judge_record") or {}
        if record.get("raw_response_redacted"):
            semantic_body += '<h4>本次评审原始输出（已脱敏）</h4><pre>' + esc(record["raw_response_redacted"]) + '</pre>'
        body += details("四维语义评审（模型辅助，非人工验收）", semantic_body)
        body += details("数据库前后变化与规则核验", '<pre>' + jsafe(case.get("business_state_changes")) + '</pre><h4>全部规则结果</h4><pre>' + jsafe(case.get("hard")) + '</pre>')
        body += details("Token 与时间证据", '<p>原记录 completed_messages 表示已保存的发送轮次；实际完整完成数另见 execution_counts。</p><pre>' + jsafe({key: case.get(key) for key in ("usage", "judge_record", "planned_messages", "completed_messages", "execution_counts")}) + '</pre>')
    return f'<details class="case" data-status="{esc(status)}" data-category="{esc(case["category"])}" data-search="{esc(case["case_id"]+" "+text)}"><summary><span class="badge {esc(status)}">{esc(STATUSES.get(status,status))}</span> {esc(label)}</summary>{body}</details>'


def render(payload):
    summary, manifest, budget = payload["summary"], payload["manifest"], payload["budget"]
    final = payload["is_final"]
    percent = summary["task_success_rate"]*100
    headline = f'{percent:.0f}% 原始严格判分任务成功率' if final else f'{summary["success"]} / 100 已确认成功'
    status = "本轮已结束" if final else "已暂停快照 · 不是最终成绩" if manifest.get("status") in {"stopped", "checkpoint_stopped"} else "进行中快照 · 不是最终成绩"
    cards = [("计划任务", 100), ("已尝试", summary["attempted"]), ("完成执行", summary["completed"]), ("确认成功", summary["success"]), ("确认失败", summary["failure"]), ("未评分", summary["unscored"]), ("未尝试", summary["not_attempted"])]
    content = f'<header><div class="eyebrow">LIBRECHAT · SUPPORT AGENT EVALUATION</div><h1>企业客服工单 Agent<br>任务成功率评测</h1><p class="lead">检查一件事有没有真正做对：目标完成、数据库状态正确、权限安全不被破坏、回答有证据。</p><div class="run">{esc(status)}　|　{esc(manifest["run_id"])}<br>报告生成：{esc(payload["generated_at"])}</div></header>'
    content += f'<main><section class="hero"><div><span class="sub">固定分母 · 100 条任务 / 计划 110 次用户消息</span><h2>{headline}</h2><p>成功数 ÷ 100；失败、未评分和未尝试均不会从分母中删除。</p></div><div class="formula">TSR = <strong>{summary["success"]} / 100</strong><span>规则核验 ∧ 四维语义验收</span></div></section>'
    if final:
        content += '<div class="notice light">本页保留原始严格结构判分及逐题证据。独立零调用格式规范化后的确认TSR、原／新分数对照和全部格式改动见 <a href="LibreChat-检索优化与任务成功率.html">检索优化与任务成功率汇总</a>；原始结果没有覆盖。</div>'
    if not final:
        content += '<div class="notice">这是尚未完成的快照，不能将当前比例作为最终简历成绩。原始运行状态：' + esc(manifest.get("status")) + '；当前任务：' + esc(manifest.get("pending_case", "无")) + '；停止原因：' + esc(manifest.get("stop_reason") or "无已记录原因") + '。</div>'
    content += '<div class="cards">' + "".join(f'<div><span>{name}</span><strong>{count}</strong></div>' for name,count in cards) + '</div>'
    content += f'<p>用户消息：计划 <b>{summary["planned_messages"]}</b> 次，已发送并保存 <b>{summary["sent_messages"]}</b> 次，完整完成 <b>{summary["completed_messages"]}</b> 次。</p>'
    if summary["administrative_budget_interruptions"]:
        content += f'<div class="notice">预算中断 <b>{summary["administrative_budget_interruptions"]}</b> 条（{esc(", ".join(summary["administrative_interrupted_case_ids"]))}）单列未评分。原5元上限触发后，用户追加最多0.50元，仅继续未尝试的 P06–P10；P05 没有重跑。不能称全部110条消息均已完成，也不能把预算中断算作业务失败。</div>'
    content += '<p class="small">“完成执行”指用户消息已得到执行证据且无运行错误；即使语义评审格式不合格而未评分，也可能已完成执行。未尝试包含尚未入账的当前任务，最终以完整结果文件为准。</p>'
    segments = [("pass",summary["success"]),("fail",summary["failure"]),("unscored",summary["unscored"]),("not_attempted",summary["not_attempted"])]
    content += '<div class="stack">' + ''.join(f'<span class="{name}" style="width:{count}%" title="{STATUSES[name]} {count}"></span>' for name,count in segments if count) + '</div>'
    content += '<section><h2>这个成绩说明什么</h2><p>它是本项目自建工单任务集的端到端目标成功率。采用结果验收思路，检查真实工具输出和数据库前后状态，并由另一次模型调用按冻结目标审阅回答。它衡量当前固定场景下的回归表现，不是官方公开 Agent 基准认证，也不能直接等同于线上企业收益。</p><p>任务只有同时满足目标完成、事实有据、权限安全和人工确认交接要求，才计为成功。等价措辞允许通过；没有强制唯一工具路径，必要的真实结果和安全约束仍必须满足。</p><div class="notice light">写操作的合格终点是 <b>pending 草稿／更新提案 + 正确提示本人打开独立页面确认</b>。模型没有完成正式审批；预先放入数据库的已解决工单不算它处理完成。</div></section>'
    if len(payload.get("lineage", [])) > 1 or manifest.get("semantic_regrade"):
        content += '<section><h2>运行与判分版本谱系</h2><p>首次运行发现部分语义评审缺少要求的证据字段，按原协议保留为未评分；随后通过一条诊断任务定位格式问题。新版统一重评全部继承证据，旧版本结果和费用均留存。没有重跑已执行的 Agent 任务，也没有从旧、新判断中挑选较高的成绩。</p>'
        if any(case.get("hard_history") for case in payload["cases"]):
            content += '<div class="notice light">新版同时修复来源引用形式识别造成的硬规则误报，并对继承的全部证据统一重算硬规则。旧、新硬规则和语义判断均可在逐题记录中核对。评分器修复带来的分数变化，不能解释为 Agent 能力提升。</div>'
        content += '<div class="table-wrap"><table><thead><tr><th>运行版本</th><th>语义协议</th><th>继承 Agent 执行</th><th>新增 Agent 执行</th><th>保存任务数</th><th>原始证据校验</th></tr></thead><tbody>'
        for node in payload["lineage"]:
            verified = "父版本 manifest/results 指纹一致" if node["source_artifacts_verified"] else "首个运行，无父版本"
            content += '<tr>' + ''.join(f'<td>{esc(value)}</td>' for value in [node["run_id"],node["semantic_protocol"],node["inherited_execution_count"],node["new_execution_count"],node["saved_task_count"],verified]) + '</tr>'
        content += '</tbody></table></div>'
        regrade = manifest.get("semantic_regrade")
        if regrade:
            content += f'<div class="notice light">继承任务统一重评：已完成 <b>{len(regrade.get("completed_case_ids",[]))} / {len(regrade.get("planned_case_ids",[]))}</b>；格式检查点：{"通过" if regrade.get("format_checkpoint_passed") else "尚未通过"}。TSR 只读取当前协议 outcome；尚未重评的任务继续列为未评分。</div>'
        content += details("版本修改与重评清单", '<pre>' + jsafe(payload["lineage"]) + '</pre>') + '</section>'
    content += '<section><h2>8 类任务分布</h2><div class="table-wrap"><table><thead><tr><th>类别</th><th>计划</th><th>成功</th><th>失败</th><th>未评分</th><th>未尝试</th><th>成功 / 计划</th></tr></thead><tbody>'
    for row in payload["categories"]:
        content += '<tr>' + ''.join(f'<td>{esc(value)}</td>' for value in [row["label"],row["planned"],row["success"],row["failure"],row["unscored"],row["not_attempted"],f'{row["success"]}/{row["planned"]} ({row["success_rate"]:.1%})']) + '</tr>'
    content += '</tbody></table></div></section>'
    content += '<section><h2>耗时应区分单消息和双消息任务</h2><p>从 Agent 执行开始至数据库最终快照完成；包含全部工具与回答，排除场景重置、语义评审耗时和等待人工确认。仅统计已完成执行的任务。P95 使用 nearest-rank：排序后第 ceil(0.95 × N) 个值。</p><table><thead><tr><th>范围</th><th>N</th><th>P50 / 秒</th><th>P95 / 秒</th></tr></thead><tbody>'
    for key, label in (("all_completed_tasks","全部完成任务"),("single_message","一次用户消息"),("two_message","两次用户消息合计")):
        row = payload["latency"][key]
        content += f'<tr><td>{label}</td><td>{row["n"]}</td><td>{fmt(row["p50_seconds"])}</td><td>{fmt(row["p95_seconds"])}</td></tr>'
    content += '</tbody></table></section>'
    content += f'<section><h2>Agent 执行与评审费用分开记</h2><p>以下来自代理捕获的服务商 usage；按峰时输入 2 元 / 百万 token、输出 8 元 / 百万 token、输入全部未命中缓存计算。仅为保守价格估算，不是实际账单。当前获授权累计预算上限为 {fmt(budget["limit_cny"])} 元。</p><div class="table-wrap"><table><thead><tr><th>阶段</th><th>请求 / 已结算</th><th>输入 token</th><th>输出 token</th><th>保守估算 / 元</th><th>待核实预留 / 元</th></tr></thead><tbody>'
    for row in budget["stages"].values():
        content += '<tr>' + ''.join(f'<td>{esc(value)}</td>' for value in [row["label"],f'{row["requests"]} / {row["settled_requests"]}',f'{row["input_tokens"]:,}',f'{row["output_tokens"]:,}',fmt(row["settled_estimate_cny"],4),fmt(row["unresolved_reserved_cny"],4)]) + '</tr>'
    content += f'</tbody></table></div><p><b>已结算保守估算：{fmt(budget["settled_estimate_cny"],4)} 元</b>；待核实预留：{fmt(budget["unresolved_reserved_cny"],4)} 元。每次转发前预留，缺失 usage 时保留费用并阻止后续请求。正在执行时，台账可包含尚未写入逐题结果的当前任务。</p></section>'
    if budget.get("amendments"):
        content += '<section><h2>预算追加有独立授权记录</h2><p>原5元额度曾实际触发拦截；之后明确追加最多0.50元，累计5.50元。追加仅用于未尝试的最后5条，不重复已执行或中断的任务。原台账快照与授权范围一并留存。</p>' + details("授权与原账本指纹", '<pre>' + jsafe(budget["amendments"]) + '</pre>') + '</section>'
    if len(payload.get("budget_phases", [])) > 1:
        content += '<section><h2>旧评审、诊断与重评费用都保留</h2><p>按唯一全局台账的连续请求区间分组，区间互不重叠。旧版无效评审仍然消耗过 API，统一重评也产生额外费用，全部占用同一个累计授权额度。不会将各版本报表中的累计金额重复相加。</p><div class="table-wrap"><table><thead><tr><th>阶段运行</th><th>台账区间（从 0 起）</th><th>Agent 请求</th><th>评审请求</th><th>已结算估算 / 元</th><th>待核实预留 / 元</th></tr></thead><tbody>'
        for row in payload["budget_phases"]:
            content += '<tr>' + ''.join(f'<td>{esc(value)}</td>' for value in [row["run_id"],f'[{row["ledger_start_index"]}, {row["ledger_end_exclusive"]})',row["agent_requests"],row["judge_requests"],fmt(row["settled_estimate_cny"],4),fmt(row["unresolved_reserved_cny"],4)]) + '</tr>'
        content += '</tbody></table></div></section>'
    content += '<section><h2>失败和未评分如何解释</h2><p>失败意味着至少一个必要目标、数据状态、安全约束或回答依据未通过。未评分意味着证据或评审输出不足以作出合格判定；不能据此断言 Agent 回答错误，也不能把它补成成功。两者在逐题列表中分开显示。</p>'
    nonpass = [case for case in payload["cases"] if case["status"] in {"fail","unscored"}]
    if nonpass:
        content += '<div class="table-wrap"><table><thead><tr><th>任务</th><th>结果</th><th>具体原因</th></tr></thead><tbody>' + ''.join(f'<tr><td>{esc(case["case_id"])}</td><td>{STATUSES[case["status"]]}</td><td class="reason">{esc(reason_text(case))}</td></tr>' for case in nonpass) + '</tbody></table></div>'
    else:
        content += '<p>当前已保存结果中没有失败或未评分项；请同时检查是否已经完成 100 条。</p>'
    content += '</section><section><h2>能够用于简历的表述</h2>'
    if final:
        sentence = f'基于 LibreChat 二次开发企业客服工单 Agent，建立覆盖 8 类场景的 100 条自建历史回归任务（计划110次消息，实际完整完成{summary["completed_messages"]}次），通过数据库状态、安全约束与模型辅助语义验收联合评估，原始严格判分确认任务成功率 {percent:.0f}%（{summary["success"]}/100），失败 {summary["failure"]} 条、未评分 {summary["unscored"]} 条；本机串行实测完整任务P95={payload["latency"]["all_completed_tasks"]["p95_seconds"]:.2f}秒（{summary["completed"]}条），写操作采用草稿／提案与独立人工确认。'
        content += '<pre class="resume">' + esc(sentence) + '</pre><p class="small">面试时需主动说明：这是自建已见回归集和模型辅助评分，不是官方 benchmark，也不是 100 条人工验收。业务使用价值与测试成绩分开陈述。</p>'
    else:
        content += '<div class="notice">当前运行未结束，不提供带最终成功率的简历句子。可先写：建立覆盖 8 类业务与安全场景的 100 条 Agent 回归任务，联合校验数据库状态、权限约束与回答依据。</div>'
    content += '</section><section><h2>逐题证据与复核</h2><p>展开可看用户问题、完整回答、真实工具参数与返回、数据库变化以及冻结目标。测试禁披露标记和凭证已隐藏。建议优先查看失败／未评分项，再检查预先规定的成功样本。</p><div class="filters"><label>结果 <select id="status"><option value="">全部</option>' + ''.join(f'<option value="{key}">{value}</option>' for key,value in STATUSES.items()) + '</select></label><label>类别 <select id="category"><option value="">全部</option>' + ''.join(f'<option value="{key}">{value}</option>' for key,value in CATEGORIES.items()) + '</select></label><label>查找 <input id="search" placeholder="任务编号或用户问题"></label><span id="visible"></span></div><div id="cases">'
    content += ''.join(render_case(case) for case in payload["cases"]) + '</div></section>'
    content += '<section><h2>边界与可追溯性</h2>' + bullets(payload["limitations_zh"])
    content += details("冻结运行信息、价格和源码指纹", '<pre>' + jsafe(manifest) + '</pre>')
    content += details("建议复核队列", '<pre>' + jsafe(payload.get("review_queue", [])) + '</pre>')
    content += '<p class="small">报告仅读取运行结果，未调用模型、改动判分或重置数据库。逐题答案与工具证据保存在本地运行目录：<br>' + esc(payload["run_directory"]) + '</p></section></main><footer>任务成功率 × 可复核证据 × 真实口径</footer>'
    css = '''*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:#18243b;font:15px/1.75 "Segoe UI","Microsoft YaHei",sans-serif}header,main,footer{max-width:1180px;margin:auto}header{padding:52px 28px 38px}.eyebrow{font-size:12px;letter-spacing:2px;color:#536b91;font-weight:700}h1{font-size:38px;line-height:1.35;margin:12px 0 16px}.lead{font-size:17px;max-width:850px}.run,.small,.meta{font-size:12px;color:#596c83}.run{padding-top:16px}main{padding:0 28px}section{background:white;border:1px solid #dce4ef;border-radius:16px;padding:28px;margin:24px 0;box-shadow:0 3px 14px #18365405}h2{font-size:23px;line-height:1.4;margin:0 0 16px}h4{font-size:15px;margin:20px 0 8px}p{margin:10px 0}.hero{background:#142b4d;color:#fff;display:flex;align-items:center;justify-content:space-between;gap:20px}.hero h2{font-size:38px;margin:8px 0}.sub{font-size:13px;color:#c8d8f1}.formula{min-width:230px;border:1px solid #ffffff30;padding:20px;border-radius:12px;font-size:17px}.formula strong,.formula span{display:block}.formula strong{font-size:28px}.formula span{font-size:12px;color:#c8d8f1}.cards{display:grid;grid-template-columns:repeat(7,1fr);gap:10px}.cards>div{background:#fff;border:1px solid #dce4ef;border-radius:12px;padding:14px}.cards span{display:block;font-size:12px;color:#596c83}.cards strong{font-size:28px}.notice{padding:16px 20px;border:1px solid #ebd5a6;background:#fff8e7;color:#684e16;border-radius:10px;margin:16px 0}.notice.light{background:#eef4ff;border-color:#d8e6ff;color:#264a7b}.stack{display:flex;height:12px;overflow:hidden;border-radius:8px;margin-top:20px}.pass{background:#dff4e9;color:#136548}.fail{background:#ffe3e3;color:#a12932}.unscored{background:#fff0ca;color:#8a5d08}.not_attempted{background:#e8edf4;color:#5d6c81}.stack .pass{background:#2ca677}.stack .fail{background:#de5c65}.stack .unscored{background:#dfab3d}.stack .not_attempted{background:#c9d3df}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;background:#edf2f8;color:#344d6c;font-weight:600}th,td{padding:12px 10px;border-bottom:1px solid #e1e8f1;vertical-align:top}.reason{white-space:pre-wrap}details{border:1px solid #dce4ef;border-radius:9px;margin:12px 0;padding:0 14px}summary{cursor:pointer;padding:12px 0;font-weight:600}details[open]>summary{border-bottom:1px solid #e1e8f1}.case{background:#fcfdff;margin:14px 0}.case>summary{font-size:14px}.badge{font-size:12px;padding:3px 9px;border-radius:20px;margin-right:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;border:1px solid #e2e8ef;padding:16px;border-radius:9px;font:13px/1.8 "Consolas","Microsoft YaHei",monospace;max-height:640px;overflow:auto}.question{background:#eef3fb}.answer{background:#fff;max-height:none}.resume{background:#eaf6ef;font-family:inherit;font-size:15px}.filters{display:flex;flex-wrap:wrap;gap:14px;align-items:center;position:sticky;top:0;background:#fff;padding:12px 0;z-index:1}select,input{font:inherit;border:1px solid #c8d4e4;border-radius:7px;padding:7px;background:#fff;max-width:240px}#visible{font-size:12px;color:#596c83}li{margin:8px 0}footer{text-align:center;padding:30px;color:#63748c;font-size:12px}.case[hidden]{display:none}@media(max-width:760px){header{padding-top:30px}header,main{padding-left:16px;padding-right:16px}h1{font-size:29px}section{padding:20px}.hero{display:block}.formula{margin-top:18px}.cards{grid-template-columns:repeat(3,1fr)}.hero h2{font-size:30px}.filters{position:static}}@media print{body{background:#fff}.filters{display:none}section{break-inside:avoid;box-shadow:none}header{padding-top:0}.case{break-inside:auto}.case[hidden]{display:none}}'''
    script = '''const status=document.getElementById('status'),category=document.getElementById('category'),search=document.getElementById('search');function filter(){let n=0;for(const row of document.querySelectorAll('.case')){const ok=(!status.value||row.dataset.status===status.value)&&(!category.value||row.dataset.category===category.value)&&(!search.value||row.dataset.search.toLowerCase().includes(search.value.toLowerCase()));row.hidden=!ok;if(ok)n++;}document.getElementById('visible').textContent=`显示 ${n} / 100 条`;}for(const el of [status,category,search])el.addEventListener('input',filter);filter();'''
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + esc(TITLE) + '</title><style>' + css + '</style></head><body>' + content + '<script>' + script + '</script></body></html>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Defaults to .local/task-success-v1/latest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    folder = args.run_dir or Path(read_json(LOCAL / "latest.json")["run_directory"])
    payload = make_payload(folder)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, content in ((".json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n"), (".html", render(payload))):
        path = args.output_dir / (TITLE + suffix)
        temporary = path.with_suffix(suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({"status": "final" if payload["is_final"] else "snapshot", "attempted": payload["summary"]["attempted"], "planned": 100, "output": str(args.output_dir / (TITLE + ".html"))}, ensure_ascii=True))


if __name__ == "__main__":
    main()
