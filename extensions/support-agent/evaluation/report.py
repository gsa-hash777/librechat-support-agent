"""Aggregate frozen Agent evaluations without hiding failures or missing costs."""
from collections import Counter
from collections.abc import Mapping
from html import escape
import json
import math

from .usage import aggregate_usage, normalize_usage


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _nearest_rank(values, percentile):
    return sorted(values)[max(0, math.ceil(len(values) * percentile) - 1)] if values else None


def _errors(result):
    evidence = _mapping(result.get("evidence"))
    return [str(error) for error in (result.get("error"), evidence.get("execution_error")) if error]


def _completed(result):
    return not _errors(result) and bool(_mapping(result.get("judgement")))


def _metrics(rows, planned):
    completed = sum(_completed(row) for row in rows)
    metrics = {"planned": planned, "attempted": len(rows), "completed": completed,
               "execution_failed": sum(bool(_errors(row)) for row in rows),
               "ungraded": sum(not _errors(row) and not _mapping(row.get("judgement")) for row in rows)}
    for label, key in (("strict_rule", "strict_rule_pass"), ("contract", "contract_pass"), ("heuristic", "heuristic_pass")):
        passed = sum(_completed(row) and _mapping(row.get("judgement")).get(key) is True for row in rows)
        judged = sum(_completed(row) and isinstance(_mapping(row.get("judgement")).get(key), bool) for row in rows)
        metrics[label] = {"passed": passed, "judged": judged,
                          "rate_of_planned": passed / planned if planned else None,
                          "rate_of_attempted": passed / len(rows) if rows else None,
                          "rate_of_completed": passed / completed if completed else None,
                          "rate_of_judged": passed / judged if judged else None}
    return metrics


def _cost_summary(rows):
    currencies = sorted({cost.get("currency") for row in rows
                         if (cost := _mapping(row.get("cost"))).get("currency") in {"CNY", "USD"}})
    known = []
    for row in rows:
        cost = _mapping(row.get("cost"))
        usage = _mapping(row.get("usage"))
        values = tuple(_number(cost.get(key)) for key in ("lower_bound", "upper_bound", "conservative_budget"))
        measured = usage.get("source") != "missing" and all(_number(usage.get(key)) is not None for key in ("input_tokens", "output_tokens"))
        if measured and cost.get("status") in {"estimate", "price_based_estimate"} and cost.get("currency") in {"CNY", "USD"} and all(value is not None for value in values):
            if values[0] <= values[1] <= values[2]:
                known.append((cost["currency"], values))
    by_currency = {}
    for currency in currencies:
        selected = [values for actual, values in known if actual == currency]
        by_currency[currency] = {
            "priced_cases": len(selected),
            "known_lower_subtotal": sum(value[0] for value in selected) if selected else None,
            "known_upper_subtotal": sum(value[1] for value in selected) if selected else None,
            "known_conservative_subtotal": sum(value[2] for value in selected) if selected else None,
        }
    complete = bool(rows) and len(known) == len(rows) and len(currencies) == 1
    totals = by_currency[currencies[0]] if complete else {}
    return {"currency": currencies[0] if len(currencies) == 1 else None,
            "actual_charge": False, "attempted": len(rows), "priced_cases": len(known),
            "unknown_cases": len(rows) - len(known), "complete": complete,
            "lower_bound": totals.get("known_lower_subtotal"),
            "upper_bound": totals.get("known_upper_subtotal"),
            "conservative_budget": totals.get("known_conservative_subtotal"),
            "known_subtotals_by_currency": by_currency,
            "status": "price_based_estimate" if complete else "mixed_currencies_not_combined" if len(currencies) > 1 else "incomplete_cost_coverage"}


def _failure(result):
    reasons = _errors(result)
    judgement = _mapping(result.get("judgement"))
    for name, check in _mapping(judgement.get("checks")).items():
        check = _mapping(check)
        if check.get("passed") is False:
            reasons.append(f"{name}: {check.get('reason') or '规则未通过'}")
    if not reasons and judgement.get("strict_rule_pass") is False:
        reasons.append("严格规则未通过；请核对原始判断记录。")
    if not reasons and not judgement:
        reasons.append("缺少规则判断记录。")
    return {"case_id": result["case_id"], "category": result.get("category", "未分类"),
            "reasons": list(dict.fromkeys(reasons))} if reasons else None


def build_report(manifest, results):
    """Use the frozen manifest as the primary denominator; preserve all results.

    Each result is one attempted frozen case, including execution failures.
    Duplicate/unplanned IDs are rejected so retries cannot alter the denominator.
    Raw result evidence is retained in JSON; only the HTML preview is shortened.
    """
    manifest = dict(manifest)
    results = list(results)
    planned = manifest.get("planned_cases")
    if isinstance(planned, bool) or not isinstance(planned, int) or planned < 0:
        raise ValueError("manifest.planned_cases must be a nonnegative integer.")
    planned_ids = manifest.get("case_ids", [])
    if not isinstance(planned_ids, list) or any(not isinstance(identifier, str) or not identifier for identifier in planned_ids):
        raise ValueError("manifest.case_ids must contain nonempty string IDs.")
    if len(set(planned_ids)) != len(planned_ids) or planned_ids and len(planned_ids) != planned:
        raise ValueError("Frozen case IDs must be unique and match planned_cases.")
    identifiers = [row.get("case_id") for row in results]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("Each result requires a nonempty case_id.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate case results would corrupt evaluation denominators.")
    if len(results) > planned or planned_ids and set(identifiers) - set(planned_ids):
        raise ValueError("Results must belong to the frozen case manifest.")

    metrics = _metrics(results, planned)
    metrics["not_attempted"] = planned - len(results)
    categories = {}
    for category in sorted({str(row.get("category", "未分类")) for row in results}):
        rows = [row for row in results if str(row.get("category", "未分类")) == category]
        # The manifest only lists IDs, so no invented per-category planned count.
        categories[category] = _metrics(rows, None)
    latencies = [value for row in results if (value := _number(row.get("elapsed_seconds"))) is not None]
    review = Counter(_mapping(row.get("judgement")).get("semantic_review", "unavailable") for row in results)
    failures = [failure for row in results if (failure := _failure(row)) is not None]
    usage = aggregate_usage([row.get("usage") if isinstance(row.get("usage"), Mapping)
                             else normalize_usage(model=manifest.get("model")) for row in results])
    return {
        "schema_version": 1, "manifest": manifest, "summary": metrics,
        "categories": categories,
        "latency_seconds": {"method": "nearest-rank", "scope": "all_attempted_including_failed",
                            "observed": len(latencies), "missing": len(results) - len(latencies),
                            "p50": _nearest_rank(latencies, .5), "p95": _nearest_rank(latencies, .95)},
        "usage": usage, "cost": _cost_summary(results),
        "semantic_review": {"pending": review["pending"], "unavailable": review["unavailable"],
                            "statuses": dict(review)},
        "failures": failures, "results": results,
        "limitations": [
            "数据为合成业务场景；运行前冻结用例不等于独立盲测。",
            "严格规则、接口约束和文本启发式检查分别统计，均不等于语义准确率。",
            "人工语义复核仍需完成；不得把自动规则通过率称为100%语义准确。",
            "单并发本地实验的延迟不能代表生产吞吐或并发性能。",
            "费用是价格表估算，不是服务商账单；未知用量或费用不可记为0。",
            "软件单元测试验证代码约束，真实Agent任务评测验证模型执行表现，两者不能混为一个通过率。",
            "延迟按全部已尝试样本计算（包含失败），P50/P95采用nearest-rank；缺失延迟另计。",
            "用量只涵盖捕获或保存的模型调用；独立标题任务及未返回用量的失败调用可能不在记录内。",
        ],
    }


def _text(value):
    return escape(str(value), quote=True)


def _json(value):
    return _text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _display(value, precision=3):
    return "未知" if value is None else f"{value:.{precision}f}"


def _preview(result):
    evidence = _mapping(result.get("evidence"))
    texts = [str(_mapping(answer).get("text", "")) for answer in evidence.get("answers", [])]
    answer = "\n\n".join(texts)
    if len(answer) > 1500:
        answer = answer[:1500] + "\n……预览截断；完整原文保留在JSON报告。"
    facts = _mapping(_mapping(result.get("judgement")).get("facts"))
    names = facts.get("tool_names")
    if names is None:
        names = [call.get("name") or _mapping(call.get("function")).get("name")
                 for call in evidence.get("tool_calls", []) if isinstance(call, Mapping)]
    delta = evidence.get("db_delta", facts.get("db_delta"))
    if delta is None:
        before = _mapping(_mapping(result.get("fixture_before")).get("rows"))
        after = _mapping(_mapping(result.get("fixture_after")).get("rows"))
        if before and after:
            delta = {table: {"before": len(before[table]), "after": len(after[table]),
                             "row_count_delta": len(after[table]) - len(before[table])}
                     for table in ("drafts", "tickets", "ticket_updates", "ticket_events", "audit_events")
                     if isinstance(before.get(table), list) and isinstance(after.get(table), list)}
        else:
            delta = facts if facts else "未提供数据库变化摘要；请查JSON中的证据路径。"
    return f'<details><summary>{_text(result["case_id"])} · {_text(result.get("category", "未分类"))}：原文与执行证据</summary><h4>模型回答（预览）</h4><pre>{_text(answer or "未返回回答")}</pre><h4>工具名称</h4><pre>{_json(names)}</pre><h4>数据库变化 / 事实摘要</h4><pre>{_json(delta)}</pre><h4>规则判断</h4><pre>{_json(result.get("judgement"))}</pre></details>'


def render_html(report):
    """Render escaped, self-contained Chinese HTML without JavaScript or links."""
    summary, manifest = report["summary"], report["manifest"]
    latency, cost = report["latency_seconds"], report["cost"]
    labels = {"strict_rule": "严格规则", "contract": "接口约束", "heuristic": "文本启发式"}
    rows = []
    for key, label in labels.items():
        value = summary[key]
        planned_rate = "未知" if value["rate_of_planned"] is None else f'{value["rate_of_planned"]:.1%}'
        rows.append(f'<tr><th>{label}</th><td>{value["passed"]} / {summary["planned"]}</td><td>{planned_rate}</td><td>{value["passed"]} / {summary["attempted"]}</td><td>{value["passed"]} / {summary["completed"]}</td></tr>')
    category_rows = ''.join(f'<tr><th>{_text(name)}</th><td>{value["attempted"]}</td><td>{value["completed"]}</td><td>{value["strict_rule"]["passed"]}</td><td>{value["contract"]["passed"]}</td><td>{value["heuristic"]["passed"]}</td></tr>' for name, value in report["categories"].items())
    failure_rows = ''.join(f'<li><strong>{_text(item["case_id"])}</strong><ul>' + ''.join(f'<li>{_text(reason)}</li>' for reason in item["reasons"]) + '</ul></li>' for item in report["failures"])
    limits = ''.join(f'<li>{_text(note)}</li>' for note in report["limitations"])
    details = ''.join(_preview(result) for result in report["results"])
    currency = cost["currency"] or "未确认币种"
    cost_text = f'{_display(cost["lower_bound"], 6)} — {_display(cost["upper_bound"], 6)} {_text(currency)}'
    status_note = "评测已完成" if manifest.get("status") == "completed" and summary["attempted"] == summary["planned"] else "评测尚未完成或已提前停止"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'"><title>Agent 真实任务评测报告</title><style>
body{{font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif;background:#f3f6fa;color:#17283e;margin:0}}main{{max-width:1120px;margin:auto;padding:38px 24px}}h1{{font-size:30px}}h2{{font-size:21px;margin-top:0}}section,details{{background:white;border:1px solid #dce4ec;border-radius:12px;padding:24px;margin:20px 0}}.notice{{background:#fff7df;border-color:#ecd38b}}.metrics{{display:flex;gap:20px;flex-wrap:wrap}}.metrics div{{flex:1;min-width:150px}}.metrics strong{{font-size:28px;display:block}}.muted{{color:#52647b}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #dce4ec;padding:12px;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:16px;border-radius:8px;font:13px/1.6 monospace}}summary{{cursor:pointer;font-weight:600}}.scroll{{overflow:auto}}@media(max-width:650px){{main{{padding:22px 14px}}section,details{{padding:18px}}}}
</style></head><body><main><p class="muted">真实模型调用 · 独立本地评测环境</p><h1>Agent 真实任务评测报告</h1><p>{_text(status_note)} · run_id：{_text(manifest.get("run_id", "未知"))}</p>
<section class="notice"><h2>先看统计边界</h2><p>合成数据：{_text(manifest.get("synthetic", "未声明"))}；运行前冻结：{_text(manifest.get("frozen_before_run", "未声明"))}；数据集：{_text(manifest.get("dataset", "未知"))}。本报告不属于独立盲测，也不声明语义准确率。</p><p>人工语义复核待办：<strong>{report["semantic_review"]["pending"]}</strong>；缺少复核状态：{report["semantic_review"]["unavailable"]}。并发设置：{_text(manifest.get("concurrency", "未知"))}。</p><ul>{limits}</ul></section>
<section><h2>样本分母</h2><div class="metrics"><div>原计划<strong>{summary["planned"]}</strong></div><div>已尝试<strong>{summary["attempted"]}</strong></div><div>完成执行与判定<strong>{summary["completed"]}</strong></div><div>尚未尝试<strong>{summary["not_attempted"]}</strong></div></div><p>执行失败：{summary["execution_failed"]}；执行返回但缺少判断：{summary["ungraded"]}。停止原因：{_text(manifest.get("stop_reason") or "无")}</p><div class="scroll"><table><thead><tr><th>自动检查</th><th>通过 / 原计划</th><th>占原计划</th><th>通过 / 已尝试</th><th>通过 / 已完成</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>分类结果</h2><p class="muted">分类按已尝试样本统计；未运行类别不会被当作通过。</p><div class="scroll"><table><thead><tr><th>类别</th><th>已尝试</th><th>已完成</th><th>严格规则通过</th><th>接口约束通过</th><th>文本启发式通过</th></tr></thead><tbody>{category_rows}</tbody></table></div></section>
<section><h2>延迟、用量与估算费用</h2><p>全部已尝试样本（包括失败）的 P50：{_display(latency["p50"])} 秒；P95：{_display(latency["p95"])} 秒。方法：nearest-rank；有记录 {latency["observed"]} 条，缺失 {latency["missing"]} 条。</p><p>费用估算区间：<strong>{cost_text}</strong>；保守预算：{_display(cost["conservative_budget"], 6)} {_text(currency)}。已知费用 {cost["priced_cases"]}/{cost["attempted"]} 条；未知 {cost["unknown_cases"]} 条。任一未知费用会使总费用上界保持未知，已知小计不能代替总额。</p><details><summary>Token 统计与覆盖率</summary><pre>{_json(report["usage"])}</pre></details><details><summary>费用区间与已知小计</summary><pre>{_json(cost)}</pre></details></section>
<section><h2>失败或未完成的样本</h2><ul>{failure_rows or '<li>当前已尝试样本未记录规则失败；仍需检查未运行样本与待人工复核项。</li>'}</ul></section>
<section><h2>冻结配置与来源</h2><p class="muted">以下路径、标识和哈希仅作为文本展示。</p><pre>{_json(manifest)}</pre></section><h2>逐条执行证据</h2>{details}<p class="muted">HTML 只缩短回答预览；JSON 报告保留传入的完整原始证据。软件测试通过数不计入本报告的 Agent 任务分母。</p></main></body></html>'''
