from copy import deepcopy

import pytest

from evaluation.report import build_report, render_html
from evaluation.usage import estimate_cost, normalize_usage


def manifest(count=4):
    return {"run_id": "frozen-test", "planned_cases": count,
            "case_ids": [f"C{number}" for number in range(count)],
            "dataset": "test", "synthetic": True, "frozen_before_run": True,
            "status": "running", "concurrency": 1, "model": "test-model"}


def result(number, *, strict=True, contract=True, heuristic=True, elapsed=1, error=None, currency="CNY"):
    usage = normalize_usage(usage_events=[{"input_tokens": 100, "output_tokens": 20,
        "input_token_details": {"cache_read": 20}, "model": "test-model"}])
    cost = estimate_cost(usage, {"currency": currency, "model": "test-model", "input_per_million": 2,
                                "cache_hit_per_million": .5, "output_per_million": 4})
    return {"case_id": f"C{number}", "category": "knowledge" if number % 2 == 0 else "safety",
            "evidence": {"answers": [{"text": "这是测试回答"}], "tool_calls": [{"name": "searchSupportKnowledge"}]},
            "judgement": {"strict_rule_pass": strict, "contract_pass": contract, "heuristic_pass": heuristic,
                          "semantic_review": "pending", "checks": {}},
            "usage": usage, "cost": cost, "elapsed_seconds": elapsed, "error": error}


def test_primary_denominator_stays_frozen_and_failures_are_not_dropped():
    values = [result(0), result(1, strict=False, heuristic=False), result(2, error="timeout")]
    report = build_report(manifest(), values)
    summary = report["summary"]
    assert (summary["planned"], summary["attempted"], summary["completed"], summary["not_attempted"]) == (4, 3, 2, 1)
    assert summary["execution_failed"] == 1
    assert summary["strict_rule"]["passed"] == 1
    assert summary["strict_rule"]["rate_of_planned"] == .25
    assert summary["strict_rule"]["rate_of_attempted"] == pytest.approx(1 / 3)
    assert summary["contract"]["passed"] == 2
    assert summary["heuristic"]["passed"] == 1
    assert report["semantic_review"]["pending"] == 3


def test_missing_judgement_is_not_completed_and_missing_checks_are_not_passes():
    first, second = result(0), result(1)
    first["judgement"] = None
    second["judgement"] = {"semantic_review": "pending"}
    report = build_report(manifest(), [first, second])
    assert report["summary"]["completed"] == 1
    assert report["summary"]["ungraded"] == 1
    assert report["summary"]["strict_rule"]["passed"] == 0
    assert report["summary"]["strict_rule"]["judged"] == 0
    assert report["semantic_review"]["unavailable"] == 1


def test_p50_p95_nearest_rank_include_failed_latency():
    values = [result(number, elapsed=number + 1) for number in range(20)]
    values[-1]["error"] = "timeout"
    report = build_report(manifest(20), values)
    assert report["latency_seconds"] == {"method": "nearest-rank", "scope": "all_attempted_including_failed",
                                          "observed": 20, "missing": 0, "p50": 10, "p95": 19}
    assert report["summary"]["completed"] == 19


def test_invalid_latency_is_missing_not_zero():
    values = [result(0, elapsed=None), result(1, elapsed=-1), result(2, elapsed=True), result(3, elapsed=float("nan"))]
    latency = build_report(manifest(), values)["latency_seconds"]
    assert latency["missing"] == 4 and latency["p50"] is None and latency["p95"] is None


def test_known_costs_are_summed_without_changing_currency():
    values = [result(0), result(1)]
    cost = build_report(manifest(), values)["cost"]
    assert cost["complete"] is True and cost["actual_charge"] is False
    assert cost["lower_bound"] == pytest.approx(2 * values[0]["cost"]["lower_bound"])
    assert cost["conservative_budget"] == pytest.approx(2 * values[0]["cost"]["conservative_budget"])
    assert cost["currency"] == "CNY"


def test_missing_cost_makes_total_unknown_but_keeps_known_subtotal():
    values = [result(0), result(1)]
    values[1]["cost"] = None
    cost = build_report(manifest(), values)["cost"]
    assert cost["unknown_cases"] == 1 and cost["priced_cases"] == 1
    assert cost["lower_bound"] is None and cost["upper_bound"] is None and cost["conservative_budget"] is None
    assert cost["known_subtotals_by_currency"]["CNY"]["known_upper_subtotal"] == values[0]["cost"]["upper_bound"]


def test_missing_usage_cannot_be_hidden_as_zero_cost():
    row = result(0)
    row["usage"] = normalize_usage(model="test-model")
    row["cost"].update(lower_bound=0, upper_bound=0, conservative_budget=0)
    report = build_report(manifest(), [row])
    assert report["cost"]["unknown_cases"] == 1 and report["cost"]["upper_bound"] is None
    assert report["usage"]["source"] == "missing"


def test_explicit_zero_token_usage_can_have_known_zero_estimate():
    row = result(0)
    row["usage"] = normalize_usage(usage_events=[{"input_tokens": 0, "output_tokens": 0, "model": "test-model"}])
    row["cost"].update(lower_bound=0, upper_bound=0, conservative_budget=0)
    report = build_report(manifest(), [row])
    assert report["cost"]["upper_bound"] == 0 and report["cost"]["complete"] is True


def test_mixed_currencies_are_never_added_or_implicitly_converted():
    report = build_report(manifest(), [result(0), result(1, currency="USD")])
    cost = report["cost"]
    assert cost["currency"] is None and cost["upper_bound"] is None
    assert cost["status"] == "mixed_currencies_not_combined"
    assert set(cost["known_subtotals_by_currency"]) == {"CNY", "USD"}


def test_categories_use_attempted_counts_without_inventing_planned_categories():
    report = build_report(manifest(), [result(0), result(1, strict=False), result(2)])
    assert report["categories"]["knowledge"]["planned"] is None
    assert report["categories"]["knowledge"]["attempted"] == 2
    assert report["categories"]["safety"]["strict_rule"]["passed"] == 0


def test_failed_checks_and_execution_errors_are_listed():
    row = result(0, strict=False)
    row["judgement"]["checks"] = {"no_mutation": {"passed": False, "reason": "Unexpected update", "kind": "safety"}}
    report = build_report(manifest(), [row, result(1, error="stream disconnected")])
    assert report["failures"][0]["reasons"] == ["no_mutation: Unexpected update"]
    assert report["failures"][1]["reasons"] == ["stream disconnected"]


def test_html_escapes_model_output_tool_names_paths_and_errors():
    row = result(0, error='<img src=x onerror="alert(1)">')
    row["evidence"]["answers"] = [{"text": "</pre><script>alert(1)</script>"}]
    row["evidence"]["tool_calls"] = [{"name": '<svg onload="alert(1)">'}]
    frozen = manifest()
    frozen["source_path"] = 'x"><iframe src="https://attacker">'
    html = render_html(build_report(frozen, [row]))
    for unsafe in ("<script", "<img", "<svg", "<iframe", 'href="https://attacker'):
        assert unsafe not in html
    assert "&lt;script&gt;" in html and "&lt;img" in html
    assert "Content-Security-Policy" in html


def test_html_boundary_disclosures_and_raw_json_preservation():
    row = result(0)
    row["evidence"]["answers"] = [{"text": "完整证据" * 1000}]
    original = deepcopy(row)
    report = build_report(manifest(), [row])
    html = render_html(report)
    assert report["results"][0] == original
    assert "预览截断" in html
    for statement in ("合成数据", "不属于独立盲测", "人工语义复核待办", "单并发", "不是服务商账单", "软件单元测试", "全部已尝试", "未知用量或费用不可记为0"):
        assert statement in html


def test_empty_partial_run_retains_planned_denominator_and_unknown_cost():
    report = build_report(manifest(), [])
    assert report["summary"]["not_attempted"] == 4
    assert report["latency_seconds"]["p50"] is None
    assert report["cost"]["upper_bound"] is None
    assert "尚未完成" in render_html(report)


@pytest.mark.parametrize("frozen,rows", [
    (manifest(), [result(0), result(0)]),
    (manifest(), [result(9)]),
    ({**manifest(), "case_ids": ["C0", "C0", "C2", "C3"]}, []),
    ({**manifest(), "planned_cases": True}, []),
    ({**manifest(), "planned_cases": -1}, []),
    ({**manifest(), "case_ids": ["C0"]}, []),
])
def test_invalid_denominators_and_duplicate_results_rejected(frozen, rows):
    with pytest.raises(ValueError):
        build_report(frozen, rows)
