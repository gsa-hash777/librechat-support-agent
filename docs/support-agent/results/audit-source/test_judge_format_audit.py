import json
from copy import deepcopy
import pytest
from audit_judge_format import DIMENSIONS, normalize, audit_row


def verdict(status="pass"):
    return {"dimensions": {key: {"status": status, "reason": "说明 {原文}，保持不变。", "evidence": "actual_answers[0].text: 明确说明。"} for key in DIMENSIONS}, "reasons": ["完整回答"]}


def misplaced(value):
    return {"dimensions": {**value["dimensions"], "reasons": value["reasons"]}}


@pytest.mark.parametrize("status", ["pass", "fail", "uncertain"])
@pytest.mark.parametrize("missing_brace", [False, True])
def test_only_structure_changes_all_status_values_are_preserved(status, missing_brace):
    original = verdict(status)
    raw = json.dumps(misplaced(original), ensure_ascii=False)
    result = normalize(raw[:-1] if missing_brace else raw, "stop")
    assert json.loads(result["normalized_json"]) == original
    assert result["semantic"]["status"] == ("unscored" if status == "uncertain" else status)
    assert result["dimension_values_unchanged"]
    assert len(result["operations"]) == (2 if missing_brace else 1)


def test_one_outer_brace_can_be_repaired_without_relocating_reasons():
    original = verdict()
    result = normalize(json.dumps(original)[:-1], "stop")
    assert json.loads(result["normalized_json"]) == original
    assert len(result["operations"]) == 1


@pytest.mark.parametrize("finish", [None, "length", "error", "tool_calls"])
def test_incomplete_provider_response_is_never_repaired(finish):
    assert normalize(json.dumps(misplaced(verdict()))[:-1], finish) is None


@pytest.mark.parametrize("defect", ["missing_evidence", "extra_dimension", "missing_dimension", "invalid_status", "extra_root", "two_braces", "unfinished_string", "duplicate_key"])
def test_ambiguous_or_semantically_incomplete_outputs_stay_unscored(defect):
    value = misplaced(verdict())
    first = DIMENSIONS[0]
    if defect == "missing_evidence": del value["dimensions"][first]["evidence"]
    if defect == "extra_dimension": value["dimensions"]["unknown"] = {}
    if defect == "missing_dimension": del value["dimensions"][first]
    if defect == "invalid_status": value["dimensions"][first]["status"] = "success"
    if defect == "extra_root": value["extra"] = "not allowed"
    raw = json.dumps(value)
    if defect == "two_braces": raw = raw[:-2]
    if defect == "unfinished_string": raw = raw[:raw.index('actual_answers') + 5]
    if defect == "duplicate_key": raw = raw.replace('"status": "pass"', '"status": "fail", "status": "pass"', 1)
    assert normalize(raw, "stop") is None


def test_valid_original_response_is_not_reconsidered():
    assert normalize(json.dumps(verdict()), "stop") is None


def test_hard_violation_remains_failure_after_semantic_format_fix():
    row = {"case_id": "X", "category": "test", "outcome": {"status": "fail"},
           "hard": {"input_valid": True, "passed": False, "checks": {"safety": {"passed": False}}},
           "semantic": {"status": "unscored"},
           "judge_record": {"finish_reason": "stop", "raw_response_redacted": json.dumps(misplaced(verdict()))}}
    original = deepcopy(row)
    audited = audit_row(row)
    assert audited["after"]["status"] == "fail"
    assert audited["format_repair_applied"]
    assert row == original
