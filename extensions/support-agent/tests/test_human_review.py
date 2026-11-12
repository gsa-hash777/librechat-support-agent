"""No model, server, container or real local-state access is required."""
import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from evaluation.review import (
    DECLARATION, KIND, ReviewError, main, render_review_html, validate_decisions,
)


@pytest.fixture
def sources():
    ids = [f"T{i:03d}" for i in range(100)]
    report = {
        "manifest": {"run_id": "unit-only", "planned_cases": 100, "case_ids": ids},
        "results": [{"case_id": cid, "category": "test", "turns": [{"prompt": "问题"}],
                     "evidence": {"answers": [{"text": "回答"}], "tool_calls": []},
                     "fixture_before": {"rows": {"tickets": [], "knowledge_docs": []}},
                     "fixture_after": {"rows": {"tickets": [], "knowledge_docs": []}},
                     "judgement": {"contract_pass": True, "strict_rule_pass": True}}
                    for cid in ids],
    }
    refs = {"rules": [{"id": "R1", "rule": "逐题核对"}], "policy": [["pending", "未评"]],
            "cases": [{"case_id": cid, "semantic_verdict": "pass", "reasons": ["辅助意见"],
                       "reference_expectations": {"necessary_facts": ["须查证"], "acceptable_answers": [],
                                                  "forbidden_inferences": [], "tool_constraints": []}}
                      for cid in ids]}
    return report, refs


@pytest.fixture
def decision_file(sources):
    report, _ = sources
    return {"schema_version": 1, "kind": KIND, "report_sha256": "a" * 64, "reference_sha256": "b" * 64,
            "reviewer": {"name": "测试复核者", "human_self_attestation": True, "declaration": DECLARATION},
            "signed_at": "2026-11-09T00:00:00Z",
            "decisions": [{"case_id": c["case_id"], "verdict": "pending", "reason": "",
                           "reference_assessment": "pending", "reference_notes": "",
                           "rule_assessment": "pending", "rule_notes": ""} for c in report["results"]]}


def validate(payload, sources):
    return validate_decisions(payload, *sources, report_sha256="a" * 64, reference_sha256="b" * 64)


def mark(item, verdict="pass"):
    item.update(verdict=verdict, reason="已核对工具返回与最终回答。", reference_assessment="agree", rule_assessment="agree")


def test_all_pending_never_inherits_agent_pass(sources, decision_file):
    out = validate(decision_file, sources)
    assert out["summary"]["reviewed"] == 0
    assert out["summary"]["pass"] == 0
    assert out["summary"]["pending"] == 100
    assert out["summary"]["self_attested_semantic_pass_rate"] is None
    assert out["reviewer"]["identity_verified"] is False


def test_partial_preserves_pending_and_has_no_accuracy(sources, decision_file):
    mark(decision_file["decisions"][0])
    mark(decision_file["decisions"][1], "fail")
    out = validate(decision_file, sources)
    assert out["summary"]["coverage"] == .02
    assert out["summary"]["pending"] == 98
    assert out["summary"]["self_attested_semantic_pass_rate"] is None


@pytest.mark.parametrize("field", ["report_sha256", "reference_sha256"])
def test_hash_drift_rejected(sources, decision_file, field):
    decision_file[field] = "c" * 64
    with pytest.raises(ReviewError, match="SHA-256"):
        validate(decision_file, sources)


@pytest.mark.parametrize("mutate", ["unknown", "duplicate", "missing", "extra"])
def test_complete_unique_exact_id_roster_required(sources, decision_file, mutate):
    if mutate == "unknown": decision_file["decisions"][0]["case_id"] = "unrelated"
    if mutate == "duplicate": decision_file["decisions"][0]["case_id"] = decision_file["decisions"][1]["case_id"]
    if mutate == "missing": decision_file["decisions"].pop()
    if mutate == "extra": decision_file["decisions"].append(copy.deepcopy(decision_file["decisions"][0]))
    with pytest.raises(ReviewError):
        validate(decision_file, sources)


@pytest.mark.parametrize("field,value", [("verdict", "approved"), ("verdict", None),
                                         ("reference_assessment", "yes"), ("rule_assessment", True)])
def test_decision_enums_are_explicit(sources, decision_file, field, value):
    decision_file["decisions"][0][field] = value
    with pytest.raises(ReviewError):
        validate(decision_file, sources)


@pytest.mark.parametrize("problem", ["no_reason", "rule_pending", "reference_pending", "revision_without_notes"])
def test_reviewed_case_requires_reason_and_reference_rule_opinion(sources, decision_file, problem):
    d = decision_file["decisions"][0]
    mark(d)
    if problem == "no_reason": d["reason"] = "  "
    if problem == "rule_pending": d["rule_assessment"] = "pending"
    if problem == "reference_pending": d["reference_assessment"] = "pending"
    if problem == "revision_without_notes": d["reference_assessment"] = "needs_revision"
    with pytest.raises(ReviewError):
        validate(decision_file, sources)


@pytest.mark.parametrize("problem", ["name", "attestation", "declaration", "timestamp"])
def test_signature_is_required_and_self_attested(sources, decision_file, problem):
    if problem == "name": decision_file["reviewer"]["name"] = " "
    if problem == "attestation": decision_file["reviewer"]["human_self_attestation"] = "true"
    if problem == "declaration": decision_file["reviewer"]["declaration"] = "I am an automated agent"
    if problem == "timestamp": decision_file["signed_at"] = "2026-11-09T00:00:00"
    with pytest.raises(ReviewError):
        validate(decision_file, sources)


def test_full_review_only_publishes_rate_without_uncertainty_or_dispute(sources, decision_file):
    for d in decision_file["decisions"]: mark(d)
    mark(decision_file["decisions"][0], "fail")
    assert validate(decision_file, sources)["summary"]["self_attested_semantic_pass_rate"] == .99
    mark(decision_file["decisions"][1], "uncertain")
    assert validate(decision_file, sources)["summary"]["self_attested_semantic_pass_rate"] is None
    mark(decision_file["decisions"][1])
    decision_file["decisions"][1].update(rule_assessment="needs_revision", rule_notes="规则过宽，需要重新界定。")
    assert validate(decision_file, sources)["summary"]["self_attested_semantic_pass_rate"] is None


def test_reference_id_mismatch_and_bad_template_rejected(sources, decision_file):
    report, refs = sources
    refs["cases"][0]["case_id"] = "other"
    with pytest.raises(ReviewError): validate(decision_file, sources)
    refs["cases"][0]["case_id"] = report["results"][0]["case_id"]
    refs["cases"][0]["reference_expectations"]["necessary_facts"] = "not a list"
    with pytest.raises(ReviewError): validate(decision_file, sources)


def test_structured_constraints_supported_and_reference_hash_binding_enforced(sources, decision_file):
    sources[1]["cases"][0]["reference_expectations"]["tool_constraints"] = {
        "required_tools": ["getMyTicket"], "expected_draft_delta": 0,
        "formal_state_must_remain_unchanged": True,
    }
    sources[1]["baseline_report_sha256"] = "a" * 64
    assert validate(decision_file, sources)["summary"]["pending"] == 100
    sources[1]["baseline_report_sha256"] = "c" * 64
    with pytest.raises(ReviewError, match="绑定"):
        validate(decision_file, sources)


def test_boolean_schema_version_is_not_numeric_version(sources, decision_file):
    decision_file["schema_version"] = True
    with pytest.raises(ReviewError):
        validate(decision_file, sources)


def test_html_is_inert_for_untrusted_json_and_offline(sources):
    report, refs = sources
    payload = '</script><script>fetch("https://bad.invalid")</script><img src=x onerror=alert(1)>&\u2028'
    report["results"][0]["evidence"]["answers"][0]["text"] = payload
    refs["cases"][0]["reference_expectations"]["necessary_facts"] = [payload]
    refs["cases"][0]["semantic_verdict"] = "fail"
    html = render_review_html(report, refs, report_sha256="a" * 64, reference_sha256="b" * 64)
    assert payload not in html
    assert '\\u003c/script\\u003e' in html
    assert "connect-src 'none'" in html and "default-src 'none'" in html
    assert "innerHTML" not in html and "eval(" not in html
    data = json.loads(re.search(r'<script id="reviewData" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert data["cases"][0]["answers"][0]["text"] == payload
    assert "verdict:'pending'" in html
    assert "localStorage.setItem" in html and "浏览器本地暂存失败" in html
    assert "node=el('details')" in html  # folded by default; no open attribute
    assert html.count("<script") == 2


def write_sources(tmp_path, sources):
    paths = [tmp_path / "baseline.json", tmp_path / "refs.json"]
    for path, data in zip(paths, sources): path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return paths


def test_cli_render_and_import_sidecar_do_not_mutate_inputs(tmp_path, sources, decision_file):
    report_path, ref_path = write_sources(tmp_path, sources)
    original = report_path.read_bytes(), ref_path.read_bytes()
    html_path, decision_path, sidecar_path = (tmp_path / name for name in ("review.html", "signed.json", "human.json"))
    main(["--report", str(report_path), "--references", str(ref_path), "--output", str(html_path)])
    assert html_path.exists()
    decision_file["report_sha256"] = hashlib.sha256(original[0]).hexdigest()
    decision_file["reference_sha256"] = hashlib.sha256(original[1]).hexdigest()
    mark(decision_file["decisions"][0])
    decision_path.write_text(json.dumps(decision_file), encoding="utf-8")
    decision_original = decision_path.read_bytes()
    main(["--report", str(report_path), "--references", str(ref_path), "--decisions", str(decision_path), "--adjudication", str(sidecar_path)])
    assert json.loads(sidecar_path.read_text(encoding="utf-8"))["summary"]["reviewed"] == 1
    assert (report_path.read_bytes(), ref_path.read_bytes()) == original
    assert decision_path.read_bytes() == decision_original


@pytest.mark.parametrize("target", ["report", "reference", "decision"])
def test_cli_import_cannot_overwrite_any_input(tmp_path, sources, decision_file, target):
    report_path, ref_path = write_sources(tmp_path, sources)
    decision_file["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    decision_file["reference_sha256"] = hashlib.sha256(ref_path.read_bytes()).hexdigest()
    decision_path = tmp_path / "signed.json"
    decision_path.write_text(json.dumps(decision_file), encoding="utf-8")
    destinations = {"report": report_path, "reference": ref_path, "decision": decision_path}
    originals = {p: p.read_bytes() for p in destinations.values()}
    with pytest.raises(SystemExit) as exc:
        main(["--report", str(report_path), "--references", str(ref_path), "--decisions", str(decision_path), "--adjudication", str(destinations[target])])
    assert exc.value.code == 1
    assert all(p.read_bytes() == data for p, data in originals.items())


def test_cli_bad_import_writes_no_sidecar(tmp_path, sources, decision_file):
    report_path, ref_path = write_sources(tmp_path, sources)
    path, output = tmp_path / "bad.json", tmp_path / "human.json"
    path.write_text(json.dumps(decision_file), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--report", str(report_path), "--references", str(ref_path), "--decisions", str(path), "--adjudication", str(output)])
    assert not output.exists()
