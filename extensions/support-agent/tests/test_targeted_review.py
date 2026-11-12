"""Frozen subset accounting and offline evidence handling, without model calls."""
import copy
import hashlib
import json
import re
import shutil
import subprocess

import pytest

from evaluation.review import DECLARATION, ReviewError
from evaluation.targeted_review import (
    KIND, SCOPE, SCRIPT, main, render_targeted_html, validate_targeted_decisions,
)


@pytest.fixture
def bundle():
    evidence = {"run_id": "run-old", "source_sha256": "a" * 64, "code_sha256": "b" * 64,
                "prompts": ["查询权限"], "answers": [{"text": "权限答复"}], "intermediate": [],
                "tool_calls": [], "database_summary": {}, "knowledge_docs": [], "auxiliary": {},
                "scores": {}, "checks": {}}
    cases = [{"case_id": cid, "category": "authorization", "selection_reasons": ["关键权限"],
              "review_expectations": {"necessary_facts": ["按真实身份查询"], "acceptable_answers": [],
                                      "forbidden_inferences": [], "tool_constraints": {}},
              "baseline": copy.deepcopy(evidence), "candidate": None,
              "candidate_notice": "尚无更新复测", "criterion_changes": []} for cid in ("A01", "R08")]
    cases[0]["candidate"] = {**copy.deepcopy(evidence), "run_id": "run-new", "source_sha256": "c" * 64}
    return {"schema_version": 1, "kind": "targeted-review-bundle", "selection": "只复核失败与权限",
            "policy_confirmed": ["员工只看本人"], "optional_full_review_file": "全部.html", "cases": cases}


@pytest.fixture
def payload(bundle):
    return {"schema_version": 1, "kind": KIND, "scope": SCOPE, "bundle_sha256": "d" * 64,
            "selected_case_ids": [c["case_id"] for c in bundle["cases"]],
            "signed_at": "2026-11-10T01:00:00+08:00",
            "reviewer": {"name": "复核者", "human_self_attestation": True, "declaration": DECLARATION},
            "decisions": [{"case_id": c["case_id"], "verdict": "pending", "reason": "",
                           "reference_assessment": "pending", "reference_notes": "",
                           "rule_assessment": "pending", "rule_notes": ""} for c in bundle["cases"]]}


def check(payload, bundle):
    return validate_targeted_decisions(payload, bundle, bundle_sha256="d" * 64)


def mark(value, verdict="pass"):
    value.update(verdict=verdict, reason="核对真实工具与数据库。", reference_assessment="agree", rule_assessment="agree")


def test_partial_counts_only_selected_without_inheriting_auxiliary(bundle, payload):
    out = check(payload, bundle)
    assert out["summary"] == {"selected": 2, "reviewed": 0, "pending": 2, "pass": 0,
                              "fail": 0, "uncertain": 0, "reference_or_rule_disputes": 0, "scope_complete": False}
    mark(payload["decisions"][0])
    out = check(payload, bundle)
    assert out["summary"]["reviewed"] == 1
    assert out["summary"]["pending"] == 1
    assert out["reviewer"]["identity_verified"] is False
    assert "accuracy" not in json.dumps(out) and "pass_rate" not in json.dumps(out)


def test_complete_means_reviewed_not_all_passed(bundle, payload):
    mark(payload["decisions"][0], "fail")
    mark(payload["decisions"][1], "uncertain")
    payload["decisions"][1].update(rule_assessment="needs_revision", rule_notes="应保留404歧义。")
    summary = check(payload, bundle)["summary"]
    assert summary["scope_complete"] is True
    assert summary["fail"] == summary["uncertain"] == summary["reference_or_rule_disputes"] == 1


@pytest.mark.parametrize("change", ["scope", "hash", "roster", "order", "kind", "boolean_version"])
def test_scope_or_bundle_drift_rejected(bundle, payload, change):
    if change == "scope": payload["scope"] = "all_cases"
    if change == "hash": payload["bundle_sha256"] = "e" * 64
    if change == "roster": payload["selected_case_ids"].append("A02")
    if change == "order": payload["selected_case_ids"].reverse()
    if change == "kind": payload["kind"] = "self-attested-human-review"
    if change == "boolean_version": payload["schema_version"] = True
    with pytest.raises(ReviewError): check(payload, bundle)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "unknown"])
def test_decisions_cover_exact_selected_roster(bundle, payload, change):
    if change == "missing": payload["decisions"].pop()
    if change == "extra": payload["decisions"].append(copy.deepcopy(payload["decisions"][0]))
    if change == "duplicate": payload["decisions"][1]["case_id"] = "A01"
    if change == "unknown": payload["decisions"][1]["case_id"] = "not-selected"
    with pytest.raises(ReviewError): check(payload, bundle)


@pytest.mark.parametrize("change", ["reason", "assessment", "revision", "name", "attestation", "time", "enum"])
def test_signed_review_needs_evidence_reason_and_explicit_assessments(bundle, payload, change):
    item = payload["decisions"][0]
    mark(item)
    if change == "reason": item["reason"] = " "
    if change == "assessment": item["rule_assessment"] = "pending"
    if change == "revision": item["reference_assessment"] = "needs_revision"
    if change == "name": payload["reviewer"]["name"] = ""
    if change == "attestation": payload["reviewer"]["human_self_attestation"] = "true"
    if change == "time": payload["signed_at"] = "2026-11-10T01:00:00"
    if change == "enum": item["verdict"] = "approved"
    with pytest.raises(ReviewError): check(payload, bundle)


@pytest.mark.parametrize("change", ["source_hash", "duplicate_id", "empty", "unsafe_link"])
def test_bundle_requires_traceable_sources_and_safe_optional_link(bundle, change):
    if change == "source_hash": bundle["cases"][0]["candidate"]["source_sha256"] = "missing"
    if change == "duplicate_id": bundle["cases"][1]["case_id"] = "A01"
    if change == "empty": bundle["cases"] = []
    if change == "unsafe_link": bundle["optional_full_review_file"] = "javascript:bad.html"
    with pytest.raises(ReviewError): render_targeted_html(bundle, bundle_sha256="d" * 64)


def test_offline_html_has_inert_evidence_and_selected_denominator(bundle):
    attack = '</script><script>fetch("https://bad.invalid")</script><img src=x onerror=alert(1)>'
    bundle["cases"][0]["candidate"]["answers"][0]["text"] = attack
    html = render_targeted_html(bundle, bundle_sha256="d" * 64)
    assert attack not in html and "\\u003c/script\\u003e" in html
    assert "connect-src 'none'" in html and "default-src 'none'" in html
    assert "innerHTML" not in html and "eval(" not in html
    data = json.loads(re.search(r'<script id="reviewData" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert data["bundle"]["cases"][0]["candidate"]["answers"][0]["text"] == attack
    assert "只复核失败样本与关键权限：2 题" in html
    assert "本轮只需审核选定的 2 题" in html
    assert "已确认权限规则（供对照，无需再确认）" in html
    assert "verdict:'pending'" in html and "localStorage.setItem" in html
    assert "浏览器本地暂存失败" in html and "其余原始题目不计入待办" in html
    assert "最近有记录的答案" in html and "无需再次独立评分" in html
    assert html.count("<script") == 2


def test_cli_keeps_inputs_and_imports_partial_separately(tmp_path, bundle, payload):
    bpath, dpath = tmp_path / "bundle.json", tmp_path / "signed.json"
    bpath.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    original = bpath.read_bytes()
    payload["bundle_sha256"] = hashlib.sha256(original).hexdigest()
    mark(payload["decisions"][0])
    dpath.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    decision_original = dpath.read_bytes()
    out, html = tmp_path / "human.json", tmp_path / "review.html"
    main(["--bundle", str(bpath), "--output", str(html)])
    main(["--bundle", str(bpath), "--decisions", str(dpath), "--adjudication", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["reviewed"] == 1
    assert bpath.read_bytes() == original and dpath.read_bytes() == decision_original
    assert html.exists()


@pytest.mark.parametrize("target", ["bundle", "decision"])
def test_cli_cannot_overwrite_inputs(tmp_path, bundle, payload, target):
    bpath, dpath = tmp_path / "bundle.json", tmp_path / "signed.json"
    bpath.write_text(json.dumps(bundle), encoding="utf-8")
    payload["bundle_sha256"] = hashlib.sha256(bpath.read_bytes()).hexdigest()
    dpath.write_text(json.dumps(payload), encoding="utf-8")
    paths = {"bundle": bpath, "decision": dpath}
    before = [bpath.read_bytes(), dpath.read_bytes()]
    with pytest.raises(SystemExit):
        main(["--bundle", str(bpath), "--decisions", str(dpath), "--adjudication", str(paths[target])])
    assert [bpath.read_bytes(), dpath.read_bytes()] == before


def test_cli_render_cannot_overwrite_bundle(tmp_path, bundle):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(SystemExit): main(["--bundle", str(path), "--output", str(path)])
    assert path.read_bytes() == original


def test_optional_expanded_review_does_not_become_required(bundle):
    bundle["optional_review"] = True
    html = render_targeted_html(bundle, bundle_sha256="d" * 64)
    assert "可选扩展复核：2 题（不计入本轮必审）" in html
    assert "这是可选扩展的 2 题，不计入本轮必审，无需完成。" in html
    assert "本轮只需审核选定的 2 题" not in html


def test_browser_state_navigation_restore_and_scope_validation(bundle, tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable; Python validation remains covered")
    script = tmp_path / "targeted.js"
    script.write_text(SCRIPT, encoding="utf-8")
    syntax = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    harness = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.listeners={};this.value='';this.textContent='';}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=nodes;}
 addEventListener(name,fn){this.listeners[name]=fn;}
 remove(){} click(){if(this.listeners.click)this.listeners.click();}
}
function setup(saved=new Map()){
 const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);};
 get('reviewData').textContent=JSON.stringify(input.data);get('filter').value='all';
 const context=vm.createContext({document:{getElementById:get,createElement:tag=>new Element(tag),createTextNode:text=>({textContent:text}),body:new Element('body')},localStorage:{getItem:key=>saved.get(key),setItem:(key,value)=>saved.set(key,value)},window:{scrollTo(){}},setTimeout:fn=>fn(),URL:{createObjectURL:()=>'',revokeObjectURL(){}},Blob:class{}});
 vm.runInContext(input.script,context);return {nodes,get,context,saved,run:s=>vm.runInContext(s,context)};
}
(async()=>{
 const app=setup();assert(app.get('progress').textContent.includes('0/2'));
 app.run("Object.assign(state.decisions[0],{verdict:'pass',reason:'checked',reference_assessment:'agree',rule_assessment:'agree'});state.reviewer.name='Reviewer';state.reviewer.human_self_attestation=true;save();validate(envelope(),true)");
 assert(app.get('progress').textContent.includes('1/2'));
 // Keep the current case while its reason is still being edited; Next moves to the pending case.
 app.run("$('filter').value='pending';selected='A01';populate()");
 assert.strictEqual(app.run('selected'),'A01');assert.strictEqual(app.run('visible().length'),2);
 app.get('next').listeners.click();assert.strictEqual(app.run('selected'),'R08');assert.strictEqual(app.run('visible().length'),1);
 app.get('filter').value='all';app.get('filter').listeners.change();assert.strictEqual(app.run('visible().length'),2);
 assert.throws(()=>app.run("validate({...envelope(),scope:'all_cases'},true)"));
 const restored=setup(app.saved);assert(restored.get('progress').textContent.includes('1/2'));
 const good=app.run('JSON.stringify(envelope())');
 await restored.get('import').listeners.change({target:{files:[{size:good.length,text:async()=>good}],value:'x'}});
 assert(restored.get('message').textContent.includes('校验通过'));
 const bad=JSON.parse(good);bad.bundle_sha256='wrong';
 await restored.get('import').listeners.change({target:{files:[{size:100,text:async()=>JSON.stringify(bad)}],value:'x'}});
 assert(restored.get('message').textContent.includes('导入失败'));assert(restored.get('progress').textContent.includes('1/2'));
 console.log('DOM state, navigation, restore and import checks passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    data = {"kind": KIND, "scope": SCOPE, "bundle_sha256": "d" * 64,
            "declaration": DECLARATION, "bundle": bundle}
    completed = subprocess.run([node, "-e", harness], input=json.dumps({"script": SCRIPT, "data": data}),
                               capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
    assert "checks passed" in completed.stdout
