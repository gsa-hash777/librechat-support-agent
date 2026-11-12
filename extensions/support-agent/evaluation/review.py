"""Offline, self-attested human review; never changes a baseline or model verdict.

Render: python -m evaluation.review --report REPORT --references REFERENCES --output review.html
Import: python -m evaluation.review --report REPORT --references REFERENCES \
        --decisions signed.json --adjudication human-review.json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
KIND = "self-attested-human-review"
CASE_COUNT = 100
VERDICTS = {"pending", "pass", "fail", "uncertain"}
ASSESSMENTS = {"pending", "agree", "needs_revision"}
DECLARATION = "我声明：已判定的题目由我本人逐题阅读证据后填写；未评题目保持待评审。此声明和署名不构成系统对身份的认证。"
MAX_TEXT = 10_000
MAX_DECISION_BYTES = 10 * 1024 * 1024


class ReviewError(ValueError):
    """A human-readable validation error without echoing imported content."""


def _unique_cases(values, label):
    if not isinstance(values, list) or len(values) != CASE_COUNT:
        raise ReviewError(f"{label}必须包含完整的 {CASE_COUNT} 题。")
    by_id = {}
    for value in values:
        if not isinstance(value, dict):
            raise ReviewError(f"{label}题目格式无效。")
        case_id = value.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or len(case_id) > 200:
            raise ReviewError(f"{label}题目 ID 无效。")
        if case_id in by_id:
            raise ReviewError(f"{label}含重复题目 ID。")
        by_id[case_id] = value
    return by_id


def _validate_sources(report, references):
    if not isinstance(report, dict) or not isinstance(references, dict):
        raise ReviewError("报告与参考答案必须为 JSON 对象。")
    results = _unique_cases(report.get("results"), "报告")
    refs = _unique_cases(references.get("cases"), "参考答案")
    if set(results) != set(refs):
        raise ReviewError("报告与参考答案的全部题目 ID 必须一致。")
    manifest = report.get("manifest", {})
    planned = manifest.get("case_ids")
    if (manifest.get("planned_cases") != CASE_COUNT or not isinstance(planned, list)
            or len(planned) != CASE_COUNT or any(not isinstance(x, str) for x in planned)
            or len(set(planned)) != CASE_COUNT or set(planned) != set(results)):
        raise ReviewError("报告清单与结果的 100 个唯一题目 ID 不一致。")
    for ref in refs.values():
        expectations = ref.get("reference_expectations")
        if not isinstance(expectations, dict):
            raise ReviewError("参考答案缺少逐题要求。")
        for key in ("necessary_facts", "acceptable_answers", "forbidden_inferences"):
            value = expectations.get(key)
            if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                raise ReviewError("参考答案要求必须为文本列表。")
        constraints = expectations.get("tool_constraints")
        if not (isinstance(constraints, dict) or isinstance(constraints, list) and all(isinstance(x, str) for x in constraints)):
            raise ReviewError("工具约束必须为文本列表或结构化对象。")
    return results, refs


def _validate_reference_binding(references, report_sha256):
    declared = references.get("baseline_report_sha256")
    if declared is not None and declared != report_sha256:
        raise ReviewError("参考答案绑定的基线报告 SHA-256 与当前报告不一致。")


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ReviewError("SHA-256 格式无效。")
    return value


def _text(value, label, *, required=False, limit=MAX_TEXT):
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise ReviewError(f"{label}缺失或长度不合要求。")
    return value.strip()


def validate_decisions(payload, report, references, *, report_sha256, reference_sha256):
    """Validate a complete ID roster with partial human decisions, returning a sidecar.

    A name and declaration are self-attestation only. No identity authentication
    occurs. Unreviewed, uncertain or disputed cases never become implicit passes.
    """
    results, _ = _validate_sources(report, references)
    _hash(report_sha256)
    _hash(reference_sha256)
    _validate_reference_binding(references, report_sha256)
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != KIND:
        raise ReviewError("人工复核文件类型或版本无效。")
    if payload.get("report_sha256") != report_sha256 or payload.get("reference_sha256") != reference_sha256:
        raise ReviewError("报告或参考答案 SHA-256 不匹配，不能导入本次复核。")
    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, dict):
        raise ReviewError("缺少人工复核署名。")
    name = _text(reviewer.get("name"), "复核者署名", required=True, limit=200)
    if reviewer.get("human_self_attestation") is not True or reviewer.get("declaration") != DECLARATION:
        raise ReviewError("需要明确的人类自述声明；程序不能认证真实身份。")
    signed_at = _text(payload.get("signed_at"), "署名时间", required=True, limit=80)
    try:
        signed_time = datetime.fromisoformat(signed_at.replace("Z", "+00:00"))
        if signed_time.tzinfo is None:
            raise ValueError
    except ValueError:
        raise ReviewError("署名时间必须为含时区的 ISO 时间。") from None
    decisions = _unique_cases(payload.get("decisions"), "人工复核")
    if set(decisions) != set(results):
        raise ReviewError("人工复核含未知或缺失的题目 ID。")
    clean = []
    for case_id in results:
        item = decisions[case_id]
        verdict = item.get("verdict")
        ref_check = item.get("reference_assessment")
        rule_check = item.get("rule_assessment")
        if not isinstance(verdict, str) or verdict not in VERDICTS:
            raise ReviewError("人工判定必须为 pending/pass/fail/uncertain。")
        if any(not isinstance(v, str) or v not in ASSESSMENTS for v in (ref_check, rule_check)):
            raise ReviewError("标准答案和规则意见必须为 pending/agree/needs_revision。")
        reason = _text(item.get("reason"), "逐题判定理由", required=verdict != "pending")
        ref_notes = _text(item.get("reference_notes"), "标准答案修订意见", required=ref_check == "needs_revision")
        rule_notes = _text(item.get("rule_notes"), "规则修订意见", required=rule_check == "needs_revision")
        if verdict != "pending" and "pending" in (ref_check, rule_check):
            raise ReviewError("已判定题目还需填写标准答案和规则的同意/需修改意见。")
        clean.append({"case_id": case_id, "verdict": verdict, "reason": reason,
                      "reference_assessment": ref_check, "reference_notes": ref_notes,
                      "rule_assessment": rule_check, "rule_notes": rule_notes})
    counts = Counter(item["verdict"] for item in clean)
    reviewed = CASE_COUNT - counts["pending"]
    disputed = sum("needs_revision" in (x["reference_assessment"], x["rule_assessment"]) for x in clean)
    fully_adjudicated = reviewed == CASE_COUNT and not counts["uncertain"] and not disputed
    return {"schema_version": SCHEMA_VERSION, "kind": KIND,
            "report_sha256": report_sha256, "reference_sha256": reference_sha256,
            "run_id": report["manifest"].get("run_id"), "signed_at": signed_at,
            "reviewer": {"name": name, "human_self_attestation": True, "declaration": DECLARATION,
                         "identity_verified": False}, "decisions": clean,
            "summary": {"planned": CASE_COUNT, "reviewed": reviewed, "pending": counts["pending"],
                        "pass": counts["pass"], "fail": counts["fail"], "uncertain": counts["uncertain"],
                        "reference_or_rule_disputes": disputed, "coverage": reviewed / CASE_COUNT,
                        "fully_adjudicated": fully_adjudicated,
                        "self_attested_semantic_pass_rate": counts["pass"] / CASE_COUNT if fully_adjudicated else None},
            "notice": "仅保存人类自述署名与判定，不能认证身份。未完成、存在uncertain或标准答案/规则争议时不输出语义通过率。基线与代理判定未被覆盖。"}


def _database_summary(result):
    before = (result.get("fixture_before") or {}).get("rows", {})
    after = (result.get("fixture_after") or {}).get("rows", {})
    summary = {}
    for table in ("tickets", "drafts", "ticket_updates", "ticket_events", "users", "knowledge_docs", "audit_events"):
        left, right = before.get(table), after.get(table)
        if not isinstance(left, list) or not isinstance(right, list):
            summary[table] = {"status": "missing"}
            continue
        old = {str(row.get("id")): row for row in left}
        new = {str(row.get("id")): row for row in right}
        added = [row for key, row in new.items() if key not in old]
        removed = [row for key, row in old.items() if key not in new]
        changed = [{"before": old[key], "after": new[key]} for key in old.keys() & new.keys() if old[key] != new[key]]
        summary[table] = {"before_count": len(left), "after_count": len(right),
                          "unchanged": not (added or removed or changed), "added": added, "removed": removed, "changed": changed}
    return summary


def _safe_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


STYLE = r"""
:root{font-family:system-ui,'Microsoft YaHei',sans-serif;color:#163047;background:#eef2f6;line-height:1.6}body{max-width:1100px;margin:0 auto;padding:24px}h1{font-size:27px}h2{font-size:21px}h3{font-size:17px}section,.card{background:white;border:1px solid #d7e1eb;border-radius:12px;padding:20px;margin:16px 0}button,select,input,textarea{font:inherit}button{cursor:pointer;padding:8px 14px;background:#174f6b;color:white;border:0;border-radius:6px}button:disabled{opacity:.45;cursor:default}select,input[type=text]{max-width:100%;padding:8px;border:1px solid #93a5b7;border-radius:5px}textarea{box-sizing:border-box;width:100%;min-height:85px;border:1px solid #93a5b7;border-radius:5px;padding:8px}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.muted{color:#536b80;font-size:14px}.notice{border-left:4px solid #b87918;background:#fff8e9;padding:12px}.error{color:#8f1c1c;white-space:pre-wrap}#storageNotice:empty{display:none}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;border-radius:6px;padding:12px;font:14px/1.7 ui-monospace,Consolas,monospace;max-height:650px;overflow:auto}details{margin:12px 0;border-top:1px solid #dbe3eb;padding-top:10px}summary{cursor:pointer;font-weight:600}fieldset{border:1px solid #d7e1eb;border-radius:8px;margin:14px 0;padding:12px}label{display:block;margin:6px 0}.choices label{display:inline-block;margin-right:18px}a{color:inherit}#progress{font-weight:600}#message{white-space:pre-wrap}code{overflow-wrap:anywhere}nav{position:sticky;top:0;background:#eef2f6;padding:10px 0;z-index:1}li{margin:6px 0}.audit{font-size:13px}.hidden{display:none}
"""

SCRIPT = r"""
'use strict';
const DATA=JSON.parse(document.getElementById('reviewData').textContent);
const ids=DATA.cases.map(c=>c.case_id), byId=new Map(DATA.cases.map(c=>[c.case_id,c]));
const declaration=DATA.declaration, key='support-human-review:'+DATA.report_sha256+':'+DATA.reference_sha256;
const pending=()=>({verdict:'pending',reason:'',reference_assessment:'pending',reference_notes:'',rule_assessment:'pending',rule_notes:''});
let state={reviewer:{name:'',human_self_attestation:false,declaration},decisions:ids.map(case_id=>({case_id,...pending()}))};
let selected=null;
const $=id=>document.getElementById(id);
const el=(tag,text)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node;};
const text=value=>typeof value==='string'?value:JSON.stringify(value,null,2);
const details=(title,content)=>{const node=el('details');node.append(el('summary',title),el('pre',text(content)));return node;};
const message=(value,error=false)=>{$('message').textContent=value;$('message').className=error?'error':'muted';};
function validate(data,signed){
  if(!data||data.schema_version!==1||data.kind!==DATA.kind)throw Error('文件类型或版本无效。');
  if(data.report_sha256!==DATA.report_sha256||data.reference_sha256!==DATA.reference_sha256)throw Error('报告或参考答案哈希不匹配。');
  if(!Array.isArray(data.decisions)||data.decisions.length!==100)throw Error('必须包含完整100题；未评题应保持pending。');
  const seen=new Set(), decisions=[];
  for(const raw of data.decisions){
    if(!raw||typeof raw.case_id!=='string'||!byId.has(raw.case_id)||seen.has(raw.case_id))throw Error('含未知或重复题目ID。');
    seen.add(raw.case_id);
    if(!['pending','pass','fail','uncertain'].includes(raw.verdict))throw Error('人工判定枚举无效。');
    if(!['pending','agree','needs_revision'].includes(raw.reference_assessment)||!['pending','agree','needs_revision'].includes(raw.rule_assessment))throw Error('标准答案或规则意见枚举无效。');
    for(const field of ['reason','reference_notes','rule_notes'])if(typeof raw[field]!=='string'||raw[field].length>10000)throw Error('逐题文本缺失或过长。');
    if(signed&&raw.verdict!=='pending'&&(!raw.reason.trim()||raw.reference_assessment==='pending'||raw.rule_assessment==='pending'))throw Error('已判题必须填写理由，并评审标准答案和规则。');
    if(signed&&((raw.reference_assessment==='needs_revision'&&!raw.reference_notes.trim())||(raw.rule_assessment==='needs_revision'&&!raw.rule_notes.trim())))throw Error('选择需修改时必须填写修订意见。');
    decisions.push({case_id:raw.case_id,verdict:raw.verdict,reason:raw.reason,reference_assessment:raw.reference_assessment,reference_notes:raw.reference_notes,rule_assessment:raw.rule_assessment,rule_notes:raw.rule_notes});
  }
  if(!data.reviewer||typeof data.reviewer.name!=='string'||data.reviewer.name.length>200||typeof data.reviewer.human_self_attestation!=='boolean'||data.reviewer.declaration!==declaration)throw Error('署名声明格式无效。');
  if(signed&&(!data.reviewer.name.trim()||!data.reviewer.human_self_attestation))throw Error('请署名并勾选本人逐题复核声明；系统不能认证身份。');
  if(signed&&(typeof data.signed_at!=='string'||!/(Z|[+-]\d\d:\d\d)$/.test(data.signed_at)||Number.isNaN(Date.parse(data.signed_at))))throw Error('署名时间缺失或无时区。');
  return {reviewer:{name:data.reviewer.name,human_self_attestation:data.reviewer.human_self_attestation,declaration},decisions};
}
function envelope(){return {schema_version:1,kind:DATA.kind,report_sha256:DATA.report_sha256,reference_sha256:DATA.reference_sha256,reviewer:state.reviewer,signed_at:new Date().toISOString(),decisions:state.decisions};}
function save(){
  try{localStorage.setItem(key,JSON.stringify(envelope()));$('saveStatus').textContent='已暂存到本浏览器。关闭前仍建议下载JSON备份。';}
  catch(e){$('storageNotice').textContent='浏览器本地暂存失败（可能禁用或空间不足）。当前内容只在本页内存中，请下载JSON备份，关闭页面会丢失。';$('saveStatus').textContent='未能本地暂存。';}
  updateProgress();
}
function currentDecision(){return state.decisions.find(x=>x.case_id===selected);}
function updateProgress(){
  const counts={pending:0,pass:0,fail:0,uncertain:0};for(const d of state.decisions)counts[d.verdict]++;
  const disputes=state.decisions.filter(d=>d.reference_assessment==='needs_revision'||d.rule_assessment==='needs_revision').length;
  $('progress').textContent=`人工判定进度 ${100-counts.pending}/100；待评 ${counts.pending}，通过 ${counts.pass}，不通过 ${counts.fail}，不确定 ${counts.uncertain}，标准/规则需修改 ${disputes}。`;
  $('metricNotice').textContent='页面进度不等于校验通过或语义准确率。下载与Python导入会检查理由、署名和全部ID；未全评或仍有不确定/争议时不输出语义通过率。';
}
function visibleCases(){
  const filter=$('filter').value;
  return DATA.cases.filter(c=>filter==='priority'?c.priority:filter==='pending'?state.decisions.find(d=>d.case_id===c.case_id).verdict==='pending':true);
}
function populate(){
  const list=visibleCases();$('caseSelect').replaceChildren();
  for(const c of list){const o=el('option',c.case_id+' · '+c.category);o.value=c.case_id;$('caseSelect').append(o);}
  if(!list.some(c=>c.case_id===selected))selected=list[0]?.case_id||null;
  if(selected)$('caseSelect').value=selected;
  render();
}
function listBlock(title,values){const block=el('section');block.append(el('h3',title));if(Array.isArray(values)){const list=el('ul');for(const v of values)list.append(el('li',v));block.append(list);}else{block.append(el('pre',text(values)));}return block;}
function radioGroup(title,field,options,d){
  const group=el('fieldset');group.append(el('legend',title));const line=el('div');line.className='choices';
  for(const [value,label] of options){const wrapper=el('label');const input=el('input');input.type='radio';input.name=field;input.value=value;input.checked=d[field]===value;input.addEventListener('change',()=>{d[field]=value;save();});wrapper.append(input,document.createTextNode(' '+label));line.append(wrapper);}
  group.append(line);return group;
}
function notes(label,field,d){const wrapper=el('label',label);const area=el('textarea');area.maxLength=10000;area.value=d[field];area.addEventListener('input',()=>{d[field]=area.value;save();});wrapper.append(area);return wrapper;}
function render(){
  const root=$('case');root.replaceChildren();const list=visibleCases();const at=list.findIndex(c=>c.case_id===selected);$('prev').disabled=at<=0;$('next').disabled=at<0||at>=list.length-1;
  if(!selected){root.append(el('p','当前筛选没有题目。'));return;}
  const c=byId.get(selected), d=currentDecision();
  root.append(el('h2',c.case_id+' · '+c.category));
  const req=el('section');req.append(el('h3','用户请求与最终回答'));
  for(let n=0;n<c.prompts.length;n++){req.append(el('h4','用户请求 '+(n+1)),el('pre',text(c.prompts[n])));const a=c.answers[n];if(a)req.append(el('h4','最终回答 '+(n+1)),el('pre',text(a.text??a)));}
  if(c.answers.length>c.prompts.length)for(const a of c.answers.slice(c.prompts.length))req.append(el('pre',text(a.text??a)));
  root.append(req,details('中间回答与轮次信息',c.intermediate),details('实际工具调用、参数与返回（不执行）',c.tool_calls),details('前后数据库摘要与变更记录',c.database_summary),details('本题知识库原文（含合成不可信资料，仅供核对）',c.knowledge_docs));
  root.append(el('h3','参考要求：请独立判断这些要求是否正确'));
  for(const [key,label] of [['necessary_facts','必要事实'],['acceptable_answers','允许的说法'],['forbidden_inferences','禁止推断'],['tool_constraints','工具约束']])root.append(listBlock(label,c.reference_expectations[key]));
  root.append(details('辅助代理意见与原始规则结果（默认折叠，非人工标准答案认证）',{agent_review:c.auxiliary,original_scores:c.original_scores,checks:c.original_checks}));
  const form=el('section');form.append(el('h3','你的逐题复核'));
  form.append(radioGroup('人工语义判定（默认待评）','verdict',[['pending','待评审'],['pass','通过'],['fail','不通过'],['uncertain','不确定']],d));
  form.append(notes('判定理由：非待评题必须填写，建议引用具体文本或工具证据','reason',d));
  form.append(radioGroup('本题参考答案要求是否合理','reference_assessment',[['pending','待评审'],['agree','同意'],['needs_revision','需修改']],d));
  form.append(notes('参考答案修改意见（选择需修改时必填）','reference_notes',d));
  form.append(radioGroup('本题适用评审规则是否合理','rule_assessment',[['pending','待评审'],['agree','同意'],['needs_revision','需修改']],d));
  form.append(notes('规则修改意见（选择需修改时必填）','rule_notes',d));root.append(form);
}
$('declaration').textContent=declaration;
$('hashes').textContent='报告 SHA-256：'+DATA.report_sha256+'\n参考答案 SHA-256：'+DATA.reference_sha256;
$('rules').append(details('查看整体评审规则与判定政策', {rules:DATA.rules,policy:DATA.policy}));
$('name').addEventListener('input',()=>{state.reviewer.name=$('name').value;save();});
$('attest').addEventListener('change',()=>{state.reviewer.human_self_attestation=$('attest').checked;save();});
$('filter').addEventListener('change',populate);$('caseSelect').addEventListener('change',()=>{selected=$('caseSelect').value;render();});
for(const [button,delta] of [['prev',-1],['next',1]])$(button).addEventListener('click',()=>{const list=visibleCases();const at=list.findIndex(c=>c.case_id===selected);if(list[at+delta]){selected=list[at+delta].case_id;populate();window.scrollTo(0,$('navigation').offsetTop);}});
$('download').addEventListener('click',()=>{try{const data=envelope();validate(data,true);const blob=new Blob([JSON.stringify(data,null,2)+'\n'],{type:'application/json'});const url=URL.createObjectURL(blob);const a=el('a');a.href=url;a.download='human-review-'+String(DATA.run_id||'baseline').replace(/[^a-zA-Z0-9_-]/g,'_')+'.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);message('已生成署名JSON下载。请保留文件，之后由Python校验并写入独立人工复核记录。');}catch(e){message(e.message,true);}});
$('import').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{if(file.size>10*1024*1024)throw Error('复核JSON不能超过10MiB。');const next=validate(JSON.parse(await file.text()),true);state=next;$('name').value=state.reviewer.name;$('attest').checked=state.reviewer.human_self_attestation;save();populate();message('已验证全部ID和两个哈希，并恢复人工判定。导入文件中的内容仅当数据处理。');}catch(e){message('导入失败：'+e.message,true);}finally{event.target.value='';}});
try{const saved=localStorage.getItem(key);if(saved){state=validate(JSON.parse(saved),false);message('已恢复本浏览器草稿；下载前仍需完成署名与逐题理由校验。');}}
catch(e){$('storageNotice').textContent='无法读取或恢复本地暂存。没有导入不匹配或损坏的状态；可手动导入已下载的复核JSON。';}
$('name').value=state.reviewer.name;$('attest').checked=state.reviewer.human_self_attestation;
updateProgress();populate();
"""


def render_review_html(report, references, *, report_sha256, reference_sha256):
    results, refs = _validate_sources(report, references)
    _hash(report_sha256)
    _hash(reference_sha256)
    _validate_reference_binding(references, report_sha256)
    cases = []
    for case_id, result in results.items():
        ref = refs[case_id]
        turns = result.get("turns", [])
        evidence = result.get("evidence", {})
        cases.append({"case_id": case_id, "category": result.get("category", ""),
                      "priority": ref.get("semantic_verdict") in {"fail", "uncertain"},
                      "prompts": [t.get("prompt", "") for t in turns] or [result.get("case", {}).get("prompt", "")],
                      "answers": evidence.get("answers", []),
                      "intermediate": [{"turn": i + 1, "answers": turn.get("intermediate_answers", []),
                                        "execution_error": turn.get("execution_error")} for i, turn in enumerate(turns)],
                      "tool_calls": evidence.get("tool_calls", []), "database_summary": _database_summary(result),
                      "knowledge_docs": (result.get("fixture_before") or {}).get("rows", {}).get("knowledge_docs", []),
                      "reference_expectations": ref["reference_expectations"],
                      "auxiliary": {k: ref.get(k) for k in ("semantic_verdict", "reasons", "evidence")},
                      "original_scores": {k: result.get("judgement", {}).get(k) for k in ("contract_pass", "heuristic_pass", "strict_rule_pass")},
                      "original_checks": result.get("judgement", {}).get("checks", {})})
    cases.sort(key=lambda c: not c["priority"])
    data = {"schema_version": SCHEMA_VERSION, "kind": KIND, "declaration": DECLARATION,
            "report_sha256": report_sha256, "reference_sha256": reference_sha256,
            "run_id": report["manifest"].get("run_id"), "rules": references.get("rules", []),
            "policy": references.get("policy", []), "cases": cases}
    script_hash = base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
    style_hash = base64.b64encode(hashlib.sha256(STYLE.encode()).digest()).decode()
    csp = (f"default-src 'none'; script-src 'sha256-{script_hash}'; style-src 'sha256-{style_hash}'; "
           "connect-src 'none'; img-src 'none'; object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none'")
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="{csp}"><title>LibreChat Agent 人工复核工作台</title><style>{STYLE}</style></head><body>
<h1>100 题人工复核工作台</h1>
<p class="notice">本页离线运行，不发送网络请求，不调用模型，不修改基线。参考要求与辅助结论由参与开发的代理编写，非人工、非独立盲评。请逐题独立阅读；所有人工选项初始为待评审。</p>
<section><h2>署名与备份</h2><label>复核者姓名或署名 <input id="name" type="text" maxlength="200" autocomplete="off"></label><label><input id="attest" type="checkbox"> <span id="declaration"></span></label>
<div class="row"><button id="download" type="button">下载署名复核 JSON</button><label>导入已下载 JSON <input id="import" type="file" accept=".json,application/json"></label></div><p id="saveStatus" class="muted">尚未写入本地暂存。</p><p id="storageNotice" class="error" role="alert"></p><p id="message" role="status"></p><details><summary>数据版本与哈希</summary><pre id="hashes"></pre></details><div id="rules"></div></section>
<p id="progress"></p><p id="metricNotice" class="muted"></p>
<nav id="navigation"><div class="row"><label>范围 <select id="filter"><option value="all">全部 100 题（优先核查项在前）</option><option value="priority">仅辅助提示的优先核查项</option><option value="pending">仅人工待评题</option></select></label><select id="caseSelect" aria-label="选择题目"></select><button id="prev" type="button">上一题</button><button id="next" type="button">下一题</button></div></nav>
<main id="case"></main><footer class="muted">没有自动接受全部结论的功能。浏览器暂存可能失效，请下载 JSON 备份；导入只形成独立人工复核记录。署名与声明不等于身份认证。</footer>
<script id="reviewData" type="application/json">{_safe_json(data)}</script><script>{SCRIPT}</script></body></html>'''


def _read_json(path, *, limit=100 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ReviewError("JSON 文件超过允许大小。")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError):
        raise ReviewError("JSON 文件无法解码或格式无效。") from None
    return value, hashlib.sha256(raw).hexdigest()


def _write_distinct(path, text, source_paths):
    path = Path(path).resolve()
    for original in source_paths:
        original = Path(original).resolve()
        if path == original or (path.exists() and original.exists() and path.samefile(original)):
            raise ReviewError("输出路径不能覆盖报告、参考答案或导入的决定文件。")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description="离线人工复核导出/导入，保持原始基线不变。")
    parser.add_argument("--report", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", help="生成离线 HTML")
    parser.add_argument("--decisions", help="人类下载的署名 JSON")
    parser.add_argument("--adjudication", help="独立人工复核 sidecar JSON")
    args = parser.parse_args(argv)
    if bool(args.decisions) != bool(args.adjudication) or (bool(args.output) == bool(args.decisions)):
        parser.error("选择 --output，或同时提供 --decisions 与 --adjudication；两个模式不能混用。")
    try:
        report, report_hash = _read_json(args.report)
        references, reference_hash = _read_json(args.references)
        if args.output:
            rendered = render_review_html(report, references, report_sha256=report_hash, reference_sha256=reference_hash)
            _write_distinct(args.output, rendered, [args.report, args.references])
            print("已生成离线人工复核工作台；全部人工判定初始为 pending。")
        else:
            decisions, _ = _read_json(args.decisions, limit=MAX_DECISION_BYTES)
            sidecar = validate_decisions(decisions, report, references, report_sha256=report_hash, reference_sha256=reference_hash)
            _write_distinct(args.adjudication, json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", [args.report, args.references, args.decisions])
            print(json.dumps({"saved": True, "summary": sidecar["summary"], "identity_verified": False}, ensure_ascii=False))
    except (ReviewError, OSError, TypeError) as exc:
        parser.exit(1, f"人工复核处理失败：{exc}\n")


if __name__ == "__main__":
    main()
