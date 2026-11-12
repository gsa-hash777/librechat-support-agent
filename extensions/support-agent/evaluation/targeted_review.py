"""Offline human review of an explicit, frozen subset; never a full-set score."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.review import (
    ASSESSMENTS, DECLARATION, MAX_DECISION_BYTES, STYLE, VERDICTS, ReviewError,
    _hash, _read_json, _safe_json, _text, _write_distinct,
)

KIND = "self-attested-targeted-review"
SCOPE = "selected_cases_only"


def _validate_bundle(bundle):
    if (not isinstance(bundle, dict) or type(bundle.get("schema_version")) is not int
            or bundle["schema_version"] != 1 or bundle.get("kind") != "targeted-review-bundle"):
        raise ReviewError("定向复核资料格式或版本无效。")
    cases = bundle.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 1000:
        raise ReviewError("定向复核必须包含明确的非空选题范围。")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ReviewError("定向复核题目格式无效。")
        cid = _text(case.get("case_id"), "题目 ID", required=True, limit=200)
        if cid != case["case_id"] or cid in seen:
            raise ReviewError("定向复核含重复或无效题目 ID。")
        seen.add(cid)
        for name in ("baseline", "candidate"):
            source = case.get(name)
            if source is None and name == "candidate":
                continue
            if not isinstance(source, dict):
                raise ReviewError("题目缺少来源证据。")
            _hash(source.get("source_sha256"))
            _text(source.get("run_id"), "来源运行编号", required=True, limit=200)
            for field in ("prompts", "answers", "intermediate", "tool_calls", "knowledge_docs"):
                if not isinstance(source.get(field), list):
                    raise ReviewError("来源证据列表缺失。")
        expectations = case.get("review_expectations")
        if not isinstance(expectations, dict):
            raise ReviewError("缺少逐题参考要求。")
        for field in ("necessary_facts", "acceptable_answers", "forbidden_inferences"):
            if not isinstance(expectations.get(field), list) or any(not isinstance(x, str) for x in expectations[field]):
                raise ReviewError("参考要求必须为文本列表。")
        if not isinstance(expectations.get("tool_constraints"), (dict, list)):
            raise ReviewError("工具约束必须为列表或对象。")
        for field in ("selection_reasons", "criterion_changes"):
            if not isinstance(case.get(field), list) or any(not isinstance(x, str) for x in case[field]):
                raise ReviewError("选题与判定变更说明必须为文本列表。")
    optional = bundle.get("optional_full_review_file", "")
    if not isinstance(optional, str) or (optional and not re.fullmatch(r"[^/\\:#?<>]+\.html", optional)):
        raise ReviewError("可选全量页面必须为同目录 HTML 文件。")
    return [c["case_id"] for c in cases]


def validate_targeted_decisions(payload, bundle, *, bundle_sha256):
    """Check the selected roster, retaining partial reviews and all negative labels."""
    ids = _validate_bundle(bundle)
    _hash(bundle_sha256)
    if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 1 or payload.get("kind") != KIND):
        raise ReviewError("定向人工复核文件类型或版本无效。")
    if payload.get("scope") != SCOPE or payload.get("selected_case_ids") != ids:
        raise ReviewError("复核范围与当前选题清单不匹配。")
    if payload.get("bundle_sha256") != bundle_sha256:
        raise ReviewError("资料 SHA-256 不匹配，不能混用不同答案或选题版本。")
    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, dict):
        raise ReviewError("缺少复核署名。")
    name = _text(reviewer.get("name"), "复核者署名", required=True, limit=200)
    if reviewer.get("human_self_attestation") is not True or reviewer.get("declaration") != DECLARATION:
        raise ReviewError("需要本人阅读并填写的明确自述声明。")
    signed_at = _text(payload.get("signed_at"), "署名时间", required=True, limit=80)
    try:
        if datetime.fromisoformat(signed_at.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError
    except ValueError:
        raise ReviewError("署名时间必须为含时区的 ISO 时间。") from None
    values = payload.get("decisions")
    if not isinstance(values, list) or len(values) != len(ids):
        raise ReviewError("须保存选定范围的完整清单；未审核题目保持 pending。")
    found, clean = set(), []
    for item in values:
        if (not isinstance(item, dict) or not isinstance(item.get("case_id"), str)
                or item["case_id"] not in ids or item["case_id"] in found):
            raise ReviewError("人工复核含未知或重复题目 ID。")
        found.add(item["case_id"])
        verdict, ref_check, rule_check = (item.get(k) for k in ("verdict", "reference_assessment", "rule_assessment"))
        if not isinstance(verdict, str) or verdict not in VERDICTS:
            raise ReviewError("人工判定枚举无效。")
        if any(not isinstance(v, str) or v not in ASSESSMENTS for v in (ref_check, rule_check)):
            raise ReviewError("参考要求或规则意见枚举无效。")
        reason = _text(item.get("reason"), "判定理由", required=verdict != "pending")
        ref_notes = _text(item.get("reference_notes"), "参考要求修订意见", required=ref_check == "needs_revision")
        rule_notes = _text(item.get("rule_notes"), "规则修订意见", required=rule_check == "needs_revision")
        if verdict != "pending" and "pending" in (ref_check, rule_check):
            raise ReviewError("已判题还需给出该题参考要求和判断规则的意见。")
        clean.append({"case_id": item["case_id"], "verdict": verdict, "reason": reason,
                      "reference_assessment": ref_check, "reference_notes": ref_notes,
                      "rule_assessment": rule_check, "rule_notes": rule_notes})
    order = {cid: n for n, cid in enumerate(ids)}
    clean.sort(key=lambda item: order[item["case_id"]])
    counts = Counter(d["verdict"] for d in clean)
    reviewed = len(ids) - counts["pending"]
    return {"schema_version": 1, "kind": KIND, "scope": SCOPE,
            "bundle_sha256": bundle_sha256, "selected_case_ids": ids,
            "signed_at": signed_at, "reviewer": {"name": name, "human_self_attestation": True,
                "declaration": DECLARATION, "identity_verified": False}, "decisions": clean,
            "summary": {"selected": len(ids), "reviewed": reviewed, "pending": counts["pending"],
                "pass": counts["pass"], "fail": counts["fail"], "uncertain": counts["uncertain"],
                "reference_or_rule_disputes": sum("needs_revision" in (d["reference_assessment"], d["rule_assessment"]) for d in clean),
                "scope_complete": reviewed == len(ids)},
            "notice": "只覆盖选定题目。完成审核可以包含不通过或不确定，不代表修复全部通过；不计算或外推全量语义准确率。署名不构成身份认证，原始评测未被修改。"}


SCRIPT = r"""
'use strict';
const DATA=JSON.parse(document.getElementById('reviewData').textContent);
const ids=DATA.bundle.cases.map(c=>c.case_id), byId=new Map(DATA.bundle.cases.map(c=>[c.case_id,c]));
const key='support-targeted-review:'+DATA.bundle_sha256, declaration=DATA.declaration;
const $=id=>document.getElementById(id), el=(tag,t)=>{const n=document.createElement(tag);if(t!==undefined)n.textContent=t;return n;};
const text=v=>typeof v==='string'?v:JSON.stringify(v,null,2);
const detail=(title,v)=>{const n=el('details');n.append(el('summary',title),el('pre',text(v)));return n;};
const pending=()=>({verdict:'pending',reason:'',reference_assessment:'pending',reference_notes:'',rule_assessment:'pending',rule_notes:''});
let state={reviewer:{name:'',human_self_attestation:false,declaration},decisions:ids.map(case_id=>({case_id,...pending()}))}, selected=ids[0];
const message=(t,error=false)=>{$('message').textContent=t;$('message').className=error?'error':'muted';};
function summary(){const c={pending:0,pass:0,fail:0,uncertain:0};for(const d of state.decisions)c[d.verdict]++;return {selected:ids.length,reviewed:ids.length-c.pending,...c,reference_or_rule_disputes:state.decisions.filter(d=>d.reference_assessment==='needs_revision'||d.rule_assessment==='needs_revision').length,scope_complete:c.pending===0};}
function envelope(){return {schema_version:1,kind:DATA.kind,scope:DATA.scope,bundle_sha256:DATA.bundle_sha256,selected_case_ids:ids,reviewer:state.reviewer,signed_at:new Date().toISOString(),decisions:state.decisions,summary:summary()};}
function validate(data,signed){
 if(!data||data.schema_version!==1||data.kind!==DATA.kind)throw Error('定向复核文件类型或版本无效。');
 if(data.scope!==DATA.scope||JSON.stringify(data.selected_case_ids)!==JSON.stringify(ids))throw Error('复核范围与选题清单不匹配。');
 if(data.bundle_sha256!==DATA.bundle_sha256)throw Error('资料哈希不匹配，不能混用不同答案或选题版本。');
 if(!Array.isArray(data.decisions)||data.decisions.length!==ids.length)throw Error('须保存所选范围完整清单，未审题保持pending。');
 const seen=new Set(), decisions=[];
 for(const d of data.decisions){
  if(!d||typeof d.case_id!=='string'||!byId.has(d.case_id)||seen.has(d.case_id))throw Error('题目ID未知或重复。');seen.add(d.case_id);
  if(!['pending','pass','fail','uncertain'].includes(d.verdict))throw Error('人工判定枚举无效。');
  if(!['pending','agree','needs_revision'].includes(d.reference_assessment)||!['pending','agree','needs_revision'].includes(d.rule_assessment))throw Error('参考要求或规则意见无效。');
  for(const k of ['reason','reference_notes','rule_notes'])if(typeof d[k]!=='string'||d[k].length>10000)throw Error('逐题文本缺失或过长。');
  if(signed&&d.verdict!=='pending'&&(!d.reason.trim()||d.reference_assessment==='pending'||d.rule_assessment==='pending'))throw Error('已判题需填写理由，并判断该题参考要求与规则。');
  if(signed&&((d.reference_assessment==='needs_revision'&&!d.reference_notes.trim())||(d.rule_assessment==='needs_revision'&&!d.rule_notes.trim())))throw Error('选择需修改时请填写修订意见。');
  decisions.push({case_id:d.case_id,verdict:d.verdict,reason:d.reason,reference_assessment:d.reference_assessment,reference_notes:d.reference_notes,rule_assessment:d.rule_assessment,rule_notes:d.rule_notes});
 }
 if(!data.reviewer||typeof data.reviewer.name!=='string'||data.reviewer.name.length>200||typeof data.reviewer.human_self_attestation!=='boolean'||data.reviewer.declaration!==declaration)throw Error('署名声明格式无效。');
 if(signed&&(!data.reviewer.name.trim()||!data.reviewer.human_self_attestation))throw Error('请填写署名并勾选本人逐题复核声明。');
 if(signed&&(typeof data.signed_at!=='string'||!/(Z|[+-]\d\d:\d\d)$/.test(data.signed_at)||Number.isNaN(Date.parse(data.signed_at))))throw Error('署名时间无效。');
 decisions.sort((a,b)=>ids.indexOf(a.case_id)-ids.indexOf(b.case_id));return {reviewer:{name:data.reviewer.name,human_self_attestation:data.reviewer.human_self_attestation,declaration},decisions};
}
function progress(){const s=summary();$('progress').textContent=`本次定向复核 ${s.reviewed}/${s.selected}；待评 ${s.pending}，通过 ${s.pass}，不通过 ${s.fail}，不确定 ${s.uncertain}，参考要求/规则需修改 ${s.reference_or_rule_disputes}。`;$('completion').textContent=s.scope_complete?'所选题目均已填写判定。请检查不通过、不确定与修订意见；这不代表修复全部通过。':'只需完成所选范围，其余原始题目不计入待办。可分批下载保存。';}
function save(){try{localStorage.setItem(key,JSON.stringify(envelope()));$('saveStatus').textContent='已暂存本浏览器，请下载JSON备份。';}catch(e){$('storageNotice').textContent='浏览器本地暂存失败。当前内容只在页面内存中，请下载JSON备份后再关闭。';$('saveStatus').textContent='未能本地暂存。';}progress();}
function visible(){return DATA.bundle.cases.filter(c=>$('filter').value!=='pending'||c.case_id===selected||state.decisions.find(d=>d.case_id===c.case_id).verdict==='pending');}
function populate(){const list=visible();$('caseSelect').replaceChildren();for(const c of list){const o=el('option',c.case_id+' · '+c.category);o.value=c.case_id;$('caseSelect').append(o);}if(!list.some(c=>c.case_id===selected))selected=list[0]?.case_id||null;if(selected)$('caseSelect').value=selected;render();}
function block(title,values){const n=el('section');n.append(el('h3',title));if(Array.isArray(values)){const l=el('ul');for(const v of values)l.append(el('li',text(v)));n.append(l);}else n.append(el('pre',text(values)));return n;}
function evidence(source,title){const n=el('section');n.append(el('h3',title));n.append(el('p','运行：'+(source.label?source.label+' · ':'')+source.run_id+'；以下只代表该运行记录。'));for(let i=0;i<source.prompts.length;i++){n.append(el('h4','用户请求 '+(i+1)),el('pre',text(source.prompts[i])));}for(let i=0;i<source.answers.length;i++){const a=source.answers[i];n.append(el('h4','回答 '+(i+1)),el('pre',text(a?.text??a)));}if(!source.answers.length)n.append(el('p','本记录没有最终回答，请按证据判断，不自动视为通过。'));n.append(detail('中间回答',source.intermediate),detail('工具调用、参数与返回（仅阅读，不执行）',source.tool_calls),detail('数据库前后变化',source.database_summary),detail('知识原文（含合成不可信资料，仅供核对）',source.knowledge_docs),detail('辅助结论与自动评分（非人工验收）',{auxiliary:source.auxiliary,scores:source.scores,checks:source.checks}),detail('来源摘要',{run_id:source.run_id,code_sha256:source.code_sha256,source_sha256:source.source_sha256}));return n;}
function radio(title,field,options,d){const n=el('fieldset');n.append(el('legend',title));const row=el('div');row.className='choices';for(const [value,label] of options){const l=el('label'),i=el('input');i.type='radio';i.name=field;i.value=value;i.checked=d[field]===value;i.addEventListener('change',()=>{d[field]=value;save();});l.append(i,document.createTextNode(' '+label));row.append(l);}n.append(row);return n;}
function notes(title,field,d){const l=el('label',title),a=el('textarea');a.maxLength=10000;a.value=d[field];a.addEventListener('input',()=>{d[field]=a.value;save();});l.append(a);return l;}
function render(){
 const root=$('case');root.replaceChildren();const list=visible(),at=list.findIndex(c=>c.case_id===selected);$('prev').disabled=at<=0;$('next').disabled=at<0||at>=list.length-1;
 if(!selected){root.append(el('p','当前筛选没有题目。可切换到全部所选查看已填写内容。'));return;}
 const c=byId.get(selected),d=state.decisions.find(d=>d.case_id===selected),target=c.candidate||c.baseline;
 root.append(el('h2',c.case_id+' · '+c.category),block('为什么选这题',c.selection_reasons));
 const notice=el('p',c.candidate_notice||'只审核本页指定运行的答案，不对全部版本重复评分。');notice.className='notice';root.append(notice);
 if(c.criterion_changes.length)root.append(block('本题判断要求的变化',c.criterion_changes));
 root.append(evidence(target,c.candidate?'本次审核：最近有记录的答案':'本次审核：原始答案（尚无更新复测）'));
 if(c.candidate){const old=el('details');old.append(el('summary','对照原始答案与证据（无需再次独立评分）'),evidence(c.baseline,'原始记录'));root.append(old);}
 root.append(el('h3','逐题参考要求：用于判断答案，不代表已完成人工验收'));
 for(const [k,t] of [['necessary_facts','必要事实'],['acceptable_answers','允许的说法'],['forbidden_inferences','禁止推断'],['tool_constraints','工具约束']])root.append(block(t,c.review_expectations[k]));
 const form=el('section');form.append(el('h3','只对上方“本次审核”答案填写一次'));
 form.append(radio('答案、工具与数据库证据是否符合要求','verdict',[['pending','待评'],['pass','通过'],['fail','不通过'],['uncertain','不确定']],d),notes('判定理由（非待评必填，可引用回答或工具证据）','reason',d));
 form.append(radio('本题参考要求是否合理','reference_assessment',[['pending','待评'],['agree','同意'],['needs_revision','需修改']],d),notes('参考要求修改意见（需修改时必填）','reference_notes',d));
 form.append(radio('本题判断规则是否合理','rule_assessment',[['pending','待评'],['agree','同意'],['needs_revision','需修改']],d),notes('规则修改意见（需修改时必填）','rule_notes',d));root.append(form);
}
$('declaration').textContent=declaration;$('selection').textContent=DATA.bundle.selection;
$('policy').append(block('已确认权限规则（供对照，无需再确认）',DATA.bundle.policy_confirmed));
$('hashes').textContent='资料 SHA-256：'+DATA.bundle_sha256+'\n选定题目：'+ids.join(', ');
$('name').addEventListener('input',()=>{state.reviewer.name=$('name').value;save();});$('attest').addEventListener('change',()=>{state.reviewer.human_self_attestation=$('attest').checked;save();});
$('filter').addEventListener('change',()=>{selected=null;populate();});$('caseSelect').addEventListener('change',()=>{selected=$('caseSelect').value;render();});
for(const [button,delta] of [['prev',-1],['next',1]])$(button).addEventListener('click',()=>{const list=visible(),at=list.findIndex(c=>c.case_id===selected);if(list[at+delta]){selected=list[at+delta].case_id;populate();window.scrollTo(0,$('navigation').offsetTop);}});
$('download').addEventListener('click',()=>{try{const value=envelope();validate(value,true);const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'}));const a=el('a');a.href=url;a.download='targeted-human-review-'+DATA.bundle_sha256.slice(0,12)+'.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);message('已生成定向复核JSON；未审题保持pending。请保留下载文件。');}catch(e){message(e.message,true);}});
$('import').addEventListener('change',async event=>{const f=event.target.files[0];if(!f)return;try{if(f.size>10*1024*1024)throw Error('复核JSON不能超过10MiB。');state=validate(JSON.parse(await f.text()),true);$('name').value=state.reviewer.name;$('attest').checked=state.reviewer.human_self_attestation;save();populate();message('选题范围、资料哈希和逐题内容校验通过，已恢复。');}catch(e){message('导入失败：'+e.message,true);}finally{event.target.value='';}});
try{const saved=localStorage.getItem(key);if(saved){state=validate(JSON.parse(saved),false);message('已恢复本浏览器的定向复核草稿。');}}catch(e){$('storageNotice').textContent='无法读取或恢复本地暂存；没有载入不匹配状态。可导入已下载的定向复核JSON。';}
$('name').value=state.reviewer.name;$('attest').checked=state.reviewer.human_self_attestation;progress();populate();
"""


def render_targeted_html(bundle, *, bundle_sha256):
    ids = _validate_bundle(bundle)
    _hash(bundle_sha256)
    data = {"kind": KIND, "scope": SCOPE, "bundle_sha256": bundle_sha256,
            "declaration": DECLARATION, "bundle": bundle}
    script_hash = base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
    style_hash = base64.b64encode(hashlib.sha256(STYLE.encode()).digest()).decode()
    csp = (f"default-src 'none'; script-src 'sha256-{script_hash}'; style-src 'sha256-{style_hash}'; "
           "connect-src 'none'; img-src 'none'; object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none'")
    optional = bundle.get("optional_full_review_file")
    link = f'<a href="{escape(optional, quote=True)}">可选：查看原始全量复核资料</a>（不计入本次待办）' if optional else ''
    expanded = bundle.get("optional_review") is True
    title = f"可选扩展复核：{len(ids)} 题（不计入本轮必审）" if expanded else f"只复核失败样本与关键权限：{len(ids)} 题"
    scope_note = f"这是可选扩展的 {len(ids)} 题，不计入本轮必审，无需完成。" if expanded else f"本轮只需审核选定的 {len(ids)} 题。"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="{csp}"><title>LibreChat Agent 定向人工复核</title><style>{STYLE}</style></head><body>
<h1>{title}</h1>
<p id="selection"></p><p class="notice">{scope_note}每题看原始问题与最近有记录的答案，只填写一次判断；没有后续记录则审核原答案。不同题可能来自不同运行，不代表都经过最新代码重测。辅助意见不是人工结论。</p>
<p>{link}</p><div id="policy"></div>
<section><h2>署名与保存</h2><label>复核者署名 <input id="name" type="text" maxlength="200" autocomplete="off"></label><label><input id="attest" type="checkbox"> <span id="declaration"></span></label><div class="row"><button id="download" type="button">下载定向复核 JSON</button><label>导入已下载 JSON <input id="import" type="file" accept=".json,application/json"></label></div><p id="saveStatus" class="muted">支持分批保存，无需一次审完。</p><p id="storageNotice" class="error" role="alert"></p><p id="message" role="status"></p><details><summary>复核范围与资料版本</summary><pre id="hashes"></pre></details></section>
<p id="progress"></p><p id="completion"></p><p class="muted">只报告所选范围的审核数量和结论，不计算全量语义准确率。填写进度不等于下载校验通过。</p>
<nav id="navigation"><div class="row"><label>显示 <select id="filter"><option value="all">全部所选题目</option><option value="pending">仅所选待评题</option></select></label><select id="caseSelect" aria-label="选择复核题目"></select><button id="prev" type="button">上一题</button><button id="next" type="button">下一题</button></div></nav>
<main id="case"></main><footer class="muted">本页离线运行，不调用模型，不执行工具，不改原评测。署名仅为自述，不能认证身份。请保留下载文件。</footer>
<script id="reviewData" type="application/json">{_safe_json(data)}</script><script>{SCRIPT}</script></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description="只审核选定失败样本与关键权限，保留原始基线。")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output")
    parser.add_argument("--decisions")
    parser.add_argument("--adjudication")
    args = parser.parse_args(argv)
    if bool(args.decisions) != bool(args.adjudication) or bool(args.output) == bool(args.decisions):
        parser.error("选择 --output，或同时提供 --decisions 与 --adjudication。")
    try:
        bundle, bundle_hash = _read_json(args.bundle)
        if args.output:
            html = render_targeted_html(bundle, bundle_sha256=bundle_hash)
            _write_distinct(args.output, html, [args.bundle])
            print(f"已生成定向复核工作台，共 {len(bundle['cases'])} 题；人工判定初始均为待评。")
        else:
            payload, _ = _read_json(args.decisions, limit=MAX_DECISION_BYTES)
            out = validate_targeted_decisions(payload, bundle, bundle_sha256=bundle_hash)
            _write_distinct(args.adjudication, json.dumps(out, ensure_ascii=False, indent=2) + "\n", [args.bundle, args.decisions])
            print(json.dumps(out["summary"], ensure_ascii=False))
    except (ReviewError, OSError, TypeError) as exc:
        parser.exit(1, f"定向复核处理失败：{exc}\n")


if __name__ == "__main__":
    main()
