"""Frozen V3.1 / V4 evidence-policy experiment with shared raw retrieval.

No generated answer is scored. Both policies consume the exact same top-three
source observations; only deterministic disposition and evidence checks differ.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from html import escape
import importlib
import json
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
from zipfile import ZipFile, ZIP_DEFLATED

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
from evaluation import hybrid_run as evidence
from app.content import suspicious_reasons
from app.knowledge import chunk_text

DATASET = SERVICE / 'evaluation/evidence_cases_v4.json'
DATASET_SHA256 = '20e4c74ff692a269919c32eb7a15ea5d4bd397a0030cd4d5060e605080709b32'
SUPPLEMENT = SERVICE / 'data/documents/supplement.json'
SUPPLEMENT_MD = SERVICE / 'data/documents/support-supplement.md'
SUPPLEMENT_MANIFEST = SERVICE / 'data/documents/supplement-manifest.json'
FROZEN = SERVICE / 'evaluation/frozen_v31'
KINDS = ('answerable','exact_missing','negative','partial','clarify')
POLICIES = ('baseline','candidate')
FIELD_PATTERNS = {
    'object':r'产品|功能|模块|业务|页面|哪.{0,4}(?:文件|记录|对象)|什么.{0,4}(?:对象|内容)',
    'symptom':r'现象|提示|错误|报错|状态|结果|发生了什么|具体.{0,4}情况|诉求|目标|需求|希望.{0,8}(?:完成|处理)',
    'operation':r'操作|步骤|动作|做了什么|做什么|怎么做',
}


def prepare_corpus(stage):
    corpus = evidence.prepare_corpus('handbook')
    if stage == 'regression':
        return corpus
    if stage != 'expanded':
        raise ValueError('Unknown corpus stage.')
    rows = evidence.read(SUPPLEMENT)
    manifest = evidence.read(SUPPLEMENT_MANIFEST)
    text = SUPPLEMENT_MD.read_text(encoding='utf-8')
    if manifest.get('org_id') != 'demo' or evidence.digest(SUPPLEMENT) != manifest['source_sha256'] or hashlib.sha256(text.encode()).hexdigest() != manifest['handbook_sha256']:
        raise ValueError('Supplement source/scope/fingerprint mismatch.')
    originals = {row['id']:row for row in rows}
    if len(originals) != len(rows) or set(originals) & set(corpus['originals']) or any(row.get('org_id') != 'demo' for row in rows):
        raise ValueError('Supplement must contain distinct same-organization sources.')
    document_id = manifest.get('document_id','KD-EVALUATION-SUPPLEMENT-V4')
    document = {'id':document_id,'org_id':'demo','title':manifest.get('title','星桥云办公补充业务手册（模拟资料）'),
                'source':'模拟客服业务补充 / data/documents/supplement.json（原文格式转换）','content':text}
    lineage = {}
    for row in manifest['articles']:
        identifier = row['id']
        start,end = row['handbook_content_start'],row['handbook_content_end']
        if (identifier in lineage or identifier not in originals or type(start) is not int or type(end) is not int
                or text[start:end] != originals[identifier]['content']
                or hashlib.sha256(originals[identifier]['content'].encode()).hexdigest() != row['content_sha256']):
            raise ValueError('Supplement lineage is not a unique verbatim source span.')
        lineage[identifier] = {'doc_id':document_id,'start':start,'end':end}
    if set(lineage) != set(originals):
        raise ValueError('Supplement lineage is incomplete.')
    unsafe = any(suspicious_reasons(str(document[key])) for key in ('title','content','source'))
    new_chunks = [{**document,**chunk,'id':document_id,'doc_id':document_id,'document_unsafe':unsafe}
                  for chunk in chunk_text(text,doc_id=document_id,revision=1)]
    return {'mode':'expanded','documents':corpus['documents']+[document],'chunks':corpus['chunks']+new_chunks,
            'originals':{**corpus['originals'],**originals},'provenance':{**corpus['provenance'],**lineage}}


def validate_cases(data, corpus):
    if data.get('schema_version') != 1 or not data.get('frozen_before_comparison'):
        raise ValueError('Expected a pre-frozen evidence-policy dataset.')
    for name,fingerprint in data.get('source_sha256',{}).items():
        # Frozen Windows manifests retain their bytes; normalize only lookup.
        path = (SERVICE/name.replace('\\', '/')).resolve()
        if not path.is_relative_to(SERVICE.resolve()) or evidence.digest(path) != fingerprint:
            raise ValueError('A source changed after case authoring was frozen.')
    seen = set()
    for row in data['cases']:
        if row['id'] in seen or row['kind'] not in KINDS or row['split'] not in ('dev','test'):
            raise ValueError('Invalid case identity or category.')
        seen.add(row['id'])
        if not row.get('query') or not row.get('allowed_dispositions'):
            raise ValueError('Cases require a query and allowed disposition set.')
        for allowed in row['allowed_dispositions']:
            if allowed.get('status') not in ('candidates','insufficient','clarify'):
                raise ValueError('Invalid allowed status.')
            if 'coverage' in allowed and allowed['coverage'] not in ('full','partial','none'):
                raise ValueError('Invalid allowed coverage.')
        units = set()
        for unit in row['evidence']:
            identifier = unit['id']
            if identifier in units or identifier not in corpus['originals'] or not unit.get('quotes'):
                raise ValueError('Invalid evidence unit.')
            units.add(identifier)
            if any(not quote or quote not in corpus['originals'][identifier]['content'] for quote in unit['quotes']):
                raise ValueError('Reference quotes must occur in the frozen source.')
        if row['kind'] in ('answerable','negative','partial') and not units:
            raise ValueError('Answerable evidence needs an original source reference.')
        if row['kind'] in ('exact_missing','clarify') and units:
            raise ValueError('Missing/clarification cases cannot claim complete answer evidence.')
        for need in row.get('missing_needs',[]):
            if not need.get('patterns'):
                raise ValueError('A missing need requires accepted semantic word families.')
            for pattern in need['patterns']:
                re.compile(pattern)
        for allowed_fields in row.get('allowed_missing_field_sets',[]):
            if not set(allowed_fields) <= FIELD_PATTERNS.keys():
                raise ValueError('Unknown clarification field.')
    return data['cases']


def regression_cases():
    # Preserve the historical labels exactly. In particular HD10 has a known
    # conservative-label boundary; this run is regression, not fresh validation.
    rows = evidence.validate_dataset(evidence.read(evidence.DATASET),evidence.read(evidence.CORPUS))
    result = []
    for row in rows:
        result.append({**row,'split':'regression','kind':{'no_answer':'exact_missing'}.get(row['kind'],row['kind']),
                       'allowed_dispositions':[{'status':row['desired_decision']}],
                       'missing_needs':[],'allowed_missing_field_sets':[],'historical_label':True})
    return result


def assessment_of(response):
    value = response.get('evidence_assessment')
    return value if isinstance(value,dict) else {}


def citation_checks(response,corpus):
    actual = {row['chunk_id']:row for row in corpus['chunks']}
    sources = {row['id']:row for row in corpus['documents']}
    returned = {row.get('chunk_id') for row in response.get('results',[]) if isinstance(row,dict)}
    checks = []
    for claim in assessment_of(response).get('supported_claims',[]):
        if not isinstance(claim,dict) or not isinstance(claim.get('evidence'),list):
            checks.append({'correct':False,'reason':'malformed_claim'})
            continue
        if not claim['evidence']:
            checks.append({'correct':False,'reason':'claim_without_source'})
        for cited in claim['evidence']:
            valid = isinstance(cited,dict)
            chunk = actual.get(cited.get('chunk_id')) if valid else None
            document = sources.get(cited.get('doc_id')) if valid else None
            valid = bool(chunk and document and cited.get('chunk_id') in returned)
            if valid:
                first,last = cited.get('start_offset'),cited.get('end_offset')
                valid = (cited.get('doc_id') == chunk['doc_id'] and type(cited.get('revision')) is int
                         and cited['revision'] == chunk['revision'] and type(first) is int and type(last) is int
                         and chunk['start_offset'] <= first < last <= chunk['end_offset']
                         and isinstance(cited.get('quote'),str) and cited['quote'] == document['content'][first:last])
            checks.append({'correct':bool(valid),'chunk_id':cited.get('chunk_id') if isinstance(cited,dict) else None})
    return checks


def asks_for_sensitive_information(questions):
    secret = r'密码|口令|验证码|API\s*Key|access.?token|访问令牌|重置链接|私钥|密钥|身份证|银行卡|手机号|整份业务正文|完整业务正文|完整客户数据'
    for question in questions:
        for sentence in re.split(r'[。！？!?；;，,\n]|但是|然而|但',question):
            for found in re.finditer(secret,sentence,re.I):
                before = sentence[:found.start()]
                if re.search(r'(?:不要|别|不得|不能).{0,2}(?:省略|遗漏|隐瞒|隐藏|遮挡|脱敏)',before):
                    return True
                if re.search(r'(?:不要|无需|不得|请勿|不能|禁止|不需要)\s*(?:提供|发送|输入|贴出|分享|告诉|上传|给出|包含|附上|索取|泄露|显示|携带|填写|提交)|隐藏|遮盖|遮挡|脱敏',before):
                    continue
                if re.search(r'请|提供|发送|告诉|输入|贴出|给我|报出|上传|是多少|是什么',sentence):
                    return True
    return False


def grade(case,response,corpus):
    if not isinstance(response,dict) or response.get('evidence_status') not in ('candidates','insufficient','clarify') or not isinstance(response.get('results'),list):
        raise ValueError('Invalid evidence policy response.')
    status = response['evidence_status']
    assessment = assessment_of(response)
    coverage = assessment.get('coverage')
    allowed = any(status == choice['status'] and ('coverage' not in choice or coverage == choice['coverage'])
                  for choice in case['allowed_dispositions'])
    observed = evidence.evidence_coverage(case,response['results'],corpus)
    all_locations = evidence.validate_locations(response['results'],corpus)
    all_locations_valid = all(position['correct'] for position in all_locations)
    known_preserved = observed['all_required_evidence_covered'] if case['evidence'] else None
    missing_text = '\n'.join(str(value) for value in assessment.get('missing_information',[]))
    missing_text += '\n'+'\n'.join(str(value) for value in assessment.get('missing_requirements',[]))
    missing_text += '\n'+'\n'.join(str(value) for value in response.get('reason_codes',[]))
    missing_text = unicodedata.normalize('NFKC',missing_text).casefold()
    needs = [{'name':need['name'],'identified':any(re.search(pattern,missing_text,re.I) for pattern in need['patterns'])}
             for need in case.get('missing_needs',[])]
    questions = assessment.get('clarification_questions',[])
    if not questions and isinstance(response.get('clarification'),str):
        questions = [response['clarification']]
    if not isinstance(questions,list) or any(not isinstance(value,str) for value in questions):
        raise ValueError('Clarification questions must be strings.')
    observed_fields = [field for field,pattern in FIELD_PATTERNS.items() if re.search(pattern,'\n'.join(questions))]
    fields = assessment.get('missing_fields')
    inferred_fields = fields is None
    if inferred_fields:
        fields = observed_fields
    if not isinstance(fields,list) or any(not isinstance(value,str) for value in fields):
        raise ValueError('Clarification fields must be strings.')
    accepted_fields = case.get('allowed_missing_field_sets',[])
    field_set_allowed = (any(set(fields)==set(expected) for expected in accepted_fields)
                         and any(set(observed_fields)==set(expected) for expected in accepted_fields)) if accepted_fields else None
    questions_cover_fields = set(fields) == set(observed_fields)
    sensitive = asks_for_sensitive_information(questions)
    citations = citation_checks(response,corpus)
    citations_valid = all(row['correct'] for row in citations)
    rule_pass = allowed and citations_valid and all_locations_valid and not sensitive
    if case['kind'] in ('answerable','negative'):
        rule_pass = rule_pass and known_preserved is True
    if case['kind'] in ('exact_missing','partial'):
        rule_pass = rule_pass and all(row['identified'] for row in needs)
    if case['kind'] == 'partial' and status == 'candidates':
        rule_pass = rule_pass and coverage == 'partial' and known_preserved is True
    if case['kind'] == 'clarify' and accepted_fields:
        rule_pass = rule_pass and field_set_allowed is True and bool(questions) and questions_cover_fields
    return {'disposition_allowed':allowed,'grounded_policy_conformance':bool(rule_pass),
            'evidence_status':status,'assessment_coverage':coverage,'verified_evidence':observed,
            'all_returned_source_locations_valid':all_locations_valid,
            'all_returned_source_locations':all_locations,
            'known_evidence_preserved':known_preserved,'missing_needs':needs,
            'all_missing_needs_identified':all(row['identified'] for row in needs) if needs else None,
            'clarification_fields':fields,'legacy_fields_inferred_from_question':inferred_fields,
            'question_observed_fields':observed_fields,
            'clarification_fields_allowed':field_set_allowed,'questions_cover_fields':bool(questions_cover_fields),
            'requests_sensitive_information':sensitive,'assessment_citations':citations}


def metric(rows,planned,policy):
    complete = [row for row in rows if row.get('execution_error') is None]
    mean = lambda values: statistics.mean(values) if values else None
    grades = [row['policies'][policy]['grade'] for row in complete]
    result = {'planned':planned,'attempted':len(rows),'completed':len(complete),'execution_errors':len(rows)-len(complete),
              'not_attempted':planned-len(rows),'disposition_allowed_rate':mean([row['disposition_allowed'] for row in grades]),
              'grounded_policy_conformance_rate':mean([row['grounded_policy_conformance'] for row in grades]),
              'sensitive_information_requests':sum(row['requests_sensitive_information'] for row in grades),'by_kind':{}}
    locations = [position for row in grades for position in row['all_returned_source_locations']]
    citations = [position for row in grades for position in row['assessment_citations']]
    result['source_locations'] = {'checked':len(locations),'correct':sum(row['correct'] for row in locations)}
    result['assessment_citations'] = {'checked':len(citations),'correct':sum(row['correct'] for row in citations)}
    for kind in KINDS:
        subset = [row['policies'][policy]['grade'] for row in complete if row['kind']==kind]
        retained = [row['known_evidence_preserved'] for row in subset if row['known_evidence_preserved'] is not None]
        missing = [row['all_missing_needs_identified'] for row in subset if row['all_missing_needs_identified'] is not None]
        targeted = [row['clarification_fields_allowed'] and row['questions_cover_fields'] and not row['requests_sensitive_information'] for row in subset if row['clarification_fields_allowed'] is not None]
        result['by_kind'][kind] = {'completed':len(subset),'conformance_rate':mean([row['grounded_policy_conformance'] for row in subset]),
                                  'allowed_disposition_rate':mean([row['disposition_allowed'] for row in subset]),
                                  'known_evidence_preservation_rate':mean(retained),'missing_needs_identification_rate':mean(missing),
                                  'targeted_clarification_rate':mean(targeted),
                                  'overrefusal_rate':mean([row['evidence_status']!='candidates' for row in subset]) if kind in ('answerable','negative','partial') else None}
    samples = [row['policies'][policy]['wall_ms'] for row in complete]
    result['gate_only_wall_ms'] = {'samples':len(samples),'p50':statistics.median(samples) if samples else None,'p95':evidence.percentile(samples,.95)}
    return result


def evaluate(engine,policies,cases,corpus,mode):
    rows = []
    for case in cases:
        row = {**case,'execution_error':None}
        try:
            start,cpu = time.perf_counter(),time.process_time()
            retrieved = engine.retrieve(corpus['chunks'],case['query'],mode=mode,limit=3)
            row['shared_retrieval_wall_ms'] = (time.perf_counter()-start)*1000
            row['shared_retrieval_cpu_ms'] = (time.process_time()-cpu)*1000
            raw = retrieved.get('ranked_results')
            if not isinstance(raw,list):
                raise ValueError('Retriever must expose raw evidence.')
            if not all(position['correct'] for position in evidence.validate_locations(raw,corpus)):
                raise ValueError('The shared raw ranking contains a forged source location.')
            frozen = json.dumps(raw,sort_keys=True,ensure_ascii=False)
            row['ranked_results'] = deepcopy(raw)
            row['raw_sha256'] = hashlib.sha256(frozen.encode()).hexdigest()
            row['raw_coverage'] = evidence.evidence_coverage(case,raw,corpus)
            row['retrieval_metadata'] = retrieved.get('retrieval_metadata')
            row['policies'] = {}
            for name,policy in policies.items():
                offered = []
                def ranker(*_):
                    offered.append(row['raw_sha256'])
                    return deepcopy(raw)
                start,cpu = time.perf_counter(),time.process_time()
                response = policy.retrieve(deepcopy(corpus['chunks']),case['query'],limit=3,ranker=ranker)
                cpu_ms,wall_ms = (time.process_time()-cpu)*1000,(time.perf_counter()-start)*1000
                if json.dumps(raw,sort_keys=True,ensure_ascii=False) != frozen:
                    raise ValueError('A policy mutated shared raw evidence.')
                if isinstance(response,dict) and response.get('evidence_status') == 'candidates' and not offered:
                    raise ValueError('A candidate policy bypassed the shared ranker.')
                if isinstance(response,dict) and isinstance(response.get('results'),list):
                    raw_ids = {item.get('chunk_id') for item in raw if isinstance(item,dict)}
                    if any(not isinstance(item,dict) or item.get('chunk_id') not in raw_ids for item in response['results']):
                        raise ValueError('A policy introduced evidence outside the shared raw ranking.')
                row['policies'][name] = {'response':response,'grade':grade(case,response,corpus),
                                         'wall_ms':wall_ms,'cpu_ms':cpu_ms,
                                         'shared_raw_sha256':row['raw_sha256'],'ranker_invocations':len(offered)}
        except Exception as error:
            row['execution_error'] = type(error).__name__
            rows.append(row)
            break
        rows.append(row)
    return rows


def load_cases(stage,split,corpus):
    if stage == 'regression':
        return regression_cases()
    cases = validate_cases(evidence.read(DATASET),corpus)
    return [row for row in cases if row['split']==split]


def worker(args):
    corpus = prepare_corpus(args.stage)
    cases = load_cases(args.stage,args.split,corpus)
    baseline = importlib.import_module('evaluation.frozen_v31.retrieval')
    candidate = importlib.import_module('app.retrieval')
    module = importlib.import_module('app.hybrid')
    before = evidence.memory()
    start,cpu = time.perf_counter(),time.process_time()
    engine = module.HybridRetriever(model_dir=args.model_dir,cache_dir=args.cache_dir,policy=baseline)
    initialization = {'wall_ms':(time.perf_counter()-start)*1000,'cpu_ms':(time.process_time()-cpu)*1000}
    start,cpu = time.perf_counter(),time.process_time()
    engine.retrieve(corpus['chunks'],evidence.WARMUP_QUERY,mode=args.modes[0],limit=3)
    cold = {'wall_ms':(time.perf_counter()-start)*1000,'cpu_ms':(time.process_time()-cpu)*1000,'query':evidence.WARMUP_QUERY,'scored':False}
    after_cold = evidence.memory()
    rows = evaluate(engine,{'baseline':baseline,'candidate':candidate},cases,corpus,args.modes[0])
    result = {'mode':args.modes[0],'stage':args.stage,'split':args.split,'planned':len(cases),
              'documents':len(corpus['documents']),'chunks':len(corpus['chunks']),'source_units':len(corpus['originals']),
              'status':'completed' if len(rows)==len(cases) and all(row['execution_error'] is None for row in rows) else 'incomplete',
              'initialization':initialization,'cold_first_query':cold,'memory':{'before':before,'after_cold':after_cold,'after_queries':evidence.memory()},
              'configuration':engine.metadata(),'metrics':{name:metric(rows,len(cases),name) for name in POLICIES},'rows':rows}
    save = evidence.save
    save(args.worker_output,result)
    engine.close()
    return result


def input_paths(model_dir):
    paths = evidence.input_paths(model_dir)+[Path(__file__),DATASET,SUPPLEMENT,SUPPLEMENT_MD,SUPPLEMENT_MANIFEST]
    paths += sorted(FROZEN.glob('*'))
    paths += [SERVICE/'tests/test_evidence_evaluation.py',SERVICE/'tests/test_hybrid_evaluation.py']
    return sorted(set(path for path in paths if path.is_file()))


def render(report):
    def percent(value):
        return 'N/A' if value is None else f'{value:.2%}'
    lines = []
    for cell in report['runs']:
        for policy,metrics in cell['metrics'].items():
            values = [cell['mode'],policy,f"{metrics['completed']}/{metrics['planned']}",percent(metrics['grounded_policy_conformance_rate']),percent(metrics['disposition_allowed_rate'])]
            lines.append('<tr>'+''.join('<td>'+escape(str(value))+'</td>' for value in values)+'</tr>')
    details = ''.join('<details><summary>'+escape(cell['mode'])+'</summary><pre>'+escape(json.dumps(cell,ensure_ascii=False,indent=2))+'</pre></details>' for cell in report['runs'])
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>缺证与追问策略评测</title><style>body{max-width:1200px;margin:auto;padding:24px;font:16px/1.7 system-ui;background:#f4f7fb;color:#19384b}table{border-collapse:collapse;background:white}td,th{padding:12px;border:1px solid #ccd4df}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}details{margin:20px 0}</style><h1>缺证与针对性追问：固定策略比较</h1><p>两策略使用同一份原始检索结果，只比较证据处置。candidates 本身不算回答成功；原文引用、已知证据保留、缺项识别和追问字段分别验证。没有生成答案，不能把本页比例称为问答准确率或幻觉率。</p><table><tr><th>检索方式</th><th>策略</th><th>完成/计划</th><th>证据规则符合率</th><th>允许处置符合率</th></tr>'+''.join(lines)+'</table><details><summary>冻结输入与边界</summary><pre>'+escape(json.dumps({key:value for key,value in report.items() if key!='runs'},ensure_ascii=False,indent=2))+'</pre></details>'+details+'</html>'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('regression','expanded'),required=True)
    parser.add_argument('--split',choices=('dev','test'),default='dev')
    parser.add_argument('--modes',nargs='+',choices=evidence.ARMS)
    parser.add_argument('--model-dir',type=Path,default=evidence.MODELS)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker-output',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--cache-dir',type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.modes = args.modes or (list(evidence.ARMS) if args.stage=='regression' else ['lexical','rerank'])
    if len(set(args.modes)) != len(args.modes):
        parser.error('Modes must be unique.')
    if not DATASET.exists() or evidence.digest(DATASET) != DATASET_SHA256:
        parser.error('The new 48 cases must be frozen before any model experiment.')
    for name,fingerprint in evidence.read(FROZEN/'manifest.json').items():
        if evidence.digest(FROZEN/name) != fingerprint:
            parser.error('The frozen baseline changed.')
    if args.worker_output:
        if len(args.modes)!=1 or args.cache_dir is None or args.cache_dir.exists():
            parser.error('Each worker requires one mode and a new private cache.')
        return worker(args)
    if args.output is None or args.output.exists():
        parser.error('Output must be a new directory; observations cannot be overwritten.')
    corpus = prepare_corpus(args.stage)
    cases = load_cases(args.stage,args.split,corpus)
    paths = input_paths(args.model_dir)
    frozen = evidence.fingerprints(paths)
    args.output.mkdir(parents=True,exist_ok=False)
    # Preserve the pre-experiment code even when a later worker fails.
    with ZipFile(args.output/'frozen-sources.zip','x',ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_relative_to(SERVICE):
                archive.write(path,path.relative_to(SERVICE).as_posix())
        archive.writestr('model-and-source-fingerprints.json',json.dumps(frozen,ensure_ascii=False,indent=2))
    runs = []
    for mode in args.modes:
        target = args.output/(mode+'.json')
        cache = args.output/'runtime-cache'/mode
        command = [sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--stage',args.stage,'--split',args.split,'--modes',mode,
                   '--model-dir',str(args.model_dir.resolve()),'--worker-output',str(target.resolve()),'--cache-dir',str(cache.resolve())]
        completed = subprocess.run(command,cwd=SERVICE,capture_output=True,text=True,encoding='utf-8')
        if completed.returncode or not target.exists():
            evidence.save(args.output/'incomplete.json',{'status':'incomplete','mode':mode,'completed_workers':len(runs),'input_sha256':frozen})
            (args.output/'worker-error.log').write_text(completed.stderr,encoding='utf-8')
            raise RuntimeError('Policy worker failed; partial evidence was retained.')
        runs.append(evidence.read(target))
        print(json.dumps({'completed_mode':mode,'stage':args.stage,'status':runs[-1]['status']}),flush=True)
    if frozen != evidence.fingerprints(paths):
        evidence.save(args.output/'incomplete.json',{'status':'incomplete','reason':'Frozen inputs changed during experiment.','input_sha256':frozen})
        raise RuntimeError('Frozen sources changed; refusing to publish a complete report.')
    report = {'schema_version':1,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'stage':args.stage,'split':args.split,
              'status':'completed' if all(run['status']=='completed' for run in runs) else 'incomplete',
              'protocol':{'cases_per_mode':len(cases),'case_kind_counts':dict(Counter(row['kind'] for row in cases)),
                          'raw_retrieval_shared_exactly':True,'baseline':'Frozen V3.1','candidate':'Current app.retrieval',
                          'modes':args.modes,'k':3,'shared_raw_calls_per_case':1,'policies_per_observation':2,
                          'primary_grounding':'Complete verified original-source quote spans, not candidates or a matching document ID alone.',
                          'policy_conformance':'Allowed disposition plus preserved required evidence, recognized missing needs, targeted safe clarification, and valid emitted citations.',
                          'paid_generative_calls':0,'new_holdout_use':'Test split is used once after source/configuration freeze; later use is regression.',
                          'cold_scope':'Fresh process and empty vector cache; OS file cache is not flushed.',
                          'latency_scope':'Shared retrieval includes frozen-gate execution; separate gate-only measurements are not full HTTP or answer latency.'},
              'input_sha256':frozen,'environment':{'python':platform.python_version(),'platform':platform.platform()},'runs':runs,
              'limitations':['Project-authored synthetic business documents and tasks, not independent production data.',
                             'No generated answer or semantic entailment is judged; heuristic rule conformance is not answer accuracy.',
                             'String word families approximate missing-need/question semantics; source evidence uses verified exact spans.',
                             'The old 36 are seen regression cases, including the HD10 conservative negative-evidence label boundary.',
                             'Completed/planned and errors must accompany conditional metric rates.',
                             'Pre-filtered demo scope; HTTP authorization and secret leakage require separate service tests.']}
    evidence.save(args.output/'report.json',report)
    (args.output/'report.html').write_text(render(report),encoding='utf-8')
    return report


if __name__=='__main__':
    main()
