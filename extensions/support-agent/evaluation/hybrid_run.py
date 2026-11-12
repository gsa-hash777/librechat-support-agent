"""Fixed lexical / vector-fusion / reranker comparison on business evidence.

Each arm/corpus runs in a fresh process with an empty document-vector cache.
Local embedding/reranking inference is real; no paid/generative model is called.
Raw top-k evidence coverage and the common evidence gate are distinct metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
import importlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

SERVICE = Path(__file__).resolve().parents[1]
ROOT = SERVICE.parents[1]
sys.path.insert(0, str(SERVICE))
from app.content import suspicious_reasons
from app.knowledge import chunk_text

DATASET = SERVICE / 'evaluation/hybrid_cases.json'
CORPUS = SERVICE / 'data/knowledge.json'
HANDBOOK = SERVICE / 'data/documents/knowledge.md'
HANDBOOK_MANIFEST = SERVICE / 'data/documents/manifest.json'
MODELS = ROOT / '.local/models/support-retrieval'
ARMS = ('lexical', 'hybrid', 'rerank')
CORPORA = ('handbook', 'seed')
DATASET_SHA256 = '515c12ff2e65a7077cedda5f557596142ae44d5c045ebac7fb2327bc3f5e995b'
HANDBOOK_ID = 'KD-8E45E617F1BC34DE98323D2A'
WARMUP_QUERY = '如何处理登录失败？'


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def validate_dataset(dataset, original):
    if dataset.get('schema_version') != 1 or dataset.get('frozen_before_comparison') is not True:
        raise ValueError('Expected a pre-frozen hybrid dataset.')
    corpus = {row['id']: row for row in original}
    seen = set()
    cases = dataset.get('cases', [])
    if not cases:
        raise ValueError('Empty dataset.')
    for row in cases:
        identifier = row.get('id')
        if not isinstance(identifier, str) or identifier in seen:
            raise ValueError('Missing or duplicate case identifier.')
        seen.add(identifier)
        if row.get('split') not in ('dev', 'test') or row.get('kind') not in ('answerable', 'no_answer', 'clarify'):
            raise ValueError('Invalid case classification.')
        if not isinstance(row.get('query'), str) or not 2 <= len(row['query']) <= 300 or not row.get('reference'):
            raise ValueError('Case needs a bounded query and rationale.')
        expected_status = {'answerable':'candidates', 'no_answer':'insufficient', 'clarify':'clarify'}[row['kind']]
        if row.get('desired_decision') != expected_status or not isinstance(row.get('evidence'), list):
            raise ValueError('Invalid evidence decision or evidence list.')
        if bool(row['evidence']) != (row['kind'] == 'answerable'):
            raise ValueError('Only answerable cases have required evidence.')
        units = set()
        for unit in row['evidence']:
            identifier = unit.get('id')
            quotes = unit.get('quotes')
            if identifier not in corpus or identifier in units or not isinstance(quotes, list) or not quotes:
                raise ValueError('Invalid or duplicate evidence unit.')
            units.add(identifier)
            if any(not isinstance(quote, str) or not quote or quote not in corpus[identifier]['content'] for quote in quotes):
                raise ValueError('Reference evidence must occur verbatim in the frozen source.')
    return cases


def prepare_corpus(mode, originals=None, handbook_text=None, handbook_manifest=None):
    originals = read(CORPUS) if originals is None else originals
    if mode not in CORPORA or any(row.get('org_id') != 'demo' for row in originals):
        raise ValueError('Only the declared demo organization corpus is benchmarked.')
    provenance = {}
    if mode == 'seed':
        documents = originals
        for row in originals:
            provenance[row['id']] = {'doc_id':row['id'], 'start':0, 'end':len(row['content'])}
    else:
        handbook_text = HANDBOOK.read_text(encoding='utf-8') if handbook_text is None else handbook_text
        handbook_manifest = read(HANDBOOK_MANIFEST) if handbook_manifest is None else handbook_manifest
        if handbook_manifest.get('org_id') != 'demo' or hashlib.sha256(handbook_text.encode()).hexdigest() != handbook_manifest['handbook_sha256']:
            raise ValueError('Handbook scope or content fingerprint mismatch.')
        records = {row['id']:row for row in originals}
        for item in handbook_manifest['articles']:
            if item['id'] in provenance:
                raise ValueError('Handbook source lineage must not contain duplicate IDs.')
            start, end = item['handbook_content_start'], item['handbook_content_end']
            record = records.get(item['id'])
            if (record is None or type(start) is not int or type(end) is not int
                    or handbook_text[start:end] != record['content']
                    or hashlib.sha256(record['content'].encode()).hexdigest() != item['content_sha256']):
                raise ValueError('Handbook lineage does not preserve original business evidence.')
            provenance[item['id']] = {'doc_id':HANDBOOK_ID, 'start':start, 'end':end}
        if set(provenance) != set(records):
            raise ValueError('Handbook must represent each source exactly once.')
        documents = [{'id':HANDBOOK_ID, 'org_id':'demo', 'title':handbook_manifest['title'],
                      'content':handbook_text, 'source':'模拟客服业务知识 / data/knowledge.json（16条原文；仅格式转换）'}]
    chunks = []
    for source in documents:
        unsafe = any(suspicious_reasons(source[field]) for field in ('title', 'content', 'source'))
        for chunk in chunk_text(source['content'], doc_id=source['id'], revision=1):
            chunks.append({**source, **chunk, 'id':source['id'], 'doc_id':source['id'], 'document_unsafe':unsafe})
    return {'mode':mode, 'documents':documents, 'chunks':chunks, 'provenance':provenance,
            'originals':{row['id']:row for row in originals}}


def validate_locations(results, corpus):
    sources = {row['id']:row for row in corpus['documents']}
    actual = {row['chunk_id']:row for row in corpus['chunks']}
    checks = []
    for row in results:
        source = sources.get(row.get('doc_id')) if isinstance(row, dict) else None
        expected = actual.get(row.get('chunk_id')) if isinstance(row, dict) else None
        valid = source is not None and expected is not None
        if valid:
            valid = row.get('id') == row.get('doc_id') == expected['doc_id']
            valid = valid and all(type(row.get(key)) is int and row[key] == expected[key] for key in ('revision', 'start_offset', 'end_offset', 'start_line', 'end_line'))
            valid = valid and all(row.get(key) == expected[key] for key in ('content', 'content_sha256', 'title', 'source'))
            # The common gate omits org_id in its public projection. Source
            # IDs, text and positions still resolve in the prefiltered corpus.
            # An explicit contradictory/null org_id is never accepted.
            valid = valid and ('org_id' not in row or row['org_id'] == expected['org_id'])
        checks.append({'chunk_id':row.get('chunk_id') if isinstance(row,dict) else None, 'correct':bool(valid)})
    return checks


def covered(intervals, first, last):
    cursor = first
    for start, end in sorted(intervals):
        if end <= cursor:
            continue
        if start > cursor:
            return False
        cursor = end
        if cursor >= last:
            return True
    return False


def evidence_units(case, spans, corpus):
    """A unit is complete only when all its required quotes are covered."""
    units = []
    for unit in case['evidence']:
        source = corpus['originals'][unit['id']]['content']
        location = corpus['provenance'][unit['id']]
        quote_checks = []
        for quote in unit['quotes']:
            positions, cursor = [], 0
            while (position := source.find(quote, cursor)) >= 0:
                positions.append(position)
                cursor = position + 1
            quote_checks.append(any(covered(spans.get(location['doc_id'], []), location['start'] + position,
                                            location['start'] + position + len(quote)) for position in positions))
        units.append({'source_unit':unit['id'], 'quotes_covered':quote_checks, 'covered':all(quote_checks)})
    return units


def evidence_coverage(case, results, corpus, limit=3):
    selected = results[:limit]
    checks = validate_locations(selected, corpus)
    spans, prefix_hits = {}, []
    for result, check in zip(selected, checks, strict=True):
        if check['correct']:
            spans.setdefault(result['doc_id'], []).append((result['start_offset'], result['end_offset']))
        # The rank is the original returned position, including an invalid
        # source. Multiple top-r chunks may jointly cover a required unit.
        prefix_hits.append(any(unit['covered'] for unit in evidence_units(case, spans, corpus)))
    units = evidence_units(case, spans, corpus)
    # Duplicate returned source characters are an overlap measure, not semantic
    # redundancy or a count of independent corroborating sources.
    total, unique = 0, 0
    for values in spans.values():
        total += sum(end-start for start,end in values)
        merged = []
        for start,end in sorted(values):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end,merged[-1][1])
            else:
                merged.append([start,end])
        unique += sum(end-start for start,end in merged)
    return {'units':units, 'hit':any(unit['covered'] for unit in units) if units else None,
            'hit_at_1':bool(prefix_hits and prefix_hits[0]) if units else None,
            'reciprocal_rank_at_3':next((1/rank for rank,hit in enumerate(prefix_hits,1) if hit),0) if units else None,
            'recall':sum(unit['covered'] for unit in units)/len(units) if units else None,
            'all_required_evidence_covered':all(unit['covered'] for unit in units) if units else None,
            'locations':checks, 'duplicate_source_characters':total-unique, 'returned_source_characters':total,
            'duplicate_source_character_fraction':(total-unique)/total if total else None}


def percentile(values, quantile):
    return sorted(values)[math.ceil(quantile*len(values))-1] if values else None


def metrics(rows, planned):
    complete = [row for row in rows if row.get('execution_error') is None]
    answerable = [row for row in complete if row['kind'] == 'answerable']
    missing = [row for row in complete if row['kind'] == 'no_answer']
    vague = [row for row in complete if row['kind'] == 'clarify']
    mean = lambda values: statistics.mean(values) if values else None
    values = {'planned':planned, 'attempted':len(rows), 'completed':len(complete),
              'execution_failed':len(rows)-len(complete), 'not_attempted':planned-len(rows),
              'answerable_cases':len(answerable), 'no_answer_cases':len(missing), 'clarify_cases':len(vague),
              'required_evidence_units':sum(len(row['evidence']) for row in answerable),
              'decision_match_rate':mean([row['decision_matches'] for row in complete]),
              'no_answer_abstain_rate':mean([row['evidence_status']=='insufficient' for row in missing]),
              'no_answer_candidate_rate':mean([row['evidence_status']=='candidates' for row in missing]),
              'clarify_rate_on_ambiguous':mean([row['evidence_status']=='clarify' for row in vague])}
    for key in ('raw','final'):
        positions = [position for row in complete for position in row[key+'_coverage']['locations']]
        values[key] = {'hit_at_3':mean([row[key+'_coverage']['hit'] for row in answerable]),
                       'evidence_units_covered':sum(unit['covered'] for row in answerable for unit in row[key+'_coverage']['units']),
                       'hit_at_1':mean([row[key+'_coverage']['hit_at_1'] for row in answerable]),
                       'mrr_at_3':mean([row[key+'_coverage']['reciprocal_rank_at_3'] for row in answerable]),
                       'macro_evidence_recall_at_3':mean([row[key+'_coverage']['recall'] for row in answerable]),
                       'all_evidence_rate_at_3':mean([row[key+'_coverage']['all_required_evidence_covered'] for row in answerable]),
                       'returned_chunks':len(positions), 'source_location_correct_rate':mean([p['correct'] for p in positions]),
                       'duplicate_source_character_fraction':mean([r[key+'_coverage']['duplicate_source_character_fraction'] for r in complete if r[key+'_coverage']['duplicate_source_character_fraction'] is not None])}
    wall = [sample for row in complete for sample in row['warm_wall_ms']]
    cpu = [sample for row in complete for sample in row['warm_cpu_ms']]
    values['warm_latency_ms'] = {'samples':len(wall), 'wall_p50':statistics.median(wall) if wall else None,
                                  'wall_p95':percentile(wall,.95),'cpu_p50':statistics.median(cpu) if cpu else None,
                                  'cpu_p95':percentile(cpu,.95)}
    return values


def memory():
    """Current process working set and high-water RSS; no child aggregation."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
                        *[(name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]]
        info = Counters()
        info.cb = ctypes.sizeof(info)
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi',use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb):
            return {'rss_bytes':info.WorkingSetSize,'peak_rss_bytes':info.PeakWorkingSetSize,'method':'GetProcessMemoryInfo'}
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {'rss_bytes':None,'peak_rss_bytes':peak if sys.platform=='darwin' else peak*1024,'method':'getrusage peak only'}
    except ImportError:
        return {'rss_bytes':None,'peak_rss_bytes':None,'method':'unavailable'}


def evaluate(engine, cases, corpus, arm, repeats=3):
    rows = []
    for case in cases:
        row = {**case, 'execution_error':None}
        try:
            responses, wall, cpu = [], [], []
            for _ in range(repeats):
                start_wall,start_cpu = time.perf_counter(),time.process_time()
                response = engine.retrieve(corpus['chunks'],case['query'],mode=arm,limit=3)
                cpu.append(round((time.process_time()-start_cpu)*1000,6))
                wall.append(round((time.perf_counter()-start_wall)*1000,6))
                if not isinstance(response,dict) or not all(isinstance(response.get(key),list) for key in ('results','ranked_results')):
                    raise ValueError('The adapter must expose raw and gated source lists separately.')
                if response.get('evidence_status') not in ('candidates','insufficient','clarify'):
                    raise ValueError('Unknown evidence decision.')
                responses.append(response)
            result = responses[0]
            signature = lambda value: ([item.get('chunk_id') for item in value['ranked_results']],
                                       [item.get('chunk_id') for item in value['results']],value['evidence_status'])
            row.update(evidence_status=result['evidence_status'],reason_codes=result.get('reason_codes',[]),
                       decision_matches=result['evidence_status']==case['desired_decision'],
                       ranked_results=result['ranked_results'][:3],results=result['results'][:3],
                       raw_coverage=evidence_coverage(case,result['ranked_results'],corpus),
                       final_coverage=evidence_coverage(case,result['results'],corpus),
                       warm_wall_ms=wall,warm_cpu_ms=cpu,stable_ranking=all(signature(v)==signature(result) for v in responses),
                       retrieval_metadata=result.get('retrieval_metadata'))
        except Exception as error:
            row['execution_error'] = type(error).__name__
            rows.append(row)
            break
        rows.append(row)
    return rows


def input_paths(model_dir):
    paths = [DATASET,CORPUS,HANDBOOK,HANDBOOK_MANIFEST,Path(__file__)]
    paths += sorted((SERVICE/'app').glob('*.py'))
    paths += sorted(SERVICE.glob('requirements*.txt'))
    # Model weights/tokenizer/config are hashed, not copied into the source ZIP.
    paths += sorted(path for path in model_dir.rglob('*') if path.is_file() and '.cache' not in path.parts)
    return paths


def fingerprints(paths):
    return {str(path.resolve()):digest(path) for path in paths}


def worker(args):
    dataset = read(DATASET)
    all_cases = validate_dataset(dataset,read(CORPUS))
    cases = [row for row in all_cases if args.split=='all' or row['split']==args.split]
    corpus = prepare_corpus(args.corpora[0])
    before = memory()
    start_wall,start_cpu = time.perf_counter(),time.process_time()
    module = importlib.import_module('app.hybrid')
    engine = module.HybridRetriever(model_dir=args.model_dir,cache_dir=args.cache_dir)
    initialized = {'wall_ms':(time.perf_counter()-start_wall)*1000,'cpu_ms':(time.process_time()-start_cpu)*1000}
    start_wall,start_cpu = time.perf_counter(),time.process_time()
    engine.retrieve(corpus['chunks'],WARMUP_QUERY,mode=args.arms[0],limit=3)
    cold = {'wall_ms':(time.perf_counter()-start_wall)*1000,'cpu_ms':(time.process_time()-start_cpu)*1000,
            'query':WARMUP_QUERY,'scored':False,'document_cache_initially_empty':True}
    after_cold = memory()
    rows = evaluate(engine,cases,corpus,args.arms[0],args.repeats)
    result = {'arm':args.arms[0],'corpus':args.corpora[0],'status':'completed' if len(rows)==len(cases) and all(row['execution_error'] is None for row in rows) else 'incomplete',
              'cases':len(cases),'documents':len(corpus['documents']),'chunks':len(corpus['chunks']),
              'initialization':initialized,'cold_first_query':cold,'memory':{'before_engine':before,'after_cold_query':after_cold,'after_warm_queries':memory()},
              'models_and_configuration':engine.metadata(),'shared_evidence_gate_sha256':digest(SERVICE/'app/retrieval.py'),
              'metrics':metrics(rows,len(cases)),
              'metrics_by_split':{split:metrics([row for row in rows if row['split']==split],sum(c['split']==split for c in cases)) for split in dict.fromkeys(c['split'] for c in cases)},'rows':rows}
    save(args.worker_output,result)
    engine.close()
    return result


def render(report):
    fmt = lambda value: '未知' if value is None else f'{value:.2%}'
    lines = []
    for result in report['runs']:
        for split,m in result['metrics_by_split'].items():
            columns = (result['corpus'],result['arm'],split,m['completed'],
                       fmt(m['raw']['hit_at_1']),fmt(m['raw']['mrr_at_3']),
                       fmt(m['raw']['hit_at_3']),fmt(m['raw']['macro_evidence_recall_at_3']),
                       fmt(m['final']['hit_at_1']),fmt(m['final']['mrr_at_3']),
                       fmt(m['final']['macro_evidence_recall_at_3']),fmt(m['no_answer_abstain_rate']),
                       fmt(m['clarify_rate_on_ambiguous']),m['warm_latency_ms']['wall_p95'])
            lines.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in columns)+'</tr>')
    details = ''.join('<details><summary>'+escape(run['corpus']+' / '+run['arm'])+'</summary><pre>'+escape(json.dumps(run,ensure_ascii=False,indent=2))+'</pre></details>' for run in report['runs'])
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>业务知识混合检索与重排比较</title><style>body{font:16px/1.7 system-ui;max-width:1250px;margin:auto;padding:28px;color:#18354a;background:#f6f8fb}table{border-collapse:collapse;width:100%;background:#fff}td,th{border:1px solid #ccd7df;padding:9px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}details{margin:18px 0}.scroll{overflow:auto}</style></head><body><h1>业务知识：词项、向量融合与重排</h1><p>固定合成题；主语料是上传的业务 Markdown 手册，种子16篇是独立对照。未混入功能说明文档。局部CPU embedding/重排是真实模型推理，没有调用付费生成模型。</p><p>先看 raw 原文证据召回，再看共用门槛后的 final；返回 candidates 不代表最终答案正确。KD 手册的正确原文片段与原 KB 证据等价，不按文档编号简单判错。</p><div class="scroll"><table><tr><th>语料</th><th>路径</th><th>集合</th><th>完成</th><th>raw Hit@1</th><th>raw MRR@3</th><th>raw Hit@3</th><th>raw证据Recall@3</th><th>final Hit@1</th><th>final MRR@3</th><th>final证据Recall@3</th><th>缺证识别</th><th>明确追问</th><th>warm P95 ms</th></tr>'+''.join(lines)+'</table></div><p>冷启动是每个独立进程的初始化与首个未评分查询，文档向量缓存从空开始；warm复用文档缓存但不缓存查询向量。模型下载不计入推理耗时。RSS是各进程当前/峰值工作集，不能等同整机可用内存或生产容量。</p><details><summary>协议、摘要和局限</summary><pre>'+escape(json.dumps({k:v for k,v in report.items() if k!='runs'},ensure_ascii=False,indent=2))+'</pre></details>'+details+'</body></html>'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['dev','test','all'],default='dev')
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    parser.add_argument('--corpora',nargs='+',choices=CORPORA,default=list(CORPORA))
    parser.add_argument('--model-dir',type=Path,default=MODELS)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--worker-output',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--cache-dir',type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1<=args.repeats<=10 or len(set(args.arms))!=len(args.arms) or len(set(args.corpora))!=len(args.corpora):
        parser.error('Use unique arms/corpora and repeats in 1..10.')
    if digest(DATASET)!=DATASET_SHA256:
        parser.error('Frozen hybrid dataset changed; create a versioned dataset/runner instead.')
    if args.worker_output:
        if len(args.arms)!=1 or len(args.corpora)!=1 or args.cache_dir is None:
            parser.error('A worker needs exactly one arm/corpus and a private fresh cache path.')
        if args.cache_dir.exists():
            parser.error('Cold-start cache path already exists.')
        return worker(args)
    if args.output is None or args.output.exists():
        parser.error('Output must be a new directory; historical runs cannot be overwritten.')
    validate_dataset(read(DATASET),read(CORPUS))
    paths = input_paths(args.model_dir)
    frozen = fingerprints(paths)
    frozen_sources = {str(path.relative_to(SERVICE)):path.read_bytes() for path in paths if path.is_relative_to(SERVICE)}
    args.output.mkdir(parents=True,exist_ok=False)
    runs = []
    for corpus in args.corpora:
        for arm in args.arms:
            output = args.output / (corpus+'-'+arm+'.json')
            cache = args.output / 'runtime-cache' / (corpus+'-'+arm)
            command = [sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--split',args.split,'--arms',arm,'--corpora',corpus,
                       '--model-dir',str(args.model_dir.resolve()),'--repeats',str(args.repeats),'--worker-output',str(output.resolve()),'--cache-dir',str(cache.resolve())]
            completed = subprocess.run(command,cwd=SERVICE,capture_output=True,text=True,encoding='utf-8')
            if completed.returncode or not output.exists():
                save(args.output/'incomplete.json',{'status':'incomplete','arm':arm,'corpus':corpus,'worker_exit_code':completed.returncode,
                                                   'completed_workers':len(runs),'input_sha256':frozen,
                                                   'reason':'Worker failed; no complete comparison published. Inspect local model availability and interfaces.'})
                # Local stack traces contain no benchmark credentials; preserve
                # diagnostics in a separate file, not the user summary.
                (args.output/'worker-error.log').write_text(completed.stderr,encoding='utf-8')
                raise RuntimeError('Hybrid worker failed; incomplete evidence was retained.')
            runs.append(read(output))
            print(json.dumps({'completed_arm':arm,'corpus':corpus,'status':runs[-1]['status']},ensure_ascii=True),flush=True)
    if frozen!=fingerprints(paths):
        save(args.output/'incomplete.json',{'status':'incomplete','reason':'Source or model artifacts changed during comparison.','input_sha256':frozen})
        raise RuntimeError('Frozen inputs changed; refusing to publish a complete comparison.')
    report = {'schema_version':1,'dataset_id':read(DATASET)['dataset_id'],'generated_at_utc':datetime.now(timezone.utc).isoformat(),
              'status':'completed' if all(run['status']=='completed' for run in runs) else 'incomplete',
              'protocol':{'split':args.split,'arms':args.arms,'corpora':args.corpora,'k':3,'repeats':args.repeats,
                          'fresh_process_per_arm_and_corpus':True,'fresh_document_cache_per_worker':True,'paid_generative_model_calls':0,
                          'warm_timing_includes':'Actual per-request model session switches required by the single-session low-memory implementation, plus uncached query embeddings.',
                          'primary_corpus':'handbook','control_corpus':'seed','holdout_policy':'Test cases are evaluated once after configuration freeze; subsequent use is regression, not fresh validation.',
                          'coverage':'Required verbatim quotes must be covered by verified raw source intervals; all quotes per KB unit are required. Handbook lineage is equivalent evidence.',
                          'hit_at_1':'Only the first returned chunk must completely cover at least one required evidence unit. Supplementary metric; Recall@3 remains primary.',
                          'mrr_at_3':'1/r for the smallest returned prefix of rank r<=3 whose verified source intervals jointly cover at least one complete required evidence unit; zero if none. This includes cross-chunk coverage.',
                          'overlap_metric':'Duplicate returned characters within the same source document; not semantic redundancy.'},
              'environment':{'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor()},
              'input_sha256':frozen,'runs':runs,
              'limitations':['Small synthetic corpus and agent-authored split, not externally independent or production data.',
                             'No generated answer is graded; evidence gate is a separate heuristic shared across arms.',
                             'CPU cold/warm query timings exclude download, HTTP, authentication and business database.',
                             'The declared organization is prefiltered; HTTP authorization needs separate service tests.',
                             'No claim of lower hallucination rate, enterprise throughput or statistically stable model ranking.']}
    save(args.output/'report.json',report)
    (args.output/'report.html').write_text(render(report),encoding='utf-8')
    with ZipFile(args.output/'frozen-sources.zip','x',ZIP_DEFLATED) as archive:
        for name,payload in frozen_sources.items():
            archive.writestr(name,payload)
        archive.writestr('model-and-source-fingerprints.json',json.dumps(frozen,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    main()
