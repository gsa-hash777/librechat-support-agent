"""Frozen synthetic retrieval comparison; no model, network or business database."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
import importlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
from evaluation.knowledge_baseline import search_documents
from app.content import suspicious_reasons, tool_view

DATASET = SERVICE / 'evaluation/knowledge_cases.json'
CORPUS = SERVICE / 'data/knowledge.json'
LEGACY = SERVICE / 'evaluation/cases.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_dataset(data):
    if not isinstance(data, dict) or data.get('schema_version') != 1 or data.get('frozen_before_candidate') is not True:
        raise ValueError('Expected a frozen knowledge dataset.')
    seen = set()
    for row in data.get('cases', []):
        if row.get('id') in seen or not isinstance(row.get('id'), str):
            raise ValueError('Duplicate or missing case ID.')
        seen.add(row['id'])
        if row.get('split') not in {'dev', 'test'} or row.get('desired_decision') not in {'candidates', 'insufficient', 'clarify'}:
            raise ValueError('Invalid split or expected evidence decision.')
        if not isinstance(row.get('query'), str) or not 2 <= len(row['query']) <= 300:
            raise ValueError('Invalid query.')
        if (not isinstance(row.get('expected_ids'), list) or any(not isinstance(x, str) for x in row['expected_ids'])
                or len(set(row['expected_ids'])) != len(row['expected_ids'])):
            raise ValueError('Invalid expected document IDs.')
        if not row.get('reference'):
            raise ValueError('Each case requires a reference rationale.')
    if not seen:
        raise ValueError('Empty dataset.')
    return data['cases']


def baseline(documents, query, limit):
    results = tool_view(search_documents(documents, query, limit))
    # The old endpoint had no evidence decision. This is an explicit proxy only.
    return {'results': results, 'evidence_status': 'candidates' if results else 'insufficient',
            'clarification': None, 'reason_codes': ['legacy_nonempty_result_proxy'],
            'decision_origin': 'inferred_from_nonempty_results_not_an_old_API_field'}


def document_chunks(documents):
    """Use the production pure chunker, never train or label from case expectations."""
    module = importlib.import_module('app.knowledge')
    rows = []
    for doc in documents:
        chunks = module.chunk_text(doc['content'], doc_id=doc['id'], revision=1)
        unsafe = bool(suspicious_reasons(doc['title'] + '\n' + doc['source'] + '\n' + doc['content']))
        for part in chunks:
            rows.append({**doc, **part, 'id': doc['id'], 'doc_id': doc['id'], 'revision': 1,
                         'content_sha256': hashlib.sha256(doc['content'].encode()).hexdigest(),
                         'document_unsafe': unsafe})
    return rows


def candidate(documents, query, limit):
    module = importlib.import_module('app.retrieval')
    return module.retrieve(document_chunks(documents), query, limit=limit)


def location_check(results, documents):
    sources = {d['id']: d for d in documents}
    checks = []
    for result in results:
        if not all(key in result for key in ('start_offset', 'end_offset', 'start_line', 'end_line', 'content_sha256')):
            checks.append({'doc_id': result.get('doc_id', result.get('id')), 'available': False, 'correct': None})
            continue
        source = sources.get(result.get('doc_id', result.get('id')))
        correct = False
        if source:
            content = source['content']
            start, end = result['start_offset'], result['end_offset']
            if type(start) is int and type(end) is int and 0 <= start < end <= len(content):
                expected_start_line = content[:start].count('\n') + 1
                expected_end_line = content[:max(start, end - 1)].count('\n') + 1
                correct = (content[start:end] == result['content'] and result['start_line'] == expected_start_line
                           and result['end_line'] == expected_end_line
                           and result['content_sha256'] == hashlib.sha256(content.encode()).hexdigest())
        checks.append({'doc_id': result.get('doc_id', result.get('id')), 'available': True, 'correct': correct})
    return checks


def unique_ids(results):
    return list(dict.fromkeys(row.get('doc_id', row.get('id')) for row in results))


def metrics(rows, cutoff=3):
    answerable = [r for r in rows if r['desired_decision'] == 'candidates']
    unsupported = [r for r in rows if r['desired_decision'] == 'insufficient']
    absent_facts = [r for r in rows if r['kind'] == 'no_answer']
    ambiguous = [r for r in rows if r['desired_decision'] == 'clarify']
    hits, recalls = [], []
    for row in answerable:
        expected = set(row['expected_ids'])
        found = set(row['retrieved_ids'][:cutoff])
        hits.append(int(bool(expected & found)))
        recalls.append(len(expected & found) / len(expected) if expected else 0)
    all_locations = [loc for r in rows for loc in r['location_checks']]
    available = [loc for loc in all_locations if loc['available']]
    mean = lambda values: statistics.mean(values) if values else None
    return {
        'cases': len(rows), 'answerable_cases': len(answerable), 'insufficient_cases': len(unsupported), 'clarify_cases': len(ambiguous),
        'hit_at_3': mean(hits), 'macro_recall_at_3': mean(recalls),
        'answerable_candidate_rate': mean([r['evidence_status'] == 'candidates' for r in answerable]),
        'unsupported_candidate_rate': mean([r['evidence_status'] == 'candidates' for r in unsupported]),
        'unsupported_abstain_rate': mean([r['evidence_status'] == 'insufficient' for r in unsupported]),
        'no_answer_cases': len(absent_facts),
        'no_answer_candidate_rate': mean([r['evidence_status'] == 'candidates' for r in absent_facts]),
        'no_answer_abstain_rate': mean([r['evidence_status'] == 'insufficient' for r in absent_facts]),
        'clarify_rate_on_ambiguous': mean([r['evidence_status'] == 'clarify' for r in ambiguous]),
        'decision_match_rate': mean([r['decision_matches'] for r in rows]),
        'location_available': len(available), 'returned_chunks': len(all_locations),
        'location_correct_rate': mean([loc['correct'] for loc in available]),
        'cross_org_leaks': sum(r['cross_org_leak'] for r in rows),
        'poison_presented_as_candidate': sum(r['kind'] == 'poison_only' and r['evidence_status'] == 'candidates' for r in rows),
    }


def evaluate(cases, corpus, implementation, repeats=3):
    rows = []
    for case in cases:
        all_docs = ([] if case.get('only_extra_documents') else corpus) + case.get('extra_documents', [])
        # Mirrors the authorized endpoint's organization scoping; does not test HTTP auth.
        scoped = [d for d in all_docs if d.get('org_id', 'demo') == 'demo']
        results, samples, stable = None, [], True
        for _ in range(repeats):
            started = time.perf_counter()
            actual = implementation(scoped, case['query'], 3)
            samples.append((time.perf_counter() - started) * 1000)
            if not isinstance(actual, dict) or not isinstance(actual.get('results'), list):
                raise ValueError('Invalid retrieval response, refusing to score.')
            if actual.get('evidence_status') not in {'candidates', 'insufficient', 'clarify'}:
                raise ValueError('Unknown retrieval evidence status.')
            if results is not None:
                stable = stable and actual == results
            results = actual
        outside = {d['id'] for d in all_docs if d.get('org_id', 'demo') != 'demo'}
        rows.append({**case, 'retrieved_ids': unique_ids(results['results']), 'results': results['results'],
                     'evidence_status': results['evidence_status'], 'reason_codes': results.get('reason_codes', []),
                     'clarification': results.get('clarification'), 'decision_origin': results.get('decision_origin', 'explicit_candidate_API_field'),
                     'decision_matches': case['desired_decision'] == results['evidence_status'],
                     'location_checks': location_check(results['results'], scoped),
                     'cross_org_leak': bool(outside & set(unique_ids(results['results']))),
                     'stable': stable, 'samples_ms': [round(v, 6) for v in samples]})
    return rows


def legacy_cases():
    return [{**case, 'split': 'legacy40', 'kind': 'answerable' if case['expected_ids'] else 'no_answer',
             'desired_decision': 'candidates' if case['expected_ids'] else 'insufficient',
             'reference': '原40题检索冒烟集合，原文件及原报告保持不变。'} for case in read(LEGACY)]


def render(report):
    def fmt(value):
        return '未提供' if value is None else f'{value:.1%}'
    table = []
    for split, stats in report['metrics'].items():
        table.append('<tr>' + ''.join('<td>' + escape(str(value)) + '</td>' for value in (
            split, stats['cases'], fmt(stats['hit_at_3']), fmt(stats['macro_recall_at_3']),
            fmt(stats['unsupported_candidate_rate']), fmt(stats['clarify_rate_on_ambiguous']), fmt(stats['location_correct_rate']))) + '</tr>')
    rows = []
    for row in report['rows']:
        rows.append('<details><summary>' + escape(row['id'] + ' · ' + row['query']) + '</summary><pre>' + escape(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre></details>')
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>知识检索固定集比较</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#18344b;background:#f7f9fc}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cbd7df;padding:10px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}details{margin:16px 0}summary{cursor:pointer}strong{color:#174f72}</style></head><body><h1>知识检索固定集：''' + escape(report['mode']) + '''</h1><p>公开合成样本，非独立盲测。无模型调用；这里只衡量检索、证据处置与位置元数据，不是Agent答案准确率或模型幻觉率。</p><p>旧系统没有证据处置字段，baseline使用“结果非空即候选”的显式代理口径。候选结果仅表示可供核查，不是答案已被证明；跨组织测试使用入口同等组织预过滤，不代替HTTP鉴权测试。</p><table><tr><th>集合</th><th>题数</th><th>Hit@3</th><th>宏Recall@3</th><th>无证据却给候选比例</th><th>模糊问题追问比例</th><th>位置正确率</th></tr>''' + ''.join(table) + '''</table><p>耗时为进程内检索函数（含分块，不含模型、网络、数据库/鉴权），不能与旧服务HTTP耗时直接比较。</p><details><summary>冻结信息与口径</summary><pre>''' + escape(json.dumps({k: v for k, v in report.items() if k != 'rows'}, ensure_ascii=False, indent=2)) + '</pre></details>' + ''.join(rows) + '</body></html>'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['baseline', 'candidate'], required=True)
    parser.add_argument('--cases', type=Path, default=DATASET, help='Explicit frozen dataset; original 48-case set remains the default.')
    parser.add_argument('--split', choices=['all', 'dev', 'test'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, help='Required for candidate to verify pre-frozen inputs.')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--include-legacy', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 20:
        parser.error('repeats must be 1..20')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be new or empty; prior runs are immutable.')
    dataset, corpus = read(args.cases), read(CORPUS)
    cases = validate_dataset(dataset)
    fingerprints = {'dataset_sha256': digest(args.cases), 'corpus_sha256': digest(CORPUS),
                    'legacy_dataset_sha256': digest(LEGACY), 'baseline_source_sha256': digest(SERVICE / 'evaluation/knowledge_baseline.py')}
    comparison_baseline = None
    if args.mode == 'candidate':
        if not args.baseline_report:
            parser.error('Candidate requires --baseline-report to bind frozen inputs.')
        original = read(args.baseline_report)
        if original['mode'] != 'baseline' or any(original['fingerprints'].get(k) != v for k, v in fingerprints.items()):
            parser.error('Frozen input mismatch; do not silently change the dataset after baseline.')
        comparison_baseline = {'path': str(args.baseline_report.resolve()), 'sha256': digest(args.baseline_report)}
    cases = [c for c in cases if args.split == 'all' or c['split'] == args.split]
    if args.include_legacy:
        cases += legacy_cases()
    source_files = [SERVICE / 'evaluation/knowledge_run.py', SERVICE / 'evaluation/knowledge_baseline.py', SERVICE / 'app/content.py']
    if args.mode == 'candidate':
        source_files += [SERVICE / 'app' / name for name in ('retrieval.py', 'knowledge.py', 'search.py', 'db.py', 'config.py', 'security.py')]
    source_hashes = {str(p.relative_to(SERVICE)): digest(p) for p in source_files}
    frozen_bytes = {str(p.relative_to(SERVICE)): p.read_bytes() for p in source_files}
    frozen_bytes.update({'inputs/dataset.json': args.cases.read_bytes(), 'inputs/corpus.json': CORPUS.read_bytes(),
                         'inputs/legacy_cases.json': LEGACY.read_bytes()})
    implementation = baseline if args.mode == 'baseline' else candidate
    rows = evaluate(cases, corpus, implementation, args.repeats)
    if source_hashes != {str(p.relative_to(SERVICE)): digest(p) for p in source_files}:
        raise RuntimeError('Source changed during retrieval evaluation; no result published.')
    grouped = {split: metrics([r for r in rows if r['split'] == split]) for split in dict.fromkeys(r['split'] for r in rows)}
    elapsed = sorted(sample for row in rows for sample in row['samples_ms'])
    report = {
        'schema_version': 1, 'mode': args.mode, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_id': dataset['dataset_id'], 'dataset_path': str(args.cases.resolve()), 'fingerprints': fingerprints,
        'comparison_baseline': comparison_baseline,
        'source_sha256': source_hashes,
        'environment': {'python': platform.python_version(), 'platform': platform.platform()},
        'protocol': {'split': args.split, 'k': 3, 'repeats': args.repeats, 'model_calls': 0,
                     'latency_scope': 'in-process retrieval function, including candidate chunking; excludes network/auth/database/model',
                     'baseline_decision': 'nonempty results proxy; not a historical API evidence-status output',
                     'candidate_means': 'potential evidence only, not a factual-answer guarantee',
                     'selection': 'public hand-authored synthetic dev/test split, not independent blind evaluation'},
        'metrics': grouped, 'latency_ms': {'samples': len(elapsed), 'p50': statistics.median(elapsed), 'p95': elapsed[math.ceil(.95 * len(elapsed)) - 1]},
        'rows': rows,
        'limitations': ['No LLM answers are graded', 'No production accuracy or hallucination-rate claim',
                        'Organization scoping is mirrored in the harness; HTTP permissions need service tests',
                        'Original 40 and 100 task reports are never overwritten'],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output / 'frozen-sources.zip', 'x', ZIP_DEFLATED) as archive:
        for name, payload in frozen_bytes.items():
            archive.writestr(name, payload)
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.output / 'report.html').write_text(render(report), encoding='utf-8')
    print(json.dumps({'mode': args.mode, 'fingerprints': fingerprints, 'metrics': grouped, 'output': str(args.output)}, ensure_ascii=True))
    return report


if __name__ == '__main__':
    main()
