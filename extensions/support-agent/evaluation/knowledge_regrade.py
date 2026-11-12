"""Offline, separately archived citation regrading; never invokes the model."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
from evaluation.knowledge_judge import JUDGE_VERSION, judge
from evaluation.report import build_report, render_html


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new directory; previous results are immutable.')
    original_hash = digest(args.input_report)
    original = json.loads(args.input_report.read_text(encoding='utf-8'))
    rows = deepcopy(original['results'])
    for row in rows:
        row['original_judgement'] = deepcopy(row['judgement'])
        row['judgement'] = judge(row['case'], row['evidence'], row['fixture_before'], row['fixture_after'],
                                 trusted_source_base_url='http://localhost:8091')
    manifest = deepcopy(original['manifest'])
    manifest['original_run_id'] = manifest['run_id']
    manifest['run_id'] += '-knowledge-regrade-v' + JUDGE_VERSION.split('-')[0].replace('.', '')
    manifest['judge_version'] = JUDGE_VERSION
    manifest['offline_regrade'] = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'raw_report_path': str(args.input_report.resolve()), 'raw_report_sha256': original_hash,
        'model_calls': 0, 'posthoc': True,
        'scope': 'Same recorded answers, tools and snapshots; new citation checks, not a new Agent run.',
        'execution_code_hash': 'code_sha256 remains the original execution fingerprint, not this grader fingerprint.',
        'grader_sha256': {name: digest(SERVICE / 'evaluation' / name) for name in ('knowledge_judge.py', 'judge.py', 'knowledge_regrade.py')},
    }
    report = build_report(manifest, rows)
    report['limitations'].insert(0, '离线复算：未重跑模型；保留原自动判分，不与原100题成绩合并。自然语言事实与权限解释仍需语义复核。')
    if digest(args.input_report) != original_hash:
        raise RuntimeError('Input evidence changed during regrading; no report published.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'report.html').write_text(render_html(report), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'summary': report['summary'], 'model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
