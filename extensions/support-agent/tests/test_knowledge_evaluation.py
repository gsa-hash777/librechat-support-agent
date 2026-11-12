"""Offline evidence-scoring checks; no network, credentials or model calls."""
import copy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from evaluation import knowledge_run as run


def document(identifier='KB-X', org='demo', content='第一行\n第二行正文\n第三行'):
    return {'id': identifier, 'org_id': org, 'title': '密码重置说明', 'content': content, 'source': 'fixture://knowledge'}


def case(identifier='case', decision='candidates', kind='answerable'):
    return {'id': identifier, 'split': 'dev', 'kind': kind, 'query': '密码如何重置？',
            'expected_ids': ['KB-X'] if decision == 'candidates' else [],
            'desired_decision': decision, 'reference': '固定证据要求'}


def test_baseline_frozen_behavior_is_lexical_not_an_answer_judge():
    docs = [document(content='密码通过邮箱重置。')]
    response = run.baseline(docs, '密码重置后的幸运数字是多少？', 3)
    assert response['evidence_status'] == 'candidates'
    assert response['results'][0]['id'] == 'KB-X'
    assert response['decision_origin'] == 'inferred_from_nonempty_results_not_an_old_API_field'
    assert run.baseline(docs, '星舰引擎', 3)['evidence_status'] == 'insufficient'


def test_unsupported_fact_score_is_not_document_recall_or_hallucination_rate():
    query = case(decision='insufficient', kind='no_answer')
    query['expected_ids'] = ['KB-X']  # Related, but not sufficient to answer the requested fact.
    rows = run.evaluate([query], [document()], lambda *_: {'results': [document()], 'evidence_status': 'candidates'}, 1)
    stats = run.metrics(rows)
    assert stats['answerable_cases'] == 0
    assert stats['hit_at_3'] is None
    assert stats['no_answer_candidate_rate'] == 1
    assert stats['decision_match_rate'] == 0
    assert not any('hallucination' in key or 'accuracy' in key for key in stats)


def test_partial_document_coverage_and_multichunk_deduplication():
    query = case()
    query['expected_ids'] = ['KB-X', 'KB-Y']
    rows = run.evaluate([query], [document()], lambda *_: {
        'results': [{**document(), 'chunk_id': 'chunk-1'}, {**document(), 'chunk_id': 'chunk-2'}],
        'evidence_status': 'candidates'}, 1)
    assert rows[0]['retrieved_ids'] == ['KB-X']
    assert run.metrics(rows)['hit_at_3'] == 1
    assert run.metrics(rows)['macro_recall_at_3'] == .5


def test_location_requires_exact_raw_slice_lines_and_full_hash():
    source = document()
    start, end = 4, 9
    row = {**source, 'doc_id': source['id'], 'content': source['content'][start:end],
           'start_offset': start, 'end_offset': end, 'start_line': 2, 'end_line': 2,
           'content_sha256': hashlib.sha256(source['content'].encode()).hexdigest()}
    assert run.location_check([row], [source])[0]['correct'] is True
    for key, value in [('content', '改写而非原文'), ('start_line', 1), ('end_offset', 100), ('content_sha256', '0' * 64)]:
        assert run.location_check([{**row, key: value}], [source])[0]['correct'] is False
    assert run.location_check([source], [source])[0] == {'doc_id': 'KB-X', 'available': False, 'correct': None}


def test_harness_org_scoping_and_poison_case_remain_separate():
    outside = document('PRIVATE', 'other')
    query = {**case(decision='insufficient', kind='cross_org'), 'extra_documents': [outside]}
    seen = []
    def retrieve(docs, query, limit):
        seen.extend(docs)
        return {'results': [], 'evidence_status': 'insufficient'}
    rows = run.evaluate([query], [document()], retrieve, 1)
    assert [d['id'] for d in seen] == ['KB-X']
    assert not rows[0]['cross_org_leak']
    assert run.metrics(rows)['no_answer_cases'] == 0  # Security fixtures do not inflate absent-fact denominator.
    assert run.metrics(rows)['insufficient_cases'] == 1


def test_decision_counts_distinguish_clarify_from_refusal():
    query = case(decision='clarify', kind='clarify')
    rows = run.evaluate([query], [], lambda *_: {'results': [], 'evidence_status': 'insufficient'}, 1)
    assert run.metrics(rows)['clarify_rate_on_ambiguous'] == 0
    assert run.metrics(rows)['decision_match_rate'] == 0


@pytest.mark.parametrize('invalid', [None, [], {'results': [], 'evidence_status': 'certain'}, {'results': 'wrong', 'evidence_status': 'candidates'}])
def test_malformed_candidate_response_never_silently_passes(invalid):
    with pytest.raises(ValueError):
        run.evaluate([case()], [document()], lambda *_: invalid, 1)


@pytest.mark.parametrize('problem', ['duplicate', 'split', 'decision', 'reference', 'id_type'])
def test_dataset_validation_rejects_ambiguous_contract(problem):
    sample = {'schema_version': 1, 'frozen_before_candidate': True, 'cases': [case()]}
    if problem == 'duplicate': sample['cases'].append(copy.deepcopy(sample['cases'][0]))
    if problem == 'split': sample['cases'][0]['split'] = 'latest'
    if problem == 'decision': sample['cases'][0]['desired_decision'] = 'answer_proved'
    if problem == 'reference': sample['cases'][0].pop('reference')
    if problem == 'id_type': sample['cases'][0]['expected_ids'] = [{}]
    with pytest.raises(ValueError): run.validate_dataset(sample)


def test_candidate_requires_matching_frozen_input_and_new_output(tmp_path, monkeypatch):
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({'mode': 'baseline', 'fingerprints': {}}), encoding='utf-8')
    with pytest.raises(SystemExit):
        run.main(['--mode', 'candidate', '--split', 'dev', '--baseline-report', str(baseline), '--output', str(tmp_path / 'next')])
    assert not (tmp_path / 'next').exists()
    existing = tmp_path / 'existing'
    existing.mkdir()
    (existing / 'report.json').write_text('original', encoding='utf-8')
    with pytest.raises(SystemExit): run.main(['--mode', 'baseline', '--output', str(existing)])
    assert (existing / 'report.json').read_text() == 'original'


def test_public_frozen_dataset_and_legacy_inputs_have_expected_rosters():
    dataset = run.read(run.DATASET)
    cases = run.validate_dataset(dataset)
    assert len(cases) == 48
    assert len([c for c in cases if c['split'] == 'dev']) == 16
    assert len([c for c in cases if c['split'] == 'test']) == 32
    assert run.digest(run.DATASET) == 'd4279e55ef26421bedbf7bb1822d320aa6a9dfe29d254340393aeb6a36bcb409'
    assert run.digest(run.LEGACY) == '77bec8a76164ee17d7ce84b5f09d25edd11e9a70f4dee0adf10cd5d881004151'


def test_report_escapes_untrusted_query_and_evidence():
    attack = '</pre><script>alert(1)</script>'
    row = {**case(), 'query': attack, 'results': [{'content': attack}]}
    report = {'mode': 'candidate', 'metrics': {}, 'rows': [row]}
    html = run.render(report)
    assert '<script>' not in html
    assert '&lt;script&gt;' in html


def test_midrun_source_change_refuses_to_publish(tmp_path, monkeypatch):
    original = run.digest
    calls = 0
    def changing_digest(path):
        nonlocal calls
        value = original(path)
        if Path(path).name == 'content.py':
            calls += 1
            return value if calls == 1 else 'f' * 64
        return value
    monkeypatch.setattr(run, 'digest', changing_digest)
    output = tmp_path / 'unstable'
    with pytest.raises(RuntimeError, match='Source changed'):
        run.main(['--mode', 'baseline', '--split', 'dev', '--repeats', '1', '--output', str(output)])
    assert not output.exists()


def test_explicit_cases_and_default_remain_separate_frozen_inputs(tmp_path, monkeypatch):
    path = tmp_path / 'custom.json'
    data = {'schema_version': 1, 'frozen_before_candidate': True, 'dataset_id': 'unit-custom', 'cases': [case()]}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    original_default = run.DATASET.read_bytes()
    custom_dir = tmp_path / 'custom-baseline'
    report = run.main(['--mode', 'baseline', '--cases', str(path), '--split', 'dev', '--repeats', '1', '--output', str(custom_dir)])
    assert report['dataset_id'] == 'unit-custom'
    assert len(report['rows']) == 1
    assert report['fingerprints']['dataset_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    with ZipFile(custom_dir / 'frozen-sources.zip') as archive:
        assert archive.read('inputs/dataset.json') == path.read_bytes()
    # A custom baseline cannot authorize running the original default set.
    with pytest.raises(SystemExit):
        run.main(['--mode', 'candidate', '--split', 'dev', '--baseline-report', str(custom_dir / 'report.json'),
                  '--output', str(tmp_path / 'mismatched')])
    monkeypatch.setattr(run, 'candidate', lambda *_: {'results': [], 'evidence_status': 'insufficient'})
    matched = run.main(['--mode', 'candidate', '--cases', str(path), '--split', 'dev', '--repeats', '1',
                        '--baseline-report', str(custom_dir / 'report.json'), '--output', str(tmp_path / 'matched')])
    assert matched['dataset_id'] == 'unit-custom'
    assert run.DATASET.read_bytes() == original_default


def test_v31_holdout_roster_is_frozen_without_running_it():
    path = run.SERVICE / 'evaluation/knowledge_holdout_v31.json'
    cases = run.validate_dataset(run.read(path))
    assert len(cases) == 16 and all(row['split'] == 'test' for row in cases)
    assert sum(row['kind'] == 'answerable' for row in cases) == 8
    assert sum(row['kind'] == 'no_answer' for row in cases) == 4
    assert sum(row['kind'] == 'clarify' for row in cases) == 2
    assert sum(row['kind'] in {'cross_org', 'poison_only'} for row in cases) == 2
    assert run.digest(path) == '35455f8cd13eb7ec57f73d517b82a045278d87f85b9199c505c0c770c4106233'
