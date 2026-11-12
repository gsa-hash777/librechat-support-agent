"""Evidence scoring and experiment controls; no downloads or neural inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import hybrid_run as run


def original(content='账户锁定十五分钟。重置链接只能使用一次。', identifier='KB-001', org='demo'):
    return {'id':identifier,'org_id':org,'title':'账户说明','content':content,'source':'fixture://account'}


def case(kind='answerable'):
    return {'id':'unit','split':'dev','kind':kind,'query':'账户锁住后多久可以重试？',
            'tags':[],'desired_decision':{'answerable':'candidates','no_answer':'insufficient','clarify':'clarify'}[kind],
            'evidence':[{'id':'KB-001','quotes':['账户锁定十五分钟','重置链接只能使用一次']}] if kind=='answerable' else [],
            'reference':'Synthetic exact source requirements.'}


def handbook(rows):
    text, articles = '# 模拟业务手册\n\n', []
    for row in rows:
        text += '## '+row['id']+' '+row['title']+'\n\n'
        first = len(text)
        text += row['content']+'\n\n'
        articles.append({'id':row['id'],'handbook_content_start':first,'handbook_content_end':first+len(row['content']),
                         'content_sha256':hashlib.sha256(row['content'].encode()).hexdigest()})
    manifest = {'org_id':'demo','title':'模拟业务手册','handbook_sha256':hashlib.sha256(text.encode()).hexdigest(),'articles':articles}
    return text,manifest


class Engine:
    def __init__(self, decision='candidates'):
        self.calls = []
        self.decision = decision

    def retrieve(self, chunks, query, mode, limit):
        self.calls.append((mode,query,limit))
        return {'ranked_results':deepcopy(chunks[:limit]),'results':deepcopy(chunks[:limit]) if self.decision=='candidates' else [],
                'evidence_status':self.decision,'reason_codes':[],'retrieval_metadata':{'mode':mode}}


def test_fixed_split_counts_and_verbatim_reference_quotes():
    assert run.digest(run.DATASET)==run.DATASET_SHA256
    rows = run.validate_dataset(run.read(run.DATASET),run.read(run.CORPUS))
    expected = {'dev':{'answerable':8,'no_answer':2,'clarify':2},'test':{'answerable':16,'no_answer':5,'clarify':3}}
    assert {split:{kind:sum(row['split']==split and row['kind']==kind for row in rows) for kind in counts} for split,counts in expected.items()}==expected


@pytest.mark.parametrize('mutator',[
    lambda row:row.update(id=None),
    lambda row:row.update(desired_decision='clarify'),
    lambda row:row.update(evidence=[]),
    lambda row:row['evidence'][0]['quotes'].append('原文没有这项价格'),
])
def test_bad_case_or_invented_reference_rejected(mutator):
    row=case()
    mutator(row)
    with pytest.raises(ValueError):
        run.validate_dataset({'schema_version':1,'frozen_before_comparison':True,'cases':[row]},[original()])


def test_correct_handbook_source_counts_as_same_business_evidence():
    originals=[original()]
    text,manifest=handbook(originals)
    seed=run.prepare_corpus('seed',originals)
    book=run.prepare_corpus('handbook',originals,text,manifest)
    assert seed['chunks'][0]['doc_id']=='KB-001'
    assert book['chunks'][0]['doc_id'].startswith('KD-')
    assert run.evidence_coverage(case(),seed['chunks'],seed)['recall']==1
    assert run.evidence_coverage(case(),book['chunks'],book)['recall']==1


def test_quoted_id_in_wrong_handbook_section_does_not_prove_evidence():
    originals=[original(),original('只是提到 KB-001；并未包含账号处理方法。'+'无关内容。'*150,'KB-002')]
    text,manifest=handbook(originals)
    book=run.prepare_corpus('handbook',originals,text,manifest)
    selected=[chunk for chunk in book['chunks'] if chunk['start_offset']>manifest['articles'][0]['handbook_content_end']]
    assert selected
    result=run.evidence_coverage(case(),selected,book)
    assert result['hit'] is False and result['recall']==0


def test_required_quote_can_be_covered_by_multiple_adjacent_verified_chunks():
    source=original('甲'*590+'账户锁定十五分钟。重置链接只能使用一次。'+'乙'*590)
    corpus=run.prepare_corpus('seed',[source])
    full=run.evidence_coverage(case(),corpus['chunks'][:2],corpus)
    partial=run.evidence_coverage(case(),corpus['chunks'][:1],corpus)
    assert full['recall']==1
    assert partial['recall']==0
    assert full['hit_at_1'] is False and full['reciprocal_rank_at_3']==.5


def test_ranking_head_metrics_require_complete_evidence_at_actual_rank():
    corpus=run.prepare_corpus('seed',[original(),original('无关材料。','KB-002')])
    good,wrong=corpus['chunks']
    for result,expected_hit,expected_rr in (([good],True,1),([wrong,good],False,.5),([wrong,wrong,good],False,1/3),([wrong],False,0),([],False,0)):
        coverage=run.evidence_coverage(case(),result,corpus)
        assert coverage['hit_at_1'] is expected_hit
        assert coverage['reciprocal_rank_at_3']==expected_rr
    forged=deepcopy(good)
    forged['content']='invalid'
    assert run.evidence_coverage(case(),[forged,good],corpus)['reciprocal_rank_at_3']==.5


def test_ranking_head_metrics_separate_raw_and_final_and_exclude_unanswerable():
    corpus=run.prepare_corpus('seed',[original()])
    rows=run.evaluate(Engine('insufficient'),[case(),case('no_answer')],corpus,'hybrid',repeats=1)
    summary=run.metrics(rows,2)
    assert summary['answerable_cases']==1
    assert summary['raw']['hit_at_1']==summary['raw']['mrr_at_3']==1
    assert summary['final']['hit_at_1']==summary['final']['mrr_at_3']==0
    assert rows[1]['raw_coverage']['reciprocal_rank_at_3'] is None


def test_multi_unit_recall_is_macro_evidence_coverage_not_handbook_id_count():
    originals=[original(),original('退款比例没有给出。','KB-002')]
    text,manifest=handbook(originals)
    corpus=run.prepare_corpus('handbook',originals,text,manifest)
    row=case()
    row['evidence'].append({'id':'KB-002','quotes':['退款比例没有给出']})
    result=run.evidence_coverage(row,corpus['chunks'],corpus)
    assert len(corpus['documents'])==1
    assert len(result['units'])==2 and result['recall']==1


@pytest.mark.parametrize('field,value',[
    ('content','伪造原文'),('content_sha256','0'*64),('start_offset',False),('end_offset',99999),
    ('start_line',999),('revision',2),('chunk_id','KB-001-v1-fake-0001'),('doc_id','PRIVATE'),('org_id','other'),('org_id',None),
])
def test_forged_source_location_cannot_count_as_retrieved_evidence(field,value):
    corpus=run.prepare_corpus('seed',[original()])
    rows=deepcopy(corpus['chunks'])
    rows[0][field]=value
    coverage=run.evidence_coverage(case(),rows,corpus)
    assert coverage['locations'][0]['correct'] is False
    assert coverage['recall']==0


def test_raw_recall_and_shared_gate_output_are_separate():
    corpus=run.prepare_corpus('seed',[original()])
    engine=Engine('insufficient')
    rows=run.evaluate(engine,[case()],corpus,'hybrid',repeats=2)
    metrics=run.metrics(rows,1)
    assert metrics['raw']['hit_at_3']==1
    assert metrics['final']['hit_at_3']==0
    assert metrics['decision_match_rate']==0
    assert len(engine.calls)==2 and metrics['warm_latency_ms']['samples']==2


def test_public_gate_projection_may_omit_org_id_but_keeps_verified_source():
    from app.retrieval import retrieve
    corpus=run.prepare_corpus('seed',[original()])
    public=retrieve(corpus['chunks'],'账户锁定',limit=3,ranker=lambda *_:deepcopy(corpus['chunks']))['results']
    assert public and all('org_id' not in row for row in public)
    coverage=run.evidence_coverage(case(),public,corpus)
    assert coverage['recall']==1 and all(item['correct'] for item in coverage['locations'])


def test_no_answer_candidate_rate_does_not_become_answer_recall():
    corpus=run.prepare_corpus('seed',[original()])
    rows=run.evaluate(Engine(),[case('no_answer')],corpus,'lexical',repeats=1)
    metric=run.metrics(rows,1)
    assert metric['no_answer_candidate_rate']==1
    assert metric['raw']['hit_at_3'] is None
    assert metric['raw']['macro_evidence_recall_at_3'] is None


def test_invalid_adapter_stops_without_removing_unattempted_denominator():
    class Bad:
        def retrieve(self,*_,**kwargs):
            return {'results':[],'evidence_status':'candidates'}
    corpus=run.prepare_corpus('seed',[original()])
    rows=run.evaluate(Bad(),[case(),dict(case(),id='next')],corpus,'lexical',repeats=1)
    metric=run.metrics(rows,2)
    assert metric['planned']==2 and metric['attempted']==1 and metric['not_attempted']==1
    assert metric['execution_failed']==1
    assert rows[0]['execution_error']=='ValueError'


def test_overlapping_chunks_count_shared_characters_without_claiming_independence():
    corpus=run.prepare_corpus('seed',[original('甲'*1500)])
    row=case()
    row['evidence']=[{'id':'KB-001','quotes':['甲'*10]}]
    coverage=run.evidence_coverage(row,corpus['chunks'][:3],corpus)
    assert coverage['duplicate_source_characters']==160
    assert 0<coverage['duplicate_source_character_fraction']<1


def test_scope_mismatch_and_bad_handbook_lineage_are_rejected():
    with pytest.raises(ValueError):
        run.prepare_corpus('seed',[original(org='other')])
    originals=[original()]
    text,manifest=handbook(originals)
    manifest['articles'][0]['handbook_content_start']+=1
    with pytest.raises(ValueError):
        run.prepare_corpus('handbook',originals,text,manifest)


def test_duplicate_handbook_source_cannot_silently_replace_lineage():
    originals=[original()]
    text,manifest=handbook(originals)
    manifest['articles'].append(deepcopy(manifest['articles'][0]))
    with pytest.raises(ValueError,match='duplicate'):
        run.prepare_corpus('handbook',originals,text,manifest)


def test_real_business_modes_exclude_function_description_and_do_not_mix_copies():
    seed,book=run.prepare_corpus('seed'),run.prepare_corpus('handbook')
    assert len(seed['documents'])==16 and len(seed['chunks'])==16
    assert len(book['documents'])==1 and len(book['chunks'])==7
    assert 'KD-13FD5B5BBDC8DB1F26B3265F' not in {row['id'] for mode in (seed,book) for row in mode['documents']}


def test_output_and_cold_cache_cannot_be_reused(tmp_path):
    with pytest.raises(SystemExit):
        run.main(['--output',str(tmp_path)])
    with pytest.raises(SystemExit):
        run.main(['--arms','lexical','--corpora','seed','--worker-output',str(tmp_path/'out.json'),'--cache-dir',str(tmp_path)])


def test_changed_inputs_refuse_complete_report(tmp_path,monkeypatch):
    state={'calls':0}
    def hashes(paths):
        state['calls']+=1
        return {'frozen':'before' if state['calls']==1 else 'after'}
    def child(command,**kwargs):
        output=Path(command[command.index('--worker-output')+1])
        run.save(output,{'status':'completed'})
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr(run,'fingerprints',hashes)
    monkeypatch.setattr(run.subprocess,'run',child)
    output=tmp_path/'new'
    with pytest.raises(RuntimeError,match='Frozen inputs'):
        run.main(['--arms','lexical','--corpora','seed','--output',str(output),'--model-dir',str(tmp_path/'models')])
    assert (output/'incomplete.json').exists()
    assert not (output/'report.json').exists()


def test_report_html_escapes_untrusted_case_text():
    corpus=run.prepare_corpus('seed',[original()])
    row=case()
    row['query']='<script>alert(1)</script>'
    rows=run.evaluate(Engine(),[row],corpus,'lexical',repeats=1)
    summary=run.metrics(rows,1)
    text=run.render({'runs':[{'arm':'lexical','corpus':'seed','metrics_by_split':{'dev':summary},'rows':rows}]})
    assert '<script>' not in text and '&lt;script&gt;' in text
