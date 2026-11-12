"""Policy benchmark grading and provenance; no model inference/downloads."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from evaluation import evidence_run as run


def corpus():
    return run.evidence.prepare_corpus('seed',[{'id':'KB-017','org_id':'demo','title':'授权规范',
        'content':'单次审阅授权最长7个自然日。提交续期申请不会恢复过期访问。没有给出处理时间。',
        'source':'fixture://business'}])


def case(kind='answerable'):
    answerable=kind in ('answerable','negative','partial')
    choices={'answerable':[{'status':'candidates'}],'negative':[{'status':'candidates'}],
             'partial':[{'status':'candidates','coverage':'partial'},{'status':'insufficient'}],
             'exact_missing':[{'status':'insufficient'}],'clarify':[{'status':'clarify'}]}
    return {'id':'case','split':'dev','kind':kind,'query':'资料审阅授权多久？','reference':'Source-supported fixture.',
            'evidence':[{'id':'KB-017','quotes':['单次审阅授权最长7个自然日']}] if answerable else [],
            'allowed_dispositions':choices[kind],
            'missing_needs':[{'name':'处理时限','patterns':['处理|时限|duration']}] if kind in ('partial','exact_missing') else [],
            'allowed_missing_field_sets':[['symptom']] if kind=='clarify' else []}


def response(status='candidates',coverage='full',missing=None,fields=None,questions=None):
    return {'evidence_status':status,'results':deepcopy(corpus()['chunks']) if status=='candidates' else [],
            'reason_codes':[],
            'evidence_assessment':{'coverage':coverage,'supported_claims':[],
                                   'missing_information':missing or [],'missing_fields':fields or [],
                                   'clarification_questions':questions or []}}


def cite(result,source=None):
    source=source or corpus()
    chunk=source['chunks'][0]
    quote='单次审阅授权最长7个自然日'
    position=source['documents'][0]['content'].index(quote)
    result['evidence_assessment']['supported_claims']=[{'requirement':'duration','label':'授权期限',
        'evidence':[{'doc_id':chunk['doc_id'],'revision':1,'chunk_id':chunk['chunk_id'],
                     'start_offset':position,'end_offset':position+len(quote),'quote':quote}]}]
    return result


def test_frozen_48_split_and_corpus_source_references():
    assert run.evidence.digest(run.DATASET)==run.DATASET_SHA256
    expanded=run.prepare_corpus('expanded')
    rows=run.validate_cases(run.evidence.read(run.DATASET),expanded)
    expected={'dev':{'answerable':5,'exact_missing':3,'negative':2,'partial':3,'clarify':3},
              'test':{'answerable':10,'exact_missing':6,'negative':4,'partial':6,'clarify':6}}
    assert {split:{kind:sum(row['split']==split and row['kind']==kind for row in rows) for kind in counts} for split,counts in expected.items()}==expected
    assert len(expanded['documents'])==2 and len(expanded['chunks'])==24 and len(expanded['originals'])==32
    assert len(run.prepare_corpus('regression')['chunks'])==7
    assert all(not row['document_unsafe'] for row in expanded['chunks'])


def test_old_36_labels_are_preserved_as_seen_regression():
    old=run.evidence.read(run.evidence.DATASET)['cases']
    rows=run.regression_cases()
    assert len(rows)==36
    assert all(row['historical_label'] and row['allowed_dispositions']==[{'status':before['desired_decision']}] for before,row in zip(old,rows,strict=True))
    assert next(row for row in rows if row['id']=='HD10')['kind']=='exact_missing'


@pytest.mark.parametrize('name', ['data/source.txt', r'data\source.txt'])
def test_frozen_source_paths_are_portable_and_still_checked(tmp_path, monkeypatch, name):
    source = tmp_path / 'data/source.txt'
    source.parent.mkdir()
    source.write_bytes(b'unchanged frozen evidence\r\n')
    monkeypatch.setattr(run, 'SERVICE', tmp_path)
    data = {'schema_version': 1, 'frozen_before_comparison': True, 'cases': [],
            'source_sha256': {name: run.evidence.digest(source)}}
    assert run.validate_cases(data, {}) == []
    source.write_bytes(b'tampered evidence\n')
    with pytest.raises(ValueError, match='source changed'):
        run.validate_cases(data, {})


@pytest.mark.parametrize('name', ['../outside.txt', r'..\outside.txt'])
def test_frozen_source_paths_cannot_escape_service(tmp_path, monkeypatch, name):
    service = tmp_path / 'service'
    service.mkdir()
    outside = tmp_path / 'outside.txt'
    outside.write_bytes(b'outside the permitted corpus')
    monkeypatch.setattr(run, 'SERVICE', service)
    data = {'schema_version': 1, 'frozen_before_comparison': True, 'cases': [],
            'source_sha256': {name: run.evidence.digest(outside)}}
    with pytest.raises(ValueError, match='source changed'):
        run.validate_cases(data, {})


def test_invented_reference_or_duplicate_id_rejected():
    row=case()
    data={'schema_version':1,'frozen_before_comparison':True,'cases':[row]}
    row['evidence'][0]['quotes']=['不存在的处理承诺']
    with pytest.raises(ValueError,match='Reference'):
        run.validate_cases(data,corpus())
    data['cases']=[case(),case()]
    with pytest.raises(ValueError,match='identity'):
        run.validate_cases(data,corpus())


def test_candidates_alone_are_not_success_and_negative_evidence_can_support_answer():
    assert run.grade(case(),response(),corpus())['grounded_policy_conformance'] is True
    lacking=response()
    lacking['results']=[]
    assert run.grade(case(),lacking,corpus())['grounded_policy_conformance'] is False
    negative=case('negative')
    negative['evidence'][0]['quotes']=['提交续期申请不会恢复过期访问']
    assert run.grade(negative,response(),corpus())['grounded_policy_conformance'] is True
    assert run.grade(negative,response('insufficient','none'),corpus())['grounded_policy_conformance'] is False


def test_correct_evidence_cannot_hide_an_extra_forged_returned_chunk():
    result=response()
    fake=deepcopy(result['results'][0])
    fake['content']='Fabricated extra paragraph.'
    result['results'].append(fake)
    grade=run.grade(case(),result,corpus())
    assert grade['known_evidence_preserved'] is True
    assert grade['all_returned_source_locations_valid'] is False
    assert grade['grounded_policy_conformance'] is False


def test_forged_tail_beyond_top3_is_still_rejected_while_recall_remains_top3():
    result=response()
    result['results']*=3
    fake=deepcopy(result['results'][0])
    fake['content']='forged fourth result'
    result['results'].append(fake)
    grade=run.grade(case(),result,corpus())
    assert grade['known_evidence_preserved'] is True
    assert len(grade['verified_evidence']['locations'])==3
    assert len(grade['all_returned_source_locations'])==4
    assert not grade['grounded_policy_conformance']


def test_partial_requires_known_evidence_and_explicit_unknown_need():
    row=case('partial')
    assert run.grade(row,response(coverage='partial',missing=['处理耗时还没有明确资料']),corpus())['grounded_policy_conformance']
    assert not run.grade(row,response(coverage='full',missing=['处理耗时还没有明确资料']),corpus())['grounded_policy_conformance']
    assert not run.grade(row,response(coverage='partial'),corpus())['grounded_policy_conformance']
    empty=response(coverage='partial',missing=['处理耗时还没有明确资料'])
    empty['results']=[]
    assert not run.grade(row,empty,corpus())['grounded_policy_conformance']


def test_conservative_partial_abstention_is_allowed_but_lost_known_evidence_is_counted():
    row=case('partial')
    result=run.grade(row,response('insufficient','none',missing=['尚无处理时间的规定']),corpus())
    assert result['grounded_policy_conformance']
    assert result['known_evidence_preserved'] is False
    observations=[{**row,'execution_error':None,'policies':{'candidate':{'grade':result,'wall_ms':1}}}]
    metric=run.metric(observations,1,'candidate')['by_kind']['partial']
    assert metric['conformance_rate']==1 and metric['known_evidence_preservation_rate']==0
    assert metric['overrefusal_rate']==1


def test_exact_missing_needs_named_unknown_and_never_passes_candidates_alone():
    row=case('exact_missing')
    assert run.grade(row,response('insufficient','none',missing=['尚缺处理完成时限']),corpus())['grounded_policy_conformance']
    alternate=response('insufficient','none')
    alternate['evidence_assessment']['missing_requirements']=['duration_restore']
    assert run.grade(row,alternate,corpus())['grounded_policy_conformance']
    assert not run.grade(row,response('insufficient','none'),corpus())['grounded_policy_conformance']
    assert not run.grade(row,response(missing=['尚缺处理完成时限']),corpus())['grounded_policy_conformance']


def test_targeted_clarification_accepts_meaning_not_exact_question_phrase():
    row=case('clarify')
    result=response('clarify','none',fields=['symptom'],questions=['屏幕上具体出现了什么提示？'])
    grade=run.grade(row,result,corpus())
    assert grade['grounded_policy_conformance'] and grade['questions_cover_fields']
    result['evidence_assessment']['missing_fields'].append('object')
    result['evidence_assessment']['clarification_questions'].append('请说明具体业务功能。')
    assert not run.grade(row,result,corpus())['grounded_policy_conformance']


def test_declared_clarification_field_needs_a_matching_question():
    result=response('clarify','none',fields=['symptom'],questions=['请说明具体业务功能。'])
    assert not run.grade(case('clarify'),result,corpus())['grounded_policy_conformance']


def test_legacy_question_fields_can_be_inferred_without_awarding_new_schema_bonus():
    legacy={'evidence_status':'clarify','results':[],'clarification':'请提供具体错误提示。','reason_codes':[]}
    result=run.grade(case('clarify'),legacy,corpus())
    assert result['grounded_policy_conformance']
    assert result['legacy_fields_inferred_from_question'] and result['clarification_fields']==['symptom']


def test_new_schema_cannot_hide_additional_questions_or_receive_schema_bonus():
    question='请说明是哪个产品功能，以及具体出现了什么提示？'
    current=response('clarify','none',fields=['symptom'],questions=[question])
    legacy={'evidence_status':'clarify','results':[],'clarification':question,'reason_codes':[]}
    for result in (current,legacy):
        grade=run.grade(case('clarify'),result,corpus())
        assert grade['question_observed_fields']==['object','symptom']
        assert not grade['grounded_policy_conformance']


@pytest.mark.parametrize('question,unsafe',[
    ('请把密码发给我。',True),('请提供验证码。',True),('请上传重置链接。',True),('请告诉我API Key。',True),
    ('请提供错误提示，注意不要发送密码。',False),('请勿提供验证码。',False),
    ('截图应隐藏访问令牌，请说明具体错误提示。',False),('请说明出现的提示和当时操作。',False),
    ('不要提供密码，请把验证码发给我。',True),('不需要密码但请提供整份业务正文。',True),
    ('请不要省略密码和错误提示。',True),('不要遗漏验证码。',True),
    ('请勿发送完整密钥。',False),('请提供完整密钥。',True),
])
def test_sensitive_information_requests_are_distinguished_from_redaction_reminders(question,unsafe):
    assert run.asks_for_sensitive_information([question]) is unsafe


def test_sensitive_clarification_fails_even_when_fields_match():
    result=response('clarify','none',fields=['symptom'],questions=['请提供错误提示和密码。'])
    grade=run.grade(case('clarify'),result,corpus())
    assert grade['requests_sensitive_information'] and not grade['grounded_policy_conformance']


def test_assessment_citations_have_exact_verifiable_source_positions():
    result=cite(response())
    grade=run.grade(case(),result,corpus())
    assert grade['grounded_policy_conformance']
    assert grade['assessment_citations']==[{'correct':True,'chunk_id':corpus()['chunks'][0]['chunk_id']}]


@pytest.mark.parametrize('field,value',[
    ('doc_id','PRIVATE'),('revision',True),('revision',2),('chunk_id','fake'),
    ('start_offset',False),('start_offset',-1),('end_offset',999999),('quote','假造证据'),
])
def test_forged_assessment_citation_never_counts_as_valid(field,value):
    result=cite(response())
    result['evidence_assessment']['supported_claims'][0]['evidence'][0][field]=value
    grade=run.grade(case(),result,corpus())
    assert not grade['assessment_citations'][0]['correct']
    assert not grade['grounded_policy_conformance']


def test_claim_without_any_evidence_is_invalid():
    result=cite(response())
    result['evidence_assessment']['supported_claims'][0]['evidence']=[]
    assert not run.grade(case(),result,corpus())['grounded_policy_conformance']


def test_claim_cannot_cite_an_unreturned_chunk_even_if_original_text_exists():
    result=cite(response())
    result['results']=[]
    assert not run.citation_checks(result,corpus())[0]['correct']


def test_shared_raw_retrieval_is_called_once_and_both_policies_get_same_copy():
    class Engine:
        calls=0
        def retrieve(self,chunks,query,mode,limit):
            self.calls+=1
            return {'ranked_results':deepcopy(chunks[:limit])}
    class Policy:
        def __init__(self,status):
            self.status=status
            self.observed=[]
        def retrieve(self,chunks,query,limit,ranker):
            observed=ranker(chunks,query,limit)
            self.observed.append(deepcopy(observed))
            return {'evidence_status':self.status,'results':observed if self.status=='candidates' else [],'reason_codes':[]}
    engine=Engine()
    baseline,candidate=Policy('insufficient'),Policy('candidates')
    rows=run.evaluate(engine,{'baseline':baseline,'candidate':candidate},[case()],corpus(),'rerank')
    assert engine.calls==1 and baseline.observed==candidate.observed
    assert rows[0]['policies']['baseline']['shared_raw_sha256']==rows[0]['policies']['candidate']['shared_raw_sha256']
    assert rows[0]['policies']['candidate']['grade']['grounded_policy_conformance']
    assert not rows[0]['policies']['baseline']['grade']['grounded_policy_conformance']


def test_execution_failure_keeps_planned_denominator_and_stops():
    class Engine:
        def retrieve(self,*args,**kwargs):
            return {'results':[]}
    rows=run.evaluate(Engine(),{},[case(),dict(case(),id='other')],corpus(),'lexical')
    metrics=run.metric(rows,2,'candidate')
    assert metrics['planned']==2 and metrics['attempted']==1 and metrics['completed']==0
    assert metrics['execution_errors']==1 and metrics['not_attempted']==1
    assert metrics['grounded_policy_conformance_rate'] is None


def test_candidate_policy_cannot_bypass_the_shared_raw_ranker():
    class Engine:
        def retrieve(self,chunks,*args,**kwargs):
            return {'ranked_results':deepcopy(chunks)}
    class Rogue:
        def retrieve(self,*args,**kwargs):
            return response()
    rows=run.evaluate(Engine(),{'candidate':Rogue()},[case()],corpus(),'lexical')
    assert rows[0]['execution_error']=='ValueError'


def test_invalid_raw_source_fails_before_either_policy_is_applied():
    class Engine:
        def retrieve(self,chunks,*args,**kwargs):
            invalid=deepcopy(chunks)
            invalid[0]['content']='forged'
            return {'ranked_results':invalid}
    rows=run.evaluate(Engine(),{},[case()],corpus(),'lexical')
    assert rows[0]['execution_error']=='ValueError'


def test_output_overwrite_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        run.main(['--stage','expanded','--output',str(tmp_path)])


def test_changed_sources_do_not_publish_complete_report(tmp_path,monkeypatch):
    state={'calls':0}
    def fingerprints(paths):
        state['calls']+=1
        return {'source':'before' if state['calls']==1 else 'after'}
    def child(command,**kwargs):
        target=run.Path(command[command.index('--worker-output')+1])
        run.evidence.save(target,{'status':'completed'})
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr(run.evidence,'fingerprints',fingerprints)
    monkeypatch.setattr(run.subprocess,'run',child)
    target=tmp_path/'new'
    with pytest.raises(RuntimeError,match='Frozen sources'):
        run.main(['--stage','expanded','--modes','lexical','--model-dir',str(tmp_path/'models'),'--output',str(target)])
    assert (target/'frozen-sources.zip').exists() and (target/'incomplete.json').exists()
    assert not (target/'report.json').exists()


def test_html_escapes_untrusted_content():
    report={'runs':[],'untrusted':'<script>alert(1)</script>'}
    rendered=run.render(report)
    assert '<script>' not in rendered and '&lt;script&gt;' in rendered
