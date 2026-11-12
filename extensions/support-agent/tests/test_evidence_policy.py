"""Fact precision and source provenance requirements for evidence-v4."""
import copy

import pytest

from app.retrieval import retrieve
from app.hybrid import HybridRetriever
from evaluation.frozen_v31 import retrieval as baseline


def assess(query, content, title='模拟业务规则'):
    row = {'id':'spec','doc_id':'spec','chunk_id':'spec-v2-part','revision':2,'org_id':'demo',
           'title':title,'content':content,'source':'fixture://rules','start_offset':80,
           'end_offset':80+len(content),'start_line':7,'end_line':9}
    return retrieve([row],query,ranker=lambda safe,q,k:safe[:k]),row


@pytest.mark.parametrize('query,content,code', [
    ('税率是多少？','该套餐报价800元，含税。','tax'),
    ('CSV最多能导出多少行？','CSV文件大小上限为100MB。','count_limit'),
    ('审计记录留存多久？','审计记录保留期限尚未提供。','retention'),
    ('申请处理多久完成？','链接有效期30分钟。','duration'),
    ('知识库刷新间隔多少分钟？','上传完成后通常5分钟可检索。','duration'),
    ('台账的最大规模是多少条？','首次试验时每批不超过20条记录。','count_limit'),
    ('批量导入最多支持多少条？','首次先用不超过20条核对映射，再决定是否继续导入。','count_limit'),
])
def test_related_value_or_mention_is_not_the_requested_fact(query,content,code):
    result,_ = assess(query,content)
    assert result['evidence_status']=='insufficient'
    assert code in result['evidence_assessment']['missing_requirements']
    assert not result['results']


@pytest.mark.parametrize('query,content', [
    ('税率是多少？','标准业务税率为6%。'),
    ('CSV最多能导出多少行？','CSV单次导出行数上限为500行。'),
    ('审批申请多久完成？','审批申请处理时限为2个工作日。'),
    ('审阅权限有效多久？','单次外部审阅授权最长7个自然日。'),
])
def test_precise_values_have_verifiable_quotes(query,content):
    result,row = assess(query,content)
    assert result['evidence_status']=='candidates'
    assert result['evidence_assessment']['coverage']=='full'
    for claim in result['evidence_assessment']['supported_claims']:
        for ref in claim['evidence']:
            assert ref['doc_id']==row['doc_id'] and ref['revision']==2
            assert ref['chunk_id']==row['chunk_id']
            assert row['content'][ref['start_offset']-80:ref['end_offset']-80]==ref['quote']


def test_separate_events_do_not_share_a_duration():
    result,_=assess('外部审阅最长几天？续期申请要处理多久？','外部审阅授权最长7个自然日。续期需要管理员核对。')
    checked=result['evidence_assessment']
    assert result['evidence_status']=='candidates' and checked['coverage']=='partial'
    assert len(checked['supported_claims'])==1
    assert any('续期' in item for item in checked['missing_information'])


def test_retaining_access_is_not_retaining_data_for_a_duration():
    result,_=assess('离职后可以保留账号访问权限吗？','离职后不允许保留账号访问权限，应立即停用。')
    assert result['evidence_status']=='candidates'
    assert 'retention' not in result['evidence_assessment']['missing_requirements']


def test_platform_without_business_context_needs_a_question():
    result,_=assess('这里不行了，帮我看看。','该产品有账号管理功能。')
    assert result['evidence_status']=='clarify'
    assert result['evidence_assessment']['missing_fields']==['object','symptom']


@pytest.mark.parametrize('query',['那样操作是否正确？','刚才那个提示什么意思？','我又试了一次还是那样。','这边的业务过不去了。'])
def test_unresolved_references_cannot_select_business_evidence(query):
    result,_=assess(query,'审批流程由管理员确认。')
    assert result['evidence_status']=='clarify'
    assert result['evidence_assessment']['clarification_questions']


def test_short_timing_clause_keeps_its_explicit_prior_event():
    result,_=assess('密码输错太多被锁住了，过多久才能重试？','账号连续输错密码5次后锁定15分钟。')
    assert result['evidence_status']=='candidates'
    assert result['evidence_assessment']['coverage']=='full'


def test_draft_retention_can_use_explicit_expiration_not_other_data_retention():
    result,_=assess('工单草稿可以保留多久？','草稿默认30分钟过期。')
    assert result['evidence_status']=='candidates'
    result,_=assess('审计日志可以保留多久？','草稿默认30分钟过期。',title='日志与草稿')
    assert result['evidence_status']=='insufficient'


def test_general_answer_can_be_retained_alongside_missing_limit():
    result,_=assess('统计周期按哪一天归属？最大台账规模是多少条？','已完成事项按完成日期归属统计周期。')
    assert result['evidence_status']=='candidates'
    assert result['evidence_assessment']['coverage']=='partial'
    assert result['evidence_assessment']['missing_requirements']==['count_limit']


def test_baseline_injection_keeps_the_previous_clarification_policy(tmp_path):
    row={'id':'kb','doc_id':'kb','chunk_id':'c1','revision':1,'org_id':'demo','title':'业务排查','content':'遇到异常先记录具体错误。','source':'fixture://business'}
    docs=copy.deepcopy([row])
    q='批量登记不成功。'
    old=HybridRetriever(cache_dir=tmp_path,policy=baseline).retrieve(docs,q,'lexical')
    current=HybridRetriever(cache_dir=tmp_path).retrieve(docs,q,'lexical')
    assert old['evidence_status']!=current['evidence_status']=='clarify'
    assert old['retrieval_version']=='knowledge-v3.1'
    assert docs==[row]


@pytest.mark.parametrize('query,content', [
    ('恢复操作完成需要多久？','恢复操作完成后，审阅链接有效期为30分钟。'),
    ('索引任务多久完成？','索引任务结束后，下载链接有效期为2小时。'),
])
def test_special_event_duration_does_not_borrow_a_following_link_lifetime(query,content):
    result,_=assess(query,content)
    assert result['evidence_status']=='insufficient'
    assert not result['evidence_assessment']['supported_claims']
