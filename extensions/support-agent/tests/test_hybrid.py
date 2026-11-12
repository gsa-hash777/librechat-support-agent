import copy
import sqlite3

import numpy as np
import pytest

from app.hybrid import HybridRetriever, fuse
from app.retrieval import retrieve


class Backend:
    def __init__(self):
        self.documents = []
        self.queries = []
        self.pairs = []

    def embed(self, texts, query=False):
        (self.queries if query else self.documents).extend(texts)
        return np.asarray([[1., 0.] if '邮箱' in text else [0., 1.] for text in texts], dtype=np.float32)

    def rerank(self, query, passages):
        self.pairs.extend(passages)
        return [4. if '网络' in text else -2. for text in passages]

    def close(self):
        pass

    def metadata(self):
        return {'test_backend': True}


@pytest.fixture
def chunks():
    return [dict(id='email', doc_id='email', chunk_id='email-1', revision=1, org_id='demo',
                 title='邮箱登录', content='登录失败时，请通过邮箱验证后重置密码。', source='fixture://email',
                 start_char=0, end_char=24, start_line=1, end_line=1),
            dict(id='network', doc_id='network', chunk_id='network-1', revision=1, org_id='demo',
                 title='网络连接', content='网络连接失败时请检查本地设置。', source='fixture://network')]


@pytest.mark.parametrize('query', ['登录失败怎么办？', 'HTTP 429怎么处理？', '这个怎么办？', '邮箱验证后的退款价格多少？', 'qzxv不可识别'])
def test_lexical_gate_is_identical(chunks, tmp_path, query):
    actual = HybridRetriever(cache_dir=tmp_path).retrieve(chunks, query, 'lexical')
    expected = retrieve(chunks, query, 3)
    assert {key: actual[key] for key in expected} == expected
    assert not tmp_path.joinpath('vectors.sqlite3').exists()


def test_rrf_combines_ranks_not_incomparable_scores(chunks):
    a, b = chunks
    lexical = [{**a, 'score': 999}, {**b, 'score': 1}]
    dense = [{**b, 'dense_score': .8}, {**a, 'dense_score': .2}]
    ranked = fuse(lexical, dense)
    assert ranked[0]['score'] == pytest.approx(1/61 + 1/62)
    assert ranked[0]['score'] == ranked[1]['score']
    assert ranked[0]['lexical_rank'] == 1
    assert ranked[0]['dense_rank'] == 2
    assert 'dense_score' not in a


def test_cache_reuses_documents_but_not_queries_and_tracks_content(chunks, tmp_path):
    backend = Backend()
    engine = HybridRetriever(cache_dir=tmp_path, backend=backend)
    original = copy.deepcopy(chunks)
    first = engine.retrieve(chunks, '邮箱登录失败', 'hybrid')
    second = engine.retrieve(chunks, '邮箱登录失败', 'hybrid')
    assert len(backend.documents) == 2 and len(backend.queries) == 2
    assert first['results'] == second['results']
    assert chunks == original
    chunks[0]['content'] += ' 新增段落。'
    engine.retrieve(chunks, '邮箱登录失败', 'hybrid')
    assert len(backend.documents) == 3
    engine.fingerprint = 'different-model'
    engine.retrieve(chunks, '邮箱登录失败', 'hybrid')
    assert len(backend.documents) == 5
    assert engine.metadata()['query_embedding_cached'] is False


def test_cache_is_scoped_and_mixed_organization_rejected(chunks, tmp_path):
    backend = Backend()
    engine = HybridRetriever(cache_dir=tmp_path, backend=backend)
    engine.retrieve(chunks, '邮箱登录失败', 'hybrid')
    other = [{**row, 'org_id': 'other'} for row in chunks]
    engine.retrieve(other, '邮箱登录失败', 'hybrid')
    assert len(backend.documents) == 4
    with pytest.raises(ValueError, match='organization'):
        engine.retrieve([chunks[0], other[1]], '邮箱登录失败', 'hybrid')
    with pytest.raises(ValueError, match='organization'):
        engine.retrieve([{**chunks[0], 'org_id': None}], '邮箱登录失败', 'hybrid')


def test_unsafe_document_is_excluded_before_embedding_and_reranking(chunks, tmp_path):
    backend = Backend()
    engine = HybridRetriever(cache_dir=tmp_path, backend=backend)
    chunks[0]['document_unsafe'] = True
    output = engine.retrieve(chunks, '邮箱登录失败', 'rerank')
    assert all('邮箱' not in text for text in backend.documents + backend.pairs)
    assert [row['id'] for row in output['ranked_results']] == ['network']
    assert output['excluded_unsafe_chunks'] == 1


def test_raw_rank_is_not_released_by_evidence_gate(chunks, tmp_path):
    engine = HybridRetriever(cache_dir=tmp_path, backend=Backend())
    output = engine.retrieve(chunks, '邮箱套餐价格多少？', 'rerank')
    assert output['ranked_results']
    assert output['results'] == [] and output['evidence_status'] == 'insufficient'
    assert output['retrieval_metadata']['scores_are_answer_confidence'] is False


def test_reranker_reorders_fusion_and_keeps_source_offsets(chunks, tmp_path):
    backend = Backend()
    engine = HybridRetriever(cache_dir=tmp_path, backend=backend)
    hybrid = engine.retrieve(chunks, '邮箱登录失败', 'hybrid')
    reranked = engine.retrieve(chunks, '邮箱登录失败', 'rerank')
    assert hybrid['ranked_results'][0]['id'] == 'email'
    assert reranked['ranked_results'][0]['id'] == 'network'
    original = chunks[0]
    actual = next(row for row in reranked['ranked_results'] if row['id'] == original['id'])
    for field in ('content', 'start_char', 'end_char', 'start_line', 'end_line', 'source', 'revision', 'chunk_id'):
        assert actual[field] == original[field]


def test_model_failure_is_explicit_and_never_falls_back(chunks, tmp_path):
    engine = HybridRetriever(model_dir=tmp_path/'missing', cache_dir=tmp_path/'cache')
    with pytest.raises(RuntimeError, match='not prepared'):
        engine.retrieve(chunks, '邮箱登录失败', 'hybrid')
    assert engine.retrieve(chunks, '邮箱登录失败', 'lexical')['results']


def test_corrupt_cache_is_rejected(chunks, tmp_path):
    engine = HybridRetriever(cache_dir=tmp_path, backend=Backend())
    engine.retrieve(chunks, '邮箱登录失败', 'hybrid')
    with sqlite3.connect(tmp_path/'vectors.sqlite3') as connection:
        connection.execute('UPDATE vectors SET dim=3')
    with pytest.raises(RuntimeError, match='cache is invalid'):
        engine.retrieve(chunks, '邮箱登录失败', 'hybrid')


def test_nonfinite_reranker_rejected(chunks, tmp_path):
    backend = Backend()
    backend.rerank = lambda q, passages: [float('nan')] * len(passages)
    with pytest.raises(RuntimeError, match='invalid scores'):
        HybridRetriever(cache_dir=tmp_path, backend=backend).retrieve(chunks, '邮箱登录失败', 'rerank')


def test_no_model_load_on_clarification(chunks, tmp_path):
    output = HybridRetriever(model_dir=tmp_path/'missing').retrieve(chunks, '这个怎么办？', 'rerank')
    assert output['evidence_status'] == 'clarify'
    assert not output['ranked_results']
