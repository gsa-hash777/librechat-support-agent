"""Optional local dense retrieval + reciprocal-rank fusion + cross-encoder.

All modes share the existing evidence policy. Internal ranked_results are for
evaluation only; HTTP callers must remove them before returning a tool result.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import threading
import time

from .content import suspicious_reasons
from . import retrieval as default_policy


MODES = ('lexical', 'hybrid', 'rerank')
RRF_K = 60
CANDIDATE_LIMIT = 10


def safe_documents(documents):
    return [row for row in documents if not row.get('document_unsafe') and not any(
        suspicious_reasons(str(row.get(field, ''))) for field in ('title', 'content', 'source', 'heading'))]


def text_of(row):
    return '\n'.join(str(row.get(field, '')) for field in ('title', 'heading', 'content'))


def identity(row):
    return (row.get('doc_id', row.get('id')), row.get('revision', 1), row.get('chunk_id', row.get('id')))


def fuse(lexical, dense, limit=CANDIDATE_LIMIT):
    scored = {}
    for channel, rows in (('lexical', lexical), ('dense', dense)):
        for rank, row in enumerate(rows, 1):
            key = identity(row)
            if key not in scored:
                scored[key] = {**row, 'score':0.0}
            scored[key]['score'] += 1 / (RRF_K + rank)
            scored[key][channel + '_rank'] = rank
            if channel == 'dense':
                scored[key]['dense_score'] = row['dense_score']
    return sorted(scored.values(), key=lambda row:(-row['score'], str(identity(row))))[:limit]


class HybridRetriever:
    def __init__(self, model_dir=None, cache_dir=None, *, backend=None, threads=2, policy=None):
        service = Path(__file__).resolve().parent.parent
        root = service.parents[1] if service.name == 'support-agent' and service.parent.name == 'extensions' else Path.cwd()
        self.model_dir = Path(model_dir or os.environ.get('SUPPORT_MODELS_PATH', root / '.local/models/support-retrieval'))
        self.cache_dir = Path(cache_dir or os.environ.get('SUPPORT_VECTOR_CACHE_PATH', root / '.local/retrieval-vectors'))
        self.backend = backend
        self.policy = policy or default_policy
        self.threads = threads
        self.lock = threading.RLock()
        self.manifest = None
        self.fingerprint = 'injected-test-backend' if backend is not None else None
        self.cache_hits = self.cache_misses = 0

    def _backend(self):
        if self.backend is None:
            manifest_path = self.model_dir / 'manifest.json'
            if not manifest_path.exists():
                raise RuntimeError('Local retrieval models are not prepared.')
            self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            for role in ('embedding', 'reranker'):
                for local, entry in self.manifest['models'][role]['files'].items():
                    if Path(local).name != local:
                        raise RuntimeError('Invalid model manifest filename.')
                    path = self.model_dir / role / local
                    if not path.is_file():
                        raise RuntimeError('Local retrieval model files are incomplete.')
                    digest = hashlib.sha256()
                    with path.open('rb') as stream:
                        for block in iter(lambda:stream.read(1024*1024), b''):
                            digest.update(block)
                    if digest.hexdigest() != entry['sha256']:
                        raise RuntimeError('Local retrieval model integrity check failed.')
            self.fingerprint = hashlib.sha256(json.dumps(self.manifest['models'],sort_keys=True).encode()).hexdigest()
            from .hybrid_models import LocalModels
            self.backend = LocalModels(self.model_dir, threads=self.threads)
        return self.backend

    def _vectors(self, rows):
        import numpy as np
        backend = self._backend()
        organizations = {row.get('org_id') for row in rows}
        if len(organizations) != 1 or None in organizations:
            raise ValueError('Vector retrieval requires one authenticated organization scope.')
        if len(rows) > 2000:
            raise ValueError('Local experimental vector index is limited to 2000 chunks.')
        texts = [text_of(row) for row in rows]
        keys = [hashlib.sha256(json.dumps([next(iter(organizations)),self.fingerprint,'window-pooling-v1',text],ensure_ascii=False).encode()).hexdigest() for text in texts]
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.cache_dir / 'vectors.sqlite3') as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, value BLOB NOT NULL)')
            cached = {}
            for offset in range(0,len(keys),500):
                batch = keys[offset:offset+500]
                for key,dim,value in connection.execute('SELECT key,dim,value FROM vectors WHERE key IN (' + ','.join('?' for _ in batch) + ')',batch):
                    vector = np.frombuffer(value,dtype='<f4')
                    if len(vector) != dim or not np.isfinite(vector).all():
                        raise RuntimeError('Vector cache is invalid; rebuild in a new cache directory.')
                    cached[key] = vector
            missing = list(dict.fromkeys(key for key in keys if key not in cached))
            self.cache_hits += len(keys)-sum(key not in cached for key in keys)
            self.cache_misses += len(missing)
            if missing:
                lookup = dict(zip(keys,texts))
                vectors = backend.embed([lookup[key] for key in missing],query=False)
                for key,vector in zip(missing,vectors,strict=True):
                    vector = np.asarray(vector,dtype='<f4')
                    if vector.ndim != 1 or not np.isfinite(vector).all():
                        raise RuntimeError('Embedding returned an invalid vector.')
                    cached[key] = vector
                    connection.execute('INSERT OR REPLACE INTO vectors VALUES(?,?,?)',(key,len(vector),vector.tobytes()))
        return np.stack([cached[key] for key in keys])

    def retrieve(self, chunks, query, mode='lexical', limit=3):
        if mode not in MODES:
            raise ValueError('Unknown retrieval mode.')
        limit = max(1,min(10,int(limit)))
        began = time.perf_counter()
        with self.lock:
            rows = safe_documents(chunks)
            ranked = []
            if rows and not self.policy._clarification(query):
                lexical = self.policy._rank(rows,query,CANDIDATE_LIMIT if mode != 'lexical' else limit)
                if mode == 'lexical':
                    ranked = lexical
                else:
                    import numpy as np
                    matrix = self._vectors(rows)
                    vector = self._backend().embed([query],query=True)[0]
                    if matrix.shape[1] != len(vector):
                        raise RuntimeError('Cached vector dimension differs from the query model.')
                    scores = matrix @ vector
                    if not np.isfinite(scores).all():
                        raise RuntimeError('Dense retrieval returned invalid scores.')
                    dense = sorted([{**row,'dense_score':float(score)} for row,score in zip(rows,scores,strict=True)],
                                   key=lambda row:(-row['dense_score'],str(identity(row))))[:CANDIDATE_LIMIT]
                    ranked = fuse(lexical,dense)
                    if mode == 'rerank':
                        scores = self._backend().rerank(query,[text_of(row) for row in ranked])
                        if not np.isfinite(scores).all():
                            raise RuntimeError('Reranking returned invalid scores.')
                        ranked = sorted([{**row,'rerank_score':float(score)} for row,score in zip(ranked,scores,strict=True)],
                                        key=lambda row:(-row['rerank_score'],-row['score'],str(identity(row))))
            ranked = ranked[:limit]
            result = self.policy.retrieve(chunks,query,limit,ranker=lambda *_:ranked)
            result.update(ranked_results=ranked,retrieval_mode=mode,
                          retrieval_metadata={'engine':'local-hybrid-v1','rrf_k':RRF_K,'candidate_limit':CANDIDATE_LIMIT,
                                              'elapsed_ms':round((time.perf_counter()-began)*1000,3),
                                              'document_cache_hits_total':self.cache_hits,'document_cache_misses_total':self.cache_misses,
                                              'scores_are_answer_confidence':False})
            return result

    def metadata(self):
        versions = {}
        for name in ('numpy','onnxruntime','tokenizers'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return {'engine':'local-hybrid-v1','models':(self.manifest or {}).get('models',{}),
                'device':'CPU','threads':self.threads,'rrf_k':RRF_K,'candidate_limit':CANDIDATE_LIMIT,
                'document_cache_hits':self.cache_hits,'document_cache_misses':self.cache_misses,
                'versions':versions,'backend':self.backend.metadata() if self.backend else None,
                'gate':getattr(self.policy,'RETRIEVAL_VERSION','unknown'),
                'offline_inference':True,'query_embedding_cached':False}

    def close(self):
        with self.lock:
            if self.backend:
                self.backend.close()
