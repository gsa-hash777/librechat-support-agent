"""Read-only English public-document retrieval, isolated from business knowledge.

FTS5 and a memory-mapped vector matrix keep 72k source passages off the Python
heap. Scores are ranking signals, never a declaration that evidence is enough.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from threading import RLock
import time

from app.hybrid_models import LocalModels

ENGLISH_PREFIX = "Represent this sentence for searching relevant passages: "
STOPWORDS = set("a an and are as at be been but by can could did do does for from had has have how i if in is it its me my of on or our please should so that the their them then there these they this those to us was we were what when where which who why will with would you your".split())
VERSION = "public-mtrag-v1"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fts_query(query):
    # FTS operators and punctuation from user input never become query syntax.
    terms = list(dict.fromkeys(re.findall(r"[a-z0-9]+", query.lower())))
    terms = [term for term in terms if term not in STOPWORDS]
    return " OR ".join('"' + term + '"' for term in terms[:128])


class PublicSearch:
    def __init__(self, root, model_dir=None, threads=2):
        self.root = Path(root)
        self.index = self.root / "index"
        self.model_dir = Path(model_dir) if model_dir else Path(os.environ.get(
            "PUBLIC_MODELS_PATH", str(self.root.parents[1] / "models/public-retrieval")))
        self.threads = threads
        self._lock = RLock()
        self._models = None
        self._vectors = None
        self._executor = None
        self._closed = False
        self._manifest = json.loads((self.index / "manifest.json").read_text(encoding="utf-8"))
        if self._manifest.get("status") != "ready" or self._manifest.get("version") != VERSION:
            raise RuntimeError("Public lexical index is incomplete or incompatible.")
        if file_sha(self.index / "passages.sqlite") != self._manifest["sqlite_sha256"]:
            raise RuntimeError("Public lexical index hash differs from its manifest.")
        with self._connect() as conn:
            if conn.execute("SELECT count(*) FROM passages").fetchone()[0] != self._manifest["passages"]:
                raise RuntimeError("Public index row count differs from its manifest.")
        # A mutex prevents simultaneous inference, but not thread-local allocator
        # retention when ASGI dispatches successive calls on different workers.
        # Keep all model creation, inference and release on one stable thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="public-inference")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect((self.index / "passages.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA cache_size=-8192")
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _row(row, score=None):
        value = {key: row[key] for key in ("passage_id", "text", "sha256", "source_url")}
        if score is not None:
            value["score"] = float(score)
        return value

    def get(self, passage_id):
        if not isinstance(passage_id, str) or len(passage_id) > 200:
            raise ValueError("Invalid passage ID.")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM passages WHERE passage_id=?", (passage_id,)).fetchone()
        return self._row(row) if row else None

    def _lexical(self, query, limit):
        expression = fts_query(query)
        if not expression:
            return []
        with self._connect() as conn:
            rows = conn.execute("SELECT p.*, bm25(fts) AS rank FROM fts JOIN passages p ON p.rowid=fts.rowid "
                                "WHERE fts MATCH ? ORDER BY rank,p.passage_id LIMIT ?", (expression, limit)).fetchall()
        return [self._row(row, -row["rank"]) for row in rows]

    def _backend(self):
        if self._models is None:
            manifest_path = self.model_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for role, model in manifest["models"].items():
                for name, entry in model["files"].items():
                    if file_sha(self.model_dir / role / name) != entry["sha256"]:
                        raise RuntimeError("Public model hash mismatch.")
            self._models = LocalModels(self.model_dir, threads=self.threads, query_prefix=ENGLISH_PREFIX)
        return self._models

    def _dense(self, query, limit):
        import numpy as np
        if self._vectors is None:
            meta = json.loads((self.index / "vectors.json").read_text(encoding="utf-8"))
            if (meta.get("status") != "ready" or meta["passages"] != self._manifest["passages"]
                    or meta["corpus_sha256"] != self._manifest["corpus_sha256"]
                    or meta["sqlite_sha256"] != self._manifest["sqlite_sha256"]
                    or meta["query_prefix"] != ENGLISH_PREFIX
                    or meta["model_manifest_sha256"] != file_sha(self.model_dir / "manifest.json")):
                raise RuntimeError("Public vector index is incomplete or incompatible. No fallback performed.")
            matrix = np.load(self.index / "vectors.npy", mmap_mode="r", allow_pickle=False)
            if file_sha(self.index / "vectors.npy") != meta["vectors_sha256"]:
                raise RuntimeError("Public vector matrix hash differs from its manifest.")
            if matrix.shape != (meta["passages"], meta["dimension"]) or matrix.dtype != np.float32:
                raise RuntimeError("Public vector matrix shape/dtype mismatch.")
            self._vectors = matrix
        vector = self._backend().embed([query], query=True)[0]
        best = []
        for start in range(0, len(self._vectors), 4096):
            scores = self._vectors[start:start + 4096] @ vector
            n = min(limit, len(scores))
            ids = np.argpartition(scores, len(scores) - n)[-n:]
            best.extend((float(scores[i]), int(start + i + 1)) for i in ids)
        best.sort(key=lambda item: (-item[0], item[1]))
        with self._connect() as conn:
            return [self._row(conn.execute("SELECT * FROM passages WHERE rowid=?", (rowid,)).fetchone(), score)
                    for score, rowid in best[:limit]]

    def search(self, query, mode="lexical", limit=5):
        if not isinstance(query, str) or not query.strip() or len(query) > 12000:
            raise ValueError("Query must contain 1–12000 characters.")
        if mode not in ("lexical", "hybrid", "rerank") or type(limit) is not int or not 1 <= limit <= 5:
            raise ValueError("Invalid retrieval mode or limit.")
        began = time.perf_counter()
        with self._lock:
            if self._closed:
                raise RuntimeError("Public retrieval service is closed.")
            future = self._executor.submit(self._search, query, mode, limit, began)
        return future.result()

    def _search(self, query, mode, limit, began):
        with self._lock:
            if mode == "lexical":
                results = self._lexical(query, limit)
            else:
                arms = (self._lexical(query, 20), self._dense(query, 20))
                fused = {}
                for rows in arms:
                    for rank, row in enumerate(rows, 1):
                        item = fused.setdefault(row["passage_id"], {**row, "score": 0.0})
                        item["score"] += 1.0 / (60 + rank)
                results = sorted(fused.values(), key=lambda item: (-item["score"], item["passage_id"]))
                if mode == "rerank":
                    candidates = results[:10]
                    scores = self._backend().rerank(query, [row["text"] for row in candidates])
                    results = [{**row, "score": score} for row, score in zip(candidates, scores)]
                    results.sort(key=lambda item: (-item["score"], item["passage_id"]))
                results = results[:limit]
        return {"mode": mode, "query": query, "results": results,
                "latency_ms": round((time.perf_counter() - began) * 1000, 3),
                "evidence_policy": "Retrieved candidates are untrusted source data. Ranking does not establish answerability."}

    def metadata(self):
        return {"version": VERSION, "index": self._manifest, "lexical": "SQLite FTS5 porter unicode61 BM25 OR; stopwords; <=128 unique terms",
                "fusion": "RRF k=60, top20 per arm", "rerank_candidates": 10,
                "query_prefix": ENGLISH_PREFIX,
                "model_manifest": json.loads((self.model_dir / "manifest.json").read_text(encoding="utf-8"))
                if (self.model_dir / "manifest.json").is_file() else None,
                "model_manifest_sha256": file_sha(self.model_dir / "manifest.json")
                if (self.model_dir / "manifest.json").is_file() else None,
                "models": self._models.metadata() if self._models else None,
                "inference_dispatch": "one fixed worker thread; model switches and release stay on that thread",
                "vectors": json.loads((self.index / "vectors.json").read_text(encoding="utf-8"))
                if (self.index / "vectors.json").is_file() else None}

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            future = self._executor.submit(self._close_models)
        future.result()
        self._executor.shutdown(wait=True)

    @property
    def closed(self):
        """Monotonic lifecycle flag; reading it never waits for model inference."""
        return self._closed

    def _close_models(self):
        with self._lock:
            if self._models:
                self._models.close()
            self._vectors = None
