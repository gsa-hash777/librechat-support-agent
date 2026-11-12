import hashlib
import json
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import threading

from fastapi.testclient import TestClient
import pytest

from app.public_api import create_app
from app.public_search import PublicSearch, VERSION, file_sha, fts_query


@pytest.fixture
def corpus(tmp_path):
    index = tmp_path / "index"
    index.mkdir()
    with sqlite3.connect(index / "passages.sqlite") as conn:
        conn.execute("CREATE TABLE passages (rowid INTEGER PRIMARY KEY, passage_id TEXT UNIQUE, source_url TEXT, text TEXT, sha256 TEXT)")
        conn.execute("CREATE VIRTUAL TABLE fts USING fts5(text,content='passages',content_rowid='rowid',tokenize='porter unicode61')")
        for number, text in enumerate(("Reset account passwords with the recovery email.",
                                       "Back up encrypted databases daily.", "<script>alert(1)</script> cloud passwords"), 1):
            conn.execute("INSERT INTO passages VALUES(?,?,?,?,?)", (number, f"cloud-{number}",
                         "https://cloud.ibm.com/docs/example", text, hashlib.sha256(text.encode()).hexdigest()))
        conn.execute("INSERT INTO fts(fts) VALUES('rebuild')")
    (index / "manifest.json").write_text(json.dumps({"version": VERSION, "status": "ready", "passages": 3,
                                                   "sqlite_sha256": file_sha(index / "passages.sqlite"),
                                                   "corpus_sha256": "fixture"}))
    return tmp_path


def test_fts_relevance_and_original_evidence(corpus):
    engine = PublicSearch(corpus)
    results = engine.search("How can I reset a password?")["results"]
    assert results[0]["passage_id"] == "cloud-1"
    assert engine.get("cloud-1")["sha256"] == hashlib.sha256(results[0]["text"].encode()).hexdigest()
    assert engine.get("unknown") is None
    assert engine.search("the or and")["results"] == []


def test_injection_is_only_data(corpus):
    engine = PublicSearch(corpus)
    assert "NEAR(" not in fts_query('"NEAR(password)" OR text:*')
    assert engine.get("' OR 1=1 --") is None
    assert engine.search('" OR NOT NEAR(reset) *')["results"]


def test_integrity_before_search(corpus):
    with sqlite3.connect(corpus / "index/passages.sqlite") as conn:
        conn.execute("UPDATE passages SET text='tampered' WHERE rowid=1")
    with pytest.raises(RuntimeError, match="hash"):
        PublicSearch(corpus)


@pytest.mark.parametrize("query,mode,limit", [("", "lexical", 5), ("x", "dense", 5), ("x", "lexical", 6), ("x", "lexical", True)])
def test_invalid_queries_fail(corpus, query, mode, limit):
    with pytest.raises(ValueError):
        PublicSearch(corpus).search(query, mode, limit)


def test_missing_vectors_does_not_silently_use_bm25(corpus):
    with pytest.raises(OSError):
        PublicSearch(corpus).search("account", "hybrid")


def test_public_readonly_api_and_escaped_console(corpus):
    config = corpus / "config.json"
    config.write_text(json.dumps({"tool_token": "x" * 48}))
    with TestClient(create_app(config, PublicSearch(corpus))) as client:
        assert client.get("/health").json()["read_only"]
        schema = client.get("/openapi.json").json()
        assert set(schema["paths"]) == {"/search", "/evidence/{passage_id}"}
        assert all(set(item) == {"get"} for item in schema["paths"].values())
        assert client.get("/search?q=account").status_code == 401
        headers = {"Authorization": "Bearer " + "x" * 48}
        assert client.get("/search?q=account", headers=headers).json()["results"]
        assert client.get("/evidence/cloud-1", headers=headers).status_code == 200
        assert client.post("/search?q=account", headers=headers).status_code == 405
        assert client.get("/search?q=x&mode=hybrid", headers=headers).status_code == 503
        result = client.get("/passage/cloud-3")
        assert "<script>" not in result.text and "&lt;script&gt;" in result.text
        assert "x" * 48 not in result.text
        assert client.get("/evidence/unknown", headers=headers).status_code == 404


def test_model_switches_and_release_have_one_thread_owner_across_callers(corpus, monkeypatch):
    engine = PublicSearch(corpus)
    owners, calls = [], []
    row = engine.get("cloud-1")
    def dense(query, limit):
        owners.append(threading.get_ident())
        calls.append(("embedding", query))
        return [{**row, "score": 1.0}]
    class Models:
        def rerank(self, query, texts):
            owners.append(threading.get_ident())
            calls.append(("reranker", query))
            return [2.0] * len(texts)
        def close(self):
            owners.append(threading.get_ident())
            calls.append(("close", None))
    engine._models = Models()
    monkeypatch.setattr(engine, "_dense", dense)
    with ThreadPoolExecutor(max_workers=4) as callers:
        results = list(callers.map(lambda i: engine.search("password " + str(i), "rerank"), range(12)))
    engine.close()
    engine.close()
    assert len(set(owners)) == 1 and owners[0] != threading.get_ident()
    assert calls[-1] == ("close", None)
    assert len([x for x in calls if x[0] == "embedding"]) == 12
    assert all(result["results"][0]["score"] == 2.0 for result in results)
    with pytest.raises(RuntimeError, match="closed"):
        engine.search("password")


def test_liveness_does_not_query_locked_model_metadata(corpus, monkeypatch):
    config = corpus / "config.json"
    config.write_text(json.dumps({"tool_token": "x" * 48}))
    engine = PublicSearch(corpus)
    app = create_app(config, engine)
    def forbidden_metadata():
        raise AssertionError("Liveness must not wait for model metadata")
    monkeypatch.setattr(engine, "metadata", forbidden_metadata)
    with TestClient(app) as client:
        assert client.get("/health").json()["passages"] == 3
        engine.close()
        assert client.get("/health").status_code == 503
