"""Ensure development experiments preserve the question/label boundary."""
import pytest
import json
from evaluation import retrieval_v2 as module
from evaluation.retrieval_v2 import Variant, fuse, partition, task_queries


def task(messages):
    return {"task_id": "q", "conversation_id": "c", "turn": len(messages), "messages": messages}


def test_queries_never_use_assistant_reference_answers():
    value = task([{"speaker": "user", "text": "How do I configure the service?"},
                  {"speaker": "assistant", "text": "REFERENCE ANSWER CANARY"},
                  {"speaker": "user", "text": "What about its limits?"}])
    queries = task_queries(value)
    assert all("CANARY" not in query for query in queries.values())
    assert queries["conditional"] == "How do I configure the service?\nWhat about its limits?"


def test_reject_gold_fields_and_invalid_roles():
    value = task([{"speaker": "user", "text": "Question"}])
    with pytest.raises(ValueError):
        task_queries({**value, "targets": ["gold"]})
    with pytest.raises(ValueError):
        task_queries(task([{"speaker": "system", "text": "Question"}]))


def test_same_query_not_double_weighted_by_history_blend():
    row = {"queries": {"lastturn": "q", "conditional": "q"},
           "candidates": {"conditional": {"lexical": [{"id": "a"}], "dense": [{"id": "b"}]}}}
    assert fuse(row, Variant("blend", policy="conditional", current_weight=1)) == fuse(row, Variant("plain", policy="conditional"))


def test_candidate_limit_controls_available_ids_and_no_duplicates():
    row = {"queries": {"lastturn": "q"}, "candidates": {"lastturn": {
        "lexical": [{"id": "a"}, {"id": "b"}], "dense": [{"id": "a"}, {"id": "c"}]}}}
    assert [r["id"] for r in fuse(row, Variant("one", candidates=1))] == ["a"]
    row["candidates"]["lastturn"]["dense"].append({"id": "a"})
    with pytest.raises(ValueError):
        fuse(row, Variant("invalid"))


def test_conversation_partition_does_not_depend_on_turn():
    assert partition("conversation-1") == partition("conversation-1")
    assert partition("conversation-1") in {"train", "validation"}


def test_rerank_execution_error_remains_in_denominator(tmp_path, monkeypatch):
    def row(identifier):
        return {"task_id": identifier, "partition": "train", "queries": {"lastturn": identifier},
                "candidates": {"lastturn": {"lexical": [{"id": identifier}], "dense": [{"id": identifier}]}}}
    rows = [row("good"), row("bad")]
    monkeypatch.setattr(module, "checked_candidates", lambda *args: rows)
    monkeypatch.setattr(module.data, "load_qrels", lambda *args: {"good": {"good": 1}, "bad": {"bad": 1}})
    class Engine:
        def __init__(self, **kwargs):
            pass
        def get(self, identifier):
            if identifier == "bad":
                raise RuntimeError("a sensitive error body that must not be copied")
            return {"text": identifier}
        def _backend(self):
            return self
        def rerank(self, query, texts):
            return [1.0] * len(texts)
        def close(self):
            pass
    import app.public_search
    monkeypatch.setattr(app.public_search, "PublicSearch", Engine)
    source = tmp_path / "source"
    source.mkdir()
    (source / "candidates.jsonl").write_text("candidate provenance", encoding="utf-8")
    output = tmp_path / "run"
    lock = tmp_path / "cpu.lock"
    module.rerank(tmp_path, tmp_path, source, output, lock, "wide_hybrid100", 50, None)
    report = json.loads((output / "report.json").read_text())
    observations = (output / "observations.jsonl").read_text()
    assert report["planned"] == 2 and report["execution_errors"] == 1
    assert report["metrics"]["all"]["macro_recall_at_5"] == .5
    assert "sensitive error" not in observations
    assert not lock.exists()


def test_cache_hash_tampering_is_rejected_before_any_inference(tmp_path):
    (tmp_path / "candidates.jsonl").write_text("tampered", encoding="utf-8")
    (tmp_path / "completion.json").write_text(json.dumps({"source_unchanged": True, "candidates_sha256": "wrong"}), encoding="utf-8")
    (tmp_path / "freeze.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        module.checked_candidates(tmp_path, tmp_path)


def test_resume_keeps_completed_prefix_and_rejects_config_change(tmp_path, monkeypatch):
    rows = [{"task_id": identifier, "partition": "train", "queries": {"lastturn": identifier},
             "candidates": {"lastturn": {"lexical": [{"id": identifier}], "dense": [{"id": identifier}]}}}
            for identifier in ("one", "two")]
    monkeypatch.setattr(module, "checked_candidates", lambda *args: rows)
    monkeypatch.setattr(module.data, "load_qrels", lambda *args: {"one": {"one": 1}, "two": {"two": 1}})
    calls = []
    interrupted = [False]
    class Engine:
        def __init__(self, **kwargs):
            pass
        def get(self, identifier):
            return {"text": identifier}
        def _backend(self):
            return self
        def rerank(self, query, texts):
            calls.append(query)
            if query == "two" and not interrupted[0]:
                interrupted[0] = True
                raise KeyboardInterrupt()
            return [1.0]
        def close(self):
            pass
    import app.public_search
    monkeypatch.setattr(app.public_search, "PublicSearch", Engine)
    source = tmp_path / "source"
    source.mkdir()
    (source / "candidates.jsonl").write_text("frozen", encoding="utf-8")
    output, lock = tmp_path / "run", tmp_path / "cpu.lock"
    with pytest.raises(KeyboardInterrupt):
        module.rerank(tmp_path, tmp_path, source, output, lock, "wide_hybrid100", 50, None)
    assert not lock.exists()
    with pytest.raises(ValueError, match="rerank_pool"):
        module.rerank(tmp_path, tmp_path, source, output, lock, "wide_hybrid100", 10, None, True)
    module.rerank(tmp_path, tmp_path, source, output, lock, "wide_hybrid100", 50, None, True)
    assert calls == ["one", "two", "two"]
    report = json.loads((output / "report.json").read_text())
    assert report["resumed_count"] == 1 and report["completed"] == 2
    assert report["metrics"]["all"]["macro_recall_at_5"] == 1.0
