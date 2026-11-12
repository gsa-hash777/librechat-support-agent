"""No model inference: frozen selection, thread dispatch and score integrity."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from threading import RLock, get_ident

import pytest

from evaluation import retrieval_v2_regression as regression
from evaluation.retrieval_v2 import Variant


def selection():
    return {"schema_version": 1, "selection_scope": "development_only",
            "source_report": {"path": "dev-report.json", "sha256": "a" * 64},
            "selected": {"variant": asdict(Variant("wide_hybrid100")), "rerank_pool": 20, "second_rrf": None}}


def task(identifier="q1"):
    return {"task_id": identifier, "conversation_id": "c1", "turn": 2,
            "messages": [{"speaker": "user", "text": "How to rotate credentials?"},
                         {"speaker": "assistant", "text": "SEALED_ASSISTANT_REFERENCE_MUST_NOT_REACH_RETRIEVER"},
                         {"speaker": "user", "text": "What about expiry?"}]}


def test_selection_is_fixed_to_predeclared_grid():
    assert regression.validate_selected(selection())["rerank_pool"] == 20
    for field, value in (("rerank_pool", 5), ("second_rrf", {"k": 11, "rerank_weight": .5}),
                         ("second_rrf", {"k": 60, "rerank_weight": .8})):
        chosen = selection()
        chosen["selected"][field] = value
        with pytest.raises(ValueError):
            regression.validate_selected(chosen)


def test_selection_rejects_implicit_or_nonfinite_variant():
    chosen = selection()
    del chosen["selected"]["variant"]["dense_weight"]
    with pytest.raises(ValueError):
        regression.validate_selected(chosen)
    chosen = selection()
    chosen["selected"]["variant"]["dense_weight"] = float("nan")
    with pytest.raises(ValueError):
        regression.validate_selected(chosen)


def test_selection_requires_complete_hashed_dev_evidence(tmp_path):
    chosen = selection()
    report = tmp_path / "dev-report.json"
    report.write_text(json.dumps({"split": "dev", "full_development": True, "planned": 188, "completed": 188, "source_unchanged": True}))
    chosen["source_report"]["sha256"] = regression.data.sha256(report)
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(chosen))
    assert regression.load_selection(path)[1] == chosen["selected"]
    report.write_text(json.dumps({"split": "dev", "full_development": False, "planned": 24, "completed": 24, "source_unchanged": True}))
    with pytest.raises(ValueError, match="hash"):
        regression.load_selection(path)
    chosen["source_report"]["sha256"] = regression.data.sha256(report)
    path.write_text(json.dumps(chosen))
    with pytest.raises(ValueError, match="188"):
        regression.load_selection(path)


def test_fusion_preserves_pool_and_deterministic_ties():
    ids = ["c", "a", "b"]
    assert regression.second_fusion(ids, [1, 3, 2], None) == ["a", "b", "c"]
    for weight in (.25, .5, .75):
        config = {"k": 10, "rerank_weight": weight}
        ranked = regression.second_fusion(ids, [1, 3, 2], config)
        assert set(ranked) == set(ids)
        rank = {"a": 1, "b": 2, "c": 3}
        expected = sorted(ids, key=lambda key: (-(weight / (10 + rank[key]) + (1 - weight) / (10 + ids.index(key) + 1)), key))
        assert ranked == expected
    assert regression.second_fusion(["b", "a"], [2, 2], None) == ["a", "b"]


@pytest.mark.parametrize("ids,scores", [(["a", "a"], [1, 2]), (["a"], []), (["a"], [float("inf")]), (["a"], [True])])
def test_invalid_rerank_results_fail(ids, scores):
    with pytest.raises(ValueError):
        regression.second_fusion(ids, scores, None)


class FakeEngine:
    def __init__(self, expected_limit=100):
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.closed = False
        self.seen = []
        self.expected_limit = expected_limit

    def _record(self, operation, query=None):
        self.seen.append((operation, get_ident(), query))

    def _lexical(self, query, limit):
        assert limit == self.expected_limit
        self._record("lexical", query)
        return [{"passage_id": "p1", "score": 1}, {"passage_id": "p2", "score": .5}]

    def _dense(self, query, limit):
        assert limit == self.expected_limit
        self._record("dense", query)
        return [{"passage_id": "p2", "score": 1}, {"passage_id": "p3", "score": .5}]

    def get(self, key):
        self._record("get")
        return {"passage_id": key, "text": "Content " + key}

    def _backend(self):
        self._record("backend")
        return self

    def rerank(self, query, passages):
        self._record("rerank", query)
        return [float(index) for index in range(len(passages))]

    def close(self):
        self._executor.submit(self._record, "close").result()
        self._executor.shutdown()
        self.closed = True


def test_registered_original_pool10_uses_original_per_arm_limit():
    selected = selection()
    selected["selected"]["variant"] = asdict(Variant("original_hybrid20", candidates=20))
    selected["selected"]["rerank_pool"] = 10
    selected["selected"]["second_rrf"] = {"k": 10, "rerank_weight": .5}
    chosen = regression.validate_selected(selected)
    engine = FakeEngine(expected_limit=20)
    try:
        result = regression.retrieve(engine, task(), chosen)
        assert set(result["ranking"]) <= {"p1", "p2", "p3"}
        assert sum(op == "dense" for op, _, _ in engine.seen) == 1
    finally:
        engine.close()
    selected["selected"]["rerank_pool"] = 20
    with pytest.raises(ValueError, match="original pool10"):
        regression.validate_selected(selected)


def test_recall_rerank_and_release_use_one_worker_and_never_assistant_reference():
    engine = FakeEngine()
    chosen = selection()["selected"]
    chosen["variant"] = asdict(Variant("blend_previous", policy="previous_last", current_weight=1.0))
    parent_thread = get_ident()
    try:
        first = regression.retrieve(engine, task(), chosen)
        second = regression.retrieve(engine, task("q2"), chosen)
        assert first["ranking"] == second["ranking"]
        assert len(first["ranking"]) <= 5
    finally:
        engine.close()
    assert len({thread for _, thread, _ in engine.seen}) == 1
    assert all(thread != parent_thread for _, thread, _ in engine.seen)
    assert not any("SEALED_ASSISTANT" in str(query) for _, _, query in engine.seen)
    assert sum(operation == "dense" for operation, _, _ in engine.seen) == 4  # no cross-task cache


def test_qrels_or_targets_cannot_be_passed_to_retrieval():
    engine = FakeEngine()
    try:
        contaminated = {**task(), "qrels": {"p1": 1}}
        with pytest.raises(ValueError, match="sanitized"):
            regression.retrieve(engine, contaminated, selection()["selected"])
        assert not engine.seen
    finally:
        engine.close()


def test_paired_comparison_keeps_failures_and_unattempted_in_denominator():
    qrels = {"q1": {"a": 1, "b": 1}, "q2": {"a": 1}, "q3": {"z": 1}}
    old = {"q1": {"ranking": ["a"]}, "q2": {"ranking": []}, "q3": {"ranking": ["z"]}}
    rows = [{"task_id": "q1", "ranking": ["a", "b"], "state": "completed"},
            {"task_id": "q2", "ranking": [], "state": "execution_error"}]
    result = regression.paired_comparison(rows, old, list(qrels), qrels, samples=1000, seed=7)
    assert result["n"] == 3
    assert result["mean_recall_delta"] == pytest.approx(-.5 / 3)
    assert result["recall_improved"] == 1
    assert result["recall_worsened"] == 1
    assert result["hit_lost"] == 1
    assert result["per_task"][-1]["state"] == "not_attempted"
    assert result == regression.paired_comparison(rows, old, list(qrels), qrels, samples=1000, seed=7)


def resume_fixture(tmp_path):
    frozen = {"source_fingerprints": {}, "selection": selection(), "planned": 2}
    regression.write_new(tmp_path / "freeze.json", frozen)
    qrels = {"q1": {"p1": 1}, "q2": {"p2": 1}}
    row = {"task_id": "q1", "state": "execution_error", "ranking": [], "candidate_ids": [],
           "metrics": regression.score_ranking([], qrels["q1"]), "latency_ms": 1}
    regression.append_row(tmp_path / "observations.jsonl", row, regression.object_sha(frozen))
    return frozen, [task("q1"), task("q2")], qrels


def test_cluster_bootstrap_keeps_turns_in_a_conversation_together():
    qrels = {key: {"p": 1} for key in ("q1", "q2", "q3")}
    old = {"q1": {"ranking": []}, "q2": {"ranking": []}, "q3": {"ranking": ["p"]}}
    rows = [{"task_id": key, "ranking": ["p"] if key != "q3" else [], "state": "completed"} for key in qrels]
    clusters = {"q1": "c1", "q2": "c1", "q3": "c2"}
    result = regression.paired_comparison(rows, old, list(qrels), qrels, samples=1000, seed=9, conversation_ids=clusters)
    assert result["mean_recall_delta"] == pytest.approx(1 / 3)
    assert result["bootstrap"]["unit"] == "paired conversation cluster"
    assert result["bootstrap"]["clusters"] == 2
    assert result["recall_delta_95_ci"] == [-1, 1]
    assert result == regression.paired_comparison(rows, old, list(qrels), qrels, samples=1000, seed=9, conversation_ids=clusters)
    with pytest.raises(ValueError, match="every task"):
        regression.paired_comparison(rows, old, list(qrels), qrels, conversation_ids={"q1": "c1"})


def test_resume_keeps_failed_prefix_and_rejects_reordered_or_tampered_rows(tmp_path):
    frozen, tasks, qrels = resume_fixture(tmp_path)
    assert regression.validate_resume(tmp_path, frozen, tasks, qrels)[0]["state"] == "execution_error"
    with pytest.raises(ValueError, match="prefix"):
        regression.validate_resume(tmp_path, frozen, list(reversed(tasks)), qrels)
    path = tmp_path / "observations.jsonl"
    path.write_text(path.read_text().replace('"latency_ms":1', '"latency_ms":999'))
    with pytest.raises(ValueError, match="prefix"):
        regression.validate_resume(tmp_path, frozen, tasks, qrels)


def test_partial_or_completed_run_cannot_be_silently_replayed(tmp_path):
    frozen, tasks, qrels = resume_fixture(tmp_path)
    path = tmp_path / "observations.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    with pytest.raises(ValueError, match="Partial"):
        regression.validate_resume(tmp_path, frozen, tasks, qrels)
    regression.write_new(tmp_path / "report.json", {"completed": 2})
    with pytest.raises(ValueError, match="unfinished"):
        regression.validate_resume(tmp_path, frozen, tasks, qrels)


def test_source_text_cannot_be_fabricated_using_correct_id_and_hash():
    original = "Pinned paragraph"
    sha = hashlib.sha256(original.encode()).hexdigest()
    response = {"ranking": ["p1"], "returned_passages": [{"passage_id": "p1", "sha256": sha, "text": original, "source_url": "https://cloud.ibm.com/docs/example"}]}
    sources = {"p1": {"sha256": sha, "source_url": response["returned_passages"][0]["source_url"]}}
    regression.validate_passages(response, sources)
    response["returned_passages"][0]["text"] = "Invented paragraph"
    with pytest.raises(ValueError, match="pinned"):
        regression.validate_passages(response, sources)
