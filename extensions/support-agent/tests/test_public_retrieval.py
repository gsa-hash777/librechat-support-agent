"""Known metrics, denominator integrity and input separation; no inference."""
from copy import deepcopy
import hashlib
import math

import pytest

from evaluation import public_retrieval as run


def task(identifier="q1"):
    return {"task_id": identifier, "conversation_id": "c", "turn": 2,
            "messages": [{"speaker": "user", "text": "service setup"},
                         {"speaker": "assistant", "text": "GOLD_EARLIER_RESPONSE"},
                         {"speaker": "user", "text": "how much?"}]}


def test_history_query_has_only_latest_four_user_messages():
    row = task()
    row["messages"] = [{"speaker": "user", "text": str(n)} for n in range(7)]
    row["messages"].insert(-1, {"speaker": "assistant", "text": "secret reference"})
    assert run.build_query(row) == "6"
    assert run.build_query(row, "previous_user3") == "3\n4\n5\n6"
    assert "GOLD" not in run.build_query(task(), "previous_user3")


def test_gold_fields_fail_closed_at_query_builder():
    row = task()
    row["targets"] = [{"text": "gold"}]
    with pytest.raises(ValueError, match="allowlisted"):
        run.build_query(row)
    row = task()
    row["messages"][-1]["enrichments"] = {"answerability": "ANSWERABLE"}
    with pytest.raises(ValueError, match="sanitized"):
        run.build_query(row)


def test_known_ranking_scores_and_macro_recall_denominator():
    scored = run.score_ranking(["wrong", "p1", "p2"], {"p1": 1, "p2": 1, "p3": 1})
    assert scored["recall_at_5"] == pytest.approx(2 / 3)
    assert scored["reciprocal_rank_at_5"] == .5
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3) + 1 / math.log2(4))
    assert scored["ndcg_at_5"] == pytest.approx(expected)
    assert scored["hit_at_5"] == 1
    assert run.score_ranking(["wrong"] * 0, {"p1": 1})["hit_at_5"] == 0


def test_ndcg_explicit_linear_gain_and_top5_cutoff():
    scored = run.score_ranking(["p2", "p1"], {"p1": 2, "p2": 1})
    assert scored["ndcg_at_5"] == pytest.approx((1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3)))
    assert run.score_ranking(["a", "b", "c", "d", "e", "p1"], {"p1": 1})["hit_at_5"] == 0


@pytest.mark.parametrize("ranking,gold", [(["p1", "p1"], {"p1": 1}), ([], {}), ([], {"p1": 0}), ([], {"p1": True})])
def test_invalid_ranking_or_qrels_rejected(ranking, gold):
    with pytest.raises(ValueError):
        run.score_ranking(ranking, gold)


def test_retriever_never_receives_tasks_qrels_or_gold_labels():
    calls = []
    def search(query, **kwargs):
        calls.append((query, kwargs))
        return {"results": [{"passage_id": "p1", "text": "document"}], "latency_ms": 5}
    rows = run.evaluate(search, [task()], {"q1": {"p1": 1}}, mode="hybrid",
                        query_policy="previous_user3", labels={"q1": {"answerability": "ANSWERABLE"}})
    assert calls == [("service setup\nhow much?", {"mode": "hybrid", "limit": 5})]
    assert rows[0]["metrics"]["recall_at_5"] == 1
    assert rows[0]["annotation"]["answerability"] == "ANSWERABLE"


def test_failures_score_zero_and_remain_in_denominator():
    counter = 0
    def search(query, **kwargs):
        nonlocal counter
        counter += 1
        if counter == 2:
            raise RuntimeError("sensitive query must not appear in error text")
        return [{"id": "p1"}]
    tasks = [task("q1"), task("q2"), task("no-qrels")]
    rows = run.evaluate(search, tasks, {"q1": {"p1": 1}, "q2": {"p1": 1}}, mode="lexical")
    summary = run.summarize(rows, 2)
    assert summary["planned"] == 2 and summary["completed"] == 1 and summary["execution_errors"] == 1
    assert summary["macro_recall_at_5"] == summary["hit_at_5"] == .5
    assert "sensitive" not in str(rows)
    assert summary["latency_ms"]["samples"] == 2
    assert run.summarize(rows[:1], 2)["macro_recall_at_5"] == .5
    assert run.summarize(rows[:1], 2)["not_attempted"] == 1


@pytest.mark.parametrize("results", [[{"id": "p1"}, {"id": "p1"}], [{"id": "unknown"}], [{}], ["p1"]])
def test_invalid_sources_become_failed_observation(results):
    rows = run.evaluate(lambda *args, **kwargs: results, [task()], {"q1": {"p1": 1}},
                        mode="lexical", known_ids={"p1"})
    assert rows[0]["state"] == "execution_error"
    assert run.summarize(rows, 1)["macro_recall_at_5"] == 0


def test_oversized_response_is_not_silently_truncated():
    rows = run.evaluate(lambda *args, **kwargs: [{"id": f"p{i}"} for i in range(6)],
                        [task()], {"q1": {"p1": 1}}, mode="rerank")
    assert rows[0]["state"] == "execution_error"


@pytest.mark.parametrize("field,value", [("text", "fabricated"), ("sha256", "fake"), ("source_url", "https://wrong.example")])
def test_correct_id_with_fabricated_text_or_location_cannot_score(field, value):
    original = {"passage_id": "p1", "text": "exact original", "source_url": "https://cloud.ibm.com/docs/item",
                "sha256": hashlib.sha256(b"exact original").hexdigest()}
    source = {"p1": {key: original[key] for key in ("sha256", "source_url")}}
    valid = run.evaluate(lambda *args, **kwargs: [original], [task()], {"q1": {"p1": 1}},
                         mode="lexical", passage_sources=source)
    assert valid[0]["metrics"]["recall_at_5"] == 1
    forged = dict(original, **{field: value})
    invalid = run.evaluate(lambda *args, **kwargs: [forged], [task()], {"q1": {"p1": 1}},
                           mode="lexical", passage_sources=source)
    assert invalid[0]["state"] == "execution_error"
    assert invalid[0]["metrics"]["recall_at_5"] == 0


def test_missing_planned_query_or_gold_corpus_row_stops_before_inference():
    def forbidden(*args, **kwargs):
        pytest.fail("inference should not start")
    with pytest.raises(ValueError, match="include every"):
        run.evaluate(forbidden, [task()], {"missing": {"p1": 1}}, mode="lexical")
    with pytest.raises(ValueError, match="absent"):
        run.evaluate(forbidden, [task()], {"q1": {"p1": 1}}, mode="lexical", known_ids={"p2"})


def test_percentiles_are_defined_and_reported_with_actual_samples():
    assert run.percentile([], .95) is None
    assert run.percentile([10], .95) == 10
    assert run.percentile([0, 10, 20], .50) == 10
    assert run.percentile([0, 10, 20], .95) == 19


def test_snapshot_files_cannot_overwrite_prior_report(tmp_path):
    path = tmp_path / "report.json"
    run.write_json(path, {"old": True})
    with pytest.raises(FileExistsError):
        run.write_json(path, {"replacement": True})
