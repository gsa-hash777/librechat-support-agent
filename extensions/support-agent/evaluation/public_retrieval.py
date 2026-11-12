"""Objective full-corpus retrieval evaluation, separate from answer correctness.

Only answerable/partial tasks with official positive qrels are scored. Missing
judgments are not invented for unanswerable/underspecified tasks. Every planned
retrieval task remains in the denominator, including execution failures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Callable

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
from evaluation import public_data as data

MODES = ("lexical", "hybrid", "rerank")
QUERY_POLICIES = ("lastturn", "previous_user3")
SCHEMA_VERSION = 1
TOP_K = 5


def build_query(task: dict, policy: str = "lastturn") -> str:
    """History arm adds up to three preceding USER turns; no reference responses."""
    if set(task) != data.INPUT_KEYS:
        raise ValueError("Retriever input must be the allowlisted task projection.")
    if policy not in QUERY_POLICIES:
        raise ValueError("Unknown query policy.")
    messages = task["messages"]
    if not messages or messages[-1].get("speaker") != "user":
        raise ValueError("Task must end in a user message.")
    for message in messages:
        if set(message) != {"speaker", "text"} or message["speaker"] not in {"user", "assistant"}:
            raise ValueError("Only sanitized conversation messages are allowed.")
        if not isinstance(message["text"], str) or not message["text"].strip():
            raise ValueError("Empty conversation message.")
    if policy == "lastturn":
        return messages[-1]["text"]
    users = [message["text"] for message in messages if message["speaker"] == "user"]
    return "\n".join(users[-4:])


def score_ranking(ranking: list[str], judgments: dict[str, int], k: int = TOP_K) -> dict:
    """Macro components; nDCG uses linear relevance gain (official qrels are binary)."""
    if k < 1 or len(ranking) != len(set(ranking)):
        raise ValueError("Ranking must contain unique passage IDs and k must be positive.")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in judgments.values()):
        raise ValueError("Relevance grades must be nonnegative integers.")
    positive = {identifier for identifier, grade in judgments.items() if grade > 0}
    if not positive:
        raise ValueError("Retrieval metrics require at least one positive qrel.")
    selected = ranking[:k]
    found = [rank for rank, identifier in enumerate(selected, 1) if identifier in positive]
    dcg = sum(judgments.get(identifier, 0) / math.log2(rank + 1) for rank, identifier in enumerate(selected, 1))
    ideal = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(sorted(judgments.values(), reverse=True)[:k], 1))
    return {"recall_at_5": len(found) / len(positive), "ndcg_at_5": dcg / ideal,
            "reciprocal_rank_at_5": 1 / found[0] if found else 0.0,
            "hit_at_5": float(bool(found)), "relevant_total": len(positive), "relevant_retrieved": len(found)}


def result_ids(response, known_ids: set[str] | None = None,
               passage_sources: dict | None = None) -> tuple[list[str], dict]:
    """Accept the public search response, requiring canonical full-corpus IDs."""
    if isinstance(response, dict):
        results = response.get("results")
        metadata = response.get("metadata", response.get("retrieval_metadata", {}))
    else:
        results, metadata = response, {}
    if not isinstance(results, list) or not isinstance(metadata, dict):
        raise ValueError("Search must return a list or a results/metadata object.")
    identifiers = []
    for row in results:
        if not isinstance(row, dict):
            raise ValueError("A search result must be an object.")
        identifier = row.get("id", row.get("passage_id", row.get("document_id")))
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("A search result needs its original passage ID.")
        if known_ids is not None and identifier not in known_ids:
            raise ValueError("Search returned a passage outside the frozen corpus.")
        if passage_sources is not None:
            original = passage_sources.get(identifier)
            text = row.get("text")
            if (original is None or not isinstance(text, str)
                    or hashlib.sha256(text.encode("utf8")).hexdigest() != original["sha256"]
                    or row.get("sha256") != original["sha256"]
                    or row.get("source_url") != original["source_url"]):
                raise ValueError("Returned passage text/source differs from the pinned original.")
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Search returned duplicate passage IDs.")
    return identifiers, metadata


def evaluate(search: Callable, tasks: list[dict], qrels: dict, *, mode: str,
             query_policy: str = "lastturn", known_ids: set[str] | None = None,
             labels: dict | None = None, on_row: Callable | None = None,
             passage_sources: dict | None = None) -> list[dict]:
    """The callable receives query/mode/limit only, never the task or gold labels."""
    if mode not in MODES or query_policy not in QUERY_POLICIES:
        raise ValueError("Unknown retrieval comparison arm.")
    identifiers = [task["task_id"] for task in tasks]
    if len(identifiers) != len(set(identifiers)) or set(qrels) - set(identifiers):
        raise ValueError("Tasks must be unique and include every planned qrel query.")
    for judgments in qrels.values():
        score_ranking([], judgments)
        if known_ids is not None and set(judgments) - known_ids:
            raise ValueError("Gold passage absent from the shared corpus.")
    rows = []
    for task in tasks:
        identifier = task["task_id"]
        if identifier not in qrels:
            continue
        query = build_query(task, query_policy)
        row = {"task_id": identifier, "conversation_id": task["conversation_id"], "turn": task["turn"],
               "query_policy": query_policy, "mode": mode, "query": query,
               "query_sha256": hashlib.sha256(query.encode("utf8")).hexdigest(), "state": "completed"}
        started = time.perf_counter()
        try:
            response = search(query, mode=mode, limit=TOP_K)
            ranking, metadata = result_ids(response, known_ids, passage_sources)
            if len(ranking) > TOP_K:
                raise ValueError("Search ignored the requested top-k bound.")
            row.update({"ranking": ranking, "retrieval_metadata": metadata,
                        "metrics": score_ranking(ranking, qrels[identifier])})
        except Exception as exc:
            # Type only: provider errors may echo a query or credentials.
            row.update({"state": "execution_error", "error_type": type(exc).__name__, "ranking": [],
                        "metrics": score_ranking([], qrels[identifier])})
        row["latency_ms"] = (time.perf_counter() - started) * 1000
        if labels is not None:
            row["annotation"] = labels.get(identifier, {})
        rows.append(row)
        if on_row is not None:
            on_row(row)
    return rows


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(rows: list[dict], planned: int) -> dict:
    if planned < len(rows) or len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("Planned denominator must contain every unique observed task.")
    completed = [row for row in rows if row["state"] == "completed"]
    latency = [row["latency_ms"] for row in rows]
    return {"planned": planned, "attempted": len(rows), "completed": len(completed),
            "execution_errors": len(rows) - len(completed), "not_attempted": planned - len(rows),
            "macro_recall_at_5": sum(row["metrics"]["recall_at_5"] for row in rows) / planned if planned else None,
            "ndcg_at_5": sum(row["metrics"]["ndcg_at_5"] for row in rows) / planned if planned else None,
            "mrr_at_5": sum(row["metrics"]["reciprocal_rank_at_5"] for row in rows) / planned if planned else None,
            "hit_at_5": sum(row["metrics"]["hit_at_5"] for row in rows) / planned if planned else None,
            "latency_ms": {"samples": len(latency), "p50": percentile(latency, .50), "p95": percentile(latency, .95),
                           "max": max(latency) if latency else None,
                           "scope": "all attempted retrieval calls, including errors; excludes index construction"},
            "failure_denominator_policy": "execution errors and unattempted planned queries score zero",
            "metric_scope": "retrieval relevance only; not answer accuracy, refusal quality or task success"}


def source_fingerprint() -> dict:
    paths = [Path(__file__), SERVICE / "evaluation/public_data.py"]
    paths.extend(sorted((SERVICE / "app").glob("*.py")))
    return {str(path.relative_to(SERVICE)).replace("\\", "/"): data.sha256(path) for path in paths}


def write_json(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run(search: Callable, *, root: Path, split: str, mode: str, query_policy: str,
        output: Path, allow_final: bool = False, retriever_metadata: dict | None = None) -> dict:
    """One immutable arm directory. Root schedules model lifetimes and final release."""
    tasks = data.load_cloud_tasks(root, split, allow_final=allow_final)
    labels = data.load_labels(root, split, allow_final=allow_final)
    qrels = data.load_qrels(root, split, allow_final=allow_final)
    passage_sources = {row["passage_id"]: {"sha256": row["sha256"], "source_url": row["source_url"]}
                       for row in data.iter_corpus(root)}
    known_ids = set(passage_sources)
    inputs = data.fingerprint(root)
    sources = source_fingerprint()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    freeze = {"schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
              "split": split, "mode": mode, "query_policy": query_policy, "top_k": TOP_K,
              "corpus_passages": len(known_ids), "input_fingerprint": inputs, "source_fingerprint": sources,
              "retriever_metadata": retriever_metadata or {}, "python": platform.python_version(),
              "platform": platform.platform(), "model_calls_paid": 0, "planned": len(qrels),
              "all_cloud_tasks": len(tasks), "non_retrieval_tasks": len(tasks) - len(qrels),
              "query_history_scope": "current user plus up to three previous user messages; never assistant references",
              "returned_source_validation": "every returned passage ID, full text SHA256 and source URL must match pinned corpus",
              "test_scope": "public external benchmark, not proof of absence from model pretraining"}
    write_json(output / "freeze.json", freeze)
    with (output / "observations.jsonl").open("x", encoding="utf8") as stream:
        def observe(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
        rows = evaluate(search, tasks, qrels, mode=mode, query_policy=query_policy,
                        known_ids=known_ids, labels=labels, on_row=observe, passage_sources=passage_sources)
    unchanged = source_fingerprint() == sources and data.fingerprint(root) == inputs
    report = {**freeze, "summary": summarize(rows, len(qrels)), "fingerprints_unchanged": unchanged,
              "by_answerability": {}, "non_retrieval_labels": dict(Counter(
                  labels[task["task_id"]]["answerability"] for task in tasks if task["task_id"] not in qrels))}
    for category in ("ANSWERABLE", "PARTIAL"):
        selected = [row for row in rows if row["annotation"]["answerability"] == category]
        planned = sum(labels[key]["answerability"] == category for key in qrels)
        report["by_answerability"][category] = summarize(selected, planned)
    write_json(output / "report.json", report)
    if not unchanged:
        raise RuntimeError("Source or input fingerprint changed during evaluation; report is invalid.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=data.DEFAULT_ROOT)
    parser.add_argument("--split", choices=("dev", "final"), default="dev")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--query-policy", choices=QUERY_POLICIES, default="lastturn")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-final", action="store_true")
    args = parser.parse_args()
    # Application adapter is loaded only for an explicit experiment invocation.
    from app.public_search import PublicSearch
    engine = PublicSearch(root=args.root)
    try:
        report = run(engine.search, root=args.root, split=args.split, mode=args.mode,
                     query_policy=args.query_policy, output=args.output, allow_final=args.allow_final,
                     retriever_metadata=engine.metadata() if hasattr(engine, "metadata") else {"metadata_unavailable": True})
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    finally:
        if hasattr(engine, "close"):
            engine.close()


if __name__ == "__main__":
    main()
