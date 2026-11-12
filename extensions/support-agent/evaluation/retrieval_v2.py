"""Isolated, development-only retrieval experiments; production defaults stay unchanged.

Candidate creation never reads qrels or reference answers. Analysis reads development
qrels only after candidates are sealed. Historical final inputs are not accepted by
this CLI. A separate, explicitly labelled seen-set regression may import the pure
query/fusion functions after a development configuration is frozen.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
from evaluation import public_data as data
from evaluation.public_retrieval import score_ranking

POLICIES = ("lastturn", "previous_last", "first_last", "previous_user3", "conditional")
ANAPHORA = re.compile(r"\b(it|its|this|that|these|those|they|them|their|same|above|former|latter)\b|^(and|also|what about|how about)\b", re.I)


def task_queries(task: dict) -> dict[str, str]:
    if set(task) != data.INPUT_KEYS:
        raise ValueError("Only the sanitized task projection is accepted.")
    users = []
    for message in task["messages"]:
        if set(message) != {"speaker", "text"} or message["speaker"] not in {"user", "assistant"}:
            raise ValueError("Invalid conversation input.")
        if message["speaker"] == "user":
            users.append(message["text"])
    if not users or task["messages"][-1]["speaker"] != "user":
        raise ValueError("The current message must be a user question.")
    last = users[-1]
    previous = "\n".join(users[-2:])
    first = "\n".join([users[0], last]) if len(users) > 1 else last
    history = "\n".join(users[-4:])
    # A generic, predeclared rule, not a list of benchmark product names.
    conditional = previous if len(users) > 1 and (ANAPHORA.search(last) or len(last.split()) <= 6) else last
    return {"lastturn": last, "previous_last": previous, "first_last": first,
            "previous_user3": history, "conditional": conditional}


@dataclass(frozen=True)
class Variant:
    name: str
    policy: str = "lastturn"
    candidates: int = 100
    rrf_k: int = 60
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    current_weight: float = 0.0


VARIANTS = (
    Variant("baseline_hybrid20", candidates=20),
    Variant("wide_hybrid100"),
    Variant("wide_dense100", lexical_weight=0.0),
    Variant("wide_dense_weighted", lexical_weight=0.25),
    Variant("wide_rrf20", rrf_k=20),
    Variant("previous_last", policy="previous_last"),
    Variant("first_last", policy="first_last"),
    Variant("previous_user3", policy="previous_user3"),
    Variant("conditional", policy="conditional"),
    Variant("blend_previous", policy="previous_last", current_weight=1.0),
    Variant("blend_first", policy="first_last", current_weight=1.0),
    Variant("blend_conditional", policy="conditional", current_weight=1.0),
)


def fuse(record: dict, variant: Variant) -> list[dict]:
    if variant.policy not in POLICIES or not 1 <= variant.candidates <= 100 or variant.rrf_k < 1:
        raise ValueError("Invalid frozen retrieval variant.")
    policies = [(variant.policy, 1.0)]
    if variant.current_weight and record["queries"][variant.policy] != record["queries"]["lastturn"]:
        policies.append(("lastturn", variant.current_weight))
    values = {}
    for policy, policy_weight in policies:
        for arm, weight in (("lexical", variant.lexical_weight), ("dense", variant.dense_weight)):
            if weight <= 0:
                continue
            rows = record["candidates"][policy][arm]
            if len({row["id"] for row in rows}) != len(rows):
                raise ValueError("Duplicate candidate passage ID.")
            for rank, row in enumerate(rows[:variant.candidates], 1):
                values[row["id"]] = values.get(row["id"], 0.0) + policy_weight * weight / (variant.rrf_k + rank)
    return [{"id": key, "score": value} for key, value in sorted(values.items(), key=lambda item: (-item[1], item[0]))]


def partition(conversation_id: str) -> str:
    # Keep all turns in the same split. The partition is fixed before scores exist.
    return "validation" if int(hashlib.sha256(conversation_id.encode()).hexdigest(), 16) % 5 == 0 else "train"


def summary(rows: list[dict], qrels: dict, variant: Variant, pool: int = 5) -> dict:
    metrics = []
    for row in rows:
        if row["task_id"] not in qrels:
            continue
        ranking = [item["id"] for item in fuse(row, variant)[:pool]]
        metrics.append(score_ranking(ranking, qrels[row["task_id"]], k=pool))
    n = len(metrics)
    return {"n": n, "macro_recall": sum(m["recall_at_5"] for m in metrics) / n if n else None,
            "hit": sum(m["hit_at_5"] for m in metrics) / n if n else None,
            "mrr": sum(m["reciprocal_rank_at_5"] for m in metrics) / n if n else None,
            "pool": pool, "top5_claim_allowed": pool == 5}


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def checked_candidates(root: Path, source: Path, models: Path | None = None) -> list[dict]:
    """Reject incomplete/mismatched caches instead of silently rerunning candidates."""
    completion = json.loads((source / "completion.json").read_text(encoding="utf-8"))
    freeze = json.loads((source / "freeze.json").read_text(encoding="utf-8"))
    if (not completion.get("source_unchanged")
            or data.sha256(source / "candidates.jsonl") != completion["candidates_sha256"]
            or freeze["input_file"] != data.verify_file(root, data.TASK_FILES["dev"])):
        raise ValueError("Candidate cache provenance mismatch.")
    rows = list(data.read_jsonl(source / "candidates.jsonl"))
    expected = {t["task_id"] for t in data.load_cloud_tasks(root, "dev")}
    if len(rows) != completion["completed"] or len(rows) != len(expected) or {r["task_id"] for r in rows} != expected:
        raise ValueError("Candidate cache is incomplete or has duplicate task IDs.")
    metadata = completion["metadata"]
    index_manifest = json.loads((root / "index/manifest.json").read_text(encoding="utf-8"))
    vector_manifest = json.loads((root / "index/vectors.json").read_text(encoding="utf-8"))
    if metadata["index"] != index_manifest or metadata["vectors"] != vector_manifest:
        raise ValueError("Corpus/vector index differs from candidate provenance.")
    if models is not None and data.sha256(models / "manifest.json") != metadata["model_manifest_sha256"]:
        raise ValueError("Model fingerprint differs from candidate provenance.")
    return rows


@contextmanager
def cpu_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat()}, stream)
    try:
        yield
    finally:
        path.unlink()


def candidates(root: Path, models: Path, output: Path, lock: Path) -> None:
    from app.public_search import PublicSearch
    tasks = data.load_cloud_tasks(root, "dev")
    output.mkdir(parents=True, exist_ok=False)
    freeze = {"split": "dev", "all_cloud_tasks": len(tasks), "candidate_limit_per_arm": 100,
              "policies": list(POLICIES), "variants": [asdict(v) for v in VARIANTS],
              "source_sha256": data.sha256(Path(__file__)), "paid_model_calls": 0,
              "partition": "sha256(conversation_id) modulo 5 equals 0: validation",
              "input_file": data.verify_file(root, data.TASK_FILES["dev"]),
              "created_at": datetime.now(timezone.utc).isoformat(),
              "source_data_untouched": True, "production_defaults_untouched": True}
    write_new(output / "freeze.json", freeze)
    cache = {}
    began = time.monotonic()
    with cpu_lock(lock):
        engine = PublicSearch(root=root, model_dir=models, threads=2)
        try:
            with (output / "candidates.jsonl").open("x", encoding="utf-8") as stream:
                for i, task in enumerate(tasks):
                    queries = task_queries(task)
                    row = {"task_id": task["task_id"], "conversation_id": task["conversation_id"],
                           "partition": partition(task["conversation_id"]), "queries": queries, "candidates": {}}
                    for policy, query in queries.items():
                        if query not in cache:
                            start = time.monotonic()
                            lexical = engine._lexical(query, 100)
                            dense = engine._dense(query, 100)
                            cache[query] = {"lexical": [{"id": r["passage_id"], "score": r["score"]} for r in lexical],
                                            "dense": [{"id": r["passage_id"], "score": r["score"]} for r in dense],
                                            "seconds": time.monotonic() - start}
                        row["candidates"][policy] = cache[query]
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                        print(json.dumps({"completed": i + 1, "planned": len(tasks), "unique_queries": len(cache),
                                          "seconds": round(time.monotonic() - began, 2)}), flush=True)
            write_new(output / "completion.json", {"completed": len(tasks), "unique_queries": len(cache),
                        "seconds": time.monotonic() - began, "candidates_sha256": data.sha256(output / "candidates.jsonl"),
                        "source_unchanged": data.sha256(Path(__file__)) == freeze["source_sha256"],
                        "metadata": engine.metadata()})
        finally:
            engine.close()


def analyze(root: Path, source: Path, output: Path) -> None:
    rows = checked_candidates(root, source)
    completion = json.loads((source / "completion.json").read_text(encoding="utf-8"))
    qrels = data.load_qrels(root, "dev")
    if set(qrels) - {r["task_id"] for r in rows}:
        raise ValueError("Incomplete development candidates.")
    report = {"scope": "development comparison; not new blind-test accuracy", "candidate_sha256": completion["candidates_sha256"],
              "variants": [], "source_sha256": data.sha256(Path(__file__))}
    for variant in VARIANTS:
        row = {"variant": asdict(variant)}
        for part in ("all", "train", "validation"):
            subset = rows if part == "all" else [r for r in rows if r["partition"] == part]
            row[part] = summary(subset, qrels, variant)
        row["candidate_recall_at_10"] = summary(rows, qrels, variant, 10)
        row["candidate_recall_at_30"] = summary(rows, qrels, variant, 30)
        row["candidate_recall_at_50"] = summary(rows, qrels, variant, 50)
        report["variants"].append(row)
    write_new(output, report)
    for row in report["variants"]:
        print(json.dumps({"name": row["variant"]["name"], "all": row["all"], "train": row["train"],
                          "validation": row["validation"], "pool50": row["candidate_recall_at_50"]["macro_recall"]}), flush=True)


def resume_rows(output: Path, expected: dict, rows: list[dict], qrels: dict) -> tuple[dict, list[dict]]:
    """Resume only a matching frozen prefix; errors are retained, never retried."""
    frozen = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    for key, value in expected.items():
        if key != "created_at" and frozen.get(key) != value:
            raise ValueError(f"Resume fingerprint mismatch: {key}")
    if data.sha256(output / "retrieval_v2.source.py") != frozen["source_sha256"]:
        raise ValueError("Archived source fingerprint mismatch.")
    if (output / "report.json").exists():
        raise ValueError("The run is already complete.")
    results = list(data.read_jsonl(output / "observations.jsonl"))
    if len(results) > len(rows):
        raise ValueError("Resume contains unexpected extra tasks.")
    for result, row in zip(results, rows):
        ids = [item["id"] for item in fuse(row, Variant(**frozen["variant"]))[:frozen["rerank_pool"]]]
        if (result["task_id"] != row["task_id"] or result["partition"] != row["partition"]
                or result["candidate_ids"] != ids or result["state"] not in {"completed", "execution_error"}
                or result["metrics"] != score_ranking(result["ranking"], qrels[row["task_id"]])
                or any(identifier not in ids for identifier in result["ranking"])
                or (result["state"] == "execution_error" and result["ranking"])):
            raise ValueError("Resume observations differ from frozen inputs.")
    return frozen, results


def rerank(root: Path, models: Path, source: Path, output: Path, lock: Path, variant_name: str, pool: int, limit: int | None, resume: bool = False) -> None:
    from app.public_search import PublicSearch
    variant = next(v for v in VARIANTS if v.name == variant_name)
    rows = checked_candidates(root, source, models)
    qrels = data.load_qrels(root, "dev")
    rows = [r for r in rows if r["task_id"] in qrels]
    if limit:
        # Deterministic selection solely for timing/prototyping, never full-score claim.
        rows = sorted(rows, key=lambda r: hashlib.sha256(r["task_id"].encode()).hexdigest())[:limit]
    if not 1 <= pool <= 100:
        raise ValueError("Invalid rerank pool.")
    freeze = {"variant": asdict(variant), "rerank_pool": pool, "planned": len(rows), "split": "dev",
              "task_ids": [r["task_id"] for r in rows], "full_development": limit is None,
              "candidate_sha256": data.sha256(source / "candidates.jsonl"), "source_sha256": data.sha256(Path(__file__)),
              "runtime_source_sha256": {name: data.sha256(SERVICE / name) for name in (
                  "app/public_search.py", "app/hybrid_models.py", "evaluation/public_data.py", "evaluation/public_retrieval.py")},
              "runtime": {"threads": 2, "batch_size": 1, "max_length": 512, "provider": "CPUExecutionProvider"},
              "created_at": datetime.now(timezone.utc).isoformat(), "paid_model_calls": 0}
    if resume:
        freeze, results = resume_rows(output, freeze, rows, qrels)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_new(output / "freeze.json", freeze)
        with (output / "retrieval_v2.source.py").open("xb") as stream:
            stream.write(Path(__file__).read_bytes())
        results = []
    resumed_count = len(results)
    began = time.monotonic()
    with cpu_lock(lock):
        engine = PublicSearch(root=root, model_dir=models, threads=2)
        try:
            with (output / "observations.jsonl").open("a" if resume else "x", encoding="utf-8") as stream:
                for i, row in enumerate(rows[resumed_count:], resumed_count):
                    start = time.monotonic()
                    ids = [r["id"] for r in fuse(row, variant)[:pool]]
                    result = {"task_id": row["task_id"], "partition": row["partition"], "candidate_ids": ids,
                              "state": "completed"}
                    try:
                        passages = [engine.get(identifier) for identifier in ids]
                        if any(p is None for p in passages):
                            raise ValueError("Candidate outside the full source corpus.")
                        query = row["queries"][variant.policy]
                        scores = engine._backend().rerank(query, [p["text"] for p in passages])
                        ranking = [key for key, score in sorted(zip(ids, scores), key=lambda item: (-item[1], item[0]))]
                        result.update({"ranking": ranking[:5], "candidate_logits": scores})
                    except Exception as exc:
                        result.update({"ranking": [], "candidate_logits": [], "state": "execution_error",
                                       "error_type": type(exc).__name__})
                    result["metrics"] = score_ranking(result["ranking"], qrels[row["task_id"]])
                    result["rerank_seconds"] = time.monotonic() - start
                    results.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    if (i + 1) % 5 == 0 or i + 1 == len(rows):
                        print(json.dumps({"completed": i + 1, "planned": len(rows), "seconds": round(time.monotonic() - began, 2)}), flush=True)
            metrics = {}
            for part in ("all", "train", "validation"):
                selected = results if part == "all" else [r for r in results if r["partition"] == part]
                n = len(selected)
                metrics[part] = {"n": n, "macro_recall_at_5": sum(r["metrics"]["recall_at_5"] for r in selected) / n if n else None,
                                 "hit_at_5": sum(r["metrics"]["hit_at_5"] for r in selected) / n if n else None}
            write_new(output / "report.json", {**freeze, "completed": len(results), "metrics": metrics,
                       "execution_errors": sum(r["state"] == "execution_error" for r in results),
                       "seconds_this_session": time.monotonic() - began, "resumed_count": resumed_count,
                       "total_observed_rerank_seconds": sum(r["rerank_seconds"] for r in results),
                       "observations_sha256": data.sha256(output / "observations.jsonl"),
                       "source_unchanged": data.sha256(Path(__file__)) == freeze["source_sha256"],
                       "latency_scope": "rerank stage over cached candidates only; not full query latency"})
            print(json.dumps(metrics), flush=True)
        finally:
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("candidates", "analyze", "rerank"))
    parser.add_argument("--root", type=Path, default=data.DEFAULT_ROOT)
    parser.add_argument("--models", type=Path, default=SERVICE.parents[1] / ".local/models/public-retrieval")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--lock", type=Path, default=SERVICE.parents[1] / ".local/retrieval-v2/cpu.lock")
    parser.add_argument("--variant", choices=[v.name for v in VARIANTS], default="wide_hybrid100")
    parser.add_argument("--pool", type=int, default=30)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "candidates":
        candidates(args.root, args.models, args.output, args.lock)
    elif args.stage == "analyze":
        analyze(args.root, args.source, args.output)
    else:
        rerank(args.root, args.models, args.source, args.output, args.lock, args.variant, args.pool, args.limit, args.resume)


if __name__ == "__main__":
    main()
