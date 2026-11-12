"""One frozen CPU regression on the historical 82-query Cloud cohort.

Development selection must already be complete. This runner never selects a
configuration from final-set scores. Retrieval sees sanitized tasks only; qrels
enter a separate scorer after retrieval. No network or paid model is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
from evaluation import public_data as data
from evaluation.public_retrieval import score_ranking, summarize
from evaluation.retrieval_v2 import Variant, cpu_lock, fuse, task_queries

WORK = SERVICE.parents[1] / ".local/retrieval-v2"
DEFAULT_BASELINE = WORK / "baseline.json"
DEFAULT_LOCK = WORK / "cpu.lock"
BOOTSTRAP_SEED = 20261112
BOOTSTRAP_SAMPLES = 10000
VERSION = "seen-cloud-retrieval-v2-regression-1"
SOURCE_FILES = ("evaluation/retrieval_v2_regression.py", "evaluation/retrieval_v2.py",
                "evaluation/public_data.py", "evaluation/public_retrieval.py",
                "app/public_search.py", "app/hybrid_models.py")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_fingerprints():
    return {name: data.sha256(SERVICE / name) for name in SOURCE_FILES}


def validate_selected(value):
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("selection_scope") != "development_only"
            or not isinstance(value.get("source_report"), dict)):
        raise ValueError("A frozen development-only selection and source report are required.")
    chosen = value.get("selected", {})
    if set(chosen) != {"variant", "rerank_pool", "second_rrf"}:
        raise ValueError("Selection must declare variant, rerank_pool and second_rrf.")
    variant = Variant(**chosen["variant"])
    if (asdict(variant) != chosen["variant"] or type(variant.candidates) is not int
            or variant.candidates not in (20, 100)):
        raise ValueError("This experiment requires explicitly frozen top20 or top100 per arm.")
    if (variant.policy not in ("lastturn", "previous_last", "first_last", "previous_user3", "conditional")
            or type(variant.rrf_k) is not int or variant.rrf_k < 1
            or any(type(weight) not in (float, int) or not math.isfinite(weight) or weight < 0
                   for weight in (variant.lexical_weight, variant.dense_weight, variant.current_weight))
            or variant.lexical_weight + variant.dense_weight <= 0):
        raise ValueError("Invalid retrieval variant.")
    if type(chosen["rerank_pool"]) is not int or chosen["rerank_pool"] not in (10, 20, 30, 50):
        raise ValueError("Registered development rerank pools are 10, 20, 30 and 50.")
    if variant.candidates == 20 and chosen["rerank_pool"] != 10:
        raise ValueError("The added top20 development arm preserves the original pool10.")
    second = chosen["second_rrf"]
    if second is not None and (not isinstance(second, dict) or set(second) != {"k", "rerank_weight"}
            or type(second["k"]) is not int or second["k"] not in (10, 60)
            or type(second["rerank_weight"]) not in (float, int) or second["rerank_weight"] not in (.25, .5, .75)):
        raise ValueError("Second-stage RRF must use the predeclared k and weight grid.")
    return chosen


def load_selection(path):
    value = read_json(path)
    chosen = validate_selected(value)
    report_path = Path(value["source_report"]["path"])
    if not report_path.is_absolute():
        report_path = Path(path).parent / report_path
    if data.sha256(report_path) != value["source_report"]["sha256"]:
        raise ValueError("Development selection report hash changed.")
    report = read_json(report_path)
    if report.get("split") != "dev" or report.get("full_development") is not True or report.get("planned") != 188:
        raise ValueError("Selection must cite a complete 188-query development result, not a timing subset.")
    if report.get("completed") != 188 or report.get("source_unchanged") is not True:
        raise ValueError("Development evidence is incomplete or source-drifted.")
    return value, chosen, report_path


def second_fusion(candidate_ids, rerank_scores, config):
    """Rank only the same candidate pool, never read labels or expand final k."""
    if len(candidate_ids) != len(rerank_scores) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate IDs/scores must be complete and unique.")
    if any(not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) for score in rerank_scores):
        raise ValueError("Nonfinite rerank score.")
    reranked = [key for key, _ in sorted(zip(candidate_ids, rerank_scores), key=lambda item: (-item[1], item[0]))]
    if config is None:
        return reranked
    ranks = {key: rank for rank, key in enumerate(reranked, 1)}
    weight, k = config["rerank_weight"], config["k"]
    scores = {key: weight / (k + ranks[key]) + (1 - weight) / (k + rank)
              for rank, key in enumerate(candidate_ids, 1)}
    return sorted(candidate_ids, key=lambda key: (-scores[key], key))


def _retrieve_on_worker(engine, task, chosen):
    """Complete model lifecycle operation; called on PublicSearch's one worker."""
    queries = task_queries(task)
    variant = Variant(**chosen["variant"])
    policies = [variant.policy]
    if variant.current_weight and queries[variant.policy] != queries["lastturn"]:
        policies.append("lastturn")
    candidates, cache = {}, {}
    for policy in policies:
        query = queries[policy]
        if not isinstance(query, str) or not query.strip() or len(query) > 12000:
            raise ValueError("Query is outside the service text bounds.")
        # No cross-task candidate cache: each task measures real recall work.
        if query not in cache:
            cache[query] = {"lexical": [{"id": row["passage_id"], "score": row["score"]} for row in engine._lexical(query, variant.candidates)],
                            "dense": [{"id": row["passage_id"], "score": row["score"]} for row in engine._dense(query, variant.candidates)]}
        candidates[policy] = cache[query]
    fused = fuse({"queries": queries, "candidates": candidates}, variant)
    ids = [row["id"] for row in fused[:chosen["rerank_pool"]]]
    if not ids:
        raise ValueError("No candidate passages.")
    passages = [engine.get(key) for key in ids]
    if any(passage is None for passage in passages):
        raise ValueError("Candidate missing from full corpus.")
    scores = engine._backend().rerank(queries[variant.policy], [row["text"] for row in passages])
    ranking = second_fusion(ids, scores, chosen["second_rrf"])
    lookup = {row["passage_id"]: row for row in passages}
    return {"ranking": ranking[:5], "candidate_ids": ids, "candidate_logits": scores,
            "candidate_retrieval": candidates, "query_policies": {key: queries[key] for key in policies},
            "returned_passages": [lookup[key] for key in ranking[:5]]}


def retrieve(engine, task, chosen):
    """No qrels/reference fields accepted; inference and release share a thread."""
    task_queries(task)  # Reject extra fields before queueing any inference.
    with engine._lock:
        if engine.closed:
            raise RuntimeError("Retrieval engine closed.")
        future = engine._executor.submit(_retrieve_on_worker, engine, task, chosen)
    return future.result()


def load_cohort(root, baseline_path):
    baseline = read_json(baseline_path)
    identifiers = baseline["overlap_audit"]["clean_retrieval_task_ids"]
    if len(identifiers) != 82 or len(set(identifiers)) != 82:
        raise ValueError("The historical cohort must contain exactly 82 unique tasks.")
    tasks = data.load_cloud_tasks(root, "final", allow_final=True)
    qrels = data.load_qrels(root, "final", allow_final=True)
    dev = data.load_cloud_tasks(root, "dev")
    # Reproduce the original exact-JSON-history overlap rule, not semantic dedup.
    history_sha = lambda task: hashlib.sha256(json.dumps(task["messages"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    dev_histories = {history_sha(task) for task in dev}
    expected = {task["task_id"] for task in tasks if task["task_id"] in qrels and history_sha(task) not in dev_histories}
    if set(identifiers) != expected:
        raise ValueError("Historical clean IDs differ from recomputed exact-overlap exclusion.")
    tasks = sorted((task for task in tasks if task["task_id"] in expected), key=lambda task: task["task_id"])
    qrels = {key: qrels[key] for key in sorted(expected)}
    arm = next(arm for arm in baseline["clean"]["retrieval"] if arm["mode"] == "rerank")
    report_path = Path(arm["artifact"]["path"])
    if data.sha256(report_path) != arm["artifact"]["sha256"]:
        raise ValueError("Historical rerank report was modified.")
    observations_path = report_path.with_name("observations.jsonl")
    old = [row for row in data.read_jsonl(observations_path) if row["task_id"] in expected]
    if len(old) != 82 or {row["task_id"] for row in old} != expected:
        raise ValueError("Historical rerank observations are incomplete or duplicated.")
    old = {row["task_id"]: {**row, "metrics": score_ranking(row["ranking"], qrels[row["task_id"]])} for row in old}
    recalculated = sum(row["metrics"]["recall_at_5"] for row in old.values()) / 82
    if not math.isclose(recalculated, arm["summary"]["macro_recall_at_5"], abs_tol=1e-12):
        raise ValueError("Historical rerank metric does not reproduce from official qrels.")
    provenance = {str(Path(baseline_path).resolve()): data.sha256(baseline_path),
                  str(report_path.resolve()): data.sha256(report_path), str(observations_path.resolve()): data.sha256(observations_path)}
    return tasks, qrels, old, provenance


def artifact_fingerprints(root, models):
    files = {"dataset:" + name: data.verify_file(root, name)["sha256"] for name in
             (data.CORPUS_FILE, data.TASK_FILES["dev"], data.TASK_FILES["final"], data.QREL_FILES["final"])}
    for name in ("manifest.json", "passages.sqlite", "vectors.json", "vectors.npy"):
        files["index:" + name] = data.sha256(root / "index" / name)
    model_manifest = read_json(models / "manifest.json")
    files["model:manifest.json"] = data.sha256(models / "manifest.json")
    for role, entry in model_manifest["models"].items():
        for name, item in entry["files"].items():
            path = models / role / name
            actual = data.sha256(path)
            if actual != item["sha256"]:
                raise ValueError("Pinned ONNX model asset changed.")
            files["model:" + role + "/" + name] = actual
    return files


def validate_passages(response, sources):
    ranking = response["ranking"]
    if len(ranking) > 5 or len(ranking) != len(set(ranking)):
        raise ValueError("Final ranking must contain at most five distinct passages.")
    rows = response["returned_passages"]
    if [row["passage_id"] for row in rows] != ranking:
        raise ValueError("Returned passage order differs from ranking.")
    for row in rows:
        truth = sources.get(row["passage_id"])
        if (truth is None or row["sha256"] != truth["sha256"] or row["source_url"] != truth["source_url"]
                or hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != truth["sha256"]):
            raise ValueError("Passage text/source does not match pinned corpus.")


def paired_comparison(rows, old, identifiers, qrels, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, conversation_ids=None):
    current = {row["task_id"]: row for row in rows}
    changes = []
    for key in identifiers:
        new = current.get(key, {"ranking": [], "state": "not_attempted"})
        new_metrics = score_ranking(new["ranking"], qrels[key])
        prior = score_ranking(old[key]["ranking"], qrels[key])
        changes.append({"task_id": key, "before_recall": prior["recall_at_5"], "after_recall": new_metrics["recall_at_5"],
                        "recall_delta": new_metrics["recall_at_5"] - prior["recall_at_5"],
                        "before_hit": bool(prior["hit_at_5"]), "after_hit": bool(new_metrics["hit_at_5"]),
                        "state": new["state"], "before_ranking": old[key]["ranking"], "after_ranking": new["ranking"]})
    deltas = [row["recall_delta"] for row in changes]
    generator = random.Random(seed)
    if conversation_ids is not None:
        if set(conversation_ids) != set(identifiers) or any(not isinstance(value, str) or not value for value in conversation_ids.values()):
            raise ValueError("Cluster bootstrap requires every task's conversation ID.")
        groups = {}
        for row in changes:
            groups.setdefault(conversation_ids[row["task_id"]], []).append(row["recall_delta"])
        clusters = [(sum(values), len(values)) for _, values in sorted(groups.items())]
        estimates = []
        for _ in range(samples):
            sampled = [generator.choice(clusters) for _ in clusters]
            estimates.append(sum(value for value, _ in sampled) / sum(size for _, size in sampled))
        estimates.sort()
        bootstrap = {"unit": "paired conversation cluster", "clusters": len(clusters),
                     "method": "percentile cluster bootstrap; paired query deltas, all turns sampled together; exploratory on seen set"}
    else:
        estimates = sorted(sum(generator.choice(deltas) for _ in deltas) / len(deltas) for _ in range(samples))
        bootstrap = {"unit": "paired task", "method": "percentile bootstrap; exploratory on seen set"}
    lo, hi = estimates[max(0, math.ceil(samples * .025) - 1)], estimates[min(samples - 1, math.ceil(samples * .975) - 1)]
    return {"n": len(identifiers), "mean_recall_delta": sum(deltas) / len(deltas), "recall_delta_95_ci": [lo, hi],
            "bootstrap": {"samples": samples, "seed": seed, **bootstrap},
            "recall_improved": sum(d > 0 for d in deltas), "recall_worsened": sum(d < 0 for d in deltas),
            "recall_equal": sum(d == 0 for d in deltas),
            "hit_gained": sum(not row["before_hit"] and row["after_hit"] for row in changes),
            "hit_lost": sum(row["before_hit"] and not row["after_hit"] for row in changes), "per_task": changes}


def seal_row(row, previous_hash):
    return {**row, "previous_sha256": previous_hash, "row_sha256": object_sha({**row, "previous_sha256": previous_hash})}


def append_row(path, row, previous_hash):
    sealed = seal_row(row, previous_hash)
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(canonical(sealed) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sealed


def validate_resume(output, expected, tasks, qrels):
    frozen = read_json(output / "freeze.json")
    if frozen != expected or (output / "report.json").exists():
        raise ValueError("Resume requires identical frozen inputs and an unfinished run.")
    for name, checksum in frozen["source_fingerprints"].items():
        if data.sha256(output / "source" / name) != checksum:
            raise ValueError("Frozen source archive changed.")
    path = output / "observations.jsonl"
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Partial observation line; manual audit required, no silent replay.")
    rows = list(data.read_jsonl(path))
    if len(rows) > len(tasks):
        raise ValueError("Unexpected extra observations.")
    previous_hash = object_sha(frozen)
    for row, task in zip(rows, tasks):
        unsigned = {key: value for key, value in row.items() if key != "row_sha256"}
        if (row.get("task_id") != task["task_id"] or row.get("previous_sha256") != previous_hash
                or row.get("row_sha256") != object_sha(unsigned)
                or row.get("state") not in ("completed", "execution_error")
                or row.get("metrics") != score_ranking(row["ranking"], qrels[task["task_id"]])
                or len(row["ranking"]) > 5 or any(key not in row.get("candidate_ids", []) for key in row["ranking"])
                or (row["state"] == "execution_error" and row["ranking"])):
            raise ValueError("Resume prefix failed its task/score/hash-chain validation.")
        previous_hash = row["row_sha256"]
    return rows


def run(*, root, models, selected_path, baseline_path, output, lock, resume=False):
    selection, chosen, dev_report = load_selection(selected_path)
    tasks, qrels, old, historical_hashes = load_cohort(root, baseline_path)
    sources = {row["passage_id"]: {"sha256": row["sha256"], "source_url": row["source_url"]} for row in data.iter_corpus(root)}
    if len(sources) != 72442 or any(set(grades) - sources.keys() for grades in qrels.values()):
        raise ValueError("The full 72442-passage corpus and official gold passages are required.")
    frozen = {"version": VERSION, "scope": "82 already-seen exact-history-deduplicated Cloud regression tasks; not new holdout",
              "selection": selection, "selection_sha256": data.sha256(selected_path), "development_report_path": str(dev_report.resolve()),
              "source_fingerprints": source_fingerprints(), "artifact_fingerprints": artifact_fingerprints(root, models),
              "historical_fingerprints": historical_hashes, "task_ids": [task["task_id"] for task in tasks],
              "sanitized_tasks_sha256": object_sha(tasks), "qrels_sha256": object_sha(qrels), "planned": 82,
              "corpus_passages": len(sources), "top_k": 5, "model_calls_paid": 0,
              "runtime": {"threads": 2, "batch_size": 1, "max_length": 512, "provider": "CPUExecutionProvider",
                          "fixed_worker": True, "warmup": False, "cross_query_candidate_cache": False},
              "latency_scope": "Full query lexical+dense+fusion+rerank and model switching on the fixed worker; excludes engine setup and grading; first query includes lazy model/vector validation.",
              "failure_policy": "Failures and unattempted queries score zero / 82; pending interrupted query becomes a zero-score failure on resume, never retried."}
    claim_path = selected_path.with_name(selected_path.name + ".regression-claim.json")
    claim = {"selected_sha256": frozen["selection_sha256"], "output": str(output.resolve()), "freeze_sha256": object_sha(frozen)}
    with cpu_lock(lock):
        if resume:
            if read_json(claim_path) != claim:
                raise ValueError("Selection has a different one-shot execution claim.")
            rows = validate_resume(output, frozen, tasks, qrels)
        else:
            output.mkdir(parents=True, exist_ok=False)
            write_new(claim_path, claim)
            write_new(output / "freeze.json", frozen)
            for name in SOURCE_FILES:
                destination = output / "source" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(SERVICE / name, destination)
            write_new(output / "selected.json", selection)
            (output / "observations.jsonl").touch(exist_ok=False)
            rows = []
        previous_hash = rows[-1]["row_sha256"] if rows else object_sha(frozen)
        pending = output / "pending.json"
        if pending.exists():
            item = read_json(pending)
            if rows and item["task_id"] == rows[-1]["task_id"]:
                pending.unlink()  # Observation was fsynced before interruption.
            elif len(rows) < len(tasks) and item["task_id"] == tasks[len(rows)]["task_id"]:
                row = {"task_id": item["task_id"], "state": "execution_error", "error_type": "InterruptedAttempt",
                       "ranking": [], "candidate_ids": [], "metrics": score_ranking([], qrels[item["task_id"]]), "latency_ms": 0,
                       "latency_unobserved": True, "query_started_at": item["started_at"]}
                row = append_row(output / "observations.jsonl", row, previous_hash)
                rows.append(row); previous_hash = row["row_sha256"]
                pending.unlink()
            else:
                raise ValueError("Pending query differs from the frozen prefix.")
        from app.public_search import PublicSearch
        engine = PublicSearch(root, models, threads=2)
        try:
            for task in tasks[len(rows):]:
                if source_fingerprints() != frozen["source_fingerprints"]:
                    raise RuntimeError("Source changed during regression; execution stopped.")
                write_new(pending, {"task_id": task["task_id"], "started_at": datetime.now(timezone.utc).isoformat()})
                started = time.perf_counter()
                row = {"task_id": task["task_id"], "state": "completed"}
                try:
                    response = retrieve(engine, task, chosen)
                    elapsed = (time.perf_counter() - started) * 1000
                    validate_passages(response, sources)
                    row.update({key: value for key, value in response.items() if key != "returned_passages"})
                except Exception as exc:
                    elapsed = (time.perf_counter() - started) * 1000
                    row.update(state="execution_error", error_type=type(exc).__name__, ranking=[], candidate_ids=[])
                row.update(latency_ms=elapsed, metrics=score_ranking(row["ranking"], qrels[task["task_id"]]))
                sealed = append_row(output / "observations.jsonl", row, previous_hash)
                rows.append(sealed); previous_hash = sealed["row_sha256"]
                pending.unlink()
                print(json.dumps({"completed": len(rows), "planned": 82, "state": row["state"], "seconds": round(elapsed / 1000, 3)}), flush=True)
        finally:
            engine.close()
        unchanged = (source_fingerprints() == frozen["source_fingerprints"]
                     and artifact_fingerprints(root, models) == frozen["artifact_fingerprints"]
                     and data.sha256(selected_path) == frozen["selection_sha256"]
                     and data.sha256(dev_report) == selection["source_report"]["sha256"]
                     and all(data.sha256(path) == checksum for path, checksum in historical_hashes.items()))
        summary = summarize(rows, 82)
        # Interrupted timing is unknown, not a zero-millisecond speed win.
        timed = [row for row in rows if not row.get("latency_unobserved")]
        summary["latency_ms"] = summarize(timed, 82)["latency_ms"]
        summary["latency_ms"]["scope"] = frozen["latency_scope"]
        report = {**frozen, "summary": summary, "fingerprints_unchanged": unchanged,
                  "completed": len(rows), "observations_sha256": data.sha256(output / "observations.jsonl"),
                  "paired_vs_original_rerank": paired_comparison(rows, old, frozen["task_ids"], qrels,
                      conversation_ids={task["task_id"]: task["conversation_id"] for task in tasks}),
                  "limitations": ["This is a previously seen regression cohort, not an unseen test or full RAG answer evaluation.",
                                  "Development selection is frozen and all 82 queries remain in the denominator.",
                                  "No Agent response, refusal accuracy or business benefit is measured here."]}
        write_new(output / "report.json", report)
        if not unchanged:
            raise RuntimeError("Regression artifact drift detected; saved report is invalid.")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=data.DEFAULT_ROOT)
    parser.add_argument("--models", type=Path, default=SERVICE.parents[1] / ".local/models/public-retrieval")
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run", action="store_true", help="Required for actual CPU inference; default only validates the selection schema.")
    args = parser.parse_args()
    selection, chosen, _ = load_selection(args.selected)
    if not args.run:
        print(json.dumps({"mode": "dry_run", "planned": 82, "paid_model_calls": 0, "cpu_model_calls": 0, "selected": chosen}))
        return
    report = run(root=args.root, models=args.models, selected_path=args.selected, baseline_path=args.baseline,
                 output=args.output, lock=args.lock, resume=args.resume)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
