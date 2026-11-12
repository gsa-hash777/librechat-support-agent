"""Build the complete, pinned IBM Cloud corpus index without loading it in RAM."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.public_search import ENGLISH_PREFIX, VERSION, file_sha
from app.hybrid_models import LocalModels
from evaluation.public_data import CORPUS_FILE, DEFAULT_ROOT, corpus_rows, verify_file


def write_json(path, value):
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def lexical(root):
    source = verify_file(root, CORPUS_FILE)
    folder = root / "index"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old.get("status") == "ready" and old.get("corpus_sha256") == source["sha256"] and old.get("version") == VERSION:
            if file_sha(folder / "passages.sqlite") != old["sqlite_sha256"]:
                raise RuntimeError("Existing lexical index differs from its manifest.")
            print(json.dumps({"lexical": "already_ready", "passages": old["passages"]}), flush=True)
            return old
        raise RuntimeError("An incompatible index exists; choose a new root.")
    path = folder / "passages.sqlite"
    if path.exists():
        raise RuntimeError("An unfinished index exists; inspect it before rebuilding.")
    began = time.perf_counter()
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA cache_size=-8192")
        conn.execute("CREATE TABLE passages (rowid INTEGER PRIMARY KEY, passage_id TEXT UNIQUE NOT NULL, source_url TEXT NOT NULL, text TEXT NOT NULL, sha256 TEXT NOT NULL)")
        conn.execute("CREATE VIRTUAL TABLE fts USING fts5(text,content='passages',content_rowid='rowid',tokenize='porter unicode61')")
        count = 0
        for count, row in enumerate(corpus_rows(root), 1):
            text = row["text"]
            conn.execute("INSERT INTO passages VALUES(?,?,?,?,?)", (count, row["id"], row["url"], text, hashlib.sha256(text.encode()).hexdigest()))
            if count % 2000 == 0:
                conn.commit()
                print(json.dumps({"indexing_passages": count}), flush=True)
        conn.execute("INSERT INTO fts(fts) VALUES('rebuild')")
        conn.commit()
        sources = conn.execute("SELECT count(DISTINCT source_url) FROM passages").fetchone()[0]
        conn.execute("PRAGMA optimize")
    result = {"version": VERSION, "status": "ready", "passages": count, "source_urls": sources,
              "corpus_sha256": source["sha256"], "sqlite_sha256": file_sha(path),
              "created_at": datetime.now(timezone.utc).isoformat(), "build_seconds": time.perf_counter() - began}
    write_json(manifest, result)
    print(json.dumps(result), flush=True)
    return result


_worker_models = None


def init_worker(model_dir, threads):
    global _worker_models
    _worker_models = LocalModels(model_dir, threads=threads, query_prefix=ENGLISH_PREFIX)


def embed_batch(rows):
    import numpy as np
    import psutil
    values = [(rowid, _worker_models.embed([text])[0] if text.strip() else np.zeros(384, dtype=np.float32))
              for rowid, text in rows]
    return values, psutil.Process().memory_info().rss


def vectors(root, model_dir, threads=2, stop_after=0, workers=1):
    import numpy as np
    import psutil
    base = lexical(root)
    folder = root / "index"
    progress = folder / "vectors.json"
    signature = {"passages": base["passages"], "corpus_sha256": base["corpus_sha256"],
                 "sqlite_sha256": base["sqlite_sha256"],
                 "model_manifest_sha256": file_sha(model_dir / "manifest.json"), "query_prefix": ENGLISH_PREFIX,
                 "dimension": 384, "dtype": "float32", "pooling": "CLS per complete overflow window; mean; L2"}
    old = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else {}
    if old and any(old.get(key) != value for key, value in signature.items()):
        raise RuntimeError("Vector checkpoint metadata differs; no resume performed.")
    path = folder / "vectors.npy"
    if path.exists() and not old:
        raise RuntimeError("Vector file without a checkpoint; inspect before resuming.")
    if old and not path.exists():
        raise RuntimeError("Vector checkpoint exists but matrix is missing; cannot resume.")
    for role, model in json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))["models"].items():
        for name, entry in model["files"].items():
            if file_sha(model_dir / role / name) != entry["sha256"]:
                raise RuntimeError("Model file differs from its pinned manifest.")
    matrix = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float32,
                                       shape=(base["passages"], 384))
    if matrix.shape != (base["passages"], 384) or matrix.dtype != np.float32:
        raise RuntimeError("Resume vector shape or dtype differs.")
    completed = old.get("completed", 0)
    if type(completed) is not int or not 0 <= completed <= base["passages"]:
        raise RuntimeError("Invalid vector checkpoint position.")
    if old.get("status") == "ready":
        if completed != base["passages"] or file_sha(path) != old.get("vectors_sha256"):
            raise RuntimeError("Completed vector index integrity failed.")
        print(json.dumps({"vectors": "already_ready"}), flush=True)
        del matrix
        return
    started_at = completed
    began = time.perf_counter()
    total_seconds = old.get("elapsed_seconds", 0)
    peak = old.get("peak_rss_bytes", 0)
    worker_peak = old.get("worker_peak_rss_bytes", 0)
    try:
        with sqlite3.connect(folder / "passages.sqlite") as conn:
            cursor = conn.execute("SELECT rowid,text FROM passages WHERE rowid>? ORDER BY rowid", (completed,))
            with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(str(model_dir), threads)) as pool:
                while True:
                    allowance = min(workers * 32, stop_after - (completed - started_at)) if stop_after else workers * 32
                    rows = list(itertools.islice(cursor, allowance))
                    if not rows:
                        break
                    batches = [rows[start:start + 32] for start in range(0, len(rows), 32)]
                    for values, rss in pool.map(embed_batch, batches):
                        for rowid, vector in values:
                            matrix[rowid - 1] = vector
                            completed = rowid
                        worker_peak = max(worker_peak, rss)
                    peak = max(peak, psutil.Process().memory_info().rss)
                    matrix.flush()
                    result = {**signature, "status": "ready" if completed == base["passages"] else "building",
                              "completed": completed, "elapsed_seconds": total_seconds + time.perf_counter() - began,
                              "peak_rss_bytes": peak, "worker_peak_rss_bytes": worker_peak,
                              "runtime": {"threads_per_worker": threads, "workers": workers, "batch_per_worker": 32},
                              "whitespace_only_passages": 3, "blank_vector_policy": "all zeros; original rows retained"}
                    if result["status"] == "ready":
                        result["vectors_sha256"] = file_sha(path)
                    write_json(progress, result)
                    print(json.dumps({"embedded": completed, "total": base["passages"], "elapsed_seconds": round(result["elapsed_seconds"], 1),
                                      "peak_rss_mib": round(peak / 1048576, 1)}), flush=True)
                    if stop_after and completed - started_at >= stop_after:
                        break
    finally:
        del matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--vectors", action="store_true")
    parser.add_argument("--models", type=Path, default=DEFAULT_ROOT.parents[1] / "models/public-retrieval")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--stop-after", type=int, default=0, help="Checkpoint after this many additional embeddings; 0 completes the corpus.")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    args = parser.parse_args()
    if args.vectors:
        vectors(args.root, args.models, args.threads, args.stop_after, args.workers)
    else:
        lexical(args.root)


if __name__ == "__main__":
    main()
