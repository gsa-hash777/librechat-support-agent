"""Verified local runtime copies avoid random SQLite I/O across Windows mounts."""
import json
from pathlib import Path
import shutil

from app.public_search import file_sha


def copy_verified(source, target, expected):
    source, target = Path(source), Path(target)
    if file_sha(source) != expected:
        raise RuntimeError("Public runtime source hash mismatch.")
    if target.is_file() and file_sha(target) == expected:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".copying")
    with source.open("rb") as src, partial.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    if file_sha(partial) != expected:
        raise RuntimeError("Public runtime copy failed its integrity check.")
    partial.replace(target)
    return True


def prepare_cache(source_index, target_index, source_models=None, target_models=None):
    source_index, target_index = Path(source_index), Path(target_index)
    meta_path = source_index / "manifest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "ready":
        raise RuntimeError("Public source index has not completed.")
    copied = int(copy_verified(source_index / "passages.sqlite", target_index / "passages.sqlite", meta["sqlite_sha256"]))
    copied += copy_verified(meta_path, target_index / "manifest.json", file_sha(meta_path))
    vector_meta = source_index / "vectors.json"
    if vector_meta.is_file():
        vectors = json.loads(vector_meta.read_text(encoding="utf-8"))
        if vectors.get("status") == "ready":
            if vectors.get("sqlite_sha256") != meta["sqlite_sha256"] or vectors.get("corpus_sha256") != meta["corpus_sha256"]:
                raise RuntimeError("Public vector and lexical indexes differ.")
            copied += copy_verified(source_index / "vectors.npy", target_index / "vectors.npy", vectors["vectors_sha256"])
            copied += copy_verified(vector_meta, target_index / "vectors.json", file_sha(vector_meta))
    if source_models is not None:
        source_models, target_models = Path(source_models), Path(target_models)
        model_meta_path = source_models / "manifest.json"
        models = json.loads(model_meta_path.read_text(encoding="utf-8"))["models"]
        if set(models) != {"embedding", "reranker"}:
            raise RuntimeError("Unexpected public model roles.")
        for role, model in models.items():
            for name, entry in model["files"].items():
                if Path(name).name != name or name in (".", ".."):
                    raise RuntimeError("Invalid public model cache filename.")
                copied += copy_verified(source_models / role / name, target_models / role / name, entry["sha256"])
        copied += copy_verified(model_meta_path, target_models / "manifest.json", file_sha(model_meta_path))
    return {"copied_files": copied, "originals_modified": False}
