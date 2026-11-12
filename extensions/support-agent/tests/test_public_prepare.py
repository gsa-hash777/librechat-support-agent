"""Real small files exercise public-index resume integrity; no model inference.

Only the expensive model and process pool are replaced. Corpus parsing, SQLite,
NPY files, checksums and checkpoint handling run through the production script.
No MTRAG task files, especially the sealed UN questions, are read here.
"""
import hashlib
import json
import sqlite3
from zipfile import ZipFile

import numpy as np
import pytest

from evaluation import public_data
from scripts import prepare_public as prepare


class FakeModels:
    calls = []

    def __init__(self, model_dir, **kwargs):
        self.model_dir = model_dir

    def embed(self, texts):
        assert all(text.strip() for text in texts), "Blank passages must not trigger neural inference."
        self.calls.extend(texts)
        return [np.full(384, 1 / np.sqrt(384), dtype=np.float32) for _ in texts]


class InlinePool:
    """Preserve initializer/map ordering without spawning expensive workers."""
    def __init__(self, *, max_workers, initializer, initargs):
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        prepare._worker_models = None

    def map(self, function, values):
        return map(function, values)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "public-data"
    archive_path = root / public_data.CORPUS_FILE
    archive_path.parent.mkdir(parents=True)
    originals = [
        {"id": "p1", "_id": "p1", "url": "https://cloud.ibm.com/docs/a", "title": "", "text": "Account recovery."},
        {"id": "p2", "_id": "p2", "url": "https://cloud.ibm.com/docs/blank", "title": "", "text": "\n\n\n"},
        {"id": "p3", "_id": "p3", "url": "https://cloud.ibm.com/docs/b", "title": "", "text": "Database backup."},
    ]
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("cloud.jsonl", "\n".join(json.dumps(row) for row in originals) + "\n")
    monkeypatch.setitem(public_data.PINNED_FILES, public_data.CORPUS_FILE,
                        (archive_path.stat().st_size, public_data.sha256(archive_path)))
    models = tmp_path / "models"
    (models / "embedding").mkdir(parents=True)
    weights = models / "embedding/model.onnx"
    weights.write_bytes(b"small fixture model, never loaded as ONNX")
    (models / "manifest.json").write_text(json.dumps({"models": {"embedding": {"files": {
        "model.onnx": {"sha256": prepare.file_sha(weights)}
    }}}}), encoding="utf8")
    monkeypatch.setattr(prepare, "LocalModels", FakeModels)
    monkeypatch.setattr(prepare, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(FakeModels, "calls", [])
    base = prepare.lexical(root)
    prepare.vectors(root, models, stop_after=1)
    return {"root": root, "models": models, "base": base, "originals": originals,
            "matrix": root / "index/vectors.npy", "checkpoint": root / "index/vectors.json"}


def set_state(fixture, status):
    path = fixture["checkpoint"]
    metadata = json.loads(path.read_text(encoding="utf8"))
    metadata["status"] = status
    if status == "ready":
        metadata["completed"] = fixture["base"]["passages"]
        metadata["vectors_sha256"] = prepare.file_sha(fixture["matrix"])
    path.write_text(json.dumps(metadata), encoding="utf8")
    return metadata


@pytest.mark.parametrize("state", ["building", "ready"])
def test_checkpoint_without_matrix_fails_instead_of_recreating_or_claiming_ready(prepared, state):
    set_state(prepared, state)
    prepared["matrix"].unlink()
    with pytest.raises((RuntimeError, OSError), match="matrix|missing|No such|cannot find"):
        prepare.vectors(prepared["root"], prepared["models"])
    assert not prepared["matrix"].exists()


@pytest.mark.parametrize("state", ["building", "ready"])
@pytest.mark.parametrize("shape,dtype", [((2, 384), np.float32), ((3, 383), np.float32), ((3, 384), np.float64)])
def test_checkpoint_shape_or_dtype_corruption_is_rejected(prepared, state, shape, dtype):
    set_state(prepared, state)
    np.save(prepared["matrix"], np.zeros(shape, dtype=dtype), allow_pickle=False)
    before = prepared["matrix"].read_bytes()
    with pytest.raises((RuntimeError, ValueError), match="shape|dtype|hash"):
        prepare.vectors(prepared["root"], prepared["models"])
    assert prepared["matrix"].read_bytes() == before


@pytest.mark.parametrize("state", ["building", "ready"])
def test_actual_model_file_hash_must_match_manifest_even_if_manifest_is_unchanged(prepared, state):
    set_state(prepared, state)
    manifest = prepared["models"] / "manifest.json"
    before_manifest = prepare.file_sha(manifest)
    (prepared["models"] / "embedding/model.onnx").write_bytes(b"tampered weights")
    assert prepare.file_sha(manifest) == before_manifest
    before_calls = len(FakeModels.calls)
    with pytest.raises(RuntimeError, match="[Mm]odel|hash|manifest"):
        prepare.vectors(prepared["root"], prepared["models"])
    assert len(FakeModels.calls) == before_calls


def test_raw_blank_passage_keeps_id_text_hash_and_row_count(prepared):
    with sqlite3.connect(prepared["root"] / "index/passages.sqlite") as conn:
        rows = conn.execute("SELECT passage_id,text,sha256 FROM passages ORDER BY rowid").fetchall()
        hits = conn.execute("SELECT rowid FROM fts WHERE fts MATCH 'backup'").fetchall()
    assert len(rows) == 3
    assert [row[0] for row in rows] == ["p1", "p2", "p3"]
    assert rows[1] == ("p2", "\n\n\n", hashlib.sha256(b"\n\n\n").hexdigest())
    assert hits == [(3,)]


def test_resume_preserves_completed_prefix_and_blank_rows_receive_zero_vectors(prepared):
    previous = np.load(prepared["matrix"], allow_pickle=False)[0].copy()
    FakeModels.calls.clear()
    prepare.vectors(prepared["root"], prepared["models"])
    matrix = np.load(prepared["matrix"], allow_pickle=False)
    assert matrix.shape == (3, 384) and matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[0], previous)
    np.testing.assert_array_equal(matrix[1], np.zeros(384, dtype=np.float32))
    assert np.any(matrix[2])
    assert FakeModels.calls == ["Database backup."]
    metadata = json.loads(prepared["checkpoint"].read_text(encoding="utf8"))
    assert metadata["status"] == "ready" and metadata["completed"] == 3
    assert metadata["vectors_sha256"] == prepare.file_sha(prepared["matrix"])


def test_resume_rejects_sqlite_content_drift_despite_same_row_count(prepared):
    with sqlite3.connect(prepared["root"] / "index/passages.sqlite") as conn:
        conn.execute("UPDATE passages SET text='changed source' WHERE rowid=3")
    assert prepare.file_sha(prepared["root"] / "index/passages.sqlite") != prepared["base"]["sqlite_sha256"]
    with pytest.raises(RuntimeError, match="[Ss][Qq][Ll]ite|hash|index"):
        prepare.vectors(prepared["root"], prepared["models"])


def test_missing_checkpoint_does_not_overwrite_existing_vector_matrix(prepared):
    prepared["checkpoint"].unlink()
    previous = prepared["matrix"].read_bytes()
    with pytest.raises(RuntimeError, match="without a checkpoint"):
        prepare.vectors(prepared["root"], prepared["models"])
    assert prepared["matrix"].read_bytes() == previous
