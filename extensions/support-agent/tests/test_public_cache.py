import hashlib
import json
from pathlib import Path

import pytest

from app.public_cache import copy_verified, prepare_cache


def test_cache_copy_is_idempotent_and_integrity_checked(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "nested/target"
    source.write_bytes(b"public source")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    assert copy_verified(source, target, checksum)
    assert target.read_bytes() == source.read_bytes()
    assert not copy_verified(source, target, checksum)
    target.write_bytes(b"corrupt cached bytes")
    assert copy_verified(source, target, checksum)
    source.write_bytes(b"changed source")
    with pytest.raises(RuntimeError, match="source hash"):
        copy_verified(source, target, checksum)
    assert target.read_bytes() == b"public source"


def test_building_vectors_never_enter_runtime_cache(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "passages.sqlite").write_bytes(b"fixture database bytes")
    checksum = hashlib.sha256((source / "passages.sqlite").read_bytes()).hexdigest()
    (source / "manifest.json").write_text(json.dumps({"status": "ready", "sqlite_sha256": checksum}))
    (source / "vectors.json").write_text(json.dumps({"status": "building"}))
    target = tmp_path / "target"
    result = prepare_cache(source, target)
    assert result["copied_files"] == 2 and not result["originals_modified"]
    assert not (target / "vectors.json").exists()
    assert prepare_cache(source, target)["copied_files"] == 0


def test_vector_row_mapping_mismatch_fails(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "passages.sqlite").write_bytes(b"db")
    checksum = hashlib.sha256(b"db").hexdigest()
    (source / "manifest.json").write_text(json.dumps({"status": "ready", "sqlite_sha256": checksum, "corpus_sha256": "c"}))
    (source / "vectors.json").write_text(json.dumps({"status": "ready", "sqlite_sha256": "different", "corpus_sha256": "c"}))
    with pytest.raises(RuntimeError, match="indexes differ"):
        prepare_cache(source, tmp_path / "target")
