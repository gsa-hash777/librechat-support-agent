"""Pinned inputs and leakage boundaries; no network or model calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from evaluation import public_data as data


def raw_task(identifier="q1", collection="ibmcloud", label="ANSWERABLE"):
    return {"task_id": identifier, "conversation_id": "c1", "turn": "2", "Collection": collection,
            "Answerability": [label], "Question Type": ["Factoid"], "Multi-Turn": ["Follow-up"],
            "input": [{"speaker": "user", "text": "What is the service?", "metadata": {"author": "hidden"}},
                      {"speaker": "agent", "text": "Earlier reference response."},
                      {"speaker": "user", "text": "How does it work?", "enrichments": {"gold": "hidden"}}],
            "targets": [{"speaker": "agent", "text": "GOLD_ANSWER"}],
            "contexts": [{"document_id": "p1", "text": "GOLD_CONTEXT"}]}


def pin(monkeypatch, root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        content = content.encode("utf8")
    path.write_bytes(content)
    monkeypatch.setitem(data.PINNED_FILES, name, (len(content), hashlib.sha256(content).hexdigest()))
    return path


def test_task_input_is_allowlist_not_blacklist():
    row = raw_task()
    row["new_future_gold_field"] = "NEVER_ALLOW"
    projected = data.task_input(row)
    assert set(projected) == data.INPUT_KEYS
    assert all(set(message) == {"speaker", "text"} for message in projected["messages"])
    assert projected["messages"][1]["speaker"] == "assistant"
    assert "GOLD" not in json.dumps(projected)
    assert "hidden" not in json.dumps(projected)
    assert projected["turn"] == 2


def test_case_insensitive_upstream_field_variants():
    old = raw_task()
    new = deepcopy(old)
    new["answerability"] = new.pop("Answerability")
    new["collection"] = new.pop("Collection")
    assert data.task_labels(old) == data.task_labels(new)
    assert data.is_cloud(old) and data.is_cloud(new)
    assert not data.is_cloud(raw_task(collection="not_really_ibmcloud"))


@pytest.mark.parametrize("mutator", [
    lambda row: row.update(task_id=""), lambda row: row.update(turn=True),
    lambda row: row.update(turn="-1"), lambda row: row.update(input=[]),
    lambda row: row["input"][-1].update(speaker="system"),
    lambda row: row["input"][-1].update(text=" "),
    lambda row: row["input"][-1].update(speaker="agent"),
])
def test_bad_task_schema_rejected(mutator):
    row = raw_task()
    mutator(row)
    with pytest.raises(ValueError):
        data.task_input(row)


def test_final_gate_precedes_file_read(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("sealed final data was accessed")
    monkeypatch.setattr(data, "verify_file", forbidden)
    for loader in (data.load_cloud_tasks, data.load_labels, data.load_qrels):
        with pytest.raises(ValueError, match="sealed"):
            loader(tmp_path, "final")


def test_only_cloud_tasks_and_separate_annotations(tmp_path, monkeypatch):
    rows = [raw_task(), raw_task("other", "fiqa")]
    pin(monkeypatch, tmp_path, data.TASK_FILES["dev"], "\n".join(json.dumps(row) for row in rows))
    inputs = data.load_cloud_tasks(tmp_path)
    labels = data.load_labels(tmp_path)
    assert len(inputs) == 1 and set(inputs[0]) == data.INPUT_KEYS
    assert labels["q1"]["answerability"] == "ANSWERABLE"
    path = tmp_path / data.TASK_FILES["dev"]
    path.write_text("corrupted", encoding="utf8")
    with pytest.raises(ValueError, match="fingerprint"):
        data.load_cloud_tasks(tmp_path)


def test_duplicate_tasks_rejected(tmp_path, monkeypatch):
    pin(monkeypatch, tmp_path, data.TASK_FILES["dev"], json.dumps(raw_task()) + "\n" + json.dumps(raw_task()))
    with pytest.raises(ValueError, match="Duplicate"):
        data.load_cloud_tasks(tmp_path)


def test_qrels_keep_grades_and_reject_duplicate_pairs(tmp_path):
    path = tmp_path / "gold.tsv"
    path.write_text("query-id\tcorpus-id\tscore\nq1\tp1\t2\nq1\tp2\t0\n", encoding="utf8")
    assert data.parse_qrels(path) == {"q1": {"p1": 2, "p2": 0}}
    path.write_text("query-id\tcorpus-id\tscore\nq1\tp1\t1\nq1\tp1\t1\n", encoding="utf8")
    with pytest.raises(ValueError, match="Duplicate"):
        data.parse_qrels(path)


def test_prepare_preserves_bytes_license_and_refuses_changed_existing_copy(tmp_path, monkeypatch):
    content = b"Apache License test fixture\n"
    monkeypatch.setattr(data, "PINNED_FILES", {"LICENSE": (len(content), hashlib.sha256(content).hexdigest())})
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "LICENSE").write_bytes(content)
    frozen = data.prepare_data(source, target)
    assert (target / "LICENSE").read_bytes() == content
    assert frozen["source_data_unmodified"] is True
    assert frozen["question_origin"].endswith("not customer tickets")
    assert data.prepare_data(source, target) == frozen
    (target / "LICENSE").write_bytes(b"modified")
    with pytest.raises(ValueError, match="fingerprint"):
        data.prepare_data(source, target)
    assert (source / "LICENSE").read_bytes() == content


def test_download_has_size_and_hash_bound_before_persisting(tmp_path, monkeypatch):
    import io
    content = b"expected"
    monkeypatch.setattr(data, "PINNED_FILES", {"LICENSE": (len(content), hashlib.sha256(content).hexdigest())})
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrongdata"))
    with pytest.raises(ValueError, match="Downloaded"):
        data.prepare_data(destination=tmp_path, download_missing=True)
    assert not (tmp_path / "LICENSE").exists()


def test_corpus_preserves_ids_and_text_without_zip_extraction(tmp_path, monkeypatch):
    archive = tmp_path / data.CORPUS_FILE
    archive.parent.mkdir(parents=True)
    row = {"id": "p1", "_id": "p1", "text": "Exact original text.\n", "title": "", "url": "https://cloud.ibm.com/docs/product"}
    with ZipFile(archive, "w") as stream:
        stream.writestr("cloud.jsonl", json.dumps(row) + "\n")
    monkeypatch.setitem(data.PINNED_FILES, data.CORPUS_FILE, (archive.stat().st_size, data.sha256(archive)))
    result = list(data.corpus_rows(tmp_path))
    assert result[0]["id"] == "p1" and result[0]["text"] == row["text"]
    assert not (tmp_path / "cloud.jsonl").exists()
    public = list(data.iter_corpus(tmp_path))
    assert public == [{"passage_id": "p1", "text": row["text"], "source_url": row["url"],
                       "sha256": hashlib.sha256(row["text"].encode()).hexdigest()}]


def test_docs_root_url_with_query_is_a_valid_original_source(tmp_path, monkeypatch):
    archive = tmp_path / data.CORPUS_FILE
    archive.parent.mkdir(parents=True)
    row = {"id": "p1", "_id": "p1", "text": "Original text", "url": "https://cloud.ibm.com/docs?topic=index"}
    with ZipFile(archive, "w") as stream:
        stream.writestr("cloud.jsonl", json.dumps(row) + "\n")
    monkeypatch.setitem(data.PINNED_FILES, data.CORPUS_FILE, (archive.stat().st_size, data.sha256(archive)))
    assert list(data.corpus_rows(tmp_path))[0]["url"] == row["url"]


def test_official_empty_passages_are_preserved_without_replacing_text(tmp_path, monkeypatch):
    archive = tmp_path / data.CORPUS_FILE
    archive.parent.mkdir(parents=True)
    rows = [{"id": f"p{n}", "_id": f"p{n}", "text": text, "url": "https://cloud.ibm.com/docs/test"}
            for n, text in enumerate(("", "\n\n", "actual content"))]
    with ZipFile(archive, "w") as stream:
        stream.writestr("cloud.jsonl", "\n".join(map(json.dumps, rows)))
    monkeypatch.setitem(data.PINNED_FILES, data.CORPUS_FILE, (archive.stat().st_size, data.sha256(archive)))
    result = list(data.iter_corpus(tmp_path))
    assert len(result) == 3
    assert [row["text"] for row in result] == ["", "\n\n", "actual content"]


def test_dev8_selection_is_deterministic_label_only_and_never_uses_final(tmp_path, monkeypatch):
    rows = []
    for category, count in (("ANSWERABLE", 6), ("PARTIAL", 3), ("UNANSWERABLE", 3)):
        for n in range(count):
            row = raw_task(f"{category}-{n}", label=category)
            row["conversation_id"] = f"c-{category}-{n}"
            rows.append(row)
    pin(monkeypatch, tmp_path, data.TASK_FILES["dev"], "\n".join(map(json.dumps, rows)))
    manifest = data.select_dev8(tmp_path)
    assert manifest == data.select_dev8(tmp_path)
    assert len(manifest["task_ids"]) == 8 and manifest["conversations"] == 8
    assert manifest["final_data_used"] is False
    assert "GOLD_ANSWER" not in json.dumps(manifest)
    assert "How does it work?" not in json.dumps(manifest)


def test_corpus_rejects_unexpected_archive_members(tmp_path, monkeypatch):
    archive = tmp_path / data.CORPUS_FILE
    archive.parent.mkdir(parents=True)
    with ZipFile(archive, "w") as stream:
        stream.writestr("../cloud.jsonl", "{}")
    monkeypatch.setitem(data.PINNED_FILES, data.CORPUS_FILE, (archive.stat().st_size, data.sha256(archive)))
    with pytest.raises(ValueError, match="members"):
        list(data.corpus_rows(tmp_path))
