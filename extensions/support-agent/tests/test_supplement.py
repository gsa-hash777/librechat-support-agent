"""Frozen synthetic business evidence and its exact Markdown source spans."""
import hashlib
import json
from pathlib import Path
import re

import pytest

from app.content import suspicious_reasons
from app.knowledge import CHUNK_CHARS, MAX_FILE_BYTES, _input, chunk_text


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "data" / "documents"
SOURCE = DOCUMENTS / "supplement.json"
HANDBOOK = DOCUMENTS / "support-supplement.md"
MANIFEST = DOCUMENTS / "supplement-manifest.json"


@pytest.fixture(scope="module")
def supplement():
    return (
        json.loads(SOURCE.read_text(encoding="utf-8")),
        HANDBOOK.read_bytes().decode("utf-8"),
        json.loads(MANIFEST.read_text(encoding="utf-8")),
    )


def test_declared_synthetic_scope_and_frozen_inputs(supplement):
    rows, text, manifest = supplement
    assert manifest["synthetic"] is True
    assert manifest["business_paragraphs_verbatim"] is True
    assert manifest["authoring"]["kind"] == "project_authored_synthetic"
    assert manifest["authoring"]["real_company_policy"] is False
    assert manifest["authoring"]["local_implementation_documentation"] is False
    assert manifest["authoring"]["real_business_owner_review_completed"] is False
    assert "不代表真实公司的制度" in text.split("## KB-017", 1)[0]
    assert "不表示本机 LibreChat" in text.split("## KB-017", 1)[0]
    assert manifest["source"] == "data/documents/supplement.json"
    assert manifest["handbook_file"] == HANDBOOK.name
    assert manifest["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert manifest["handbook_sha256"] == hashlib.sha256(HANDBOOK.read_bytes()).hexdigest()
    assert manifest["document_count"] == len(rows) == 16
    assert set(manifest["baseline_sources"]) == {
        "data/knowledge.json", "data/documents/knowledge.md", "data/documents/manifest.json",
    }
    for relative, digest in manifest["baseline_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_business_records_are_additive_and_have_complete_paragraphs(supplement):
    rows, _, manifest = supplement
    originals = json.loads((ROOT / "data" / "knowledge.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in rows] == [f"KB-{number:03d}" for number in range(17, 33)]
    assert not {row["id"] for row in rows} & {row["id"] for row in originals}
    assert not {row["content"] for row in rows} & {row["content"] for row in originals}
    assert len({row["title"] for row in rows}) == len(rows)
    assert manifest["org_id"] == "demo"
    for row in rows:
        assert set(row) == {"id", "org_id", "title", "content", "source"}
        assert row["org_id"] == "demo"
        assert row["source"].startswith("项目自编模拟业务规范 / ")
        assert 2 <= len(row["content"].split("\n\n")) <= 4
        assert 300 <= len(re.findall(r"[\u3400-\u9fff]", row["content"])) <= 500
        assert row["content"] == row["content"].strip()


def test_manifest_spans_preserve_every_body_byte_and_line(supplement):
    rows, text, manifest = supplement
    assert "zero-based, end-exclusive" in manifest["offset_unit"]
    assert [entry["id"] for entry in manifest["articles"]] == [row["id"] for row in rows]
    previous_end = 0
    for row, entry in zip(rows, manifest["articles"], strict=True):
        start, end = entry["handbook_content_start"], entry["handbook_content_end"]
        assert type(start) is int and type(end) is int
        assert previous_end < start < end <= len(text)
        body = text[start:end]
        assert body.encode("utf-8") == row["content"].encode("utf-8")
        assert text.count(row["content"]) == 1
        assert entry["content_sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()
        assert entry["title"] == row["title"] and entry["source"] == row["source"]
        assert f"## {row['id']} {row['title']}\n" in text[previous_end:start]
        assert entry["handbook_content_start_line"] == text.count("\n", 0, start) + 1
        assert entry["handbook_content_end_line"] == text.count("\n", 0, end - 1) + 1
        previous_end = end


def test_import_parser_preserves_utf8_markdown_without_real_database(supplement):
    _, text, manifest = supplement
    raw = HANDBOOK.read_bytes()
    assert len(raw) < MAX_FILE_BYTES
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
    filename, parsed, title, source = _input(
        HANDBOOK.name, raw, manifest["title"], "项目自编模拟业务资料 / 补充手册",
    )
    assert filename == HANDBOOK.name and parsed == text
    assert title == manifest["title"] and source.startswith("项目自编模拟业务资料")


def test_chunks_keep_complete_business_evidence_and_valid_source_ranges(supplement):
    _, text, manifest = supplement
    chunks = chunk_text(text, doc_id="supplement-import-check", revision=1)
    assert chunks and len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    covered = bytearray(len(text))
    for chunk in chunks:
        start, end = chunk["start_offset"], chunk["end_offset"]
        assert 0 <= start < end <= len(text)
        assert chunk["content"] == text[start:end]
        assert len(chunk["content"]) <= CHUNK_CHARS
        assert chunk["start_line"] == text.count("\n", 0, start) + 1
        assert chunk["end_line"] == text.count("\n", 0, end - 1) + 1
        covered[start:end] = b"\x01" * (end - start)
    assert all(covered)
    for entry in manifest["articles"]:
        assert all(covered[entry["handbook_content_start"]:entry["handbook_content_end"]])


def test_benign_business_evidence_is_not_quarantined_or_linked_to_real_contacts(supplement):
    rows, text, _ = supplement
    assert not suspicious_reasons(text)
    assert not re.search(r"https?://|mailto:|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    for row in rows:
        assert not suspicious_reasons("\n".join(row[key] for key in ("title", "source", "content")))
