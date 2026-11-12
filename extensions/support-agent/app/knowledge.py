"""Versioned UTF-8 knowledge imports and exact, organization-scoped source spans.

Only an authenticated browser administrator may import. Readers never fetch URLs
or interpret document text. ``knowledge_docs`` retains its original five columns.
"""
from bisect import bisect_right
import hashlib
import json
import re
import time
import unicodedata

from fastapi import HTTPException

from .content import suspicious_reasons
from .db import audit, execute_statements


MAX_FILE_BYTES = 1024 * 1024
MAX_SOURCE_LINES = 10_000
CHUNK_CHARS = 600
CHUNK_OVERLAP = 80

SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_assets (
    doc_id TEXT PRIMARY KEY REFERENCES knowledge_docs(id) ON DELETE CASCADE,
    org_id TEXT NOT NULL, file_key TEXT NOT NULL, filename TEXT NOT NULL,
    current_revision INTEGER NOT NULL CHECK(current_revision >= 1),
    created_at REAL NOT NULL, updated_at REAL NOT NULL,
    UNIQUE(org_id, file_key)
);
CREATE TABLE IF NOT EXISTS knowledge_revisions (
    doc_id TEXT NOT NULL REFERENCES knowledge_docs(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL CHECK(revision >= 1), filename TEXT NOT NULL,
    title TEXT NOT NULL, source TEXT NOT NULL, content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL, import_sha256 TEXT NOT NULL,
    chunk_count INTEGER NOT NULL, actor_id TEXT NOT NULL REFERENCES users(id),
    created_at REAL NOT NULL, PRIMARY KEY(doc_id, revision)
);
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    doc_id TEXT NOT NULL, revision INTEGER NOT NULL, ordinal INTEGER NOT NULL,
    chunk_id TEXT NOT NULL, content TEXT NOT NULL, heading TEXT NOT NULL,
    start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL,
    start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
    PRIMARY KEY(doc_id, revision, ordinal), UNIQUE(doc_id, revision, chunk_id),
    FOREIGN KEY(doc_id, revision) REFERENCES knowledge_revisions(doc_id, revision) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS knowledge_assets_org ON knowledge_assets(org_id, doc_id);
"""


def ensure_schema(connection) -> None:
    """Create additive tables without committing or changing existing document rows."""
    execute_statements(connection, SCHEMA)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _label(value, label: str, limit: int, *, default: str = "") -> str:
    if not isinstance(value, str):
        raise HTTPException(422, f"{label}必须为文本。")
    value = unicodedata.normalize("NFC", value.strip()) or default
    if not value or len(value) > limit or any(unicodedata.category(c) in {"Cc", "Cs"} for c in value):
        raise HTTPException(422, f"{label}为空、过长或含控制字符。")
    return value


def _input(filename, content_bytes, title, source):
    filename = _label(filename, "文件名", 180)
    if any(c in filename for c in '/\\:') or filename in {".", ".."}:
        raise HTTPException(422, "请上传普通文件名，不接受路径或网址。")
    if not filename.lower().endswith((".md", ".txt")):
        raise HTTPException(422, "当前仅支持 UTF-8 编码的 .md 和 .txt 文本文件。")
    if not isinstance(content_bytes, bytes):
        raise HTTPException(422, "上传内容必须为文件字节。")
    if len(content_bytes) > MAX_FILE_BYTES:
        raise HTTPException(413, "文件超过 1 MiB 上限，请先拆分。")
    try:
        content = content_bytes.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError:
        raise HTTPException(422, "文件不是有效的 UTF-8 文本，请转换编码后重试。") from None
    if not content.strip() or any(unicodedata.category(c) in {"Cc", "Cs"} and c not in "\n\t" for c in content):
        raise HTTPException(422, "文件为空或包含二进制控制字符。")
    if content.count("\n") + 1 > MAX_SOURCE_LINES:
        raise HTTPException(413, "文件超过 10000 行上限，请先拆分。")
    title = _label(title, "标题", 180, default=filename.rsplit(".", 1)[0])
    source = _label(source, "来源说明", 500, default="用户导入 / " + filename)
    return filename, content, title, source


def _boundaries(content: str):
    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(content) if char == "\n")
    headings, boundaries, stack = [], [], []
    fence = None
    offset = 0
    # Newlines, line numbers and source-page anchors all use LF only. Unicode
    # paragraph separators remain source characters rather than hidden new lines.
    for bare in content.split("\n"):
        line_length = len(bare) + int(offset + len(bare) < len(content))
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", bare)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence) and not bare[marker.end():].strip():
                fence = None
        elif fence is None:
            match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", bare)
            if match:
                level = len(match.group(1))
                stack = [(depth, title) for depth, title in stack if depth < level]
                stack.append((level, match.group(2)))
                headings.append((offset, " / ".join(title for _, title in stack)))
                boundaries.append(offset)
        if not bare.strip():
            boundaries.append(offset + line_length)
        offset += line_length
    return starts, headings, sorted(set(boundaries))


def chunk_text(content: str, *, doc_id: str = "document", revision: int = 1,
               max_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split exact substrings with bounded overlap, preserving every source character.

    Offsets are zero-based Python Unicode character offsets; lines are one-based,
    inclusive. End offsets are exclusive. Headings inside fenced code are ignored.
    """
    if not isinstance(content, str):
        raise ValueError("Content must be text.")
    if type(max_chars) is not int or type(overlap) is not int or max_chars < 32 or not 0 <= overlap < max_chars // 2:
        raise ValueError("Invalid chunk size or overlap.")
    if not content:
        return []
    lines, headings, boundaries = _boundaries(content)
    heading_offsets = [offset for offset, _ in headings]
    digest = _digest(content)
    chunks, start = [], 0
    while start < len(content):
        end = min(start + max_chars, len(content))
        if end < len(content):
            low = bisect_right(boundaries, start + max_chars // 2)
            high = bisect_right(boundaries, end)
            if high > low:
                end = boundaries[high - 1]
        # If overlap begins before a section heading, label from the first heading
        # inside the chunk only when no prior heading exists; offsets remain exact.
        heading_index = bisect_right(heading_offsets, start) - 1
        heading = headings[heading_index][1] if heading_index >= 0 else ""
        if not heading:
            first = bisect_right(heading_offsets, start - 1)
            if first < len(headings) and headings[first][0] < end:
                heading = headings[first][1]
        ordinal = len(chunks) + 1
        chunks.append({"id": doc_id, "doc_id": doc_id,
                       "chunk_id": f"{doc_id}-v{revision}-{digest[:12]}-{ordinal:04d}",
                       "revision": revision, "ordinal": ordinal, "content": content[start:end],
                       "heading": heading, "start_offset": start, "end_offset": end,
                       "start_line": bisect_right(lines, start),
                       "end_line": bisect_right(lines, end - 1), "content_sha256": digest})
        if end == len(content):
            break
        start = end - overlap
    return chunks


def _source_metadata(document, revision: int, *, filename=None, created_at=None, updated_at=None):
    return {"id": document["id"], "doc_id": document["id"], "org_id": document["org_id"],
            "filename": filename, "title": document["title"], "source": document["source"],
            "revision": revision, "content_sha256": _digest(document["content"]),
            "document_unsafe": bool(suspicious_reasons(document["title"] + "\n" + document["source"] + "\n" + document["content"])),
            "created_at": created_at, "updated_at": updated_at}


def _attach(chunks, document):
    unsafe = bool(suspicious_reasons(document["title"] + "\n" + document["source"] + "\n" + document["content"]))
    return [dict(chunk, title=document["title"], source=document["source"], document_unsafe=unsafe) for chunk in chunks]


def get_document(connection, org_id: str, doc_id: str, revision: int | None = None) -> dict:
    """Read the requested immutable revision within the supplied trusted org scope."""
    if revision is not None and (type(revision) is not int or not 1 <= revision <= 2**63 - 1):
        raise HTTPException(404, "知识文档或版本不存在或无权访问。")
    row = connection.execute("SELECT * FROM knowledge_docs WHERE id=? AND org_id=?", (doc_id, org_id)).fetchone()
    if row is None:
        raise HTTPException(404, "知识文档或版本不存在或无权访问。")
    document = dict(row)
    asset = connection.execute("SELECT * FROM knowledge_assets WHERE doc_id=? AND org_id=?", (doc_id, org_id)).fetchone()
    if asset is None:
        if revision not in (None, 1):
            raise HTTPException(404, "知识文档或版本不存在或无权访问。")
        chunks = _attach(chunk_text(document["content"], doc_id=doc_id), document)
        result = _source_metadata(document, 1)
        result.update(content=document["content"], chunks=chunks, chunk_count=len(chunks),
                      imported=False, versions=[{"revision": 1, "title": document["title"],
                          "source": document["source"], "filename": None,
                          "content_sha256": result["content_sha256"], "chunk_count": len(chunks),
                          "created_at": None, "actor_id": None}])
        return result
    revision = asset["current_revision"] if revision is None else revision
    stored = connection.execute("SELECT * FROM knowledge_revisions WHERE doc_id=? AND revision=?", (doc_id, revision)).fetchone()
    if stored is None:
        raise HTTPException(404, "知识文档或版本不存在或无权访问。")
    source = dict(stored, id=doc_id, org_id=org_id)
    chunks = []
    for chunk in connection.execute("SELECT * FROM knowledge_chunks WHERE doc_id=? AND revision=? ORDER BY ordinal", (doc_id, revision)):
        chunks.append(dict(chunk, id=doc_id, title=source["title"], source=source["source"], content_sha256=source["content_sha256"]))
    chunks = _attach(chunks, source)
    result = _source_metadata(source, revision, filename=source["filename"],
                              created_at=source["created_at"], updated_at=asset["updated_at"])
    versions = [dict(item) for item in connection.execute(
        "SELECT revision,filename,title,source,content_sha256,chunk_count,created_at,actor_id FROM knowledge_revisions WHERE doc_id=? ORDER BY revision DESC", (doc_id,))]
    result.update(content=source["content"], chunks=chunks, chunk_count=len(chunks), imported=True,
                  current_revision=asset["current_revision"], versions=versions)
    return result


def list_documents(connection, org_id: str) -> list[dict]:
    """List current source metadata without exposing other organizations or raw text."""
    documents = []
    for row in connection.execute("""SELECT d.*,a.filename,a.current_revision,a.created_at,a.updated_at,r.chunk_count
            FROM knowledge_docs d LEFT JOIN knowledge_assets a ON a.doc_id=d.id AND a.org_id=d.org_id
            LEFT JOIN knowledge_revisions r ON r.doc_id=a.doc_id AND r.revision=a.current_revision
            WHERE d.org_id=? ORDER BY d.id""", (org_id,)):
        item = _source_metadata(row, row["current_revision"] or 1, filename=row["filename"],
                                created_at=row["created_at"], updated_at=row["updated_at"])
        item.update(imported=row["current_revision"] is not None,
                    chunk_count=row["chunk_count"] if row["chunk_count"] is not None else len(chunk_text(row["content"], doc_id=row["id"])))
        documents.append(item)
    return documents


def search_chunks(connection, org_id: str) -> list[dict]:
    """Return current revision chunks only, retaining old seed documents unchanged."""
    # Two bounded reads rather than fetching metadata/versions once per document.
    documents = {row["id"]: dict(row) for row in connection.execute("""SELECT d.*,a.current_revision
        FROM knowledge_docs d LEFT JOIN knowledge_assets a ON a.doc_id=d.id AND a.org_id=d.org_id
        WHERE d.org_id=? ORDER BY d.id""", (org_id,))}
    grouped = {doc_id: [] for doc_id in documents}
    for row in connection.execute("""SELECT c.*,r.content_sha256 FROM knowledge_chunks c
        JOIN knowledge_assets a ON a.doc_id=c.doc_id AND a.current_revision=c.revision
        JOIN knowledge_docs d ON d.id=a.doc_id AND d.org_id=a.org_id
        JOIN knowledge_revisions r ON r.doc_id=c.doc_id AND r.revision=c.revision
        WHERE a.org_id=? ORDER BY c.doc_id,c.ordinal""", (org_id,)):
        grouped[row["doc_id"]].append(dict(row, id=row["doc_id"]))
    chunks = []
    for doc_id, document in documents.items():
        entries = grouped[doc_id] if document["current_revision"] else chunk_text(document["content"], doc_id=doc_id)
        chunks.extend(_attach(entries, document))
    return chunks


def import_document(connection, actor: dict, filename: str, content_bytes: bytes,
                    title: str, source: str, trace_id: str) -> dict:
    """Own the write transaction, rechecking browser authorization under its lock.

    Call on an idle connection. This function commits its own import or denial
    audit; it never commits a caller's pending work. The controller must also
    enforce browser CSRF before invoking this domain operation.
    """
    if connection.in_transaction:
        raise RuntimeError("Knowledge import requires an idle connection.")
    connection.execute("BEGIN IMMEDIATE")
    current = None
    try:
        row = connection.execute("SELECT * FROM users WHERE id=?", (actor.get("id"),)).fetchone()
        current = dict(row) if row else None
        if current is None or not current["active"] or current["role"] != "admin":
            raise HTTPException(403, "仅已启用的组织管理员可导入知识文档。")
        session = connection.execute("SELECT 1 FROM sessions WHERE token_hash=? AND user_id=? AND expires_at>?",
                                     (actor.get("session_hash"), current["id"], time.time())).fetchone()
        if session is None:
            raise HTTPException(401, "请使用有效的独立浏览器会话导入文档。")
        if actor.get("org_id") != current["org_id"]:
            raise HTTPException(403, "组织权限已变化，请重新登录。")
        filename, content, title, source = _input(filename, content_bytes, title, source)
        key = filename.casefold()
        old = connection.execute("SELECT * FROM knowledge_assets WHERE org_id=? AND file_key=?", (current["org_id"], key)).fetchone()
        digest = _digest(content)
        identity = json.dumps([filename, title, source, digest], ensure_ascii=False, separators=(",", ":"))
        fingerprint = _digest(identity)
        now = time.time()
        if old:
            doc_id = old["doc_id"]
            previous = connection.execute("SELECT import_sha256 FROM knowledge_revisions WHERE doc_id=? AND revision=?", (doc_id, old["current_revision"])).fetchone()
            if previous and previous["import_sha256"] == fingerprint:
                audit(connection, current["id"], "knowledge.import", doc_id, "replayed", trace_id)
                result = get_document(connection, current["org_id"], doc_id)
                connection.commit()
                return {key: value for key, value in dict(result, replayed=True).items() if key not in {"content", "chunks", "versions"}}
            revision = old["current_revision"] + 1
        else:
            doc_id = "KD-" + _digest(json.dumps([current["org_id"], key], ensure_ascii=False))[:24].upper()
            if connection.execute("SELECT 1 FROM knowledge_docs WHERE id=?", (doc_id,)).fetchone():
                raise HTTPException(409, "文档标识冲突，请更换文件名后重试。")
            revision = 1
        chunks = chunk_text(content, doc_id=doc_id, revision=revision)
        connection.execute("INSERT INTO knowledge_docs VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,content=excluded.content,source=excluded.source",
                           (doc_id, current["org_id"], title, content, source))
        connection.execute("INSERT INTO knowledge_assets VALUES(?,?,?,?,?,?,?) ON CONFLICT(doc_id) DO UPDATE SET filename=excluded.filename,current_revision=excluded.current_revision,updated_at=excluded.updated_at",
                           (doc_id, current["org_id"], key, filename, revision, now, now))
        connection.execute("INSERT INTO knowledge_revisions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (doc_id, revision, filename, title, source, content, digest, fingerprint, len(chunks), current["id"], now))
        connection.executemany("INSERT INTO knowledge_chunks VALUES(?,?,?,?,?,?,?,?,?,?)", [
            (doc_id, revision, chunk["ordinal"], chunk["chunk_id"], chunk["content"], chunk["heading"],
             chunk["start_offset"], chunk["end_offset"], chunk["start_line"], chunk["end_line"]) for chunk in chunks])
        audit(connection, current["id"], "knowledge.import", doc_id, "ok", trace_id)
        result = get_document(connection, current["org_id"], doc_id)
        connection.commit()
    except HTTPException:
        connection.rollback()
        connection.execute("BEGIN IMMEDIATE")
        audit(connection, current["id"] if current else None, "knowledge.import", None, "denied", trace_id)
        connection.commit()
        raise
    except Exception:
        connection.rollback()
        raise
    return {key: value for key, value in dict(result, replayed=False).items() if key not in {"content", "chunks", "versions"}}
