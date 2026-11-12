"""Pinned, offline-first IBM Cloud benchmark data; never imports the application.

The public corpus is real product documentation. Questions were authored by
annotators, not collected from production customer tickets. Answerability is
conditional on reference passages, not a claim about the entire document corpus.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import urlopen
from zipfile import ZipFile

REPOSITORY = "https://github.com/IBM/mt-rag-benchmark"
COMMIT = "e0970de1b1207c2a9506adf238e6e1a22037e38a"
RAW = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/{COMMIT}/"
SERVICE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.environ.get("PUBLIC_DATA_PATH", str(SERVICE.parents[1] / ".local/public-benchmark/mtrag")
                                  if SERVICE.parent.name == "extensions" else "/app/public-data"))
CORPUS_FILE = "corpora/passage_level/cloud.jsonl.zip"
TASK_FILES = {"dev": "mtrag-human/generation_tasks/reference.jsonl",
              "final": "mtragun-human/generation_tasks/reference.jsonl"}
QREL_FILES = {"dev": "mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv",
              "final": "mtragun-human/retrieval_tasks/qrels/cloud.tsv"}
LICENSE_STATEMENT = "https://ibm.github.io/mt-rag-benchmark/MT_RAG_SemEval_Proposal.pdf"
PINNED_FILES = {
    "LICENSE": (11357, "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
    "README.md": (7461, "b27a82e3c524a30a04000a79067b7049724f87262f1779009b525633b94d6720"),
    "corpora/README.md": (1241, "d2ebebf563df3218fe01a53dd6bfac686bbe05b6d6e1d59b1b3ca56cceedc985"),
    CORPUS_FILE: (21310415, "a0c5427013c65556aeddfdc246dd6068f3a7fad08ce6ae7adddaa9e87aae246d"),
    TASK_FILES["dev"]: (7608375, "b3dde4076f9f692a99513e424a0a39f57591dbe45600a4c1017aaa7a7edcfaac"),
    TASK_FILES["final"]: (3836136, "68053f709c928570718fd2c88455eccf974746f5bfa2206d7fd39b969acb21d4"),
    QREL_FILES["dev"]: (30951, "925635635bf0e60b20d8ea0b18160cff878897acf8e0bf54e7096cc0ade8bd3b"),
    QREL_FILES["final"]: (15719, "4c9e5903b0c1a702ade78f3f3bb826712614d31095f40c91121a214de6d248bf"),
    "mtrag-human/retrieval_tasks/cloud/cloud_lastturn.jsonl":
        (20870, "85851a3d24fd09d07b6817ad023604f42ed3313af34dc95431c039890ddda1cb"),
    "provenance/MT_RAG_SemEval_Proposal.pdf":
        (149481, "11ec5bc8505eb1a87ca67c835d5a3ad1376d7536febf5c137f2ceef2d65d8c75"),
}
ANSWERABILITY = {"ANSWERABLE", "PARTIAL", "UNANSWERABLE", "CONVERSATIONAL", "UNDERSPECIFIED"}
INPUT_KEYS = {"task_id", "conversation_id", "turn", "messages"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_url(name: str) -> str:
    if name not in PINNED_FILES:
        raise ValueError("File is not part of the frozen dataset.")
    return LICENSE_STATEMENT if name.startswith("provenance/") else RAW + name


def verify_file(root: Path, name: str) -> dict:
    if name not in PINNED_FILES:
        raise ValueError("Unpinned data file.")
    path = Path(root) / name
    expected_bytes, expected_hash = PINNED_FILES[name]
    if not path.is_file() or path.stat().st_size != expected_bytes or sha256(path) != expected_hash:
        raise ValueError(f"Dataset fingerprint mismatch: {name}")
    return {"bytes": expected_bytes, "sha256": expected_hash, "url": source_url(name)}


def fingerprint(root: Path = DEFAULT_ROOT) -> dict:
    return {"repository": REPOSITORY, "commit": COMMIT, "license": "Apache-2.0",
            "data_license_statement": LICENSE_STATEMENT,
            "files": {name: verify_file(root, name) for name in PINNED_FILES}}


def prepare_data(source: Path | None = None, destination: Path = DEFAULT_ROOT,
                 download_missing: bool = False) -> dict:
    """Copy verified bytes, preserving originals; downloads require an explicit flag."""
    destination = Path(destination)
    for name, (size, checksum) in PINNED_FILES.items():
        target = destination / name
        if target.exists():
            verify_file(destination, name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source is not None and (Path(source) / name).is_file():
            verify_file(source, name)
            shutil.copyfile(Path(source) / name, target)
        elif download_missing:
            with urlopen(source_url(name), timeout=120) as response:
                data = response.read(size + 1)
            if len(data) != size or hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError(f"Downloaded data fingerprint mismatch: {name}")
            with target.open("xb") as stream:
                stream.write(data)
        else:
            raise FileNotFoundError(f"Missing pinned input: {name}; provide --source or --download-missing")
        verify_file(destination, name)
    result = fingerprint(destination)
    result.update({"schema_version": 1, "source_data_unmodified": True,
                   "question_origin": "human-authored benchmark conversations, not customer tickets",
                   "answerability_scope": "reference_context_not_entire_corpus",
                   "metadata_note": "Upstream README corpus counts are inconsistent; use verified file counts."})
    manifest = destination / "freeze.json"
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if manifest.exists() and manifest.read_text(encoding="utf8") != payload:
        raise ValueError("Existing data freeze differs; use a new destination.")
    if not manifest.exists():
        manifest.write_text(payload, encoding="utf8")
    return result


def _split(split: str, allow_final: bool) -> None:
    if split not in TASK_FILES:
        raise ValueError("split must be dev or final")
    if split == "final" and not allow_final:
        raise ValueError("Final inputs are sealed; enable allow_final only after development freeze.")


def read_jsonl(path: Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at line {number}.")
                yield row


def is_cloud(row: dict) -> bool:
    value = row.get("Collection", row.get("collection"))
    return value in {"ibmcloud", "mt-rag-ibmcloud-elser-512-100-20240502"}


def task_labels(row: dict) -> dict:
    labels = row.get("Answerability", row.get("answerability"))
    if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in ANSWERABILITY:
        raise ValueError("Unsupported answerability annotation.")
    return {"answerability": labels[0], "multi_turn": row.get("Multi-Turn", []),
            "question_type": row.get("Question Type", [])}


def task_input(row: dict) -> dict:
    """Allowlist projection. Never exposes targets, contexts or annotation metadata."""
    identifier, conversation = row.get("task_id"), row.get("conversation_id")
    turn = row.get("turn")
    if not isinstance(identifier, str) or not identifier or not isinstance(conversation, str) or not conversation:
        raise ValueError("Task identifiers are required.")
    if isinstance(turn, bool) or not str(turn).isdigit() or int(turn) < 1:
        raise ValueError("Task turn must be a positive integer.")
    messages = []
    for message in row.get("input", []):
        role, text = message.get("speaker"), message.get("text")
        if role not in {"user", "agent", "assistant"} or not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid conversation input message.")
        messages.append({"speaker": "assistant" if role == "agent" else role, "text": text})
    if not messages or messages[-1]["speaker"] != "user":
        raise ValueError("A task must end with the current user message.")
    return {"task_id": identifier, "conversation_id": conversation,
            "turn": int(turn), "messages": messages}


def load_cloud_tasks(root: Path = DEFAULT_ROOT, split: str = "dev", *,
                     allow_final: bool = False) -> list[dict]:
    _split(split, allow_final)
    verify_file(root, TASK_FILES[split])
    tasks, seen = [], set()
    for row in read_jsonl(Path(root) / TASK_FILES[split]):
        if not is_cloud(row):
            continue
        item = task_input(row)
        if item["task_id"] in seen:
            raise ValueError("Duplicate benchmark task identifier.")
        task_labels(row)
        seen.add(item["task_id"])
        tasks.append(item)
    return tasks


def load_labels(root: Path = DEFAULT_ROOT, split: str = "dev", *,
                allow_final: bool = False) -> dict:
    _split(split, allow_final)
    verify_file(root, TASK_FILES[split])
    return {row["task_id"]: task_labels(row) for row in read_jsonl(Path(root) / TASK_FILES[split]) if is_cloud(row)}


def load_qrels(root: Path = DEFAULT_ROOT, split: str = "dev", *,
               allow_final: bool = False) -> dict[str, dict[str, int]]:
    _split(split, allow_final)
    verify_file(root, QREL_FILES[split])
    return parse_qrels(Path(root) / QREL_FILES[split])


def parse_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        if rows.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError("Unexpected BEIR qrels header.")
        for row in rows:
            query, passage = row["query-id"], row["corpus-id"]
            if not query or not passage or not row["score"].isdigit():
                raise ValueError("Invalid qrel.")
            grade = int(row["score"])
            values = qrels.setdefault(query, {})
            if passage in values:
                raise ValueError("Duplicate qrel pair.")
            values[passage] = grade
    return qrels


def corpus_rows(root: Path = DEFAULT_ROOT) -> Iterable[dict]:
    """Stream the exact published passages; do not rechunk or silently drop rows."""
    verify_file(root, CORPUS_FILE)
    with ZipFile(Path(root) / CORPUS_FILE) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != "cloud.jsonl":
            raise ValueError("Unexpected corpus archive members.")
        with archive.open(members[0]) as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("id") != row.get("_id") or not isinstance(row.get("id"), str):
                    raise ValueError("Ambiguous passage identifier.")
                if not isinstance(row.get("text"), str):
                    raise ValueError("Corpus passage text must be a string.")
                if not isinstance(row.get("url"), str):
                    raise ValueError("Unexpected corpus source URL.")
                source = urlparse(row["url"])
                if source.scheme != "https" or source.netloc != "cloud.ibm.com" or not (source.path == "/docs" or source.path.startswith("/docs/")):
                    raise ValueError("Unexpected corpus source URL.")
                yield {"id": row["id"], "text": row["text"], "url": row["url"], "title": row.get("title", "")}


def iter_corpus(root: Path = DEFAULT_ROOT) -> Iterable[dict]:
    """PublicSearch index adapter, maintaining the official ID and full passage."""
    for row in corpus_rows(root):
        yield {"passage_id": row["id"], "text": row["text"], "source_url": row["url"],
               "sha256": hashlib.sha256(row["text"].encode("utf8")).hexdigest()}


def select_dev8(root: Path = DEFAULT_ROOT) -> dict:
    """Deterministic label-only pilot; questions and reference answers do not select cases."""
    tasks = load_cloud_tasks(root, "dev")
    labels = load_labels(root, "dev")
    selected, used_conversations = [], set()
    for category, count in (("ANSWERABLE", 4), ("PARTIAL", 2), ("UNANSWERABLE", 2)):
        candidates = [task for task in tasks if labels[task["task_id"]]["answerability"] == category]
        for _ in range(count):
            candidates.sort(key=lambda task: (task["conversation_id"] in used_conversations, task["task_id"]))
            if not candidates:
                raise ValueError("Insufficient pilot category coverage.")
            task = candidates.pop(0)
            used_conversations.add(task["conversation_id"])
            selected.append({key: task[key] for key in ("task_id", "conversation_id", "turn")})
            selected[-1]["answerability"] = category
    return {"schema_version": 1, "split": "dev", "commit": COMMIT,
            "selection": "label-only 4 answerable / 2 partial / 2 unanswerable; prefer distinct conversations then lexical task ID",
            "source_sha256": verify_file(root, TASK_FILES["dev"])["sha256"],
            "task_ids": [task["task_id"] for task in selected], "tasks": selected,
            "conversations": len(used_conversations), "final_data_used": False}


def inspect_data(root: Path = DEFAULT_ROOT) -> dict:
    """Integrity-only check, including final data; returns no question or answer text."""
    result = fingerprint(root)
    passages, urls, empty_passages = set(), set(), set()
    for row in corpus_rows(root):
        if row["id"] in passages:
            raise ValueError("Duplicate corpus passage identifier.")
        passages.add(row["id"])
        urls.add(row["url"])
        if not row["text"].strip():
            empty_passages.add(row["id"])
    result["corpus"] = {"passages": len(passages), "source_urls": len(urls),
                        "empty_text_passages_preserved": len(empty_passages), "empty_text_ids": sorted(empty_passages)}
    result["splits"] = {}
    for split in TASK_FILES:
        # This administrative scan deliberately produces only counts and integrity facts.
        tasks = load_cloud_tasks(root, split, allow_final=True)
        labels = load_labels(root, split, allow_final=True)
        qrels = load_qrels(root, split, allow_final=True)
        if set(qrels) - {task["task_id"] for task in tasks}:
            raise ValueError("Qrel query absent from Cloud tasks.")
        if any(set(values) - passages for values in qrels.values()):
            raise ValueError("Qrel passage absent from the frozen full corpus.")
        eligible = {key for key, value in labels.items() if value["answerability"] in {"ANSWERABLE", "PARTIAL"}}
        if set(qrels) != eligible or any(not any(g > 0 for g in values.values()) for values in qrels.values()):
            raise ValueError("Retrieval qrels and answerable/partial task coverage disagree.")
        result["splits"][split] = {"tasks": len(tasks), "conversations": len({x["conversation_id"] for x in tasks}),
                                   "answerability": dict(Counter(x["answerability"] for x in labels.values())),
                                   "retrieval_tasks": len(qrels), "qrel_pairs": sum(map(len, qrels.values())),
                                   "empty_text_qrel_passages": len(empty_passages & {p for values in qrels.values() for p in values})}
    result.update({"paid_model_calls": 0, "performance_evaluation_run": False,
                   "answerability_scope": "reference_context_not_entire_corpus"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare_data(args.source, args.root, args.download_missing)
    print(json.dumps(inspect_data(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
