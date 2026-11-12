"""Offline, value-preserving normalization of two registered judge JSON defects.

No Agent/model calls, no criterion changes, no edits to a frozen run. The raw
strict-parser result remains available beside this separate audit result.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

SERVICE = Path(r"E:\LibreChat\extensions\support-agent")
sys.path.insert(0, str(SERVICE))
from evaluation.task_success import DIMENSIONS, combine, parse_judge_response

VERSION = "judge-json-syntax-audit-v1"
PROTOCOL = Path(__file__).with_name("judge-format-audit-protocol.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def loads(text):
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))


def only_outer_brace_missing(text):
    """Reject open strings, unmatched brackets and anything except one root }."""
    stack, quoted, escaped = [], False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack.pop() != ("{" if char == "}" else "["):
                return False
    return not quoted and not escaped and stack == ["{"]


def normalize(raw, finish_reason):
    """Return a new string only for the two enumerated structural repairs."""
    if finish_reason != "stop" or not isinstance(raw, str) or len(raw) > 24000:
        return None
    text, operations = raw.strip(), []
    try:
        value = loads(text)
    except json.JSONDecodeError:
        if not only_outer_brace_missing(text):
            return None
        try:
            value = loads(text + "}")
        except (ValueError, RecursionError):
            return None
        operations.append("append_one_missing_outer_closing_brace")
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, dict):
        return None
    original_dimensions = value.get("dimensions")
    if (set(value) == {"dimensions"} and isinstance(original_dimensions, dict)
            and set(original_dimensions) == set(DIMENSIONS) | {"reasons"}):
        value = {"dimensions": {key: original_dimensions[key] for key in DIMENSIONS},
                 "reasons": original_dimensions["reasons"]}
        operations.append("move_existing_reasons_from_dimensions_to_root")
    if not operations:
        return None
    normalized = canonical(value)
    semantic = parse_judge_response(normalized)
    # The original parser must accept every existing dimension and evidence.
    # It may still assign uncertain/unscored. Never fill a missing field.
    if set(semantic.get("dimensions", {})) != set(DIMENSIONS):
        return None
    original_values = {key: original_dimensions[key] for key in DIMENSIONS}
    if value["dimensions"] != original_values:
        raise AssertionError("A semantic value changed.")
    return {"operations": operations, "normalized_json": normalized,
            "semantic": semantic, "dimension_values_unchanged": True}


def audit_row(row):
    before = row["outcome"]
    semantic_before = row.get("semantic", {})
    record = row.get("judge_record") or {}
    raw = record.get("raw_response_redacted")
    repair = normalize(raw, record.get("finish_reason"))
    # Valid original results are never reconsidered; all rows use the same rule.
    eligible = semantic_before.get("status") == "unscored" and repair is not None
    semantic_after = repair["semantic"] if eligible else semantic_before
    after = combine(row["hard"], semantic_after) if eligible else before
    return {"case_id": row["case_id"], "category": row["category"],
            "before": before, "after": after,
            "semantic_before": semantic_before, "semantic_after": semantic_after,
            "format_repair_applied": eligible,
            "operations": repair["operations"] if eligible else [],
            "dimension_values_unchanged": True,
            "raw_redacted_sha256": hashlib.sha256(raw.encode()).hexdigest() if isinstance(raw, str) else None,
            "normalized_json": repair["normalized_json"] if eligible else None,
            "finish_reason": record.get("finish_reason")}


def register():
    protocol = {"version": VERSION, "registered_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(__file__),
        "selection_scope": "Evaluator syntax repair, registered before all 100 executions completed; not Agent tuning.",
        "observed_examples": ["K04", "K05", "K08", "K19", "M09"],
        "allowed_repairs": ["At most one missing root closing brace, only with complete strings and all inner brackets closed.",
                            "Move an existing reasons list from the dimensions object to root when these are exactly the four dimensions plus reasons."],
        "unchanged": ["Original Agent responses, calls, snapshots and single-attempt counts.",
                      "All four existing dimension status/reason/evidence values and existing reasons list.",
                      "Original strict parser and task acceptance criteria.",
                      "All original run artifacts and every paid call in the shared budget ledger."],
        "rejected": ["finish_reason other than stop", "missing dimensions or evidence", "duplicate keys",
                     "invalid status values", "extra keys", "more than one missing closing bracket",
                     "unfinished strings", "all other malformed structures"],
        "application": "Apply once uniformly to all 100 saved rows; retain before/after, including fail and uncertain. No model calls or retries."}
    with PROTOCOL.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)
    return protocol


def run(folder, output):
    protocol = loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["script_sha256"] != sha(__file__):
        raise ValueError("Registered audit implementation changed.")
    folder, output = Path(folder), Path(output)
    rows = loads((folder / "results.json").read_text(encoding="utf-8"))
    manifest = loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed" or len(rows) != 100
            or [row["case_id"] for row in rows] != manifest["case_ids"]):
        raise ValueError("Only a complete, unique, frozen 100-task run is auditable.")
    actual_parser = SERVICE / "evaluation/task_success.py"
    frozen_parser = folder / "frozen/extensions/support-agent/evaluation/task_success.py"
    if sha(actual_parser) != sha(frozen_parser):
        raise ValueError("The original strict parser source changed.")
    audited = [audit_row(row) for row in rows]
    raw = Counter(item["before"]["status"] for item in audited)
    normalized = Counter(item["after"]["status"] for item in audited)
    repairs = [item for item in audited if item["format_repair_applied"]]
    def counts(counter):
        return {"success": counter["pass"], "failure": counter["fail"], "unscored": counter["unscored"],
                "planned": 100, "confirmed_task_success_rate": counter["pass"] / 100}
    categories = {}
    for category in sorted({row["category"] for row in rows}):
        members = [item for item in audited if item["category"] == category]
        counter = Counter(item["after"]["status"] for item in members)
        categories[category] = {"planned": len(members), "success": counter["pass"],
            "failure": counter["fail"], "unscored": counter["unscored"], "task_success_rate": counter["pass"] / len(members)}
    report = {"version": VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(folder.resolve()), "source_manifest_sha256": sha(folder / "manifest.json"),
        "source_results_sha256": sha(folder / "results.json"), "protocol_path": str(PROTOCOL),
        "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(__file__),
        "strict_parser_sha256": sha(actual_parser), "paid_calls": 0,
        "raw_summary": counts(raw), "normalized_summary": counts(normalized),
        "raw_semantic_schema_coverage": sum(bool(item["semantic_before"].get("dimensions")) for item in audited) / 100,
        "normalized_semantic_schema_coverage": sum(bool(item["semantic_after"].get("dimensions")) for item in audited) / 100,
        "format_repairs": len(repairs), "repaired_case_ids": [item["case_id"] for item in repairs],
        "outcome_changed_case_ids": [item["case_id"] for item in audited if item["before"]["status"] != item["after"]["status"]],
        "categories": categories, "cases": audited,
        "limitations": ["Same frozen synthetic seen regression tasks and model-assisted rubric; not official benchmark or human verification.",
                        "A formatting-only evaluator repair, not an improvement in Agent behavior.",
                        "Any remaining uncertain or invalid result is unscored and remains in the denominator.",
                        "Values are preserved from the saved redacted judge response; original run judgments remain intact."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return {key: report[key] for key in ("raw_summary", "normalized_summary", "format_repairs", "repaired_case_ids")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.register:
        print(canonical(register()))
    elif args.run_dir and args.output:
        print(canonical(run(args.run_dir, args.output)))
    else:
        parser.error("Choose --register or --run-dir with --output.")
