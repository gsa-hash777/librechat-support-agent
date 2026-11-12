"""Reproducible service and retrieval checks; never calls an LLM or real tickets."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.main import create_app
from app.search import tokenize


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def overlap_baseline(documents: list[dict], query: str, limit: int) -> list[str]:
    """Unweighted unique-token overlap; intentionally simple, explicitly named baseline."""
    terms = set(tokenize(query))
    scores = [(len(terms & set(tokenize(document["title"] + " " + document["content"]))), document["id"]) for document in documents]
    return [identifier for score, identifier in sorted(scores, key=lambda value: (-value[0], value[1])) if score > 0][:limit]


def relevance_metrics(rows: list[dict], result_field: str, cutoff: int) -> dict:
    answerable = [row for row in rows if row["expected_ids"]]
    no_evidence = [row for row in rows if not row["expected_ids"]]
    hits, recalls = [], []
    for row in answerable:
        relevant = set(row["expected_ids"])
        retrieved = set(row[result_field][:cutoff])
        covered = len(relevant & retrieved)
        hits.append(int(covered > 0))
        recalls.append(covered / len(relevant))
    return {
        "answerable_cases": len(answerable),
        "hit_at_k": statistics.mean(hits) if hits else None,
        "macro_recall_at_k": statistics.mean(recalls) if recalls else None,
        "no_evidence_cases": len(no_evidence),
        "empty_result_rate_on_no_evidence": statistics.mean(not row[result_field] for row in no_evidence) if no_evidence else None,
    }


def run_service_tests(work: Path) -> dict:
    xml_path = work / "pytest.xml"
    command = [sys.executable, "-X", "utf8", "-m", "pytest", str(ROOT / "tests"), str(ROOT / "integration"), "-q", f"--junitxml={xml_path}", f"--basetemp={work / 'pytest-temp'}"]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    groups = {"service": {"tests": 0, "passed": 0}, "integration": {"tests": 0, "passed": 0}}
    cases = []
    if xml_path.exists():
        suites = ET.parse(xml_path).getroot()
        for suite in suites.iter("testsuite"):
            for name in totals:
                totals[name] += int(suite.attrib.get(name, 0))
        for case in suites.iter("testcase"):
            class_name = case.attrib.get("classname", "")
            group = "integration" if class_name.startswith("integration.") else "service"
            outcome = next((name for name in ("failure", "error", "skipped") if case.find(name) is not None), "passed")
            groups[group]["tests"] += 1
            groups[group]["passed"] += int(outcome == "passed")
            cases.append({"class": class_name, "name": case.attrib["name"], "outcome": outcome})
    return {"status": "passed" if completed.returncode == 0 else "failed", "exit_code": completed.returncode, **totals, "groups": groups, "cases": cases, "duration_seconds": round(time.perf_counter() - started, 3)}


def load_cases(path: Path, known_ids: set[str]) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("The cases file must contain a nonempty list.")
    identifiers = set()
    for case in cases:
        if not isinstance(case.get("id"), str) or case["id"] in identifiers:
            raise ValueError("Every evaluation case needs a unique string id.")
        identifiers.add(case["id"])
        if not isinstance(case.get("query"), str) or not 2 <= len(case["query"]) <= 300:
            raise ValueError(f"Invalid query length for {case['id']}.")
        expected = case.get("expected_ids")
        if not isinstance(expected, list) or any(not isinstance(value, str) for value in expected):
            raise ValueError(f"expected_ids must be a list of strings for {case['id']}.")
        if len(set(expected)) != len(expected) or not set(expected) <= known_ids:
            raise ValueError(f"Unknown or duplicate evidence IDs for {case['id']}.")
    return cases


def evaluate(args, work: Path) -> dict:
    corpus_path = ROOT / "data" / "knowledge.json"
    documents = [document for document in json.loads(corpus_path.read_text(encoding="utf-8")) if document.get("org_id", "demo") == "demo"]
    cases = load_cases(args.cases, {document["id"] for document in documents})
    api_key = "evaluation-only-api-key-" + "e" * 32
    credentials = work / "users.json"
    credentials.write_text(json.dumps({"users": [{"id": "evaluation", "display_name": "Evaluation fixture", "org_id": "demo", "password": "evaluation-only-password", "api_key": api_key}]}), encoding="utf-8")
    settings = Settings(db_path=work / "support.db", credentials_path=credentials, public_base_url="http://testserver", tool_base_url="http://testserver", seed_path=corpus_path)
    headers = {"Authorization": "Bearer " + api_key}
    rows = [{**case, "baseline_ids": overlap_baseline(documents, case["query"], args.k), "retrieved_ids": [], "samples_ms": [], "http_statuses": [], "stable_ranking": True} for case in cases]
    warmup_count = min(args.warmup, len(cases))
    with TestClient(create_app(settings)) as client:
        for case in cases[:warmup_count]:
            client.get("/api/knowledge/search", headers=headers, params={"q": case["query"], "limit": args.k})
        sequence = list(range(len(cases))) * args.repeats
        random.Random(20261109).shuffle(sequence)
        for index in sequence:
            row = rows[index]
            started = time.perf_counter()
            response = client.get("/api/knowledge/search", headers=headers, params={"q": row["query"], "limit": args.k})
            elapsed = (time.perf_counter() - started) * 1000
            row["samples_ms"].append(round(elapsed, 6))
            row["http_statuses"].append(response.status_code)
            retrieved = [item["id"] for item in response.json().get("results", [])] if response.status_code == 200 else []
            if len(row["http_statuses"]) == 1:
                row["retrieved_ids"] = retrieved
            elif retrieved != row["retrieved_ids"]:
                row["stable_ranking"] = False
    all_latencies = [sample for row in rows for sample in row["samples_ms"]]
    failures = sum(status != 200 for row in rows for status in row["http_statuses"])
    return {
        "corpus": {"total_documents": len(documents), "org_id": "demo", "sha256": digest(corpus_path)},
        "cases": {"count": len(cases), "sha256": digest(args.cases), "scope": "hand-authored synthetic smoke cases, not a held-out production benchmark"},
        "protocol": {"k": args.k, "repeats": args.repeats, "warmup_requests_excluded": warmup_count, "concurrency": 1, "timing_scope": "in-process FastAPI TestClient request completion; includes auth, SQLite, retrieval, audit and client overhead; excludes startup and network", "p95_method": "nearest rank ceil(0.95*n)", "seed": 20261109, "model_calls": 0, "cost_measurement": "not measured; no LLM/provider call"},
        "bm25": relevance_metrics(rows, "retrieved_ids", args.k),
        "token_overlap_baseline": relevance_metrics(rows, "baseline_ids", args.k),
        "latency_ms": {"count": len(all_latencies), "p50": round(statistics.median(all_latencies), 3), "p95": round(percentile(all_latencies, 0.95), 3), "min": round(min(all_latencies), 3), "max": round(max(all_latencies), 3), "non_200_requests": failures},
        "rows": rows,
    }


def format_percent(value) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def markdown(report: dict) -> str:
    retrieval = report["retrieval"]
    tests = report["service_tests"]
    k = retrieval["protocol"]["k"]
    lines = [
        "# 服务与检索评测结果", "", f"运行时间（UTC）：{report['generated_at_utc']}", "",
        "本报告使用模拟知识与独立临时数据库，没有调用大模型，不代表 LibreChat Agent 的端到端任务成功率或生产环境性能。", "",
        f"服务与集成脚本测试：{tests['status']}；用例 {tests.get('tests', 'N/A')}，失败 {tests.get('failures', 'N/A')}，错误 {tests.get('errors', 'N/A')}，跳过 {tests.get('skipped', 'N/A')}。", "",
        f"分组：服务测试 {tests.get('groups', {}).get('service', {}).get('passed', 'N/A')}/{tests.get('groups', {}).get('service', {}).get('tests', 'N/A')}；集成脚本测试 {tests.get('groups', {}).get('integration', {}).get('passed', 'N/A')}/{tests.get('groups', {}).get('integration', {}).get('tests', 'N/A')}。", "",
        f"知识文档 {retrieval['corpus']['total_documents']} 份；手工编写的冒烟查询 {retrieval['cases']['count']} 条；每条重复 {retrieval['protocol']['repeats']} 次；并发数 1。", "",
        f"| 检索算法 | Hit@{k} | Macro Recall@{k} | 无证据用例返回空结果比例 |", "|---|---:|---:|---:|",
        f"| BM25（含标题加权） | {format_percent(retrieval['bm25']['hit_at_k'])} | {format_percent(retrieval['bm25']['macro_recall_at_k'])} | {format_percent(retrieval['bm25']['empty_result_rate_on_no_evidence'])} |",
        f"| 简单不重复词项重合计数基线 | {format_percent(retrieval['token_overlap_baseline']['hit_at_k'])} | {format_percent(retrieval['token_overlap_baseline']['macro_recall_at_k'])} | {format_percent(retrieval['token_overlap_baseline']['empty_result_rate_on_no_evidence'])} |", "",
        f"服务请求完成耗时：P50 {retrieval['latency_ms']['p50']} ms；P95 {retrieval['latency_ms']['p95']} ms；样本 {retrieval['latency_ms']['count']}；非 200 响应 {retrieval['latency_ms']['non_200_requests']}。", "",
        "计时包含进程内 TestClient、身份校验、SQLite、检索与审计写入；不含服务启动、真实网络、LibreChat 或模型推理。失败响应仍计入耗时。P95 使用 nearest-rank；原始样本见 JSON。", "",
        "相关文档集合为手工标注的文档级证据；Hit 检查是否至少命中一份，Recall 检查所需证据覆盖比例。空结果比例仅检查检索器输出，不能推断模型拒答正确率。", "",
        "查询集与语料随源码公开，目前是功能冒烟样本；继续基于它调参后，不应将其称为独立盲测。端到端 Agent 评测还需要实际接入模型并核验回答与最终工单状态。", "",
        "## 可复现信息", "", f"- Python：{report['environment']['python']}", f"- 平台：{report['environment']['platform']}", f"- 语料 SHA256：`{retrieval['corpus']['sha256']}`", f"- 查询 SHA256：`{retrieval['cases']['sha256']}`", f"- 服务代码 SHA256：`{report['source_sha256']}`", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=HERE / "cases.json")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--skip-tests", action="store_true", help="Only measure retrieval; service tests will be marked not_run.")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or not 1 <= args.k <= 10:
        parser.error("repeats >= 1, warmup >= 0 and 1 <= k <= 10 are required.")
    workspace = (HERE / "work").resolve()
    assert workspace.is_relative_to(HERE.resolve())
    workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run-", dir=workspace) as directory:
        work = Path(directory)
        tests = {"status": "not_run"} if args.skip_tests else run_service_tests(work)
        retrieval = evaluate(args, work)
        source = hashlib.sha256()
        for path in sorted((ROOT / "app").glob("*.py")):
            source.update(path.name.encode("utf-8"))
            source.update(path.read_bytes())
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": {name: importlib.metadata.version(name) for name in ["fastapi", "httpx", "pytest"]}},
            "source_sha256": source.hexdigest(),
            "evaluator_sha256": digest(Path(__file__)),
            "service_tests": tests,
            "retrieval": retrieval,
            "limitations": ["synthetic smoke evaluation", "no LLM invocation", "no end-to-end Agent success claim", "no production/network/load latency claim", "no measured inference cost"],
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "report.md").write_text(markdown(report), encoding="utf-8")
    print(f"Report: {(args.output / 'report.json').resolve()}")
    print(f"Retrieval: Hit@{args.k}={format_percent(retrieval['bm25']['hit_at_k'])}, Macro Recall@{args.k}={format_percent(retrieval['bm25']['macro_recall_at_k'])}")
    print(f"In-process latency: P50={retrieval['latency_ms']['p50']} ms, P95={retrieval['latency_ms']['p95']} ms")
    return int(tests["status"] == "failed" or retrieval["latency_ms"]["non_200_requests"] > 0 or not all(row["stable_ranking"] for row in retrieval["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
