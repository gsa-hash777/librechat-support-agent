"""Validate evaluation inputs without invoking a model or changing business data."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from string import Formatter

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "evaluation"
FROZEN_PATH = EVALUATION / "agent_cases.json"
DEVELOPMENT_PATH = EVALUATION / "agent_dev_cases.json"
FROZEN = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
DEVELOPMENT = json.loads(DEVELOPMENT_PATH.read_text(encoding="utf-8"))
ALL_CASES = FROZEN["cases"] + DEVELOPMENT["cases"]
FROZEN_SHA256 = "befcb7171e7d9feb7b2d5766fc750a043ff5a51f6be73055caa48e9921c9830b"
COUNTS = {"knowledge": 20, "missing_info": 10, "no_answer": 10, "create": 10, "idempotency": 10, "authorization": 15, "lifecycle": 15, "injection": 10}
ACTORS = {"alice", "bob", "support1", "support2", "admin", "eve", "other_admin"}
FIXTURES = {"none": None, "open": "open", "assigned": "open", "in_progress": "in_progress", "resolved": "resolved", "closed": "closed", "pending_draft": None, "pending_update": "open"}
TOOLS = {"getCurrentSupportUser", "searchSupportKnowledge", "listMyTickets", "getMyTicket", "createTicketDraft", "getMyTicketDraft", "listMyAuditEvents", "listAssignableSupportStaff", "getTicketHistory", "proposeTicketUpdate", "getTicketUpdate"}
PLACEHOLDERS = {"ticket_id", "draft_id", "update_id", "request_id"}
DOC_IDS = {item["id"] for item in json.loads((ROOT / "data" / "knowledge.json").read_text(encoding="utf-8"))}
EXPECT_FIELDS = {"required_tools", "forbidden_tools", "expected_doc_ids", "answer_any_groups", "answer_forbidden", "draft_delta", "update_delta", "ticket_status", "update_action", "draft_request_id", "idempotent", "denial_expected", "min_tool_counts"}
OPTIONAL_EXPECT_FIELDS = {"allowed_tools", "draft_title", "draft_priority", "update_assignee_id"}


def text_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from text_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from text_values(child)


def test_frozen_v1_content_hash_requires_explicit_dataset_version_change():
    assert hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest() == FROZEN_SHA256, "Frozen v1 must not be tuned after observing model answers. Publish a new dataset version for substantive corrections."


def test_splits_have_declared_sizes_categories_and_provenance():
    assert len(FROZEN["cases"]) == 100
    assert len(DEVELOPMENT["cases"]) == 12
    assert Counter(case["category"] for case in FROZEN["cases"]) == COUNTS
    assert FROZEN["split"] == "frozen_regression" and FROZEN["frozen"] is True
    assert DEVELOPMENT["split"] == "development" and DEVELOPMENT["frozen"] is False
    for document in [FROZEN, DEVELOPMENT]:
        assert document["schema_version"] == 1
        assert document["dataset_id"] == "support-agent-synthetic-v1"
        assert "合成" in document["provenance"]
        assert "不是" in document["provenance"] and "盲测" in document["provenance"]


def test_cases_have_distinct_ids_and_prompts_across_splits():
    assert len({case["id"] for case in ALL_CASES}) == len(ALL_CASES)
    assert len({case["prompt"] for case in ALL_CASES}) == len(ALL_CASES)
    assert all(case["id"].startswith("DEV-") for case in DEVELOPMENT["cases"])
    assert not any(case["id"].startswith("DEV-") for case in FROZEN["cases"])


def test_new_tasks_are_not_copies_of_original_retrieval_queries():
    original_queries = {case["query"] for case in json.loads((EVALUATION / "cases.json").read_text(encoding="utf-8"))}
    assert not original_queries.intersection(case["prompt"] for case in ALL_CASES)


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case["id"])
def test_case_contract_and_fixture_are_consistent(case):
    assert {"id", "category", "actor", "fixture", "prompt", "expect"} <= case.keys()
    assert set(case) <= {"id", "category", "actor", "fixture", "prompt", "expect", "turns", "fixture_options"}
    assert case["category"] in COUNTS
    assert case["actor"] in ACTORS
    assert case["fixture"] in FIXTURES
    assert isinstance(case["prompt"], str) and len(case["prompt"]) >= 20
    expect = case["expect"]
    assert EXPECT_FIELDS <= expect.keys()
    assert set(expect) <= EXPECT_FIELDS | OPTIONAL_EXPECT_FIELDS
    for field in ["required_tools", "forbidden_tools", "expected_doc_ids", "answer_forbidden"]:
        assert isinstance(expect[field], list)
        assert all(isinstance(value, str) and value for value in expect[field])
        assert len(expect[field]) == len(set(expect[field]))
    assert set(expect["required_tools"]) <= TOOLS
    assert set(expect["forbidden_tools"]) <= TOOLS
    assert not set(expect["required_tools"]) & set(expect["forbidden_tools"])
    assert set(expect["expected_doc_ids"]) <= DOC_IDS
    assert "EVAL_FORBIDDEN_CANARY" in expect["answer_forbidden"]
    assert isinstance(expect["answer_any_groups"], list)
    assert all(isinstance(group, list) and group and all(isinstance(phrase, str) and phrase for phrase in group) for group in expect["answer_any_groups"])
    assert type(expect["draft_delta"]) is int and expect["draft_delta"] in {0, 1}
    assert type(expect["update_delta"]) is int and expect["update_delta"] in {0, 1}
    assert expect["draft_delta"] + expect["update_delta"] <= 1
    assert type(expect["idempotent"]) is bool and type(expect["denial_expected"]) is bool
    assert expect["ticket_status"] == FIXTURES[case["fixture"]]
    assert expect["update_action"] in {None, "assign", "start", "comment", "resolve", "close"}
    assert expect["draft_request_id"] in {None, "{request_id}"}
    assert isinstance(expect["min_tool_counts"], dict)
    assert set(expect["min_tool_counts"]) <= set(expect["required_tools"])
    assert all(type(count) is int and count >= 1 for count in expect["min_tool_counts"].values())

    if expect["expected_doc_ids"]:
        assert "searchSupportKnowledge" in expect["required_tools"]
    if expect["draft_delta"]:
        assert case["fixture"] == "none"
        assert "createTicketDraft" in expect["required_tools"]
        assert expect["draft_request_id"] == "{request_id}"
        assert isinstance(expect["draft_title"], str) and 2 <= len(expect["draft_title"]) <= 120
        assert expect["draft_title"] in case["prompt"]
        assert expect["draft_priority"] in {"low", "normal", "high"}
        assert expect["draft_priority"] in case["prompt"]
    else:
        assert expect["draft_request_id"] is None
        assert not {"draft_title", "draft_priority"} & expect.keys()
    if expect["update_delta"]:
        assert "proposeTicketUpdate" in expect["required_tools"]
        assert "getMyTicket" in expect["required_tools"]
        assert expect["update_action"] is not None
        assert case["fixture"] not in {"none", "pending_draft", "pending_update", "closed"}
    else:
        assert expect["update_action"] is None
    if expect["update_action"] == "assign":
        assert case["actor"] == "admin"
        assert expect["update_assignee_id"] in {"support1", "support2"}
        assert expect["update_assignee_id"] in case["prompt"]
    else:
        assert "update_assignee_id" not in expect
    if expect["update_action"] == "close":
        assert case["actor"] == "alice" and case["fixture"] == "resolved"
    if expect["update_action"] == "resolve":
        assert case["actor"] in {"support1", "admin"} and case["fixture"] == "in_progress"
    if expect["denial_expected"]:
        assert expect["required_tools"] == [], "Correct early refusal must not require an unauthorized tool call."
        assert expect["draft_delta"] == expect["update_delta"] == 0
    if case["category"] == "missing_info":
        assert expect["draft_delta"] == expect["update_delta"] == 0
        assert {"createTicketDraft", "proposeTicketUpdate"} <= set(expect["forbidden_tools"])
    if expect["idempotent"]:
        assert case["category"] == "idempotency"
        assert expect["draft_delta"] == 1
        assert expect["min_tool_counts"].get("createTicketDraft", 0) >= 2
        assert len(case["turns"]) == 2
        assert case["turns"][0] == case["prompt"]
        assert all("{request_id}" in turn for turn in case["turns"])
    else:
        assert "turns" not in case

    present = set()
    for text in text_values(case):
        for _, field, format_spec, conversion in Formatter().parse(text):
            if field is not None:
                assert field in PLACEHOLDERS and not format_spec and conversion is None
                present.add(field)
    if "ticket_id" in present:
        assert case["fixture"] not in {"none", "pending_draft"}
    if "draft_id" in present:
        assert case["fixture"] == "pending_draft"
    if "update_id" in present:
        assert case["fixture"] == "pending_update"

    options = case.get("fixture_options", {})
    assert set(options) <= {"inject_document", "inject_ticket_note"}
    if options:
        assert case["category"] == "injection"
    if "inject_document" in options:
        document = options["inject_document"]
        assert set(document) == {"id", "title", "content", "source", "org_id"}
        assert document["org_id"] == "demo"
        assert document["id"] not in DOC_IDS
        assert all(isinstance(value, str) and value for value in document.values())
        assert "EVAL_FORBIDDEN_CANARY" in document["content"]
    if "inject_ticket_note" in options:
        assert case["fixture"] not in {"none", "pending_draft"}
        assert "EVAL_FORBIDDEN_CANARY" in options["inject_ticket_note"]


def test_injection_cases_include_retrieved_data_and_ticket_content_attacks():
    injection = [case for case in FROZEN["cases"] if case["category"] == "injection"]
    assert sum("inject_document" in case.get("fixture_options", {}) for case in injection) >= 4
    assert sum("inject_ticket_note" in case.get("fixture_options", {}) for case in injection) >= 2
    assert any(case["expect"]["denial_expected"] for case in injection)
