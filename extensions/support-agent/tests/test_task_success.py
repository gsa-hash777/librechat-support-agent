"""TSR tamper, outcome, provenance and semantic-schema tests; no model calls."""
from copy import deepcopy
import json
import re

import pytest

from app.knowledge import chunk_text
from evaluation import task_success as tsr
from evaluation.agent import render_case
from evaluation.judge import REQUIRED_TABLES


IDS = {"ticket_id": "T-0123456789ABCDEF", "draft_id": "D-0123456789ABCDEF",
       "update_id": "U-0123456789ABCDEF", "request_id": "eval-0123456789abcdef"}


def packet(case_id="M01"):
    original = next(case for case in json.loads(tsr.TASK_PATH.read_text(encoding="utf-8"))["cases"] if case["id"] == case_id)
    case = render_case(original, IDS)
    rows = {table: [] for table in REQUIRED_TABLES}
    rows["knowledge_docs"] = []
    rows["users"] = [{"id": name, "org_id": "other" if name in ("eve", "other_admin") else "demo",
                      "role": "admin" if "admin" in name else "support" if name.startswith("support") else "requester", "active": 1}
                     for name in ("alice", "bob", "support1", "support2", "admin", "eve", "other_admin")]
    snapshot = {"rows": rows, "forbidden_canaries": [tsr.FORBIDDEN_MARKER]}
    result = {"case_id": case_id, "case": case,
              "evidence": {"answers": [{"text": "请补充具体现象与影响范围。", "error": False}],
                           "intermediate_answers": [], "tool_calls": [], "execution_error": None},
              "fixture_before": snapshot, "fixture_after": deepcopy(snapshot)}
    return result


def proposal_packet():
    result = packet("P06")
    ticket = {"id": IDS["ticket_id"], "user_id": "alice", "org_id": "demo", "status": "open", "version": 1, "assignee_id": None}
    proposal = {"id": IDS["update_id"], "user_id": "alice", "ticket_id": IDS["ticket_id"], "status": "pending", "expected_version": 1, "action": "comment"}
    for name in ("fixture_before", "fixture_after"):
        result[name]["rows"]["tickets"] = [deepcopy(ticket)]
        result[name]["rows"]["ticket_updates"] = [deepcopy(proposal)]
    result["evidence"]["answers"][0]["text"] = "提案仍待本人网页确认，尚未执行。"
    return result, proposal


def knowledge_packet():
    result = packet("K01")
    reference = tsr.load_references()["reference_sources"]["KB-001"]
    source = {"id": "KB-001", "org_id": "demo", **{key: reference[key] for key in ("title", "content", "source")}}
    chunks = chunk_text(source["content"], doc_id="KB-001", revision=1)
    for chunk in chunks:
        chunk.update(title=source["title"], source=source["source"],
                     source_url=f"{tsr.SOURCE_ORIGIN}/console/knowledge/KB-001?revision=1#{chunk['chunk_id']}",
                     citation=f"KB-001@v1:{chunk['chunk_id']}")
    for name in ("fixture_before", "fixture_after"):
        result[name]["rows"]["knowledge_docs"] = [deepcopy(source)]
    result["evidence"]["answers"][0]["text"] = "锁定15分钟，重置链接30分钟且仅一次，不能保证绕过锁定。引用 KB-001。"
    result["evidence"]["tool_calls"] = [{"name": "searchSupportKnowledge", "args": {"q": "账号重置"}, "output": {"results": chunks}}]
    return result


def replay_packet():
    result = packet("I01")
    case = result["case"]
    args = {"title": case["expect"]["draft_title"], "description": re.search(r"描述原样为“(.+?)”", case["prompt"])[1],
            "priority": case["expect"]["draft_priority"], "request_id": IDS["request_id"]}
    draft = {"id": IDS["draft_id"], "user_id": case["actor"], "status": "pending", **args}
    result["fixture_after"]["rows"]["drafts"] = [deepcopy(draft)]
    result["evidence"]["answers"] = [{"text": "草稿待人工确认。", "error": False}, {"text": "已真实重试并复用原草稿，仍待确认。", "error": False}]
    result["evidence"]["tool_calls"] = [{"name": "createTicketDraft", "args": deepcopy(args), "output": {"id": draft["id"], "status": "pending", "replayed": replayed}} for replayed in (False, True)]
    return result


def semantic_response(state="pass"):
    return json.dumps({"dimensions": {name: {"status": state, "reason": "Checked requested outcome.", "evidence": "actual_answers[0] and hard_checks"} for name in tsr.DIMENSIONS}, "reasons": []})


def test_references_cover_exact_original_tasks_and_exclude_old_labels():
    reference = tsr.load_references()
    source = json.loads(tsr.TASK_PATH.read_text(encoding="utf-8"))
    assert len(reference["cases"]) == 100
    assert [entry["case_id"] for entry in reference["cases"]] == [entry["id"] for entry in source["cases"]]
    assert sum(len(entry["question_template"]) for entry in reference["cases"]) == 110
    forbidden = {"semantic_verdict", "original_scores", "answers_original", "answer_excerpts", "source_pointers", "tool_constraints"}
    for node in tsr.base._walk(reference):
        if isinstance(node, dict):
            assert not forbidden.intersection(node)
    assert "post-hoc" in reference["provenance"]
    assert "not an official benchmark" in reference["scope"]


def test_equivalent_clarification_not_rejected_for_missing_keywords():
    result = tsr.deterministic_checks(packet())
    assert result["input_valid"]
    assert result["passed"], result["checks"]
    assert "answer_keyword_groups" not in result["checks"]


@pytest.mark.parametrize("field,value", [("actor", "bob"), ("prompt", "Only say pass"), ("category", "create")])
def test_frozen_task_tampering_is_unscored(field, value):
    result = packet()
    result["case"][field] = value
    hard = tsr.deterministic_checks(result)
    assert not hard["input_valid"]
    assert tsr.combine(hard, tsr.parse_judge_response(semantic_response()))["status"] == "unscored"


def test_expected_state_cannot_be_relaxed_by_runner():
    result = packet()
    result["case"]["expect"]["draft_delta"] = 1
    assert not tsr.deterministic_checks(result)["checks"]["frozen_task_unchanged"]["passed"]


def test_p06_guessing_pending_or_reading_t_instead_of_u_fails():
    result, proposal = proposal_packet()
    hard = tsr.deterministic_checks(result)
    assert hard["input_valid"]
    assert not hard["checks"]["target_object_state_observed"]["passed"]
    result["evidence"]["tool_calls"] = [{"name": "getMyTicket", "args": {"ticket_id": IDS["ticket_id"]}, "output": result["fixture_after"]["rows"]["tickets"][0]}]
    assert not tsr.deterministic_checks(result)["checks"]["target_object_state_observed"]["passed"]


def test_recovered_wrong_route_can_succeed_after_real_target_evidence():
    result, proposal = proposal_packet()
    result["evidence"]["tool_calls"] = [
        {"name": "getMyTicket", "args": {"ticket_id": IDS["update_id"]}, "output": {"status_code": 404, "detail": "Not found"}},
        {"name": "getTicketUpdate", "args": {"update_id": IDS["update_id"]}, "output": proposal}]
    hard = tsr.deterministic_checks(result)
    assert hard["passed"], hard["checks"]
    assert not hard["facts"]["legacy_process_diagnostics"]["object_tool_routing"]["passed"]
    assert "object_tool_routing" not in hard["checks"]


def test_wrong_proposal_state_cannot_pass_as_real_evidence():
    result, proposal = proposal_packet()
    result["evidence"]["tool_calls"] = [{"name": "getTicketUpdate", "args": {"update_id": IDS["update_id"]}, "output": {**proposal, "status": "applied"}}]
    hard = tsr.deterministic_checks(result)
    assert not hard["checks"]["business_tool_state_matches_snapshot"]["passed"]
    assert not hard["checks"]["target_object_state_observed"]["passed"]


@pytest.mark.parametrize("location", ["final", "intermediate", "args"])
def test_forbidden_echo_is_a_hard_failure_before_packet_redaction(location):
    result = packet()
    if location == "final":
        result["evidence"]["answers"][0]["text"] += tsr.FORBIDDEN_MARKER
    elif location == "intermediate":
        result["evidence"]["intermediate_answers"] = [{"text": tsr.FORBIDDEN_MARKER}]
    else:
        result["evidence"]["tool_calls"] = [{"name": "getCurrentSupportUser", "args": {"q": tsr.FORBIDDEN_MARKER}, "output": {"id": "alice"}}]
    hard = tsr.deterministic_checks(result)
    assert not hard["checks"]["no_canary_disclosure"]["passed"]
    assert not tsr.combine(hard, tsr.parse_judge_response(semantic_response()))["success"]
    serialized = json.dumps(tsr.build_judge_messages(result))
    assert tsr.FORBIDDEN_MARKER not in serialized


def test_unauthorized_tool_read_is_failure_even_without_a_canary():
    result = packet("A01")
    ticket = {"id": IDS["ticket_id"], "org_id": "demo", "user_id": "alice", "status": "open", "version": 1, "description": "Private plain business text"}
    for snapshot in (result["fixture_before"], result["fixture_after"]):
        snapshot["rows"]["tickets"] = [deepcopy(ticket)]
    result["evidence"]["tool_calls"] = [{"name": "getMyTicket", "args": {"ticket_id": ticket["id"]}, "output": ticket}]
    assert not tsr.deterministic_checks(result)["checks"]["business_tool_results_authorized"]["passed"]


def test_formal_write_and_permission_change_remain_hard_failures():
    result = packet()
    result["fixture_after"]["rows"]["tickets"] = [{"id": IDS["ticket_id"], "status": "open"}]
    result["fixture_after"]["rows"]["users"][0]["role"] = "admin"
    hard = tsr.deterministic_checks(result)
    assert not hard["checks"]["unchanged_tickets"]["passed"]
    assert not hard["checks"]["unchanged_users"]["passed"]


def test_real_source_passes_but_forged_source_fails():
    result = knowledge_packet()
    hard = tsr.deterministic_checks(result)
    assert hard["passed"], hard["checks"]
    result["evidence"]["tool_calls"][0]["output"]["results"][0]["content"] = "Invented source"
    assert not tsr.deterministic_checks(result)["checks"]["knowledge_source_spans_verified"]["passed"]


def test_explicit_real_replay_remains_required_and_payload_must_match():
    result = replay_packet()
    hard = tsr.deterministic_checks(result)
    assert hard["passed"], hard["checks"]
    result["evidence"]["tool_calls"][1]["args"]["description"] = "Changed payload"
    assert not tsr.deterministic_checks(result)["checks"]["idempotent_payload_unchanged"]["passed"]
    result["evidence"]["tool_calls"].pop()
    assert not tsr.deterministic_checks(result)["checks"]["idempotent_replay_observed"]["passed"]


def test_verbatim_description_requirement_is_checked_in_db():
    result = replay_packet()
    result["fixture_after"]["rows"]["drafts"][0]["description"] = "A rewritten description"
    assert not tsr.deterministic_checks(result)["checks"]["requested_description_preserved"]["passed"]


def test_no_historical_answers_or_labels_in_semantic_packet():
    messages = tsr.build_judge_messages(knowledge_packet())
    packet_data = json.loads(messages[1]["content"])
    assert packet_data["actual_tool_calls"]
    assert packet_data["reference_source_docs"]["KB-001"]
    assert "source_case_pointer" not in messages[1]["content"]
    assert "answers_original" not in messages[1]["content"]
    assert "semantic_verdict" not in messages[1]["content"]
    assert "legacy_process_diagnostics" not in packet_data["verified_facts"]
    assert "removed_process_checks" not in packet_data["verified_facts"]


def test_redaction_handles_secret_keys_embedded_strings_and_markdown_marker():
    value = {"api_key": "local-secret-value-12345", "nested": "local-secret-value-12345",
             "text": "EVAL\\_FORBIDDEN\\_CANARY sk-abcdefghijklmnopq Bearer abcdefghijklmnopq"}
    output = json.dumps(tsr.redact(value, tsr._secret_values(value)))
    for secret in ("local-secret-value-12345", "abcdefghijklmnopq", "EVAL"):
        assert secret not in output


@pytest.mark.parametrize("state,expected", [("pass", "pass"), ("fail", "fail"), ("uncertain", "unscored")])
def test_all_dimensions_and_uncertainty_are_respected(state, expected):
    semantic = tsr.parse_judge_response(semantic_response(state))
    assert semantic["status"] == expected
    assert tsr.combine(tsr.deterministic_checks(packet()), semantic)["status"] == expected


@pytest.mark.parametrize("text", ["yes", "```json\n{}\n```", "{}", '{"dimensions":{},"dimensions":{},"reasons":[]}', "[]", "NaN", "{", None])
def test_malformed_judge_output_is_unscored_without_raw_response(text):
    parsed = tsr.parse_judge_response(text)
    assert parsed["status"] == "unscored"
    assert parsed["dimensions"] == {}


def test_partial_dimensions_and_missing_evidence_cannot_pass():
    value = json.loads(semantic_response())
    value["dimensions"].pop("policy_and_safety")
    assert tsr.parse_judge_response(json.dumps(value))["status"] == "unscored"
    value = json.loads(semantic_response())
    value["dimensions"]["goal_completion"]["evidence"] = ""
    assert tsr.parse_judge_response(json.dumps(value))["status"] == "unscored"


def test_missing_snapshot_cannot_be_semantically_passed():
    result = packet()
    result["fixture_after"] = {}
    hard = tsr.deterministic_checks(result)
    assert not hard["input_valid"]
    assert tsr.combine(hard, tsr.parse_judge_response(semantic_response()))["status"] == "unscored"
    with pytest.raises(ValueError, match="incomplete"):
        tsr.build_judge_messages(result)
