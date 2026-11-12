"""Paid-run safeguards exercised with fake HTTP and temporary files only."""
from copy import deepcopy
import base64
import hashlib
import json
from types import SimpleNamespace

import pytest

from evaluation import agent


MODEL = "deepseek-test-model"


def usage_event(inputs=100, outputs=20, seq=1):
    return {"input_tokens": inputs, "output_tokens": outputs, "total_tokens": inputs + outputs,
            "input_token_details": {"cache_read": 40}, "model": MODEL,
            "provider": "openAI", "runId": "fake-run", "seq": seq}


def assistant(identifier="reply", text="最终回答", usage=None):
    message = {"messageId": identifier, "isCreatedByUser": False,
               "content": [{"type": "text", "text": text}]}
    if usage is not None:
        message["metadata"] = {"usage": usage}
    return message


def test_failed_turn_keeps_known_subtotals_but_total_usage_is_unknown():
    usage = agent.turn_usage([], [usage_event()], MODEL, "timeout")
    assert usage["input_tokens"] is None and usage["output_tokens"] is None
    assert usage["known_input_tokens"] == 100 and usage["known_output_tokens"] == 20
    assert usage["coverage"]["missing"] == 1
    assert "failed_turn_may_have_unreported_charge_total_unknown" in usage["notes"]


def test_sse_and_matching_rollup_are_not_added_twice():
    saved = assistant(usage={"input": 60, "output": 20, "cacheRead": 40, "cacheWrite": 0})
    usage = agent.turn_usage([saved], [usage_event()], MODEL, None)
    assert (usage["input_tokens"], usage["output_tokens"]) == (100, 20)
    assert usage["measurement"] == "sse-provider-usage"


def test_partial_sse_prefers_persisted_rollup_without_double_counting():
    saved = assistant(usage={"input": 120, "output": 40, "cacheRead": 80, "cacheWrite": 0})
    usage = agent.turn_usage([saved], [usage_event()], MODEL, None)
    assert (usage["input_tokens"], usage["output_tokens"]) == (200, 40)
    assert usage["measurement"] == "librechat-message-rollup"
    assert "sse_and_persisted_totals_differ_using_persisted_rollup" in usage["notes"]


@pytest.mark.parametrize("plain,output", [(10, 20), (60, 10), (160, 10)])
def test_smaller_or_conflicting_rollup_never_reduces_observed_usage(plain, output):
    saved = assistant(usage={"input": plain, "output": output, "cacheRead": 40, "cacheWrite": 0})
    usage = agent.turn_usage([saved], [usage_event()], MODEL, None)
    assert usage["input_tokens"] is None and usage["output_tokens"] is None
    assert (usage["known_input_tokens"], usage["known_output_tokens"]) == (100, 20)
    assert "persisted_rollup_below_observed_usage_total_unknown" in usage["notes"]


class FakeApi:
    base = "http://unit-test.invalid"
    token = "unit-test-token"

    def __init__(self, messages=None, *, post_error=False, active=False, abort_error=False):
        self.messages = messages or []
        self.post_error, self.active, self.abort_error = post_error, active, abort_error
        self.requests = []

    def request(self, method, path, payload=None):
        self.requests.append((method, path, payload))
        if (method, path) == ("POST", "/api/agents/chat"):
            if self.post_error:
                raise agent.BootstrapError("Submission outcome unknown.")
            return {"streamId": "fake-stream", "conversationId": "fake-conversation", "generationCreatedAt": 123}
        if path == "/api/agents/chat/abort":
            if self.abort_error:
                raise agent.BootstrapError("Abort transport failed.")
            return {"ok": True}
        if path.startswith("/api/agents/chat/status/"):
            return {"active": self.active}
        if path.startswith("/api/messages/"):
            return self.messages
        raise AssertionError("Unexpected fake request")


def run_turn(api, **changes):
    args = {"api": api, "agent_id": "fake-agent", "prompt": "测试请求", "conversation_id": None,
            "parent_id": None, "previous_ids": set(), "timeout": 30, "model": MODEL}
    return agent.turn(**{**args, **changes})


def test_each_turn_uses_only_last_final_answer_but_keeps_tool_evidence(monkeypatch):
    earlier = assistant("prior", "上轮回答")
    intermediate = assistant("intermediate", "正在检索")
    intermediate["content"].append({"type": "tool_call", "tool_call": {"id": "call-1", "name": "searchSupportKnowledge"}})
    final = assistant("final", "最后的回答")
    api = FakeApi([earlier, {"messageId": "user-new", "isCreatedByUser": True}, intermediate, final])
    monkeypatch.setattr(agent, "stream", lambda api, start, deadline, events: events.append(usage_event()))
    outcome = run_turn(api, previous_ids={"prior"})
    assert [answer["text"] for answer in outcome["answers"]] == ["最后的回答"]
    assert [answer["text"] for answer in outcome["intermediate_answers"]] == ["正在检索"]
    assert [call["name"] for call in outcome["tool_calls"]] == ["searchSupportKnowledge"]
    assert outcome["parent_id"] == "final" and outcome["execution_error"] is None


def test_timeout_aborts_and_never_retries_paid_submission(monkeypatch):
    api = FakeApi([assistant()])
    def timeout(api, start, deadline, events):
        events.append(usage_event())
        raise agent.BootstrapError("Agent evaluation timed out.")
    monkeypatch.setattr(agent, "stream", timeout)
    monkeypatch.setattr(agent.time, "sleep", lambda seconds: None)
    outcome = run_turn(api)
    assert sum(method == "POST" and path == "/api/agents/chat" for method, path, _ in api.requests) == 1
    aborts = [payload for method, path, payload in api.requests if path == "/api/agents/chat/abort"]
    assert aborts == [{"streamId": "fake-stream", "generationCreatedAt": 123}]
    assert outcome["drained"] is True and outcome["execution_error"]
    assert outcome["usage"]["input_tokens"] is None


def test_ambiguous_post_failure_never_retries_or_claims_known_cost(monkeypatch):
    api = FakeApi(post_error=True)
    monkeypatch.setattr(agent, "stream", lambda *args: pytest.fail("No stream exists after unacknowledged submission."))
    outcome = run_turn(api)
    assert len(api.requests) == 1
    assert outcome["drained"] is False
    assert outcome["usage"]["input_tokens"] is None and outcome["usage"]["output_tokens"] is None


@pytest.mark.parametrize("abort_error", [False, True])
def test_abort_failure_or_still_active_request_is_not_marked_drained(monkeypatch, abort_error):
    api = FakeApi(active=True, abort_error=abort_error)
    monkeypatch.setattr(agent, "stream", lambda *args: (_ for _ in ()).throw(agent.BootstrapError("timeout")))
    monkeypatch.setattr(agent.time, "sleep", lambda seconds: None)
    outcome = run_turn(api)
    assert outcome["drained"] is False
    assert sum(path == "/api/agents/chat" for _, path, _ in api.requests) == 1


def bindings(tool_count=11):
    raw_spec = json.dumps({"openapi": "3.1.0", "servers": [{"url": "http://support-eval:8090"}], "paths": {}})
    metadata = {"domain": "http://support-eval:8090", "raw_spec": raw_spec,
                "api_key": "fake-encrypted-binding", "auth": {"type": "service_http", "authorization_type": "bearer"}}
    tools = [f"tool_{index}" for index in range(tool_count)]
    saved = {"instructions": agent.instructions("alice"), "provider": "DeepSeek", "model": MODEL,
             "model_parameters": deepcopy(agent.MODEL_PARAMETERS), "recursion_limit": agent.RECURSION_LIMIT,
             "tools": tools, "actions": ["action-1"]}
    action = {"agent_id": "agent-1", "action_id": "action-1", "metadata": metadata}
    state = {"model": MODEL, "tool_count": tool_count, "schema_sha256": hashlib.sha256(raw_spec.encode()).hexdigest(), "agents": {"alice": {
        "agent_id": "agent-1", "action_id": "action-1", "tools": tools[:], "actions": ["action-1"],
        "stored_metadata_sha256": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()}}}
    class ConfigApi:
        def request(self, method, path):
            assert method == "GET", "Validation must remain read-only."
            if path == "/api/agents/actions":
                return [action]
            assert path == "/api/agents/agent-1/expanded"
            return saved
    return ConfigApi(), state, saved, action


@pytest.mark.parametrize('tool_count', [11, 12])
def test_validate_agents_accepts_exact_frozen_binding(tool_count):
    api, state, _, _ = bindings(tool_count)
    agent.validate_agents(api, state, {"alice"})


def test_validate_agents_rejects_changed_declared_tool_count():
    api, state, _, _ = bindings(12)
    state['tool_count'] = 11
    with pytest.raises(agent.BootstrapError):
        agent.validate_agents(api, state, {'alice'})


@pytest.mark.parametrize("field,value", [("domain", "http://support-agent:8090"),
    ("raw_spec", "different-schema"), ("api_key", "different-credential")])
def test_validate_agents_rejects_action_domain_schema_or_credential_drift(field, value):
    api, state, _, action = bindings()
    action["metadata"][field] = value
    with pytest.raises(agent.BootstrapError, match="binding changed"):
        agent.validate_agents(api, state, {"alice"})


def test_validate_agents_checks_schema_hash_independently_of_metadata_hash():
    api, state, _, action = bindings()
    action["metadata"]["raw_spec"] = "changed-schema"
    state["agents"]["alice"]["stored_metadata_sha256"] = hashlib.sha256(json.dumps(action["metadata"], sort_keys=True).encode()).hexdigest()
    with pytest.raises(agent.BootstrapError):
        agent.validate_agents(api, state, {"alice"})


@pytest.mark.parametrize("field,value", [("provider", "OtherPaidProvider"), ("model", "different-model"),
    ("recursion_limit", 999), ("model_parameters", {"temperature": 1}), ("instructions", "changed")])
def test_validate_agents_rejects_model_endpoint_or_runtime_parameter_drift(field, value):
    api, state, saved, _ = bindings()
    saved[field] = value
    with pytest.raises(agent.BootstrapError):
        agent.validate_agents(api, state, {"alice"})


def test_exclusive_run_rejects_concurrent_holder_and_releases_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "LOCAL", tmp_path / "temporary-evaluation-lock")
    with agent.exclusive_run():
        with pytest.raises(agent.BootstrapError, match="Another evaluation"):
            with agent.exclusive_run():
                pytest.fail("Two fixture owners must not run at once.")
    with agent.exclusive_run():
        pass


@pytest.mark.parametrize("remaining,should_renew", [(200, True), (600, False)])
def test_session_is_renewed_before_expiry_without_submitting_a_model_request(monkeypatch, remaining, should_renew):
    api = FakeApi()
    payload = base64.urlsafe_b64encode(json.dumps({"exp": 1000 + remaining}).encode()).decode().rstrip("=")
    api.token = "header." + payload + ".signature"
    forbidden, logins = [], []
    monkeypatch.setattr(agent.time, "time", lambda: 1000)
    def authenticate(client, demo):
        logins.append(True)
        client.token = "renewed-test-token"
    monkeypatch.setattr(agent, "authenticate", authenticate)
    agent.renew_session(api, {"password": "fake-test-password"}, forbidden, 120)
    assert bool(logins) is should_renew
    assert forbidden == (["renewed-test-token"] if should_renew else [])
    assert api.requests == []


def resumed_run(tmp_path, monkeypatch, *, prior_cost=1.0, drained=True, changed_budget=False, pending=False):
    local = tmp_path / "fake-lab"
    folder = tmp_path / "resume-folder"
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr(agent, "ROOT", tmp_path)
    monkeypatch.setattr(agent, "LOCAL", local)
    monkeypatch.setattr(agent, "code_digest", lambda: "frozen-code")
    monkeypatch.setattr(agent, "digest", lambda path: "sha-" + path.name)
    monkeypatch.setattr(agent, "LocalApi", lambda base: FakeApi())
    monkeypatch.setattr(agent, "authenticate", lambda api, demo: None)
    monkeypatch.setattr(agent, "validate_agents", lambda *args: None)
    executed = []
    monkeypatch.setattr(agent, "execute_case", lambda *args: executed.append(True) or pytest.fail("Resume must stop before another paid case."))
    monkeypatch.setattr(agent, "build_report", lambda manifest, results: {"safe": True})
    monkeypatch.setattr(agent, "render_html", lambda report: "temporary test report")
    writes = {}
    monkeypatch.setattr(agent, "save_state", lambda path, value: writes.__setitem__(path.name, deepcopy(value)))
    state = {"model": MODEL, "parameters": agent.MODEL_PARAMETERS, "recursion_limit": agent.RECURSION_LIMIT,
             "instructions_sha256": "instructions", "schema_sha256": "schema", "agents": {}}
    cases = [{"id": "C0", "actor": "alice"}, {"id": "C1", "actor": "alice"}]
    existing = {"run_id": "original-run", "dataset_sha256": "sha-dataset.json", "code_sha256": "frozen-code",
                "prices_sha256": "sha-prices.json", "case_ids": ["C0", "C1"], "planned_cases": 2,
                "model": MODEL, "parameters": agent.MODEL_PARAMETERS, "recursion_limit": agent.RECURSION_LIMIT,
                "instructions_sha256": "instructions", "schema_sha256": "schema", "timeout_seconds": 120,
                "corpus_sha256": "sha-knowledge.json", "platform_config_sha256": "sha-config.yaml", "max_budget_cny": 5.0}
    existing["reference_sha256"] = "sha-reference_answers_v2.json" if (agent.SERVICE / "evaluation/reference_answers_v2.json").exists() else None
    if pending:
        existing["pending_case"] = "C1"
    prior = [{"case_id": "C0", "cost": {"conservative_budget": prior_cost}, "drained": drained}]
    rates = {"currency": "CNY", "model": MODEL, "input_per_million": 1, "cache_hit_per_million": .1, "output_per_million": 2}
    def read_json(path):
        return deepcopy({"agents.json": state, "demo.json": {"password": "fake-password"},
                         "users.json": {"users": []}, "prices.json": {"off_peak": rates, "peak": rates, "source": "fixture://prices"},
                         "manifest.json": existing, "results.json": prior}[path.name])
    monkeypatch.setattr(agent, "read_json", read_json)
    args = SimpleNamespace(dataset="test", resume=folder, max_budget_cny=6.0 if changed_budget else 5.0, timeout=120)
    return args, tmp_path / "dataset.json", cases, writes, executed


@pytest.mark.parametrize("prior_cost,drained,reason", [
    (None, True, "unknown_usage_or_active_request"),
    (1.0, False, "unknown_usage_or_active_request"),
    (5.0, True, "estimated_budget_reached"),
])
def test_resume_rechecks_prior_unknown_cost_active_request_and_budget(tmp_path, monkeypatch, prior_cost, drained, reason):
    args, dataset, cases, writes, executed = resumed_run(tmp_path, monkeypatch, prior_cost=prior_cost, drained=drained)
    assert agent.run_paid(args, dataset, cases) == 2
    assert not executed
    assert writes["manifest.json"]["stop_reason"] == reason


@pytest.mark.parametrize("options,pattern", [({"changed_budget": True}, "configuration changed"),
                                            ({"pending": True}, "unknown outcome")])
def test_resume_refuses_changed_budget_or_unfinished_model_attempt(tmp_path, monkeypatch, options, pattern):
    args, dataset, cases, _, executed = resumed_run(tmp_path, monkeypatch, **options)
    with pytest.raises(agent.BootstrapError, match=pattern):
        agent.run_paid(args, dataset, cases)
    assert not executed
