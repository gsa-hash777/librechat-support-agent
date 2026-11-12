"""Public benchmark isolation, accounting and no-retry checks without model calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import pytest

from evaluation import public_agent as runner, public_agent_setup as setup, public_data, public_judge


def schema():
    return {"servers": [{"url": "http://support-public:8092"}], "paths": {
        "/search": {"get": {"operationId": "searchPublicKnowledge", "parameters": []}},
        "/evidence/{passage_id}": {"get": {"operationId": "getPublicKnowledgeEvidence", "parameters": []}}}}


def state():
    return {"mode": "lexical", "schema": schema(), "schema_sha256": setup.fingerprint(schema()),
            "model": setup.MODEL, "provider": setup.PROVIDER, "parameters": {role: setup.role_parameters(role) for role in ("answer", "judge")},
            "recursion_limit": setup.RECURSION_LIMIT, "owner_id": "owner",
            "agents": {"answer": {"agent_id": "private-answer", "tools": ["search", "read"],
                                  "actions": ["action"], "action_id": "action", "metadata_sha256": setup.fingerprint({})},
                       "judge": {"agent_id": "private-judge", "tools": [], "actions": []}}}


def task(identifier="one", question="How can I enable the service?"):
    return {"task_id": identifier, "conversation_id": "official-" + identifier, "turn": 3,
            "messages": [{"speaker": "user", "text": "I use the standard service."},
                         {"speaker": "assistant", "text": "Which task do you want to do?"},
                         {"speaker": "user", "text": question}]}


def passage(identifier="ibmcld_1", text="Enable the service in the console."):
    return {"passage_id": identifier, "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source_url": "https://cloud.ibm.com/docs/example", "score": .5}


def search_response(results=None):
    return {"mode": "lexical", "query": "enable", "results": [passage()] if results is None else results, "latency_ms": 1.5}


class HealthyPublic:
    def request(self, method, path, payload=None):
        assert (method, path, payload) == ("GET", "/health", None)
        return {"status": "ok", "corpus": "IBM Cloud MTRAG", "passages": 72442, "read_only": True}


def response(*, answer="Use the console [ibmcld_1].", unknown=False, calls=None):
    usage = runner.normalize_usage(model=setup.MODEL) if unknown else runner.normalize_usage(
        usage_events=[{"input_tokens": 1000, "output_tokens": 100, "input_token_details": {"cache_read": 0},
                       "provider": "openAI", "model": setup.MODEL}], model=setup.MODEL)
    return {"answers": [{"text": answer}], "tool_calls": calls if calls is not None else [
        {"name": "searchPublicKnowledge_action_unit", "args": {"q": "enable", "mode": "lexical", "limit": 5},
         "output": json.dumps(search_response())}], "usage": usage,
        "execution_error": None, "drained": True}


def execute(tmp_path, *, tasks=None, fake=None, **kwargs):
    ledger = kwargs.pop("ledger", None) or runner.Budget(tmp_path / "budget.json")
    snapshot = kwargs.pop("snapshot_fn", lambda: {"sha256": "same", "counts": {"tickets": 2}})
    freeze = kwargs.pop("freeze_fn", lambda verify: {"fixed": "fixture"})
    support = kwargs.pop("support", HealthyPublic())
    return runner.execute(tasks or [task()], None, support, state(), tmp_path / "run",
                          ledger, runner.prices_snapshot(),
                          turn_fn=fake or (lambda *args: response()), snapshot_fn=snapshot, freeze_fn=freeze,
                          **kwargs)


def test_only_whitelisted_official_dialogue_reaches_answer_prompt():
    raw = task()
    raw.update(targets=[{"text": "SECRET-GOLD"}], contexts=[{"text": "SECRET-PASSAGE"}],
               enrichment="SECRET-ENRICHMENT", Answerability="SECRET-LABEL")
    raw["messages"][0]["target"] = "SECRET-NESTED"
    result = runner.answer_prompt(raw)
    assert "SECRET" not in result
    assert "standard service" in result and "How can I enable" in result
    assert raw["conversation_id"] not in result and "turn" not in runner.project_task(raw)


def test_public_redaction_preserves_bearer_document_prose_and_evidence_hash(tmp_path):
    text = "Authenticate with a Bearer token. Keep the token private."
    expected = passage(text=text)
    item = response(answer="Use a Bearer token. [ibmcld_1]", calls=[{
        "name": "searchPublicKnowledge", "output": json.dumps(search_response([expected]))}])
    report = execute(tmp_path, fake=lambda *args: item)
    assert report["completed"] == 1 and report["stop_reason"] is None
    row = report["results"][0]
    assert row["retrieved_passages"] == [expected] and row["evidence_conflicts"] == []
    assert row["answer"] == "Use a Bearer token. [ibmcld_1]"
    saved = json.loads((tmp_path / "run/report.json").read_text(encoding="utf-8"))
    assert saved["results"][0]["retrieved_passages"] == [expected]
    assert hashlib.sha256(saved["results"][0]["retrieved_passages"][0]["text"].encode()).hexdigest() == expected["sha256"]
    assert runner.redact_public({"reference_answer": text, "reference_passages": [expected]}, []) == {
        "reference_answer": text, "reference_passages": [expected]}


def test_public_redaction_masks_exact_known_credentials_and_nested_secret_fields():
    known = ["fixture-provider-key-123", "fixture-password-456", "fixture-public-tool-789", "fixture-jwt-abc"]
    raw = {"text": "Bearer token. Bearer " + known[0] + " " + " ".join(known[1:]),
           "nested": [{"password": "unknown-password", "api_key": "unknown-api-value",
                       "authorization": "Bearer unknown-value", "refreshToken": "unknown-refresh",
                       "safe": "Bearer token."}]}
    clean = runner.redact_public(raw, known)
    serialized = json.dumps(clean)
    assert all(value not in serialized for value in known)
    assert "Bearer token. Bearer [REDACTED]" in clean["text"]
    assert clean["nested"][0] == {"password": "[REDACTED]", "api_key": "[REDACTED]",
                                    "authorization": "[REDACTED]", "refreshToken": "[REDACTED]",
                                    "safe": "Bearer token."}
    assert raw["nested"][0]["password"] == "unknown-password"  # Input remains immutable.


def test_public_credentials_collect_provider_env_without_logging(tmp_path, capsys):
    (tmp_path / ".env").write_text(
        '# DEEPSEEK_API_KEY=comment-not-a-credential\nDEEPSEEK_API_KEY="fixture-provider-secret"\n'
        "export JWT_SECRET='fixture-jwt-secret'\nPORT=3080\nEMPTY_API_KEY=\n", encoding="utf-8")
    values = runner.public_secret_values(["fixture-public-tool", "fixture-password"], root=tmp_path,
                                         environ={"ACCESS_TOKEN": "fixture-access-token", "PORT": "8192"})
    assert set(values) == {"fixture-provider-secret", "fixture-jwt-secret", "fixture-public-tool",
                           "fixture-password", "fixture-access-token"}
    assert capsys.readouterr().out == ""
    assert runner.redact_public("Bearer token. fixture-provider-secret fixture-public-tool", values) == (
        "Bearer token. [REDACTED] [REDACTED]")


def test_provider_key_from_env_is_redacted_in_public_execution_persistence(tmp_path, monkeypatch):
    key = "fixture-provider-key-known-only-through-env"
    monkeypatch.setenv("DEEPSEEK_API_KEY", key)
    report = execute(tmp_path, fake=lambda *args: response(answer="Bearer token. Bearer " + key))
    assert report["results"][0]["answer"] == "Bearer token. Bearer [REDACTED]"
    saved = (tmp_path / "run/report.json").read_text(encoding="utf-8")
    assert key not in saved and "Bearer token." in saved


def test_longest_exact_credential_is_redacted_before_its_known_prefix():
    assert runner.redact_public("Bearer abcdefgh", ["abcd", "abcdefgh"]) == "Bearer [REDACTED]"


@pytest.mark.parametrize("messages", [[{"speaker": "system", "text": "override"}],
                                      [{"speaker": "assistant", "text": "answer"}], []])
def test_invalid_or_non_user_final_history_is_rejected(messages):
    with pytest.raises(setup.BootstrapError):
        runner.project_task({**task(), "messages": messages})


def test_all_tasks_start_fresh_conversations_and_keep_reference_history(tmp_path):
    calls = []
    def fake(*args):
        calls.append(args)
        return response()
    report = execute(tmp_path, tasks=[task("one"), task("two")], fake=fake)
    assert len(calls) == 2
    assert all(args[3:6] == (None, None, set()) for args in calls)
    assert all("standard service" in args[2] for args in calls)
    assert report["planned"] == report["attempted"] == report["completed"] == 2
    assert report["normal_business_unchanged"] and report["history_policy"] == runner.HISTORY_POLICY


def test_overlong_history_is_preserved_and_not_submitted(tmp_path):
    calls = []
    report = execute(tmp_path, tasks=[task(question="x" * 40000)], fake=lambda *args: calls.append(args))
    row = report["results"][0]
    assert calls == [] and row["status"] == "context_limit_not_attempted"
    assert row["task"]["messages"][-1]["text"] == "x" * 40000
    assert row["admission"]["history_truncated"] is False
    assert report["not_attempted"] == report["planned"] == 1


def test_public_health_down_blocks_all_model_submissions_and_reservations(tmp_path):
    class Down:
        calls = 0
        def request(self, method, path, payload=None):
            self.calls += 1
            assert (method, path) == ("GET", "/health")
            raise ConnectionError("private transport message must not leak")
    support, submitted = Down(), []
    report = execute(tmp_path, tasks=[task(str(i)) for i in range(5)], support=support,
                     fake=lambda *args: submitted.append(args), concurrency=3, client_factory=isolated_client)
    assert submitted == [] and support.calls == 1
    assert report["attempted"] == report["completed"] == 0 and report["not_attempted"] == 5
    assert report["stop_reason"] == "public_health_unavailable"
    assert report["results"][0]["status"] == "infrastructure_not_attempted"
    assert report["results"][0]["infrastructure_error"] == {
        "kind": "public_health_unavailable", "error_type": "ConnectionError"}
    assert not (tmp_path / "budget.json").exists()
    assert "private transport" not in (tmp_path / "run/report.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("payload", [None, {}, {"status": "ok"},
    {"status": "ok", "corpus": "business", "passages": 72442, "read_only": True},
    {"status": "ok", "corpus": "IBM Cloud MTRAG", "passages": True, "read_only": True},
    {"status": "ok", "corpus": "IBM Cloud MTRAG", "passages": 0, "read_only": True},
    {"status": "ok", "corpus": "IBM Cloud MTRAG", "passages": 72442, "read_only": False}])
def test_invalid_public_health_identity_is_not_success(payload):
    class Service:
        def request(self, method, path):
            return payload
    with pytest.raises(setup.BootstrapError):
        runner.public_health(Service())


def test_health_is_checked_again_before_each_submission_without_retry(tmp_path):
    class StopsAfterFirst(HealthyPublic):
        calls = 0
        def request(self, method, path, payload=None):
            self.calls += 1
            if self.calls > 1:
                raise runner.ApiError(503, "/health", "private upstream body")
            return super().request(method, path, payload)
    support, submitted = StopsAfterFirst(), []
    def fake(*args):
        submitted.append(1)
        return response()
    report = execute(tmp_path, tasks=[task(str(i)) for i in range(3)], support=support, fake=fake)
    assert support.calls == 2 and submitted == [1]
    assert report["attempted"] == report["completed"] == 1 and report["not_attempted"] == 2
    assert report["stop_reason"] == "public_health_unavailable"
    assert report["results"][1]["infrastructure_error"]["http_status"] == 503
    assert report["results"][0]["public_health"]["read_only"] is True


@pytest.mark.parametrize("output", [
    "Error: connect ECONNREFUSED support-public:8092", "Request failed with status code 503",
    {"error": "Connection reset"}, {"detail": "Requested index unavailable"},
    {**search_response(), "isError": True}, {**search_response(), "mode": "rerank"},
    {"results": []}, {**search_response(), "results": "[]"},
    {**search_response(), "results": [{"passage_id": "ibmcld_1", "text": "incomplete"}]},
    {**search_response(), "latency_ms": float("nan")},
])
def test_failed_tool_output_is_execution_error_not_normal_refusal(tmp_path, output):
    submitted = []
    def fake(*args):
        submitted.append(1)
        return response(answer="I cannot find supporting evidence.", calls=[{
            "name": "searchPublicKnowledge_action_unit", "output": json.dumps(output)}])
    report = execute(tmp_path, tasks=[task("one"), task("two")], fake=fake)
    row = report["results"][0]
    assert submitted == [1] and report["attempted"] == 1 and report["completed"] == 0
    assert report["not_attempted"] == 1 and row["status"] == "execution_error"
    assert row["execution"]["execution_error"]
    assert row["infrastructure_error"]["kind"] == "public_tool_contract_failure"
    assert report["stop_reason"] == "public_tool_infrastructure_failure"
    # Observed provider usage is still settled, not discarded or falsely zeroed.
    assert row["usage"]["input_tokens"] == 1000 and report["cost"]["conservative_budget"] > 0
    ledger = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert ledger["attempts"][0]["status"] == "settled"


def test_successful_empty_search_is_not_an_infrastructure_error(tmp_path):
    item = response(answer="The search returned no evidence.", calls=[{
        "name": "searchPublicKnowledge", "output": search_response([])}])
    report = execute(tmp_path, fake=lambda *args: item)
    assert report["completed"] == 1 and report["stop_reason"] is None
    assert report["results"][0]["retrieved_passages"] == []
    assert "infrastructure_error" not in report["results"][0]


def test_malformed_tool_object_inspection_preserves_observed_provider_usage(tmp_path):
    malformed = search_response([{**passage(), "score": 10 ** 400}])
    report = execute(tmp_path, fake=lambda *args: response(calls=[{
        "name": "searchPublicKnowledge", "output": malformed}]))
    row = report["results"][0]
    assert row["status"] == "execution_error" and row["usage"]["input_tokens"] == 1000
    assert row["cost"]["conservative_budget"] > 0
    assert row["infrastructure_error"]["failures"][0]["reason"] == "tool_contract_inspection_failed"
    assert report["stop_reason"] == "public_tool_infrastructure_failure"


def test_no_tool_call_is_a_quality_issue_not_a_fabricated_infrastructure_error(tmp_path):
    report = execute(tmp_path, fake=lambda *args: response(calls=[]))
    assert report["completed"] == 1 and report["stop_reason"] is None
    assert report["results"][0]["checks"]["search_observed"] is False


def test_failed_evidence_read_is_detected_even_after_successful_search(tmp_path):
    item = response()
    item["tool_calls"].append({"name": "getPublicKnowledgeEvidence", "output": "HTTP 503: service unavailable"})
    report = execute(tmp_path, fake=lambda *args: item)
    assert report["completed"] == 0 and report["stop_reason"] == "public_tool_infrastructure_failure"
    assert report["results"][0]["infrastructure_error"]["failures"][0]["tool_name"] == "getPublicKnowledgeEvidence"


def test_parallel_tool_failure_stops_new_submissions_and_drains_known_charges(tmp_path):
    barrier, submitted = threading.Barrier(3), []
    def fake(client, agent_id, prompt, *args):
        question = prompt_question(prompt)
        submitted.append(question)
        barrier.wait(timeout=3)
        if question == "job-0":
            return response(calls=[{"name": "searchPublicKnowledge", "output": "ECONNRESET"}])
        time.sleep(.05)
        return response()
    report = execute(tmp_path, tasks=[task(str(i), "job-" + str(i)) for i in range(6)], fake=fake,
                     concurrency=3, client_factory=isolated_client)
    assert sorted(submitted) == ["job-0", "job-1", "job-2"]
    assert report["attempted"] == 3 and report["completed"] == 2 and report["not_attempted"] == 3
    assert report["stop_reason"] == "public_tool_infrastructure_failure" and report["pending_cases"] == []
    ledger = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert len(ledger["attempts"]) == 3 and all(row["status"] == "settled" for row in ledger["attempts"])


@pytest.mark.parametrize("mutate", [
    lambda value: value["servers"][0].update(url="http://support-agent:8090"),
    lambda value: value["paths"].update({"/mutation": {"post": {"operationId": "create"}}}),
    lambda value: value["paths"]["/search"]["get"].update(operationId="createTicket"),
])
def test_public_action_rejects_normal_service_or_mutation(mutate):
    value = schema()
    mutate(value)
    with pytest.raises(setup.BootstrapError):
        setup.public_schema(value)


class FakeApi:
    def __init__(self):
        self.state = state()
        self.agents = {}
        self.actions = [{"agent_id": "private-answer", "action_id": "action", "metadata": {}}]
        for role, entry in self.state["agents"].items():
            self.agents[entry["agent_id"]] = {**entry, "author": "owner", "provider": setup.PROVIDER, "model": setup.MODEL,
                "model_parameters": setup.role_parameters(role), "recursion_limit": 5 if role == "answer" else 2,
                "instructions": setup.instructions("lexical") if role == "answer" else setup.JUDGE_INSTRUCTIONS}

    def request(self, method, path, payload=None):
        assert method == "GET"
        if path == "/openapi.json":
            return schema()
        if path == "/api/agents/actions":
            return self.actions
        return self.agents[path.split("/")[-2]]


def test_validate_binds_read_only_answer_and_zero_tool_judge():
    api = FakeApi()
    setup.validate(api, api, api.state)
    api.agents["private-judge"]["tools"] = ["search"]
    with pytest.raises(setup.BootstrapError):
        setup.validate(api, api, api.state)


def test_public_roles_disable_provider_retries_and_judge_has_terminal_graph_step():
    api = FakeApi()
    assert all("maxRetries" not in setup.role_parameters(role) for role in ("answer", "judge"))
    assert setup.provider_settings()["addParams"]["maxRetries"] == 0
    assert api.agents["private-judge"]["recursion_limit"] == 2
    api.agents["private-judge"]["recursion_limit"] = 1
    with pytest.raises(setup.BootstrapError):
        setup.validate(api, api, api.state)


@pytest.mark.parametrize("field,value", [("author", "someone"), ("model", "other"), ("instructions", "tampered"),
                                          ("recursion_limit", 20), ("model_parameters", {"max_tokens": 9000})])
def test_validate_refuses_model_owner_or_prompt_drift(field, value):
    api = FakeApi()
    api.agents["private-answer"][field] = value
    with pytest.raises(setup.BootstrapError):
        setup.validate(api, api, api.state)


def test_actual_evidence_and_all_citations_are_preserved():
    observed = runner.observed_evidence(response(answer="Console [ibmcld_1]. Another [invented]. Again [ibmcld_1]."))
    assert len(observed["citations"]) == 3
    assert observed["retrieved_passages"] == [passage()]
    assert observed["checks"]["citation_ids_returned"] is False


def test_conflicting_or_tampered_tool_text_stops_run(tmp_path):
    broken = passage()
    broken["sha256"] = "0" * 64
    other = passage(text="Different content.")
    result = response(calls=[{"name": "searchPublicKnowledge", "output": search_response([broken, other])}])
    report = execute(tmp_path, tasks=[task("one"), task("two")], fake=lambda *args: result)
    assert len(report["results"][0]["evidence_conflicts"]) == 2
    assert report["attempted"] == 1 and report["stop_reason"] == "tool_evidence_integrity"


def test_shared_budget_pre_reserves_and_refuses_pending_or_unknown(tmp_path):
    path = tmp_path / "budget.json"
    budget = runner.Budget(path, limit=1)
    pending = budget.reserve("run", "one", "answer", .8)
    assert json.loads(path.read_text())["attempts"][0]["status"] == "pending"
    with pytest.raises(setup.BootstrapError):
        runner.Budget(path, limit=1)
    second = budget.reserve("run", "two", "answer", .1)
    assert budget.reserve("run", "three", "answer", .2) is None
    budget.cancel_unsubmitted(second)
    budget.settle(pending, {"conservative_budget": .6})
    other = runner.Budget(path, limit=1)
    assert other.reserve("next", "two", "judge", .5) is None
    pending = other.reserve("next", "two", "judge", .3)
    assert other.settle(pending, {"conservative_budget": None}) is False
    with pytest.raises(setup.BootstrapError):
        runner.Budget(path, limit=1)


def test_unknown_usage_never_becomes_zero_and_stops_following_tasks(tmp_path):
    report = execute(tmp_path, tasks=[task("one"), task("two")], fake=lambda *args: response(unknown=True))
    assert report["attempted"] == 1 and report["not_attempted"] == 1
    assert report["cost"]["conservative_budget"] is None and report["shared_budget_spent_cny"] is None
    assert report["usage"]["input_tokens"] is None
    assert report["stop_reason"] == "execution_or_usage_incomplete"


def test_transport_exception_is_one_submission_never_automatic_retry(tmp_path):
    calls = []
    def fail(*args):
        calls.append(args)
        raise TimeoutError("private credential must not leak")
    report = execute(tmp_path, tasks=[task("one"), task("two")], fake=fail)
    assert len(calls) == 1 and report["completed"] == 0
    assert "private credential" not in (tmp_path / "run/report.json").read_text()
    assert report["shared_budget_spent_cny"] is None
    with pytest.raises(setup.BootstrapError):
        execute(tmp_path, fake=fail)
    assert len(calls) == 1


def test_auth_failure_does_not_reserve_or_submit(tmp_path):
    def fail_auth():
        raise RuntimeError("expired")
    calls = []
    report = execute(tmp_path, fake=lambda *args: calls.append(args), before_turn=fail_auth)
    assert calls == [] and report["attempted"] == 0
    budget = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert budget["attempts"][0]["status"] == "not_submitted"
    assert report["shared_budget_spent_cny"] == 0
    assert report["stop_reason"] == "preflight_authentication"


def test_preflight_http_failure_records_safe_route_status_and_reason(tmp_path):
    def fail_auth():
        raise runner.ApiError(429, "/api/auth/login?secret=do-not-copy", "arbitrary-body-do-not-copy")
    report = execute(tmp_path, before_turn=fail_auth)
    error = report["results"][0]["preflight_error"]
    assert error == {"error_type": "ApiError", "http_status": 429, "path": "/api/auth/login", "reason": "rate_limited"}
    assert "do-not-copy" not in (tmp_path / "run/report.json").read_text(encoding="utf-8")
    assert report["attempted"] == 0


def test_worker_clients_copy_initial_token_without_login_and_own_cookie_jars(monkeypatch):
    import base64
    from urllib.request import HTTPCookieProcessor
    token = "header." + base64.urlsafe_b64encode(json.dumps({"exp": time.time() + 86400}).encode()).decode().rstrip("=") + ".signature"
    main, support = setup.LocalApi("http://localhost:3080"), setup.LocalApi("http://localhost:8192")
    main.token, support.token = token, "support-secret"
    requests, clients = [], []
    def request(client, method, path, payload=None):
        requests.append((method, path))
        assert method == "GET"
        return {}
    def validate(client, service, current):
        clients.append((client, service))
        client.request("GET", "/api/agents/actions")
        service.request("GET", "/openapi.json")
    monkeypatch.setattr(setup.LocalApi, "request", request)
    monkeypatch.setattr(setup, "validate", validate)
    factory = runner.worker_client_factory(main, support, state(), {}, [token, "support-secret"], 120)
    main.token = "changed-main-token"
    workers = [factory() for _ in range(3)]
    for worker, before, secrets in workers:
        before()  # Unexpired copied token does not authenticate again.
        assert worker.token == token
    assert len({id(client.opener) for pair in clients for client in pair}) == 6
    jars = [next(h.cookiejar for h in client.opener.handlers if isinstance(h, HTTPCookieProcessor)) for pair in clients for client in pair]
    assert len({id(jar) for jar in jars}) == 6
    workers[0][0].token = "refreshed-worker-token"
    workers[0][2].append("worker-only-secret")
    assert workers[1][0].token == token and "worker-only-secret" not in workers[1][2]
    assert len(requests) == 6 and all(method == "GET" for method, _ in requests)


def test_judging_joins_gold_only_after_answer_and_keeps_actual_evidence_separate(tmp_path):
    report = execute(tmp_path)
    gold = {"one": {"reference_answer": "SECRET-GOLD", "reference_answerability": "ANSWERABLE",
                    "reference_passages": [passage("ibmcld_gold", "Reference-only text.")]}}
    tasks, records = runner.judge_inputs(report, gold)
    record = records["one"]
    assert "SECRET-GOLD" not in runner.answer_prompt(tasks[0])
    assert record["retrieved_passages"] == [passage()]
    assert record["reference_passages"] == gold["one"]["reference_passages"]
    assert "SECRET-GOLD" in public_judge.build_judge_messages(record)[1]["content"]


def test_judge_denominator_contains_unattempted_answer_cases(tmp_path):
    record = {"case_id": "one", "conversation_id": "official-one", "mode": "full_rag", "question": "Enable?",
              "history": [], "answer": "I do not know.", "reference_answer": "Use console.",
              "reference_answerability": "ANSWERABLE", "reference_passages": [passage()],
              "retrieved_passages": [], "citations": []}
    expected = [{"case_id": key, "conversation_id": "official-" + key, "mode": "full_rag"} for key in ("one", "not_answered")]
    report = execute(tmp_path, stage="judge", judge_records={"one": record},
                     fake=lambda *args: response(answer="invalid JSON", calls=[]), manifest_extra={"expected_cases": expected})
    assert report["judge_summary"]["by_mode"]["full_rag"]["tasks"] == 2
    assert len(report["unattempted_judgments"]) == 1


def test_default_dry_run_never_connects_or_submits(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["public_agent.py", "--state", str(tmp_path / "absent.json")])
    monkeypatch.setattr(public_data, "load_cloud_tasks", lambda *args, **kwargs: [task()])
    monkeypatch.setattr(setup, "clients", lambda *args, **kwargs: pytest.fail("dry run connected"))
    monkeypatch.setattr(runner, "load_gold", lambda *args: pytest.fail("answer loaded gold"))
    runner.main()
    report = json.loads(capsys.readouterr().out)
    assert report["model_calls"] == 0 and report["tasks"] == 1
    assert "How can" not in json.dumps(report) and report["gold_passed_to_answer_model"] is False


def test_budget_lock_excludes_second_runner(tmp_path):
    with runner.ledger_lock(tmp_path / "budget.json"):
        with pytest.raises(setup.BootstrapError):
            with runner.ledger_lock(tmp_path / "budget.json"):
                pytest.fail("concurrent ledger accepted")


def prompt_question(prompt):
    return json.loads(prompt.split("\n", 1)[1])[-1]["text"]


def isolated_client():
    return object(), None, []


def test_concurrent_workers_are_isolated_once_only_and_results_keep_plan_order(tmp_path):
    barrier, lock = threading.Barrier(3), threading.Lock()
    calls, clients, completed = [], {}, []
    def fake(client, agent_id, prompt, conversation, parent, previous, timeout, model):
        question = prompt_question(prompt)
        with lock:
            calls.append(question)
            clients[threading.get_ident()] = id(client)
        assert conversation is None and parent is None and previous == set()
        barrier.wait(timeout=3)
        time.sleep({"job-0": .09, "job-1": .04, "job-2": 0}[question])
        with lock:
            completed.append(question)
        return response()
    report = execute(tmp_path, tasks=[task(str(i), "job-" + str(i)) for i in range(3)], fake=fake,
                     concurrency=3, client_factory=isolated_client)
    assert sorted(calls) == ["job-0", "job-1", "job-2"]
    assert len(clients) == len(set(clients.values())) == 3
    assert completed != calls
    assert [row["case_id"] for row in report["results"]] == ["0", "1", "2"]
    assert report["completed"] == 3 and report["pending_cases"] == []
    assert report["frozen_runtime_unchanged"]


def test_pending_reservations_never_oversell_global_or_run_budget(tmp_path):
    ledger = runner.Budget(tmp_path / "budget.json", limit=.9)
    lock, running, maximum = threading.Lock(), 0, 0
    def fake(*args):
        nonlocal running, maximum
        with lock:
            running += 1
            maximum = max(maximum, running)
        disk = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
        committed = sum(row.get("conservative_cost_cny", 0) + (row["reserved_cny"] if row["status"] == "pending" else 0)
                        for row in disk["attempts"])
        assert committed <= .9
        time.sleep(.04)
        with lock:
            running -= 1
        return response()
    report = execute(tmp_path, tasks=[task(str(i)) for i in range(6)], fake=fake,
                     ledger=ledger, concurrency=3, client_factory=isolated_client, max_run_cny=.9)
    assert report["completed"] == 6 and 1 <= maximum <= 2
    assert all(row["status"] == "settled" for row in ledger.state["attempts"])


def test_unknown_parallel_charge_stops_new_work_and_drains_all_inflight(tmp_path):
    barrier, lock = threading.Barrier(3), threading.Lock()
    calls = []
    def fake(client, agent_id, prompt, *args):
        question = prompt_question(prompt)
        with lock:
            calls.append(question)
        barrier.wait(timeout=3)
        if question == "job-0":
            return response(unknown=True)
        time.sleep(.08)
        return response()
    report = execute(tmp_path, tasks=[task(str(i), "job-" + str(i)) for i in range(6)], fake=fake,
                     concurrency=3, client_factory=isolated_client)
    assert sorted(calls) == ["job-0", "job-1", "job-2"]
    assert report["attempted"] == 3 and report["not_attempted"] == 3 and len(report["results"]) == 6
    assert report["shared_budget_spent_cny"] is None and report["pending_cases"] == []
    ledger = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert sorted(row["status"] for row in ledger["attempts"]) == ["settled", "settled", "unknown_usage"]
    with pytest.raises(setup.BootstrapError):
        runner.Budget(tmp_path / "budget.json")


def test_shared_client_factory_is_rejected_before_second_thread_submits(tmp_path):
    shared, started = object(), threading.Event()
    calls = []
    def fake(*args):
        calls.append(1)
        started.set()
        time.sleep(.1)
        return response()
    report = execute(tmp_path, tasks=[task(str(i)) for i in range(3)], fake=fake, concurrency=3,
                     client_factory=lambda: (shared, None, []))
    assert len(calls) <= 1 and report["stop_reason"] == "preflight_authentication"
    assert report["shared_budget_spent_cny"] is not None


def test_runtime_drift_blocks_next_paid_submission_and_records_all_planned_cases(tmp_path):
    frozen = {"version": 1}
    def fake(*args):
        frozen["version"] = 2
        return response()
    report = execute(tmp_path, tasks=[task("one"), task("two"), task("three")], fake=fake,
                     freeze_fn=lambda verify: dict(frozen))
    assert report["attempted"] == 1 and report["not_attempted"] == 2
    assert report["frozen_runtime_unchanged"] is False
    assert report["stop_reason"] == "input_or_frozen_runtime_changed"


def test_changed_normal_business_tables_cannot_be_reported_unchanged(tmp_path):
    snapshots = iter([{"sha256": "before"}, {"sha256": "after"}])
    report = execute(tmp_path, snapshot_fn=lambda: next(snapshots))
    assert report["normal_business_unchanged"] is False and report["status"] == "stopped"


def make_asset_tree(tmp_path):
    root, models = tmp_path / "data", tmp_path / "models"
    (root / "index").mkdir(parents=True)
    (models / "encoder").mkdir(parents=True)
    database, vector, model = root / "index/passages.sqlite", root / "index/vectors.npy", models / "encoder/model.onnx"
    database.write_bytes(b"pinned-sqlite")
    vector.write_bytes(b"pinned-vectors")
    model.write_bytes(b"pinned-model")
    index = {"status": "ready", "sqlite_sha256": runner.digest(database), "corpus_sha256": "a" * 64}
    setup.save_state(root / "index/manifest.json", index)
    setup.save_state(models / "manifest.json", {"models": {"encoder": {"files": {"model.onnx": {"sha256": runner.digest(model)}}}}})
    setup.save_state(root / "index/vectors.json", {**index, "vectors_sha256": runner.digest(vector),
                                                 "model_manifest_sha256": runner.digest(models / "manifest.json")})
    return root, models


def test_frozen_snapshot_binds_code_index_vectors_and_model_files(tmp_path):
    root, models = make_asset_tree(tmp_path)
    frozen = runner.runtime_snapshot("rerank", data_root=root, model_dir=models, verify_files=True)
    assert set(frozen["manifests"]) == {"index", "models", "vectors"}
    assert {"app/public_search.py", "app/public_api.py", "app/public_cache.py", "app/hybrid_models.py", "compose.public-evaluation.yml"} <= frozen["source_sha256"].keys()
    assert runner.runtime_snapshot("rerank", data_root=root, model_dir=models) == frozen
    (models / "encoder/model.onnx").write_bytes(b"changed model")
    assert runner.runtime_snapshot("rerank", data_root=root, model_dir=models) != frozen
    with pytest.raises(setup.BootstrapError):
        runner.runtime_snapshot("rerank", data_root=root, model_dir=models, verify_files=True)


def test_lexical_freeze_does_not_depend_on_in_progress_vectors(tmp_path):
    root, models = make_asset_tree(tmp_path)
    frozen = runner.runtime_snapshot("lexical", data_root=root, model_dir=models, verify_files=True)
    (root / "index/vectors.json").write_text("not finished", encoding="utf-8")
    assert runner.runtime_snapshot("lexical", data_root=root, model_dir=models) == frozen
    assert set(frozen["assets"]) == {"passages.sqlite"}


def test_parallel_requires_explicit_isolated_factory(tmp_path):
    with pytest.raises(setup.BootstrapError):
        execute(tmp_path, concurrency=2)


def test_worker_credentials_are_redacted_even_when_not_in_coordinator_secret_list(tmp_path):
    secret = "private-worker-token-unique"
    report = execute(tmp_path, fake=lambda *args: response(answer=secret), concurrency=2,
                     client_factory=lambda: (object(), None, [secret]))
    assert secret not in (tmp_path / "run/report.json").read_text(encoding="utf-8")
    assert secret not in report["results"][0]["answer"]


def test_final_snapshot_error_preserves_paid_outcome_and_blocks_isolation_claim(tmp_path):
    calls = []
    def snapshot():
        calls.append(1)
        if len(calls) > 1:
            raise OSError("private path")
        return {"sha256": "before"}
    report = execute(tmp_path, snapshot_fn=snapshot)
    assert report["completed"] == 1 and report["cost"]["conservative_budget"] > 0
    assert report["stop_reason"] == "final_isolation_snapshot_unavailable"
    assert report.get("normal_business_unchanged") is not True


def test_transient_local_file_replacement_retries_without_repeating_model_call(tmp_path, monkeypatch):
    original = runner._save_state
    attempts = []
    def temporarily_busy(path, value):
        attempts.append(path.name)
        if len(attempts) == 1:
            raise PermissionError("reader holds file")
        return original(path, value)
    monkeypatch.setattr(runner, "_save_state", temporarily_busy)
    calls = []
    def fake(*args):
        calls.append(1)
        return response()
    report = execute(tmp_path, fake=fake)
    assert len(calls) == 1 and report["completed"] == 1


def reconciliation_fixture(tmp_path):
    ledger = tmp_path / "budget.json"
    report_path = tmp_path / "failed.json"
    attempts, rows, entries = [], [], []
    for index in range(3):
        identifier, attempt_id, cid, prompt = str(index), "attempt-" + str(index), "conversation-" + str(index), "Official judge task " + str(index)
        attempts.append({"attempt_id": attempt_id, "run_id": "failed-run", "case_id": identifier,
                         "stage": "judge", "reserved_cny": .3356928, "status": "unknown_usage"})
        rows.append({"case_id": identifier, "attempt_id": attempt_id, "status": "execution_error",
                     "execution": {"conversation_id": cid, "drained": True, "prompt": prompt,
                                   "usage": {"input_tokens": None, "output_tokens": None}}})
        body = json.dumps(runner.offline_judge_payload(prompt), ensure_ascii=False, separators=(",", ":")).encode()
        entries.append({"attempt_id": attempt_id, "case_id": identifier, "conversation_id": cid,
                        "original_reserved_cny": .3356928,
                        "payload_capture": {"wire_bytes": len(body), "wire_sha256": hashlib.sha256(body).hexdigest(),
                                            "roles": ["system", "user"], "capture_count": 1, "tool_count": 0,
                                            "user_matches": True, "system_matches": True},
                        "budget_upper_bound_cny": round(7 * ((len(body) + 4096) * 2 + 2200 * 8) / 1_000_000 * 1.2, 8)})
    setup.save_state(ledger, {"limit_cny": 10, "attempts": attempts})
    setup.save_state(report_path, {"run_id": "failed-run", "stage": "judge", "parameters": {"max_tokens": 2200}, "results": rows})
    audit = {"version": "public-unknown-budget-v1", "approval_scope": "three_failed_dev8_judges_budget_only",
             "ledger_before_sha256": runner.digest(ledger), "source_report": {"path": str(report_path), "sha256": runner.digest(report_path)},
             "formula": {"currency": "CNY", "max_http_attempts": 7, "server_max_retries": 6, "input_margin_tokens": 4096,
                         "output_limit_tokens": 2200, "input_per_million": 2, "output_per_million": 8, "safety_multiplier": 1.2},
             "runtime_evidence": {"offline_network_calls": 0, "provider_reported_usage_found": False,
                                  "extra_model_calls_disabled": True, "title_convo": False,
                                  "no_files_memory_mcp_skills_tools": True, "sdk_source_sha256": {"fixture": "a" * 64}},
             "entries": entries}
    path = tmp_path / "audit.json"
    setup.save_state(path, audit)
    return path, ledger, audit


def test_audited_unknown_preserves_originals_and_consumes_budget_without_faking_usage(tmp_path):
    path, ledger, audit = reconciliation_fixture(tmp_path)
    original_ledger, original_report = ledger.read_bytes(), runner.digest(audit["source_report"]["path"])
    with pytest.raises(setup.BootstrapError):
        runner.Budget(ledger)
    dry = runner.reconcile_unknown(path, runner.digest(path), ledger, lambda cid: {"active": False})
    assert dry["status"] == "audit_validated_dry_run" and ledger.read_bytes() == original_ledger
    result = runner.reconcile_unknown(path, runner.digest(path), ledger, lambda cid: {"active": False}, apply=True)
    budget = runner.Budget(ledger)
    assert budget.known_spent == 0 and budget.spent == budget.unknown_upper_bound > 0
    assert result["provider_tokens"] is None and result["actual_cost_cny"] is None
    assert runner.digest(audit["source_report"]["path"]) == original_report
    assert Path(result["ledger_before_snapshot"]).read_bytes() == original_ledger
    assert all(row["old_status"] == "unknown_usage" and row["reserved_cny"] == .3356928 and not row["provider_usage_known"] for row in budget.state["attempts"])
    assert budget.reserve("next", "next", "judge", 10 - budget.spent + .01) is None
    with pytest.raises(setup.BootstrapError):
        runner.reconcile_unknown(path, runner.digest(path), ledger, lambda cid: {"active": False}, apply=True)


@pytest.mark.parametrize("failure", ["hash", "active", "case", "payload", "formula", "budget"])
def test_reconciliation_rejects_unverified_evidence_or_excess_budget_without_changing_ledger(tmp_path, failure):
    path, ledger, audit = reconciliation_fixture(tmp_path)
    if failure == "case":
        audit["entries"][0]["case_id"] = "other"
    elif failure == "payload":
        audit["entries"][0]["payload_capture"]["wire_bytes"] += 1
    elif failure == "formula":
        audit["formula"]["max_http_attempts"] = 1
    elif failure == "budget":
        data = setup.read_json(ledger)
        data["attempts"].append({"attempt_id": "prior", "run_id": "prior", "case_id": "prior", "stage": "answer",
                                 "status": "settled", "conservative_cost_cny": 9.99})
        setup.save_state(ledger, data)
        audit["ledger_before_sha256"] = runner.digest(ledger)
    setup.save_state(path, audit)
    before = ledger.read_bytes()
    with pytest.raises(setup.BootstrapError):
        runner.reconcile_unknown(path, "wrong" if failure == "hash" else runner.digest(path), ledger,
                                 lambda cid: {"active": failure == "active"}, apply=True)
    assert ledger.read_bytes() == before


def test_future_report_separates_known_estimates_from_audited_unknown_commitments(tmp_path):
    path, ledger, audit = reconciliation_fixture(tmp_path)
    runner.reconcile_unknown(path, runner.digest(path), ledger, lambda cid: {"active": False}, apply=True)
    budget = runner.Budget(ledger)
    report = execute(tmp_path, ledger=budget)
    assert report["shared_budget_actual_cost_cny"] is None
    assert report["shared_budget_estimated_total_cny"] is None
    assert report["shared_budget_spent_cny"] is None
    assert report["shared_budget_known_estimated_cost_cny"] > 0
    assert report["shared_budget_unknown_upper_bound_cny"] == budget.unknown_upper_bound
    assert report["shared_budget_committed_cny"] == budget.spent


def v2_audit_fixture(tmp_path):
    path, ledger, audit = reconciliation_fixture(tmp_path)
    audit["version"] = "public-unknown-budget-v2"
    audit["formula"].update(max_http_attempts=1, server_max_retries=0)
    audit["runtime_evidence"].update(verified_caller_max_retries=0, verified_sdk_max_retries=0,
                                      provider=setup.PROVIDER, provider_config_sha256="a" * 64)
    report = setup.read_json(Path(audit["source_report"]["path"]))
    observation = {"failed_messages": {}, "external_messages": {}}
    for row, entry in zip(report["results"], audit["entries"]):
        row["status"] = "completed"  # Historical runner defect: content:error was ignored.
        row["execution"].update(answers=[], usage_events=[], message_ids=["error-" + row["case_id"]])
        observation["failed_messages"][entry["conversation_id"]] = {
            "message_id": "error-" + row["case_id"], "content_types": ["error"], "provider_usage": None}
        entry["budget_upper_bound_cny"] = round(((entry["payload_capture"]["wire_bytes"] + 4096) * 2 + 2200 * 8) / 1_000_000 * 1.2, 8)
    external = {"message_id": "separate-response", "conversation_id": "conversation-0",
                "submission_origin": "unknown", "include_in_evaluation": False,
                "provider_usage": {"input": 7170, "output": 460}}
    observation["external_messages"][external["message_id"]] = external
    audit["external_observed_runs"] = [external]
    observed = tmp_path / "mongo-observation.json"
    setup.save_state(observed, observation)
    audit["mongo_observation"] = {"path": str(observed), "sha256": runner.digest(observed)}
    setup.save_state(Path(audit["source_report"]["path"]), report)
    audit["source_report"]["sha256"] = runner.digest(audit["source_report"]["path"])
    setup.save_state(path, audit)
    return path, ledger, audit


def test_zero_retry_audit_keeps_failed_rows_and_accounts_separate_observed_usage(tmp_path):
    path, ledger, audit = v2_audit_fixture(tmp_path)
    original = runner.digest(audit["source_report"]["path"])
    result = runner.reconcile_unknown(path, runner.digest(path), ledger, lambda cid: {"active": False}, apply=True)
    budget = runner.Budget(ledger)
    assert result["external_observed_cost_cny"] == .021624
    assert budget.known_spent == .021624
    assert result["budget_committed_cny"] == round(budget.spent, 8)
    assert runner.digest(audit["source_report"]["path"]) == original
    assert budget.state["attempts"][-1]["include_in_evaluation"] is False
    assert len([x for x in budget.state["attempts"] if x["status"] == "accounted_unknown"]) == 3


@pytest.mark.parametrize("failure", ["duplicate", "usage", "error_identity", "retry", "artifact_hash"])
def test_zero_retry_audit_rejects_incomplete_or_duplicate_observations(tmp_path, failure):
    path, ledger, audit = v2_audit_fixture(tmp_path)
    observation = setup.read_json(Path(audit["mongo_observation"]["path"]))
    if failure == "duplicate":
        audit["external_observed_runs"] *= 2
    elif failure == "usage":
        audit["external_observed_runs"][0]["provider_usage"]["input"] = None
        observation["external_messages"]["separate-response"] = audit["external_observed_runs"][0]
    elif failure == "error_identity":
        observation["failed_messages"]["conversation-0"]["message_id"] = "other-message"
    elif failure == "retry":
        audit["runtime_evidence"]["verified_caller_max_retries"] = 6
    setup.save_state(Path(audit["mongo_observation"]["path"]), observation)
    if failure != "artifact_hash":
        audit["mongo_observation"]["sha256"] = runner.digest(audit["mongo_observation"]["path"])
    else:
        audit["mongo_observation"]["sha256"] = "wrong"
    setup.save_state(path, audit)
    before = ledger.read_bytes()
    with pytest.raises(setup.BootstrapError):
        runner.reconcile_unknown(path, runner.digest(path), ledger, lambda cid: {"active": False}, apply=True)
    assert ledger.read_bytes() == before


def test_judge_cli_keeps_original_generation_retrieval_mode(monkeypatch, tmp_path, capsys):
    current = state()
    current["mode"] = "rerank"
    state_path, answers_path = tmp_path / "agents.json", tmp_path / "answers.json"
    setup.save_state(state_path, current)
    setup.save_state(answers_path, {"retrieval_mode": "lexical", "schedule": [{"case_id": "one", "conversation_id": "official-one", "mode": "full_rag"}],
                                  "results": [{"case_id": "one", "task": task(), "attempted": True, "status": "completed",
                                               "answer": "Unknown.", "retrieved_passages": [], "citations": []}]})
    monkeypatch.setattr(sys, "argv", ["public_agent.py", "--run", "--stage", "judge", "--answers", str(answers_path),
                                     "--state", str(state_path), "--ledger", str(tmp_path / "budget.json"), "--out", str(tmp_path / "judged")])
    monkeypatch.setattr(public_data, "load_cloud_tasks", lambda *args, **kwargs: [task()])
    monkeypatch.setattr(public_data, "verify_file", lambda *args: {"sha256": "fixture"})
    monkeypatch.setattr(runner, "load_gold", lambda *args: {"one": {"reference_answerability": "UNANSWERABLE", "reference_answer": "Unknown.", "reference_passages": []}})
    monkeypatch.setattr(setup, "clients", lambda *args, **kwargs: (setup.LocalApi("http://localhost:3080"), setup.LocalApi("http://localhost:8192"), {}, []))
    monkeypatch.setattr(setup, "validate", lambda *args: None)
    original_read = runner.read_json
    monkeypatch.setattr(runner, "read_json", lambda path: {} if Path(path).name == "demo.json" else original_read(path))
    captured = {}
    def execute_without_model(*args, **kwargs):
        captured.update(kwargs["manifest_extra"])
        return {"status": "fixture_only"}
    monkeypatch.setattr(runner, "execute", execute_without_model)
    runner.main()
    assert captured["retrieval_mode"] == "lexical" and captured["judge_agent_current_mode"] == "rerank"
