import json
from types import SimpleNamespace

import pytest

import oauth_smoke


def test_callback_requires_actual_registered_origin_action_and_state():
    base = "http://localhost:3080"
    valid = base + "/api/actions/action123/oauth/callback?code=private&state=native"
    assert oauth_smoke.checked_callback(valid, "action123", "native", base=base).startswith("/api/actions/")
    for invalid in (valid.replace(":3080", ":3900"), valid.replace("action123", "other"),
                    valid.replace("state=native", "state=forged"), valid + "&state=native", valid.replace("code=private", "error=denied")):
        with pytest.raises(oauth_smoke.SmokeError):
            oauth_smoke.checked_callback(invalid, "action123", "native", base=base)


def test_hidden_fields_decode_entities_without_copying_visible_text():
    form = oauth_smoke.HiddenFields('<p>private</p><input type="hidden" name="csrf" value="a&amp;b"><input name="password" value="private">')
    assert form.fields == {"csrf": "a&b"}


def test_oauth_rejects_unexpected_authorization_target_before_login_or_bind():
    api = SimpleNamespace(base="http://localhost:3080", request=lambda *args: pytest.fail("No network request allowed"))
    for url in ("https://evil.invalid/console/oauth/authorize", "http://localhost:8443/console/oauth/authorize"):
        with pytest.raises(oauth_smoke.SmokeError, match="authorization_target_invalid"):
            oauth_smoke.complete_oauth(api, url, support_user={}, action_id="a1", client_id="client", tls_context=None)


def test_storage_check_is_read_only_and_emits_only_boolean_projection(monkeypatch, tmp_path):
    expected = {"accounts": [{"account": "alice", "access_present": True, "refresh_present": True,
                              "bound_to_lc_user": True, "encrypted_at_rest": True}], "distinct_user_ciphertexts": True, "distinct_refresh_ciphertexts": True}

    def run(command, **kwargs):
        source = kwargs["input"]
        assert ".find(" in source and "decryptV2" in source
        assert not any(action in source for action in (".update", ".insert", ".delete", "jwt.sign"))
        assert "write(JSON.stringify({accounts:results" in source
        return SimpleNamespace(returncode=0, stdout=json.dumps(expected))

    monkeypatch.setattr(oauth_smoke.subprocess, "run", run)
    actual = oauth_smoke.storage_evidence(tmp_path, [{"support_user_id": "alice", "librechat_user_id": "a" * 24}], "action123")
    assert actual == expected


def test_storage_rejects_untrusted_identifiers_before_subprocess(monkeypatch, tmp_path):
    monkeypatch.setattr(oauth_smoke.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected subprocess"))
    with pytest.raises(oauth_smoke.SmokeError):
        oauth_smoke.storage_evidence(tmp_path, [{"support_user_id": "alice", "librechat_user_id": "bad"}], "action123")


def test_failed_model_submission_is_not_retried_and_usage_remains_unknown():
    calls = []

    def request(*args):
        calls.append(args)
        raise OSError("private-token")

    row = oauth_smoke.one_turn(SimpleNamespace(request=request), {"support_user_id": "alice"}, {"id": "alice"},
                               {"agent_id": "a", "action_id": "b", "model": "deepseek-flash"}, "client", None, 30)
    assert len(calls) == 1
    assert row["status"] == "failed"
    assert row["usage"]["input_tokens"] is None
    assert row["error_code"] == "chat_or_oauth_transport_failed"
    assert "private-token" not in json.dumps(row)


def test_viewer_preflight_uses_view_endpoint_and_one_login_each(monkeypatch):
    calls = []
    accounts = [{"support_user_id": name, "email": name, "password": "private", "librechat_user_id": name} for name in ("alice", "bob")]

    class Api:
        def __init__(self, base):
            self.account = None

        def request(self, method, path, payload=None):
            if path == "/api/auth/login":
                self.account = payload["email"]
                calls.append((self.account, method, path))
                return {"token": "private", "user": {"id": self.account}}
            calls.append((self.account, method, path))
            if path.endswith("/expanded"):
                assert self.account == "alice"
                return {"provider": "DeepSeek", "model": "deepseek-flash", "tools": ["getCurrentSupportUser_action"]}
            return {"id": "shared-agent"}  # Viewer cannot read model/tools.

    monkeypatch.setattr(oauth_smoke, "LocalApi", Api)
    state = {"agent_id": "shared-agent"}
    assert len(oauth_smoke.preflight_accounts(accounts, state, {"model": "deepseek-flash"})) == 2
    assert calls == [("alice", "POST", "/api/auth/login"), ("alice", "GET", "/api/agents/shared-agent/expanded"),
                     ("bob", "POST", "/api/auth/login"), ("bob", "GET", "/api/agents/shared-agent")]
    assert state["model"] == "deepseek-flash"


def test_initial_sse_attachment_replays_early_action_oauth_without_resume_snapshot():
    """Regression: native initial attach replays the early Action auth delta;
    its resume snapshot deliberately has no non-MCP OAuth replay event.
    """
    connected = []
    auth_url = "https://localhost:8443/console/oauth/authorize?state=private"
    delta = {"event": "on_run_step_delta", "data": {"delta": {"tool_calls": [{"name": "getCurrentSupportUser_action_test"}], "auth": auth_url}}}

    class Response:
        def __init__(self, events):
            self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=SimpleNamespace(settimeout=lambda value: None)))
            self.lines = iter(line for event in events for line in [("data: " + json.dumps(event) + "\n").encode(), b"\n"])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def readline(self, limit):
            return next(self.lines, b"")

    class Opener:
        def open(self, request, timeout):
            query = oauth_smoke.parse_qs(oauth_smoke.urlsplit(request.full_url).query)
            assert query["generationCreatedAt"] == ["123"]
            # Simulates the actual server's initial-vs-resume contract.
            events = [delta, {"final": True}] if query["resume"] == ["false"] else [{"sync": True, "resumeState": {"runSteps": [], "aggregatedContent": []}}]
            return Response(events)

    observed, counts = set(), {}
    result = oauth_smoke.stream(SimpleNamespace(base="http://localhost:3080", token="private", opener=Opener()),
                                {"streamId": "stream", "generationCreatedAt": 123}, oauth_smoke.time.monotonic() + 10,
                                on_auth=connected.append, usage_events=[], observed_tools=observed, event_counts=counts)
    assert connected == [auth_url] and result["oauth_observed"] is True
    assert observed == {"getCurrentSupportUser_action_test"}
    assert counts == {"on_run_step_delta": 1, "final": 1}


def test_sync_pending_events_and_duplicate_auth_are_consumed_once():
    connected, observed, counts = [], set(), {}
    consumer = oauth_smoke.StreamConsumer(connected.append, [], observed, counts)
    event = {"event": "on_run_step_delta", "data": {"delta": {"tool_calls": [{"name": "getCurrentSupportUser_action_test"}], "auth": "private-auth-url"}}}
    consumer.consume({"sync": True, "resumeState": {"runSteps": [{"stepDetails": {"tool_calls": [{"name": "getCurrentSupportUser_action_test"}]}}],
                                                   "replayEvents": [event]}, "pendingEvents": [event]})
    assert connected == ["private-auth-url"]
    assert observed == {"getCurrentSupportUser_action_test"}
    assert "private-auth-url" not in json.dumps(counts)


def test_unexpected_tool_in_sync_snapshot_is_recorded_and_rejected():
    observed = set()
    consumer = oauth_smoke.StreamConsumer(lambda value: pytest.fail("No authorization for another tool"), [], observed, {})
    with pytest.raises(oauth_smoke.SmokeError, match="unexpected_tool"):
        consumer.consume({"sync": True, "resumeState": {"aggregatedContent": [{"type": "tool_call", "tool_call": {"name": "createTicketDraft"}}]}})
    assert observed == {"createTicketDraft"}


def test_repair_budget_keeps_original_unknown_reservation_and_report_hash(tmp_path):
    baseline = {"status": "failed", "results": [{"account": "alice", "status": "passed"}, {"account": "bob", "status": "failed"}]}
    ledger = {"budget_cny": .2, "attempts": [{"account": "alice", "status": "settled", "estimated_cny": .02209, "reserved_cny": .1},
                                            {"account": "bob", "status": "unknown_usage", "estimated_cny": None, "reserved_cny": .1}]}
    original = json.dumps(baseline).encode()
    (tmp_path / "oauth-report.json").write_bytes(original)
    (tmp_path / "oauth-ledger.json").write_text(json.dumps(ledger))
    _, actual, digest, committed = oauth_smoke.recovery_context(tmp_path)
    assert actual == ledger
    assert digest == oauth_smoke.hashlib.sha256(original).hexdigest()
    assert committed == pytest.approx(.12209)
    assert committed + oauth_smoke.REPAIR_RESERVATION_CNY == pytest.approx(.17209)
    assert (tmp_path / "oauth-report.json").read_bytes() == original
    ledger["repair_attempt"] = {"status": "pending"}
    (tmp_path / "oauth-ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(oauth_smoke.SmokeError, match="unused_repair"):
        oauth_smoke.recovery_context(tmp_path)


def test_refresh_isolation_survives_expired_alice_access_but_requires_new_bob_access():
    storage = {"accounts": [{"account": name, "refresh_present": True, "refresh_encrypted": True, "bound_to_lc_user": True,
                              "access_present": name == "bob", "access_encrypted": name == "bob"} for name in ("alice", "bob")],
               "distinct_refresh_ciphertexts": True, "distinct_user_ciphertexts": False}
    assert oauth_smoke.recovery_storage_pass(storage)
    storage["accounts"][1]["access_present"] = False
    assert not oauth_smoke.recovery_storage_pass(storage)
