"""Optional retrieval preserves identity checks, evidence gates and tool boundaries."""
from dataclasses import replace

import pytest

from app.config import Settings
from app.retrieval import retrieve
from app.security import token_hash
from test_service import login


@pytest.fixture
def settings(settings, request):
    return replace(settings, **getattr(request, "param", {}))


class RecordingEngine:
    def __init__(self, callback=None, *, insufficient=False, error=False):
        self.calls = []
        self.callback = callback
        self.insufficient = insufficient
        self.error = error

    def retrieve(self, chunks, query, mode, limit):
        self.calls.append({"chunks": chunks, "mode": mode, "limit": limit})
        if self.callback:
            self.callback()
        if self.error:
            raise RuntimeError("PRIVATE_MODEL_PATH private-api-key-12345 private-query")
        result = retrieve(chunks, query, limit)
        if self.insufficient:
            result.update(results=[], evidence_status="insufficient", reason_codes=["missing_fact_evidence"])
        return {**result, "retrieval_mode": mode,
                "ranked_results": [{"content": "INTERNAL_UNGATED_SECRET", "org_id": "private"}],
                "retrieval_metadata": {"scores_are_answer_confidence": False}}


def search(client, headers, *, surface="api", mode="hybrid", query="忘记密码如何重置"):
    path = "/api/knowledge/search" if surface == "api" else "/console/knowledge"
    return client.get(path, params={"q": query, "mode": mode}, headers=headers["alice"] if surface == "api" else {})


def test_default_lexical_does_not_construct_or_call_models(client, app, headers):
    engine = RecordingEngine(error=True)
    app.state.retrieval_engine = engine
    response = client.get("/api/knowledge/search", params={"q": "忘记密码如何重置", "limit": 1}, headers=headers["alice"])
    assert response.status_code == 200
    result = response.json()
    assert result["retrieval_mode"] == "lexical"
    assert len(result["results"]) == 1
    assert result["results"][0]["citation"].startswith("reset-password@v1:")
    assert not engine.calls


@pytest.mark.parametrize("mode", ["hybrid", "rerank"])
@pytest.mark.parametrize("surface", ["api", "console"])
def test_neural_modes_require_explicit_enablement(client, app, headers, users, mode, surface):
    engine = RecordingEngine()
    app.state.retrieval_engine = engine
    login(client, users)
    response = search(client, headers, surface=surface, mode=mode)
    assert response.status_code == 503
    assert "尚未启用" in response.text
    assert "sharedmarker" not in response.text
    assert not engine.calls


@pytest.mark.parametrize("settings", [{"retrieval_enabled": True, "retrieval_mode": "rerank"}], indirect=True)
def test_configured_default_explicit_override_and_console_selector(client, app, headers, users):
    engine = RecordingEngine()
    app.state.retrieval_engine = engine
    response = client.get("/api/knowledge/search", params={"q": "忘记密码如何重置"}, headers=headers["alice"])
    assert response.status_code == 200 and response.json()["retrieval_mode"] == "rerank"
    assert engine.calls[-1]["mode"] == "rerank"
    response = search(client, headers, mode="lexical")
    assert response.status_code == 200 and response.json()["retrieval_mode"] == "lexical"
    assert len(engine.calls) == 1
    login(client, users)
    page = client.get("/console/knowledge")
    assert '<select id="mode" name="mode">' in page.text
    assert '<option value="rerank" selected>混合检索并重排</option>' in page.text
    assert "关键词与向量混合检索" in page.text


@pytest.mark.parametrize("settings", [{"retrieval_enabled": True}], indirect=True)
@pytest.mark.parametrize("actor,org", [("alice", "demo"), ("eve", "other")])
def test_engine_only_receives_authenticated_organization(client, app, headers, actor, org):
    engine = RecordingEngine()
    app.state.retrieval_engine = engine
    response = client.get("/api/knowledge/search", params={"q": "sharedmarker", "mode": "hybrid", "limit": 2}, headers=headers[actor])
    assert response.status_code == 200
    assert {chunk["org_id"] for chunk in engine.calls[0]["chunks"]} == {org}
    assert engine.calls[0]["limit"] == 2
    assert "INTERNAL_UNGATED_SECRET" not in response.text
    assert "ranked_results" not in response.text
    assert "other-secret" not in response.text if org == "demo" else "reset-password" not in response.text


@pytest.mark.parametrize("settings", [{"retrieval_enabled": True}], indirect=True)
@pytest.mark.parametrize("surface", ["api", "console"])
def test_evidence_gate_does_not_expose_internal_candidates(client, app, headers, users, surface):
    app.state.retrieval_engine = RecordingEngine(insufficient=True)
    login(client, users)
    response = search(client, headers, surface=surface)
    assert response.status_code == 200
    assert "INTERNAL_UNGATED_SECRET" not in response.text
    assert "sharedmarker" not in response.text
    if surface == "api":
        assert response.json()["results"] == []
        assert response.json()["evidence_status"] == "insufficient"
        assert response.json()["reason_codes"] == ["missing_fact_evidence"]
    else:
        assert "目前没有足够证据回答" in response.text


@pytest.mark.parametrize("settings", [{"retrieval_enabled": True}], indirect=True)
@pytest.mark.parametrize("surface", ["api", "console"])
def test_model_error_has_safe_503_and_no_lexical_fallback(client, app, headers, users, surface):
    app.state.retrieval_engine = RecordingEngine(error=True)
    login(client, users)
    response = search(client, headers, surface=surface)
    assert response.status_code == 503
    assert "本地检索模型暂不可用" in response.text
    for secret in ("PRIVATE_MODEL_PATH", "private-api-key", "private-query", "sharedmarker"):
        assert secret not in response.text


@pytest.mark.parametrize("settings", [{"retrieval_enabled": True}], indirect=True)
@pytest.mark.parametrize("surface,revocation", [
    ("api", "disabled"), ("api", "organization"), ("api", "key"),
    ("console", "disabled"), ("console", "organization"),
    ("console", "session_deleted"), ("console", "session_expired"),
])
def test_revocation_during_inference_commits_and_blocks_evidence(client, app, headers, users, surface, revocation):
    login(client, users)
    mutations = {
        "disabled": ("UPDATE users SET active=0 WHERE id='alice'", ()),
        "organization": ("UPDATE users SET org_id='other' WHERE id='alice'", ()),
        "key": ("UPDATE users SET api_key_hash=? WHERE id='alice'", (token_hash("rotated-tool-key"),)),
        "session_deleted": ("DELETE FROM sessions WHERE user_id='alice'", ()),
        "session_expired": ("UPDATE sessions SET expires_at=0 WHERE user_id='alice'", ()),
    }
    state = {"committed": False}

    def revoke_while_inference_runs():
        # A real independent writer must commit now, not after inference returns.
        with app.state.database.connection() as connection:
            connection.execute("PRAGMA busy_timeout=100")
            connection.execute(*mutations[revocation])
        state["committed"] = True

    app.state.retrieval_engine = RecordingEngine(revoke_while_inference_runs)
    response = search(client, headers, surface=surface)
    assert state["committed"], "Inference must not retain the SQLite writer/read transaction."
    assert response.status_code == 401
    assert "sharedmarker" not in response.text
    assert "INTERNAL_UNGATED_SECRET" not in response.text
    event = "knowledge.search" if surface == "api" else "knowledge.console.search"
    with app.state.database.connection() as connection:
        latest = connection.execute("SELECT outcome FROM audit_events WHERE event=? ORDER BY id DESC LIMIT 1", (event,)).fetchone()
        assert latest["outcome"] == "denied"


def test_mode_validation_and_tool_schema_remain_bounded(client, headers):
    response = search(client, headers, mode="PRIVATE_INVALID_MODE")
    assert response.status_code == 422
    assert "PRIVATE_INVALID_MODE" not in response.text
    schema = client.get("/openapi.json").json()
    operations = [operation for path in schema["paths"].values() for operation in path.values()]
    assert len(operations) == 12
    operation = schema["paths"]["/api/knowledge/search"]["get"]
    assert operation["operationId"] == "searchSupportKnowledge"
    parameter = next(item for item in operation["parameters"] if item["name"] == "mode")
    assert not parameter["required"]
    enums = [item["enum"] for item in parameter["schema"]["anyOf"] if "enum" in item]
    assert enums == [["lexical", "hybrid", "rerank"]]


def test_retrieval_environment_configuration(monkeypatch, tmp_path):
    values = {"SUPPORT_RETRIEVAL_MODE": "hybrid", "SUPPORT_RETRIEVAL_ENABLED": "true",
              "SUPPORT_MODELS_PATH": str(tmp_path / "models"),
              "SUPPORT_VECTOR_CACHE_PATH": str(tmp_path / "vectors"), "SUPPORT_RETRIEVAL_THREADS": "1"}
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    settings = Settings.from_env()
    assert settings.retrieval_enabled and settings.retrieval_mode == "hybrid"
    assert settings.models_path == tmp_path / "models"
    assert settings.vector_cache_path == tmp_path / "vectors"
    assert settings.retrieval_threads == 1
    monkeypatch.setenv("SUPPORT_RETRIEVAL_MODE", "unknown")
    with pytest.raises(ValueError, match="SUPPORT_RETRIEVAL_MODE"):
        Settings.from_env()
