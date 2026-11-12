"""Startup safety: operations overlay cannot silently revert to shared credentials."""
import io
import json
from pathlib import Path
import sys

import pytest

from scripts import project


@pytest.fixture
def local_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project, "ROOT", tmp_path)
    monkeypatch.setattr(project, "SERVICE", tmp_path / "extensions/support-agent")
    return tmp_path


def test_default_start_keeps_operations_with_or_without_retrieval(local_root):
    lexical = project.operations_compose()
    assert lexical[-2:] == ["-f", "compose.operations.yml"]
    assert "compose.retrieval.yml" not in lexical
    models = local_root / ".local/models/support-retrieval/manifest.json"
    models.parent.mkdir(parents=True)
    models.write_text("{}")
    with_models = project.operations_compose()
    assert with_models[-4:] == ["-f", "compose.retrieval.yml", "-f", "compose.operations.yml"]


def test_explicit_retrieval_start_requires_prepared_models(local_root):
    with pytest.raises(RuntimeError, match="PrepareRetrieval"):
        project.operations_compose(require_retrieval=True)


def test_start_prepares_tls_before_starting_only_business_services(local_root, monkeypatch):
    calls = []
    monkeypatch.setattr(project, "run", lambda *args: calls.append(args))
    monkeypatch.setattr(project, "prepare_local_tls", lambda: calls.append(("tls",)))
    monkeypatch.setattr(project, "bootstrap_operations", lambda: calls.append(("bootstrap",)))
    project.start_operations()
    assert Path(calls[0][-1]).parts[-2:] == ("scripts", "init.py")
    assert calls[1] == ("tls",)
    assert calls[2][-4:] == ("mongodb", "support-agent", "support-tls", "api")
    assert "compose.operations.yml" in calls[2]
    assert not any("support-public" in call for call in calls)
    assert calls[-1] == ("bootstrap",)


def test_stop_includes_tls_but_does_not_stop_public(local_root, monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["project.py", "Stop"])
    monkeypatch.setattr(project, "run", lambda *args: calls.append(args))
    assert project.main() == 0
    assert calls[0][-5:] == ("stop", "api", "support-tls", "support-agent", "mongodb")
    assert "support-public" not in calls[0]


def test_debug_retains_operations_and_only_recreates_support_after_one_start(local_root, monkeypatch):
    (local_root / "compose.debug.yml").write_text("services: {}")
    calls = []
    monkeypatch.setattr(project, "start_operations", lambda: calls.append(("normal-start",)))
    monkeypatch.setattr(project, "run", lambda *args: calls.append(args))
    monkeypatch.setattr(project, "require_operations_ready", lambda: calls.append(("ready",)))
    project.start_debug()
    assert calls[0] == ("normal-start",)
    assert calls[1].index("compose.operations.yml") < calls[1].index("compose.debug.yml")
    assert "--no-deps" in calls[1] and calls[1][-1] == "support-agent"
    assert "support-public" not in calls[1]
    assert calls[-1] == ("ready",)
    assert sum(call == ("normal-start",) for call in calls) == 1


def test_debug_missing_overlay_fails_before_start(local_root, monkeypatch):
    monkeypatch.setattr(project, "start_operations", lambda: pytest.fail("must not modify the deployment"))
    with pytest.raises(RuntimeError, match="compose.debug.yml is missing"):
        project.start_debug()


def test_open_apps_requests_both_entrypoints_in_system_browser(monkeypatch):
    calls = []
    monkeypatch.setattr(project.webbrowser, "open", lambda url, **kwargs: calls.append((url, kwargs)) or True)
    project.open_apps()
    assert calls == [("http://localhost:3080", {"new": 2}),
                     ("https://localhost:8443/console", {"new": 2})]


def test_open_apps_reports_launch_failure_without_claiming_navigation_success(monkeypatch):
    results = iter([True, False])
    monkeypatch.setattr(project.webbrowser, "open", lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match="same system-browser profile"):
        project.open_apps()


def tls_files(root):
    directory = root / ".local/support-tls"
    directory.mkdir(parents=True)
    (directory / "server.crt").write_text("fixture certificate")
    (directory / "server.key").write_text("fixture key")


def test_already_trusted_certificate_never_requests_trust_again(local_root, monkeypatch):
    tls_files(local_root)
    calls = []
    monkeypatch.setattr(project.shutil, "which", lambda name: "pwsh")
    monkeypatch.setattr(project, "run_json", lambda *args: calls.append(args) or {})
    monkeypatch.setattr(project, "tls_trusted", lambda _: True)
    project.prepare_local_tls()
    assert len(calls) == 1
    assert "-Trust" not in calls[0]


def test_untrusted_certificate_requests_trust_once_and_rechecks(local_root, monkeypatch):
    tls_files(local_root)
    calls, trusted = [], iter([False, True])
    monkeypatch.setattr(project.shutil, "which", lambda name: "pwsh")
    monkeypatch.setattr(project, "run_json", lambda *args: calls.append(args) or {})
    monkeypatch.setattr(project, "tls_trusted", lambda _: next(trusted))
    project.prepare_local_tls()
    assert sum("-Trust" in args for args in calls) == 1


def test_certificate_validity_failure_is_explicit(local_root, monkeypatch):
    monkeypatch.setattr(project, "run_json", lambda *args: {"trusted": True, "valid": False})
    with pytest.raises(RuntimeError, match="expired or not yet valid"):
        project.tls_trusted("pwsh")


@pytest.mark.parametrize("callback_changed", [True, False])
def test_bootstrap_restarts_only_support_when_callback_changed(local_root, monkeypatch, callback_changed):
    commands, readiness = [], []
    monkeypatch.setattr(project, "require_operations_ready", lambda: readiness.append(True))
    monkeypatch.setattr(project, "run_json", lambda *args: {
        "status": "ready", "authentication": "per_user_oauth", "support_restart_required": callback_changed,
        "model_called": False,
    })
    monkeypatch.setattr(project, "run", lambda *args: commands.append(args))
    result = project.bootstrap_operations()
    assert result["model_called"] is False
    if callback_changed:
        assert len(commands) == 1
        assert commands[0][-3:] == ("restart", "--no-deps", "support-agent")
        assert len(readiness) == 2
    else:
        assert commands == []
        assert len(readiness) == 1


@pytest.mark.parametrize("legacy", [True, False])
def test_bootstrap_requires_operations_runtime_flags(local_root, monkeypatch, legacy):
    responses = iter([{"status": "ready"}, {"servers": [{"url": "https://support-tls:8443"}]}])
    monkeypatch.setattr(project, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(next(responses)).encode()))
    monkeypatch.setattr(project, "run_json", lambda *args: {"operations": not legacy})
    if legacy:
        with pytest.raises(RuntimeError, match="legacy shared tool keys disabled"):
            project.require_operations_ready()
    else:
        project.require_operations_ready()


def test_workspace_has_new_no_model_tasks_and_keeps_paid_tasks_labelled():
    workspace = Path(__file__).resolve().parents[3] / "SupportAgent.code-workspace"
    tasks = json.loads(workspace.read_text(encoding="utf-8"))["tasks"]["tasks"]
    by_action = {task["args"][-1]: task for task in tasks}
    for action in ["OperationsStart", "StartDebug", "OpenApps", "Backup", "Monitor", "MultiUser"]:
        assert "no model calls" in by_action[action]["label"]
    assert "100 paid tasks" in by_action["AgentBaseline"]["label"]
