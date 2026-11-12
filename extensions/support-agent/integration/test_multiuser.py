"""Provisioning safety checks; all LibreChat calls and local admin CLI are mocked."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bootstrap
import multiuser
from test_bootstrap import FakeApi as BootstrapFakeApi

RESOURCE_ID = "0123456789abcdef01234567"

class FakeApi:
    users = {}
    calls = []
    permissions = set()
    fail_login_once = set()
    missing_id = set()
    owner_id = "lc-alice"

    def __init__(self, base):
        self.base, self.token, self.user_id = base, None, None

    def request(self, method, path, payload=None):
        cls = type(self)
        cls.calls.append((method, path, copy.deepcopy(payload), self.user_id))
        if path == "/api/auth/register":
            raise AssertionError("Multiuser provisioning must not fall back to HTTP registration.")
        if path == "/api/auth/login":
            email = payload["email"]
            if email in cls.fail_login_once:
                cls.fail_login_once.remove(email)
                raise bootstrap.ApiError(429, path)
            item = cls.users.get(email)
            if item is None:
                raise bootstrap.ApiError(404, path, "user_not_found")
            if item["password"] != payload["password"]:
                raise bootstrap.ApiError(401, path, "incorrect_password")
            self.user_id = item["id"]
            user = {"id": item["id"], "role": item.get("role", "USER")}
            if email in cls.missing_id:
                user.pop("id")
            return {"token": "session-test-secret", "user": user}
        if method == "GET" and path == "/api/agents/agent_demo/expanded":
            return {"id": "agent_demo", "_id": RESOURCE_ID, "author": cls.owner_id}
        if method == "PUT" and path == "/api/permissions/agent/" + RESOURCE_ID:
            if self.user_id != cls.owner_id:
                raise AssertionError("Only the authenticated Agent owner may share it.")
            cls.permissions.update(item["id"] for item in payload["updated"])
            cls.permissions.difference_update(item["id"] for item in payload["removed"])
            return {"success": True}
        if method == "GET" and path == "/api/agents/agent_demo":
            if self.user_id != cls.owner_id and self.user_id not in cls.permissions:
                raise bootstrap.ApiError(403, path)
            return {"id": "agent_demo"}
        raise AssertionError((method, path))


class MultiuserTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / ".local").mkdir()
        self.owner = {"support_user_id": "alice", "name": "Owner User", "username": "owner",
                      "email": "alice@local.test", "password": "owner-password-test-secret"}
        self.business = [
            {"id": "alice", "display_name": "林晓", "role": "requester"},
            {"id": "bob", "display_name": "陈明", "role": "requester"},
            {"id": "admin", "display_name": "管理员", "role": "admin"},
            {"id": "inactive", "display_name": "停用", "active": False},
        ]
        self.write("demo.json", self.owner)
        self.write("users.json", {"users": self.business})
        self.write("support-agent-bootstrap.json", {"agent_id": "agent_demo", "action_id": "action_demo"})
        FakeApi.users = {self.owner["email"]: {"id": "lc-alice", "password": self.owner["password"], "role": "USER"}}
        FakeApi.calls, FakeApi.permissions = [], set()
        FakeApi.fail_login_once, FakeApi.missing_id = set(), set()
        self.native_calls = []

    def write(self, name, value):
        (self.root / ".local" / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def native(self, root, accounts):
        self.assertEqual(root, self.root)
        self.native_calls.append(copy.deepcopy(accounts))
        result = {}
        for account in accounts:
            self.assertGreaterEqual(len(account["name"]), 3)
            email, ident = account["email"], account["support_user_id"]
            if email not in FakeApi.users:
                FakeApi.users[email] = {"id": "lc-" + ident, "password": account["password"], "role": "USER"}
            result[ident] = FakeApi.users[email]["id"]
        return result

    def run_setup(self, apply=True, native=None):
        with patch.object(multiuser, "LocalApi", FakeApi), patch.object(multiuser, "provision_local_accounts", native or self.native, create=True):
            return multiuser.provision(self.root, apply=apply)

    def stored_accounts(self):
        return json.loads((self.root / ".local/librechat-users.json").read_text(encoding="utf-8"))["accounts"]

    def permission_calls(self):
        return [call for call in FakeApi.calls if call[:2] == ("PUT", "/api/permissions/agent/" + RESOURCE_ID)]

    def test_preview_is_read_only_does_not_contact_services_or_expose_credentials(self):
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        result = self.run_setup(apply=False)
        after = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(FakeApi.calls, [])
        self.assertEqual(self.native_calls, [])
        self.assertEqual(result["permission"], "agent_viewer")
        self.assertFalse(result["public_sharing"])
        self.assertFalse(result["model_called"])
        self.assertNotIn(self.owner["password"], json.dumps(result))
        self.assertNotIn("inactive", result["business_accounts"])

    def test_each_account_logs_in_once_owner_is_reused_and_only_viewer_access_is_shared(self):
        result = self.run_setup()
        self.assertEqual(result["status"], "ready")
        logins = [call[2]["email"] for call in FakeApi.calls if call[1] == "/api/auth/login"]
        self.assertCountEqual(logins, ["alice@local.test", "bob@local.test", "admin@local.test"])
        self.assertEqual(logins.count(self.owner["email"]), 1)
        shared = self.permission_calls()[0][2]
        self.assertFalse(shared["public"])
        self.assertEqual(shared["removed"], [])
        self.assertEqual({item["id"] for item in shared["updated"]}, {"lc-bob", "lc-admin"})
        self.assertTrue(all(item["type"] == "user" and item["accessRoleId"] == "agent_viewer" for item in shared["updated"]))
        self.assertTrue(all(item["librechat_role"] == "USER" for item in result["accounts"]))
        self.assertFalse(any("role" in (call[2] or {}) for call in FakeApi.calls))
        self.assertFalse(any(call[1] == "/api/auth/register" for call in FakeApi.calls))
        self.assertNotIn("session-test-secret", json.dumps(result))
        for account in self.stored_accounts():
            self.assertNotIn(account["password"], json.dumps(result))

    def test_repeat_apply_keeps_passwords_and_does_not_create_accounts_again(self):
        self.run_setup()
        original = {item["support_user_id"]: copy.deepcopy(item) for item in self.stored_accounts()}
        first_native_count = sum(len(accounts) for accounts in self.native_calls)
        self.run_setup()
        after = {item["support_user_id"]: item for item in self.stored_accounts()}
        self.assertEqual(original, after)
        self.assertEqual(sum(len(accounts) for accounts in self.native_calls), first_native_count)
        self.assertEqual({item["id"] for item in self.permission_calls()[-1][2]["updated"]}, {"lc-bob", "lc-admin"})

    def test_partial_login_failure_keeps_generated_credentials_for_retry(self):
        FakeApi.fail_login_once.add("bob@local.test")
        with self.assertRaises(bootstrap.ApiError):
            self.run_setup()
        self.assertEqual(self.permission_calls(), [])
        saved = {item["support_user_id"]: item["password"] for item in self.stored_accounts()}
        self.run_setup()
        self.assertEqual(saved, {item["support_user_id"]: item["password"] for item in self.stored_accounts()})
        self.assertEqual(len(FakeApi.users), 3)

    def test_unexpected_global_admin_role_is_rejected_without_permission_changes(self):
        def admin_native(root, accounts):
            result = self.native(root, accounts)
            FakeApi.users["bob@local.test"]["role"] = "ADMIN"
            return result
        with self.assertRaises(bootstrap.BootstrapError):
            self.run_setup(native=admin_native)
        self.assertEqual(self.permission_calls(), [])

    def test_changed_immutable_user_id_is_rejected(self):
        self.run_setup()
        FakeApi.calls.clear()
        FakeApi.users["bob@local.test"]["id"] = "different-lc-identity"
        with self.assertRaises(bootstrap.BootstrapError):
            self.run_setup()
        self.assertEqual(self.permission_calls(), [])

    def test_missing_user_id_is_not_converted_to_none_principal(self):
        FakeApi.missing_id.add("bob@local.test")
        with self.assertRaises(bootstrap.BootstrapError):
            self.run_setup()
        self.assertEqual(self.permission_calls(), [])

    def test_two_business_accounts_cannot_silently_share_owner_identity(self):
        def duplicate_native(root, accounts):
            result = self.native(root, accounts)
            FakeApi.users["bob@local.test"]["id"] = "lc-alice"
            result["bob"] = "lc-alice"
            return result
        with self.assertRaises(bootstrap.BootstrapError):
            self.run_setup(native=duplicate_native)
        self.assertEqual(self.permission_calls(), [])

    def test_saved_inactive_and_removed_accounts_are_not_logged_in_and_managed_access_is_removed(self):
        self.run_setup()
        accounts = self.stored_accounts()
        self.business = [item for item in self.business if item["id"] != "admin"]
        next(item for item in self.business if item["id"] == "bob")["active"] = False
        self.write("users.json", {"users": self.business})
        FakeApi.calls.clear()
        before_passwords = {item["support_user_id"]: item["password"] for item in accounts}
        result = self.run_setup()
        logins = [call[2]["email"] for call in FakeApi.calls if call[1] == "/api/auth/login"]
        self.assertEqual(logins, ["alice@local.test"])
        self.assertEqual({item["business_account"] for item in result["accounts"]}, {"alice"})
        self.assertEqual({item["id"] for item in self.permission_calls()[-1][2]["removed"]}, {"lc-bob", "lc-admin"})
        self.assertEqual(before_passwords, {item["support_user_id"]: item["password"] for item in self.stored_accounts()})
        self.assertEqual(FakeApi.permissions, set())

    def test_error_cli_does_not_print_upstream_body_or_passwords(self):
        secret = "upstream-body-password-and-token"
        error = bootstrap.ApiError(429, "/api/auth/login")
        error.upstream_body = secret
        stderr = io.StringIO()
        with patch.object(multiuser, "provision", side_effect=error), patch("sys.argv", ["multiuser.py", "--root", str(self.root), "--apply"]), patch("sys.stderr", stderr):
            result = multiuser.main()
        self.assertEqual(result, 1)
        self.assertNotIn(secret, stderr.getvalue())
        self.assertNotIn(self.owner["password"], stderr.getvalue())
        self.assertIn("429", stderr.getvalue())

    def test_native_cli_passes_credentials_only_via_stdin(self):
        accounts = [{**self.owner, "support_user_id": "bob"}]
        completed = SimpleNamespace(returncode=0, stdout='non-secret startup log\nSUPPORT_PROVISION_RESULT={"bob":"lc-bob"}\n', stderr="")
        with patch.object(multiuser.subprocess, "run", return_value=completed) as run:
            result = multiuser.provision_local_accounts(self.root, accounts)
        self.assertEqual(result, {"bob": "lc-bob"})
        arguments = run.call_args.args[0]
        self.assertIsInstance(arguments, list)
        self.assertNotIn(self.owner["password"], " ".join(arguments))
        self.assertNotIn(self.owner["email"], " ".join(arguments))
        self.assertEqual(json.loads(run.call_args.kwargs["input"]), accounts)
        self.assertTrue(run.call_args.kwargs["capture_output"])
        self.assertEqual(run.call_args.kwargs["cwd"], self.root)

    def test_native_cli_failure_is_opaque_even_if_logs_contain_secrets(self):
        secret = "secret-log-body-do-not-print"
        completed = SimpleNamespace(returncode=1, stdout=secret, stderr=secret)
        with patch.object(multiuser.subprocess, "run", return_value=completed):
            with self.assertRaises(bootstrap.BootstrapError) as raised:
                multiuser.provision_local_accounts(self.root, [self.owner])
        self.assertNotIn(secret, str(raised.exception))

    def test_login_missing_user_never_falls_back_to_http_registration(self):
        def missing_native(root, accounts):
            result = self.native(root, accounts)
            FakeApi.users.pop("bob@local.test")
            return result
        with self.assertRaises(bootstrap.ApiError):
            self.run_setup(native=missing_native)
        self.assertFalse(any(call[1] == "/api/auth/register" for call in FakeApi.calls))
        self.assertEqual(self.permission_calls(), [])


class OAuthMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / ".local").mkdir()
        (self.root / ".local/demo.json").write_text(json.dumps({"support_user_id": "alice", "email": "demo@local.test",
            "name": "Demo User", "username": "demo", "password": "migration-password-test"}), encoding="utf-8")
        (self.root / ".local/operations.json").write_text(json.dumps({"oauth": {"client_id": "test-client",
            "client_secret": "migration-client-secret-" * 3, "redirect_uris": []}}), encoding="utf-8")
        BootstrapFakeApi.agent, BootstrapFakeApi.action, BootstrapFakeApi.mutations = None, None, []

    def run_bootstrap(self, base="http://localhost:3080"):
        with patch.object(bootstrap, "LocalApi", BootstrapFakeApi):
            return bootstrap.bootstrap(self.root, base, "http://localhost:8090", "deepseek-flash")

    def test_old_fixed_key_action_is_migrated_in_place(self):
        first = self.run_bootstrap()
        original_id = BootstrapFakeApi.action["action_id"]
        BootstrapFakeApi.action["metadata"].update({"domain": "http://support-agent:8090", "api_key": "old-fixed-secret",
            "auth": {"type": "service_http", "authorization_type": "bearer"}, "raw_spec": "old-spec"})
        second = self.run_bootstrap()
        self.assertEqual(first["agent_id"], second["agent_id"])
        self.assertEqual(BootstrapFakeApi.action["action_id"], original_id)
        self.assertEqual(BootstrapFakeApi.action["metadata"]["auth"]["type"], "oauth")
        self.assertEqual(BootstrapFakeApi.action["metadata"]["domain"], "https://support-tls:8443")
        self.assertNotIn("api_key", BootstrapFakeApi.action["metadata"])
        self.assertEqual(BootstrapFakeApi.mutations.count(("POST", "/api/agents")), 1)
        self.assertEqual(second["authentication"], "per_user_oauth")

    def test_manual_auth_drift_is_repaired_even_if_raw_spec_and_local_fingerprint_match(self):
        self.run_bootstrap()
        original_spec = BootstrapFakeApi.action["metadata"]["raw_spec"]
        BootstrapFakeApi.action["metadata"]["auth"] = {"type": "service_http", "authorization_type": "bearer"}
        self.run_bootstrap()
        self.assertEqual(BootstrapFakeApi.action["metadata"]["raw_spec"], original_spec)
        self.assertEqual(BootstrapFakeApi.action["metadata"]["auth"]["type"], "oauth")
        self.assertEqual(BootstrapFakeApi.mutations.count(("POST", "/api/agents/actions/agent_demo")), 2)

    def test_callback_uses_canonical_librechat_origin_and_no_secret_enters_state_or_report(self):
        report = self.run_bootstrap(base="http://127.0.0.1:3080")
        operations = json.loads((self.root / ".local/operations.json").read_text(encoding="utf-8"))
        self.assertEqual(operations["oauth"]["redirect_uris"], ["http://localhost:3080/api/actions/action_demo/oauth/callback"])
        state = (self.root / ".local/support-agent-bootstrap.json").read_text(encoding="utf-8")
        for secret in ("migration-password-test", "migration-client-secret-" * 3, "test-session"):
            self.assertNotIn(secret, state)
            self.assertNotIn(secret, json.dumps(report))


if __name__ == "__main__":
    unittest.main()
