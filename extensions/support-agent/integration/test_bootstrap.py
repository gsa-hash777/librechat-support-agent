import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import MagicMock, Mock

import bootstrap


SPEC = {
    "openapi": "3.1.0",
    "servers": [{"url": "https://support-tls:8443"}],
    "paths": {
        "/api/drafts": {"post": {
            "operationId": "createTicketDraft",
            "summary": "Create a draft only",
            "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Draft"}}}},
        }},
        "/api/tickets/{ticket_id}": {"get": {
            "operationId": "getMyTicket",
            "parameters": [
                {"name": "ticket_id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "maximum": 10}},
            ],
        }},
    },
    "components": {"schemas": {"Draft": {
        "type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"],
    }}},
}


class FakeApi:
    agent = None
    action = None
    mutations = []

    def __init__(self, base):
        self.token = None

    def request(self, method, path, payload=None):
        cls = type(self)
        if path == "/openapi.json":
            return copy.deepcopy(SPEC)
        if path == "/api/auth/login":
            return {"token": "test-session", "user": {"id": "test-user"}}
        if method == "GET" and path.startswith("/api/agents?"):
            return {"data": [copy.deepcopy(cls.agent)] if cls.agent and cls.agent["name"] == bootstrap.AGENT_NAME else []}
        if method == "GET" and path.endswith("/expanded"):
            return copy.deepcopy(cls.agent)
        if method == "GET" and path == "/api/agents/actions":
            return [copy.deepcopy(cls.action)] if cls.action else []
        cls.mutations.append((method, path))
        if method == "POST" and path == "/api/agents":
            cls.agent = {**copy.deepcopy(payload), "id": "agent_demo", "author": "test-user"}
            return copy.deepcopy(cls.agent)
        if method == "PATCH":
            cls.agent.update(copy.deepcopy(payload))
            return copy.deepcopy(cls.agent)
        if method == "POST" and path == "/api/agents/actions/agent_demo":
            cls.action = {"agent_id": "agent_demo", "action_id": "action_demo", "metadata": copy.deepcopy(payload["metadata"])}
            cls.action["metadata"].pop("api_key")
            cls.agent["tools"] = [item["function"]["name"] for item in payload["functions"]]
            cls.agent["actions"] = ["support_action_demo"]
            return [copy.deepcopy(cls.agent), copy.deepcopy(cls.action)]
        raise AssertionError((method, path))


class BootstrapTests(unittest.TestCase):
    def test_local_json_requests_include_identified_browser_compatible_user_agent(self):
        api = bootstrap.LocalApi("http://localhost:3080")
        api.opener = Mock()
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{}'
        api.opener.open.return_value = response
        for method, path in [("GET", "/api/agents"), ("POST", "/api/agents/chat/abort")]:
            api.request(method, path, {} if method == "POST" else None)
            request = api.opener.open.call_args.args[0]
            self.assertEqual(request.get_header("User-agent"), bootstrap.LOCAL_USER_AGENT)
            self.assertIn("SupportAgentLab/1.0", request.get_header("User-agent"))
            self.assertIn("Chrome/", request.get_header("User-agent"))

    def test_verified_missing_user_404_registers_then_logs_in(self):
        api = Mock()
        api.request.side_effect = [
            bootstrap.ApiError(404, "/api/auth/login", "user_not_found"),
            {"message": "Registration successful"},
            {"token": "session", "user": {"id": "new-user"}},
        ]
        demo = {"email": "demo@local.test", "password": "test-password", "name": "Demo", "username": "demo"}
        user = bootstrap.authenticate(api, demo)
        self.assertEqual(user["id"], "new-user")
        self.assertEqual(api.request.call_args_list[1].args[:2], ("POST", "/api/auth/register"))
        self.assertEqual(api.request.call_args_list[1].args[2]["confirm_password"], "test-password")

    def test_other_404_and_401_login_failures_never_register(self):
        for status, reason in [(404, None), (404, "missing_credentials"), (404, "incorrect_password"), (401, None)]:
            api = Mock()
            api.request.side_effect = bootstrap.ApiError(status, "/api/auth/login", reason)
            with self.assertRaises(bootstrap.ApiError):
                bootstrap.authenticate(api, {"email": "demo@local.test", "password": "test-password"})
            self.assertEqual(api.request.call_count, 1)

    def test_login_error_parser_keeps_only_allowlisted_reason(self):
        self.assertEqual(bootstrap.login_error_reason(404, "/api/auth/login", b'{"message":"Email does not exist."}'), "user_not_found")
        self.assertIsNone(bootstrap.login_error_reason(404, "/different-route", b'{"message":"Email does not exist."}'))
        self.assertIsNone(bootstrap.login_error_reason(404, "/api/auth/login", b'{"message":"unknown secret body"}'))
        self.assertIsNone(bootstrap.login_error_reason(404, "/api/auth/login", b'<h1>Not Found</h1>'))

    def test_only_local_http_origins_receive_credentials(self):
        for url in ["https://example.com", "http://localhost@evil.test", "http://localhost:3080/path", "http://example.com"]:
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.local_base(url)
        self.assertEqual(bootstrap.local_base("http://127.0.0.1:3080/"), "http://127.0.0.1:3080")

    def test_flat_function_parameters_match_librechat_action_contract(self):
        functions = bootstrap.action_functions(SPEC)
        body = functions[0]["function"]["parameters"]
        self.assertEqual(body["required"], ["title"])
        self.assertEqual(body["properties"]["title"], {"type": "string"})
        query = functions[1]["function"]["parameters"]
        self.assertEqual(query["required"], ["ticket_id"])
        self.assertEqual(query["properties"]["limit"]["maximum"], 10)
        self.assertNotIn("Authorization", query["properties"])

    def test_external_or_recursive_schema_references_are_rejected(self):
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.resolve_refs({"$ref": "https://example.com/schema.json"}, {})
        document = {"loop": {"$ref": "#/loop"}}
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.resolve_refs(document["loop"], document)

    def test_update_proposal_schema_keeps_version_path_and_action_parameters(self):
        document = copy.deepcopy(SPEC)
        document["paths"]["/api/tickets/{ticket_id}/updates"] = {"post": {
            "operationId": "proposeTicketUpdate",
            "summary": "Create a proposal; do not execute it",
            "parameters": [{"name": "ticket_id", "in": "path", "required": True, "schema": {"type": "string"}}],
            "requestBody": {"content": {"application/json": {"schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["assign", "start", "comment", "resolve", "close"]},
                    "expected_version": {"type": "integer", "minimum": 1},
                    "request_id": {"type": "string"},
                    "note": {"type": "string"},
                    "assignee_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["action", "expected_version", "request_id", "note"],
            }}}},
        }}
        functions = bootstrap.action_functions(document)
        self.assertEqual(len(functions), len(bootstrap.action_functions(SPEC)) + 1)
        proposal = next(item["function"] for item in functions if item["function"]["name"] == "proposeTicketUpdate")
        self.assertEqual(set(proposal["parameters"]["required"]), {"ticket_id", "action", "expected_version", "request_id", "note"})
        self.assertNotIn("assignee_id", proposal["parameters"]["required"])
        self.assertNotIn("role", proposal["parameters"]["properties"])

    def test_workflow_upgrade_reuses_saved_agent_preserving_name_model_and_action(self):
        FakeApi.agent, FakeApi.action, FakeApi.mutations = None, None, []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".local").mkdir()
            (root / ".local/demo.json").write_text(json.dumps({
                "email": "demo@local.test", "name": "Demo User", "username": "demo",
                "password": "test-password", "support_user_id": "alice",
            }), encoding="utf-8")
            (root / ".local/users.json").write_text(json.dumps({"users": [{"id": "alice", "api_key": "test-tool-key"}]}), encoding="utf-8")
            (root / ".local/operations.json").write_text(json.dumps({"oauth": {"client_id": "test-client", "client_secret": "test-client-secret-" * 3, "redirect_uris": []}}), encoding="utf-8")
            with patch.object(bootstrap, "LocalApi", FakeApi):
                first = bootstrap.bootstrap(root, "http://localhost:3080", "http://localhost:8090", "deepseek-flash")
                FakeApi.agent.update({"name": "Renamed by user", "model": "user-selected-model", "model_parameters": {"temperature": 0.7}, "instructions": "Previous lab version"})
                revised = copy.deepcopy(SPEC)
                revised["paths"]["/api/updates/{id}"] = {"get": {"operationId": "getTicketUpdate"}}
                with patch.dict(SPEC, revised):
                    second = bootstrap.bootstrap(root, "http://localhost:3080", "http://localhost:8090", "different-default")
                    third = bootstrap.bootstrap(root, "http://localhost:3080", "http://localhost:8090", "different-default")
            self.assertEqual(first["agent_id"], second["agent_id"])
            self.assertEqual(second["agent_id"], third["agent_id"])
            self.assertEqual(FakeApi.agent["name"], "Renamed by user")
            self.assertEqual(FakeApi.agent["model"], "user-selected-model")
            self.assertEqual(FakeApi.agent["model_parameters"], {"temperature": 0.7})
            self.assertEqual(FakeApi.agent["instructions"], bootstrap.INSTRUCTIONS)
            self.assertEqual(FakeApi.agent["tools"].count("getTicketUpdate"), 1)
            self.assertEqual(FakeApi.mutations.count(("POST", "/api/agents")), 1)
            self.assertEqual(FakeApi.mutations.count(("PATCH", "/api/agents/agent_demo")), 1)
            self.assertEqual(FakeApi.mutations.count(("POST", "/api/agents/actions/agent_demo")), 2)

    def test_bootstrap_rerun_keeps_agent_and_action_without_new_writes(self):
        FakeApi.agent, FakeApi.action, FakeApi.mutations = None, None, []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".local").mkdir()
            (root / ".local/demo.json").write_text(json.dumps({
                "email": "demo@local.test", "name": "Demo User", "username": "demo",
                "password": "test-password", "support_user_id": "alice",
            }), encoding="utf-8")
            (root / ".local/users.json").write_text(json.dumps({"users": [{"id": "alice", "api_key": "test-tool-key"}]}), encoding="utf-8")
            (root / ".local/operations.json").write_text(json.dumps({"oauth": {"client_id": "test-client", "client_secret": "test-client-secret-" * 3, "redirect_uris": []}}), encoding="utf-8")
            with patch.object(bootstrap, "LocalApi", FakeApi):
                first = bootstrap.bootstrap(root, "http://localhost:3080", "http://localhost:8090", "deepseek-flash")
                second = bootstrap.bootstrap(root, "http://localhost:3080", "http://localhost:8090", "deepseek-flash")
            self.assertEqual(first["agent_id"], second["agent_id"])
            self.assertEqual(len(FakeApi.mutations), 2)
            state = (root / ".local/support-agent-bootstrap.json").read_text(encoding="utf-8")
            for secret in ["test-password", "test-tool-key", "test-session"]:
                self.assertNotIn(secret, state)
            self.assertFalse(first["model_called"])


if __name__ == "__main__":
    unittest.main()
