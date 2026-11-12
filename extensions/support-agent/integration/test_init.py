import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.init import initialize, main


class InitTests(unittest.TestCase):
    def test_first_init_creates_local_roles_and_rerun_preserves_every_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize(root)
            paths = [root / ".env", root / ".local/demo.json", root / ".local/users.json"]
            original_bytes = {path: path.read_bytes() for path in paths}
            document = json.loads((root / ".local/users.json").read_text(encoding="utf-8"))
            users = {user["id"]: user for user in document["users"]}
            self.assertEqual({key: user["role"] for key, user in users.items()}, {
                "alice": "requester", "bob": "requester", "support1": "support",
                "support2": "support", "admin": "admin",
            })
            self.assertEqual(users["support1"]["display_name"], "王工")
            self.assertEqual(users["support2"]["display_name"], "陈工")
            self.assertEqual(users["admin"]["display_name"], "服务台管理员")
            self.assertTrue(all(user["org_id"] == "demo" for user in users.values()))
            self.assertEqual(len({user["password"] for user in users.values()}), len(users))
            self.assertEqual(len({user["api_key"] for user in users.values()}), len(users))
            initialize(root)
            for path, content in original_bytes.items():
                self.assertEqual(path.read_bytes(), content)

    def test_existing_records_metadata_env_and_passwords_are_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / ".local"
            local.mkdir()
            existing = [
                {"id": "alice", "display_name": "原名字", "org_id": "demo", "password": "alice-secret", "api_key": "alice-key"},
                {"id": "bob", "org_id": "demo", "role": "requester", "password": "bob-secret", "api_key": "bob-key"},
                {"id": "support1", "org_id": "existing-org", "role": "requester", "active": False, "password": "existing", "api_key": "existing-key"},
                {"id": "admin", "org_id": "demo", "role": "support", "password": "admin-existing", "api_key": "admin-existing-key"},
            ]
            document = {"users": existing, "private_metadata": {"revision": 4}}
            (local / "users.json").write_text(json.dumps(document), encoding="utf-8")
            env_content = b"# Keep comments\r\nDEEPSEEK_API_KEY=local-secret\r\nJWT_SECRET=existing-jwt\r\n"
            demo_content = b'{"password":"existing-demo-password","support_user_id":"alice"}\r\n'
            (root / ".env").write_bytes(env_content)
            (local / "demo.json").write_bytes(demo_content)
            initialize(root)
            result = json.loads((local / "users.json").read_text(encoding="utf-8"))
            self.assertEqual(result["users"][:len(existing)], existing)
            self.assertEqual(result["private_metadata"], {"revision": 4})
            self.assertEqual([user["id"] for user in result["users"]], ["alice", "bob", "support1", "admin", "support2"])
            self.assertEqual((root / ".env").read_bytes(), env_content)
            self.assertEqual((local / "demo.json").read_bytes(), demo_content)

    def test_invalid_existing_users_fail_without_changing_configuration(self):
        invalid_documents = ["not JSON", '{"users":{}}', '{"users":[{"id":"alice"},{"id":"alice"}]}']
        for raw in invalid_documents:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / ".local").mkdir()
                path = root / ".local/users.json"
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    initialize(root)
                self.assertEqual(path.read_text(encoding="utf-8"), raw)
                self.assertFalse((root / ".env").exists())
                self.assertFalse((root / ".local/demo.json").exists())

    def test_cli_output_never_contains_generated_passwords_or_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(sys, "argv", ["init.py", "--root", str(root)]), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(main(), 0)
            users = json.loads((root / ".local/users.json").read_text(encoding="utf-8"))["users"]
            output = stdout.getvalue() + stderr.getvalue()
            for user in users:
                self.assertNotIn(user["password"], output)
                self.assertNotIn(user["api_key"], output)


if __name__ == "__main__":
    unittest.main()
