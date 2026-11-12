import unittest
import time
from unittest.mock import MagicMock, Mock

from bootstrap import LOCAL_USER_AGENT, LocalApi
from smoke import message_evidence, redact, stream_events


class SmokeTests(unittest.TestCase):
    def test_sse_uses_same_identified_user_agent_as_local_api(self):
        api = LocalApi("http://localhost:3080")
        api.token = "test-session"
        api.opener = Mock()
        response = MagicMock()
        response.__enter__.return_value.__iter__.return_value = iter([b'data: {"final":true}\n', b'\n'])
        api.opener.open.return_value = response
        count, final = stream_events(api, {"streamId": "test", "generationCreatedAt": 1}, time.monotonic() + 10)
        self.assertEqual(count, 1)
        self.assertTrue(final["final"])
        request = api.opener.open.call_args.args[0]
        self.assertEqual(request.get_header("User-agent"), LOCAL_USER_AGENT)

    def test_reports_exclude_known_secrets_and_sensitive_fields(self):
        value = {"answer": "token=secret-test-123 Bearer unknown-token", "password": "other", "nested": [{"api_key": "private"}]}
        redacted = redact(value, ["secret-test-123"])
        self.assertEqual(redacted["answer"], "token=[REDACTED] Bearer [REDACTED]")
        self.assertEqual(redacted["password"], "[REDACTED]")
        self.assertEqual(redacted["nested"][0]["api_key"], "[REDACTED]")

    def test_evidence_selects_final_text_and_tool_output_without_reasoning(self):
        answers, calls = message_evidence([
            {"isCreatedByUser": True, "text": "question"},
            {"messageId": "response", "content": [
                {"type": "think", "think": "private reasoning"},
                {"type": "text", "text": "Answer [KB-002]"},
                {"type": "tool_call", "tool_call": {"id": "call1", "name": "searchSupportKnowledge", "args": {"q": "SSO"}, "output": "source"}},
            ]},
        ])
        self.assertEqual(answers[0]["text"], "Answer [KB-002]")
        self.assertEqual(calls[0]["output"], "source")
        self.assertNotIn("private reasoning", str((answers, calls)))


if __name__ == "__main__":
    unittest.main()
