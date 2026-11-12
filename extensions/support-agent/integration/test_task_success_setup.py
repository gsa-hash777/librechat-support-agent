import copy
import io
import json
import socket
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import task_success_setup as setup
from evaluation import task_success_proxy as proxy


class FakeResponse(io.BytesIO):
    pass


class DisconnectResponse(FakeResponse):
    def __init__(self, raw, first_line, continue_reading):
        super().__init__(raw)
        self.first_line, self.continue_reading = first_line, continue_reading
        self.first = True

    def __next__(self):
        if self.first:
            self.first = False
            line = super().__next__()
            self.first_line.set()
            return line
        self.continue_reading.wait(3)
        return super().__next__()


class FakeUpstream:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        if isinstance(self.raw, Exception):
            raise self.raw
        return FakeResponse(self.raw)


class SetupTests(unittest.TestCase):
    def test_policy_adapter_changes_only_identity_paragraph(self):
        original = setup.INSTRUCTIONS.splitlines()
        actual = setup.instructions("support1").splitlines()
        changed = [(a, b) for a, b in zip(original, actual) if a != b]
        self.assertEqual(len(changed), 1)
        self.assertTrue(changed[0][0].startswith("每位 LibreChat 用户须通过 OAuth"))
        self.assertIn("固定绑定 support1", changed[0][1])
        self.assertEqual(len(actual), len(original))

    def test_policy_adapter_fails_on_missing_identity_rule(self):
        with patch.object(setup, "INSTRUCTIONS", "modified policy"):
            with self.assertRaises(setup.BootstrapError):
                setup.instructions("alice")

    def test_all_seven_fixed_agents_use_guard_and_limits(self):
        self.assertEqual(len(setup.ACTORS), 7)
        self.assertEqual(len({x[1] for x in setup.ACTORS.values()}), 2)
        self.assertEqual(len({x[0] for x in setup.ACTORS.values()}), 3)
        for actor in setup.ACTORS:
            agent = setup.desired_agent(actor)
            self.assertEqual(agent["provider"], "DeepSeekTaskEval")
            self.assertEqual(agent["model_parameters"], {"temperature": .2, "max_tokens": 1200, "maxContextTokens": 32768})
            self.assertEqual(agent["recursion_limit"], 12)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "budget.json"
        self.ledger = proxy.Ledger(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_reservation_prevents_parallel_request_and_recovers_actual_cost(self):
        ident = self.ledger.reserve(b"request", 1200)
        with self.assertRaises(proxy.GuardError):
            self.ledger.reserve(b"second", 1200)
        self.ledger.finish(ident, {"prompt_tokens": 100, "completion_tokens": 20})
        self.assertAlmostEqual(self.ledger.snapshot()["committed_cny"], .00036)
        self.assertFalse(self.ledger.snapshot()["blocked"])

    def test_unknown_usage_retains_reserved_and_blocks_restart(self):
        ident = self.ledger.reserve(b"request", 1200)
        expected = self.ledger.snapshot()["committed_cny"]
        self.ledger.finish(ident, None)
        restarted = proxy.Ledger(self.path)
        self.assertEqual(restarted.snapshot()["committed_cny"], expected)
        with self.assertRaises(proxy.GuardError):
            restarted.reserve(b"second", 1200)

    def test_pending_after_crash_blocks_restart(self):
        self.ledger.reserve(b"request", 1200)
        with self.assertRaises(proxy.GuardError):
            proxy.Ledger(self.path).reserve(b"second", 1200)

    def test_budget_rejects_before_forward(self):
        with self.assertRaises(proxy.GuardError) as cm:
            self.ledger.reserve(b"x" * 2_500_000, 1200)
        self.assertEqual(cm.exception.status, 402)
        self.assertEqual(self.ledger.snapshot()["attempts"], [])

    def test_invalid_usage_and_bound_violation_never_zero_cost(self):
        ident = self.ledger.reserve(b"request", 1200)
        self.ledger.finish(ident, {"prompt_tokens": 999999, "completion_tokens": 12})
        state = self.ledger.snapshot()
        self.assertEqual(state["attempts"][0]["status"], "bound_violation")
        self.assertTrue(state["blocked"])
        self.assertGreater(state["committed_cny"], 1)


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ledger = proxy.Ledger(Path(self.temp.name) / "budget.json")
        self.server = None

    def tearDown(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()
        self.temp.cleanup()

    def start(self, raw):
        self.upstream = FakeUpstream(raw)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), proxy.handler_type(self.ledger, "fake-key", self.upstream))
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)

    def post(self, payload=None):
        body = {"model": "deepseek-flash", "messages": [{"role": "user", "content": "test"}], "max_tokens": 1200, "thinking": {"type": "disabled"}, "stream": True}
        if payload:
            body.update(payload)
        return urlopen(Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": "Bearer fake-key", "Content-Type": "application/json"}), timeout=3)

    def test_stream_usage_settled_before_done(self):
        self.start(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: {"usage":{"prompt_tokens":100,"completion_tokens":20},"choices":[]}\n\ndata: [DONE]\n\n')
        with self.post() as response:
            self.assertIn(b"[DONE]", response.read())
        self.assertEqual(self.ledger.snapshot()["attempts"][0]["status"], "settled")
        self.assertEqual(self.upstream.calls, 1)

    def test_stream_without_usage_blocks_next_request(self):
        self.start(b'data: {"choices":[]}\n\ndata: [DONE]\n\n')
        with self.post() as response:
            response.read()
        with self.assertRaises(HTTPError) as cm:
            self.post()
        self.assertEqual(cm.exception.code, 409)
        self.assertEqual(self.upstream.calls, 1)
        self.assertEqual(self.ledger.snapshot()["attempts"][0]["status"], "unknown_usage")

    def test_nonstream_usage(self):
        self.start(b'{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20}}')
        with self.post({"stream": False}) as response:
            self.assertIn(b"usage", response.read())
        self.assertFalse(self.ledger.snapshot()["blocked"])

    def test_client_disconnect_does_not_erase_provider_usage(self):
        self.start(b'')
        first_line, continue_reading = threading.Event(), threading.Event()
        completed = threading.Event()
        raw = b'data: {"choices":[]}\n\ndata: {"usage":{"prompt_tokens":100,"completion_tokens":20}}\n\ndata: [DONE]\n\n'
        self.upstream.open = lambda *_args, **_kwargs: DisconnectResponse(raw, first_line, continue_reading)
        original_finish = self.ledger.finish
        def finish(*args, **kwargs):
            original_finish(*args, **kwargs)
            completed.set()
        self.ledger.finish = finish
        response = self.post()
        self.assertTrue(first_line.wait(2))
        response.close()
        continue_reading.set()
        self.assertTrue(completed.wait(3))
        self.assertEqual(self.ledger.snapshot()["attempts"][0]["status"], "settled")

    def test_network_failure_retains_reservation(self):
        self.start(OSError("simulated"))
        with self.assertRaises(HTTPError):
            self.post({"stream": False})
        self.assertEqual(self.ledger.snapshot()["attempts"][0]["status"], "unknown_usage")
        self.assertEqual(self.upstream.calls, 1)

    def test_unapproved_model_or_unlimited_output_is_never_forwarded(self):
        self.start(b'{}')
        for changes in ({"model": "other"}, {"max_tokens": None}, {"thinking": {"type": "enabled"}}, {"max_tokens": 1501}):
            with self.assertRaises(HTTPError) as cm:
                self.post(changes)
            self.assertEqual(cm.exception.code, 400)
        self.assertEqual(self.upstream.calls, 0)
        self.assertEqual(self.ledger.snapshot()["attempts"], [])

    def test_multimodal_requests_are_never_forwarded_or_billed(self):
        self.start(b'{}')
        for content in ([{"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}], [{"type": "input_audio", "input_audio": {"data": "fake", "format": "wav"}}], [{"type": "text", "text": "ok", "image_url": "hidden"}]):
            with self.assertRaises(HTTPError) as cm:
                self.post({"messages": [{"role": "user", "content": content}]})
            self.assertEqual(cm.exception.code, 400)
        self.assertEqual(self.upstream.calls, 0)
        self.assertEqual(self.ledger.snapshot()["attempts"], [])

    def test_text_blocks_and_function_calls_are_supported(self):
        body = {"model": "deepseek-flash", "max_tokens": 1200, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "question"}]}, {"role": "assistant", "content": None, "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "getCurrentSupportUser", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "call1", "content": "result"}]}
        parsed, _ = proxy.request_payload(json.dumps(body).encode())
        self.assertEqual(len(parsed["messages"]), 3)


if __name__ == "__main__":
    unittest.main()
