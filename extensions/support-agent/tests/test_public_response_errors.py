from unittest.mock import patch

import pytest

from evaluation import agent


class ResponseApi:
    def __init__(self, content, error=False):
        self.content = content
        self.error = error
        self.posts = []

    def request(self, method, path, payload=None):
        if method == "POST":
            self.posts.append(path)
            return {"streamId": "s", "conversationId": "c", "generationCreatedAt": 1}
        if path.startswith("/api/agents/chat/status/"):
            return {"active": False}
        if path.startswith("/api/messages/"):
            return [{"messageId": "m", "isCreatedByUser": False, "content": self.content, "error": self.error}]
        raise AssertionError(path)


@pytest.mark.parametrize("content,error", [
    ([{"type": "error", "error": 'Upstream request timed out.'}], False),
    ([], False),
    ([{"type": "text", "text": "Unsuccessful response"}], True),
])
def test_provider_error_or_empty_message_cannot_complete(content, error):
    api = ResponseApi(content, error)
    with patch.object(agent, "stream"):
        result = agent.turn(api, "a", "prompt", None, None, set(), 180, "deepseek-flash")
    assert result["execution_error"]
    assert result["drained"]
    assert result["usage"]["total_tokens"] is None
    assert api.posts == ["/api/agents/chat"]
