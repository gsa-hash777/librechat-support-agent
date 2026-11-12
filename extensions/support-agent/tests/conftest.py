import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.main import create_app


@pytest.fixture
def users():
    return [
        {"id": "alice", "display_name": "林晓", "org_id": "demo", "password": "alice-test-password", "api_key": "alice-test-api-key-" + "a" * 32},
        {"id": "bob", "display_name": "陈明", "org_id": "demo", "password": "bob-test-password", "api_key": "bob-test-api-key-" + "b" * 32},
        {"id": "eve", "display_name": "其他组织", "org_id": "other", "password": "eve-test-password", "api_key": "eve-test-api-key-" + "e" * 32},
    ]


@pytest.fixture
def settings(tmp_path, users):
    credentials = tmp_path / "users.json"
    credentials.write_text(json.dumps({"users": users}, ensure_ascii=False), encoding="utf-8")
    seed = tmp_path / "knowledge.json"
    seed.write_text(json.dumps([
        {"id": "reset-password", "org_id": "demo", "title": "密码重置", "content": "sharedmarker 忘记密码后请通过邮箱验证重置密码。", "source": "fixture://password"},
        {"id": "vpn-connect", "org_id": "demo", "title": "VPN 连接排查", "content": "sharedmarker VPN 连接失败时检查网络连接与客户端版本。", "source": "fixture://vpn"},
        {"id": "other-secret", "org_id": "other", "title": "其他组织资料", "content": "sharedmarker 其他组织专属资料，仅对该组织用户可见。", "source": "fixture://other"},
    ], ensure_ascii=False), encoding="utf-8")
    return Settings(
        db_path=tmp_path / "support.db",
        credentials_path=credentials,
        public_base_url="http://testserver",
        tool_base_url="http://testserver",
        seed_path=seed,
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as active:
        yield active


@pytest.fixture
def headers(users):
    return {user["id"]: {"Authorization": "Bearer " + user["api_key"]} for user in users}
