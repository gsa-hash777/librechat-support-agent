import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('prepare_retrieval', Path(__file__).resolve().parents[1]/'scripts/prepare_retrieval.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class Response(BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status, self.headers = status, headers or {}


@pytest.fixture
def artifact():
    body = b'pinned model weights'
    return body, {'size':len(body), 'sha256':hashlib.sha256(body).hexdigest()}


def test_resume_only_matching_range(tmp_path, monkeypatch, artifact):
    body, entry = artifact
    partial = tmp_path/'model.part'
    partial.write_bytes(body[:6])
    def fetch(request, timeout):
        assert request.headers['Range'] == 'bytes=6-'
        return Response(body[6:], 206, {'Content-Range': f'bytes 6-{len(body)-1}/{len(body)}'})
    monkeypatch.setattr(download, 'urlopen', fetch)
    download.download('https://example.invalid/model', partial, entry)
    assert partial.read_bytes() == body


def test_ignored_range_replaces_partial(tmp_path, monkeypatch, artifact):
    body, entry = artifact
    partial = tmp_path/'model.part'
    partial.write_bytes(body[:6])
    monkeypatch.setattr(download, 'urlopen', lambda *a, **k: Response(body))
    download.download('https://example.invalid/model', partial, entry)
    assert partial.read_bytes() == body


def test_silent_short_read_retries_and_checks_hash(tmp_path, monkeypatch, artifact):
    body, entry = artifact
    partial = tmp_path/'model.part'
    replies = iter([Response(body[:6]), Response(body[6:], 206, {'Content-Range':f'bytes 6-{len(body)-1}/{len(body)}'})])
    monkeypatch.setattr(download, 'urlopen', lambda *a, **k: next(replies))
    download.download('https://example.invalid/model', partial, entry)
    assert download.verified(partial, entry)


def test_wrong_range_never_appended(tmp_path, monkeypatch, artifact):
    body, entry = artifact
    partial = tmp_path/'model.part'
    partial.write_bytes(body[:6])
    monkeypatch.setattr(download, 'urlopen', lambda *a, **k: Response(body[7:], 206, {'Content-Range':'bytes 7-19/20'}))
    with pytest.raises(RuntimeError, match='byte range'):
        download.download('https://example.invalid/model', partial, entry)
    assert partial.read_bytes() == body[:6]


def test_wrong_hash_never_succeeds(tmp_path, monkeypatch, artifact):
    body, entry = artifact
    partial = tmp_path/'model.part'
    monkeypatch.setattr(download, 'urlopen', lambda *a, **k: Response(b'x'*len(body)))
    with pytest.raises(RuntimeError, match='integrity'):
        download.download('https://example.invalid/model', partial, entry)


def test_git_blob_checksum_includes_header(tmp_path):
    path = tmp_path/'config.json'
    path.write_bytes(b'{}')
    entry = {'size':2, 'git_blob_sha1':hashlib.sha1(b'blob 2\0{}').hexdigest()}
    assert download.verified(path, entry)
