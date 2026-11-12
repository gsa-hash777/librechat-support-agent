"""Backup safety checks use temporary data only; no Docker or live services."""
import gzip
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile

import pytest

from scripts import backup


def database(path: Path) -> None:
    db = sqlite3.connect(path)
    db.executescript("CREATE TABLE tickets(id TEXT PRIMARY KEY, title TEXT); INSERT INTO tickets VALUES('T-1', 'private fixture');")
    db.close()


def file_archive(path: Path, extra: tuple[str, bytes] | None = None) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name in ["uploads", "data"]:
            entry = tarfile.TarInfo(name)
            entry.type = tarfile.DIRTYPE
            archive.addfile(entry)
        for name, content in [extra or ("uploads/example.txt", b"private upload"), ("data/session.bin", b"runtime")]:
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))


@pytest.fixture
def bundle(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    database(directory / "support.db")
    with gzip.open(directory / "mongo.archive.gz", "wb") as stream:
        stream.write(b"fixture: full Mongo restore is separately tested using fake Docker")
    file_archive(directory / "librechat-files.tar.gz")
    (directory / "knowledge-source").mkdir()
    (directory / "knowledge-source/knowledge.md").write_text("# private source\n", encoding="utf-8")
    backup.write_manifest(directory, {
        "sqlite": backup.sqlite_summary(directory / "support.db"),
        "completed_at": backup.utc_now(),
        "images": {"mongodb": "sha256:" + "a" * 64},
    })
    return directory


def test_sqlite_backup_captures_committed_wal_not_uncommitted_changes(tmp_path):
    source = tmp_path / "source.db"
    writer = sqlite3.connect(source)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.executescript("CREATE TABLE records(value TEXT); INSERT INTO records VALUES('committed');")
        writer.execute("INSERT INTO records VALUES('not committed')")
        snapshot = tmp_path / "snapshot.db"
        summary = backup.sqlite_snapshot(source, snapshot)
        assert summary["tables"] == {"records": 1}
        restored = sqlite3.connect(snapshot)
        try:
            assert restored.execute("SELECT value FROM records").fetchall() == [("committed",)]
        finally:
            restored.close()
        assert source.with_name("source.db-wal").exists()
        sealed = sqlite3.connect(snapshot)
        try:
            assert sealed.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        finally:
            sealed.close()
        assert not Path(str(snapshot) + "-wal").exists()
        assert not Path(str(snapshot) + "-shm").exists()
    finally:
        writer.rollback()
        writer.close()


def test_snapshot_never_overwrites_existing_destination(tmp_path):
    source, destination = tmp_path / "source.db", tmp_path / "existing.db"
    database(source)
    destination.write_bytes(b"leave this untouched")
    with pytest.raises(backup.BackupError, match="destination must be new"):
        backup.sqlite_snapshot(source, destination)
    assert destination.read_bytes() == b"leave this untouched"


def test_cold_helper_retains_uncheckpointed_wal_and_leaves_source_unchanged(tmp_path):
    source = tmp_path / "source.db"
    writer = sqlite3.connect(source)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.executescript("CREATE TABLE records(value TEXT); INSERT INTO records VALUES('committed in WAL');")
        # Keep the connection open to retain WAL; no writes occur during capture.
        wal = Path(str(source) + "-wal")
        before = (backup.sha256(source), backup.sha256(wal))
        target = tmp_path / "cold-snapshot.db"
        with target.open("xb") as stream:
            subprocess.run([sys.executable, "-c", backup.SQLITE_STREAM, str(source), "cold"],
                           stdout=stream, stderr=subprocess.PIPE, check=True)
        assert backup.sqlite_summary(target)["tables"] == {"records": 1}
        assert before == (backup.sha256(source), backup.sha256(wal))
    finally:
        writer.close()


def test_foreign_key_violation_fails_validation(tmp_path):
    source = tmp_path / "invalid.db"
    db = sqlite3.connect(source)
    db.executescript("CREATE TABLE parent(id INTEGER PRIMARY KEY); CREATE TABLE child(parent_id REFERENCES parent(id)); INSERT INTO child VALUES(999);")
    db.close()
    with pytest.raises(backup.BackupError, match="foreign_key_check"):
        backup.sqlite_summary(source)


def test_bundle_and_isolated_partial_restore_preserve_source(bundle, tmp_path):
    original = {name: backup.sha256(path) for name, path in backup.files_under(bundle).items()}
    verified = backup.verify_bundle(bundle)
    result = backup.restore_drill(bundle, tmp_path / "drills", skip_mongo=True,
                                  manifest_sha256=verified["manifest_sha256"])
    target = Path(result["restore_directory"])
    assert result["status"] == "partial_pass"
    assert result["production_modified"] is False
    assert result["mongo"]["status"] == "skipped"
    assert result["sqlite"]["tables"] == {"tickets": 1}
    assert (target / "librechat/uploads/example.txt").read_bytes() == b"private upload"
    assert (target / "knowledge-source/knowledge.md").read_text() == "# private source\n"
    assert original == {name: backup.sha256(path) for name, path in backup.files_under(bundle).items()}
    another = backup.restore_drill(bundle, tmp_path / "drills", skip_mongo=True)
    assert another["restore_directory"] != result["restore_directory"]


@pytest.mark.parametrize("mutation", ["modified", "missing", "extra", "incomplete", "manifest"])
def test_tampering_missing_and_partial_bundles_rejected(bundle, mutation):
    digest = backup.sha256(bundle / "manifest.json")
    if mutation == "modified":
        (bundle / "knowledge-source/knowledge.md").write_text("changed")
    elif mutation == "missing":
        (bundle / "mongo.archive.gz").unlink()
    elif mutation == "extra":
        (bundle / "unexpected.txt").write_text("extra")
    elif mutation == "incomplete":
        (bundle / ".incomplete").write_text("failed")
    else:
        (bundle / "manifest.json").write_text("{}")
    with pytest.raises(backup.BackupError):
        backup.verify_bundle(bundle, digest)


def test_payload_and_manifest_modified_together_fail_external_digest(bundle):
    digest = backup.sha256(bundle / "manifest.json")
    target = bundle / "knowledge-source/knowledge.md"
    target.write_text("changed")
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"]["knowledge-source/knowledge.md"] = {"bytes": target.stat().st_size, "sha256": backup.sha256(target)}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(backup.BackupError, match="independently retained"):
        backup.verify_bundle(bundle, digest)


def test_old_empty_wal_metadata_is_never_read_or_changed(bundle):
    digest = backup.sha256(bundle / "manifest.json")
    (bundle / "support.db-wal").write_bytes(b"")
    (bundle / "support.db-shm").write_bytes(b"untrusted shared-memory content")
    before = {name: backup.sha256(path) for name, path in backup.files_under(bundle).items()}
    verified = backup.verify_bundle(bundle, digest)
    assert verified["ignored_empty_wal_metadata"] == ["support.db-wal", "support.db-shm"]
    assert verified["manifest"]["sqlite"]["tables"] == {"tickets": 1}
    assert before == {name: backup.sha256(path) for name, path in backup.files_under(bundle).items()}


def test_nonempty_unverified_wal_is_rejected_before_sqlite_open(bundle, monkeypatch):
    (bundle / "support.db-wal").write_bytes(b"unverified transaction data")
    monkeypatch.setattr(backup, "sqlite_summary", lambda *args, **kwargs: pytest.fail("must reject before reading SQLite"))
    with pytest.raises(backup.BackupError, match="nonempty SQLite WAL"):
        backup.verify_bundle(bundle)


def test_old_manifested_empty_wal_metadata_is_hash_checked_but_never_consumed(bundle):
    wal, shm = bundle / "support.db-wal", bundle / "support.db-shm"
    wal.write_bytes(b"")
    shm.write_bytes(b"old private snapshot metadata")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path in [wal, shm]:
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": backup.sha256(path)}
    manifest_path.write_text(json.dumps(manifest))
    digest = backup.sha256(manifest_path)
    assert backup.verify_bundle(bundle, digest)["ignored_empty_wal_metadata"] == ["support.db-wal", "support.db-shm"]
    shm.write_bytes(b"modified metadata")
    with pytest.raises(backup.BackupError, match="size or SHA-256"):
        backup.verify_bundle(bundle, digest)


def test_verification_of_wal_mode_snapshot_never_creates_sidecars(bundle):
    connection = sqlite3.connect(bundle / "support.db")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.close()
    # This fixture snapshot has no pending WAL. Bind its current main-file bytes.
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"]["support.db"]["sha256"] = backup.sha256(bundle / "support.db")
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    before = set(backup.files_under(bundle))
    backup.verify_bundle(bundle)
    assert set(backup.files_under(bundle)) == before


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/drive", "uploads/../../escape", "uploads\\escape",
                                   "uploads/file:stream", "uploads/CON.txt", "uploads/name.", "uploads/./alias"])
def test_unsafe_archive_paths_are_rejected_before_creating_restore_directory(tmp_path, name):
    archive = tmp_path / "evil.tar.gz"
    file_archive(archive, (name, b"unsafe"))
    target = tmp_path / "restored"
    with pytest.raises(backup.BackupError):
        backup.extract_files(archive, target)
    assert not target.exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_archive_links_and_special_files_rejected(tmp_path, kind):
    path = tmp_path / "links.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("uploads/escape")
        entry.type, entry.linkname = kind, "../../escape"
        archive.addfile(entry)
    with pytest.raises(backup.BackupError, match="Links and special"):
        backup.archive_entries(path)


def test_archive_case_collision_rejected(tmp_path):
    path = tmp_path / "collision.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name in ["uploads/A", "uploads/a"]:
            archive.addfile(tarfile.TarInfo(name), io.BytesIO(b""))
    with pytest.raises(backup.BackupError, match="Duplicate"):
        backup.archive_entries(path)


def test_restore_cannot_be_nested_inside_bundle(bundle):
    with pytest.raises(backup.BackupError, match="inside its source bundle"):
        backup.restore_drill(bundle, bundle / "restore", skip_mongo=True)


class FakeDocker:
    def __init__(self):
        self.calls = []
        self.containers = {service: {"Id": service + "-id", "State": {"Running": True},
                                    "Image": "sha256:" + "a" * 64,
                                    "Config": {"Env": ["SUPPORT_DB_PATH=/data/support.db"], "Labels": {}}}
                           for service in ["api", "support-agent", "mongodb"]}
        self.temp_name = None

    def deployment(self):
        return self.containers

    def run(self, *args, output=None, timeout=300):
        self.calls.append(args)
        if output is not None:
            if args[0] == "exec" and "mongodump" in args:
                with gzip.open(output, "wb") as stream:
                    stream.write(b"mongo-fixture")
            elif backup.SQLITE_STREAM in args:
                database(output)
            elif backup.FILES_STREAM in args:
                file_archive(output)
            else:
                raise AssertionError(args)
        if "--detach" in args:
            self.temp_name = args[args.index("--name") + 1]
        if any("JSON.stringify" in value for value in args):
            return '{"users":2,"messages":5}'
        return ""

    def inspect(self, name):
        assert name == self.temp_name
        return {"Name": "/" + name, "Config": {"Labels": {backup.DRILL_LABEL: "true"}}}


def test_quiesce_recovers_only_previously_running_application_containers_on_error():
    docker = FakeDocker()
    docker.containers["api"]["State"]["Running"] = False
    with pytest.raises(RuntimeError, match="capture failed"):
        with backup.quiesced(docker, docker.containers, True):
            raise RuntimeError("capture failed")
    assert docker.calls == [("stop", "--time", "30", "support-agent-id"), ("start", "support-agent-id")]


def test_create_records_scope_and_excludes_plaintext_secret_configuration(tmp_path):
    root = tmp_path / "repo"
    data = root / "extensions/support-agent/data"
    data.mkdir(parents=True)
    (data / "knowledge.json").write_text("[]")
    (root / ".env").write_text("KEY=never-copy-this")
    (root / ".local").mkdir()
    (root / ".local/users.json").write_text("private-provisioning")
    docker = FakeDocker()
    result = backup.create_backup(root, root / ".local/backups", pause_writes=True, docker=docker)
    directory = Path(result["backup"])
    assert result["consistency"] == "application_writers_quiesced"
    manifest = backup.verify_bundle(directory)["manifest"]
    assert manifest["sensitive"] is True
    assert manifest["files"].keys() == {"support.db", "mongo.archive.gz", "librechat-files.tar.gz", "knowledge-source/knowledge.json"}
    assert "never-copy-this" not in (directory / "manifest.json").read_text()
    assert ("stop", "--time", "30", "api-id") in docker.calls
    assert ("stop", "--time", "30", "support-agent-id") in docker.calls
    assert docker.calls[-2:] == [("start", "support-agent-id"), ("start", "api-id")]
    assert all("support-public" not in call and "mongodb-id" not in call[1:2] for call in docker.calls if call[0] in {"start", "stop"})
    helpers = [call for call in docker.calls if call[0] == "run"]
    assert all("none" in call and any(arg.endswith(":ro") for arg in call) for call in helpers)


def test_online_create_does_not_stop_any_container(tmp_path):
    data = tmp_path / "extensions/support-agent/data"
    data.mkdir(parents=True)
    (data / "knowledge.json").write_text("[]")
    docker = FakeDocker()
    result = backup.create_backup(tmp_path, tmp_path / ".local/backups", docker=docker)
    assert result["consistency"] == "online_independent_store_snapshots"
    assert not any(call[0] in {"stop", "start"} for call in docker.calls)


def test_full_drill_mongo_is_isolated_and_cleaned_up(bundle, tmp_path):
    docker = FakeDocker()
    result = backup.restore_drill(bundle, tmp_path / "drills", docker=docker)
    assert result["status"] == "passed"
    assert result["mongo"]["collections"] == {"users": 2, "messages": 5}
    run = next(call for call in docker.calls if "--detach" in call)
    assert run[run.index("--network") + 1] == "none"
    assert "--pull=never" in run and "--publish" not in run and "-p" not in run
    assert docker.calls[-1] == ("stop", "--time", "20", docker.temp_name)
    assert all("--drop" not in call for call in docker.calls)


def test_retention_only_plans_bounded_verified_directories(bundle, tmp_path):
    root = tmp_path / "backups"
    root.mkdir()
    for day in [1, 2, 3]:
        shutil.copytree(bundle, root / f"backup-2026110{day}T120000Z-1234abcd")
    unrelated = root / "other-user-directory"
    unrelated.mkdir()
    plan = backup.retention_plan(root, keep=2)
    assert plan["deleted"] == 0
    assert plan["removal_candidates"] == ["backup-20261101T120000Z-1234abcd"]
    assert len(list(root.iterdir())) == 4
    with pytest.raises(backup.BackupError, match="at least one"):
        backup.retention_plan(root, keep=0)


def test_retention_skips_incomplete_backup_without_deleting_it(bundle, tmp_path):
    root = tmp_path / "backups"
    root.mkdir()
    complete = root / "backup-20261102T120000Z-1234abcd"
    shutil.copytree(bundle, complete)
    incomplete = root / "backup-20261101T120000Z-1234abcd"
    incomplete.mkdir()
    (incomplete / ".incomplete").write_text("capture failed")
    plan = backup.retention_plan(root, keep=1)
    assert plan["keep"] == [complete.name]
    assert plan["removal_candidates"] == []
    assert plan["skipped"][0]["name"] == incomplete.name
    assert (incomplete / ".incomplete").exists()
