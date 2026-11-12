"""Private local backup, integrity verification and isolated restore drills.

No command restores over a running deployment. `create` is online by default;
`create --quiesce` briefly stops only api/support-agent and restarts the containers
that this command stopped. MongoDB and the public benchmark service stay running.
"""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
from typing import Iterator
import uuid


ROOT = Path(__file__).resolve().parents[3]
SERVICE = ROOT / "extensions/support-agent"
FORMAT = "support-agent-backup-v1"
REQUIRED_PAYLOADS = {"support.db", "mongo.archive.gz", "librechat-files.tar.gz"}
DRILL_LABEL = "local.support-agent.restore-drill"


class BackupError(RuntimeError):
    """A failed precondition; messages never contain command output or credentials."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value: str) -> Path:
    """Portable paths: reject Windows drives, ADS, device names and aliases too."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise BackupError("Unsafe relative path in backup.")
    parts = value.split("/")
    if PurePosixPath(value).is_absolute() or any(p in {"", ".", ".."} for p in parts):
        raise BackupError("Unsafe relative path in backup.")
    for part in parts:
        if part.endswith((" ", ".")) or any(ord(c) < 32 or c in '<>"|?*' for c in part):
            raise BackupError("Unsafe filename in backup.")
        if re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part):
            raise BackupError("Reserved filename in backup.")
    return Path(*parts)


def no_links(path: Path) -> None:
    """Reject symlinks and Windows junctions, including existing ancestor links."""
    current = path.absolute()
    for item in [current, *current.parents]:
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise BackupError("Links/junctions are not permitted in backup or restore paths.")


def files_under(directory: Path) -> dict[str, Path]:
    no_links(directory)
    if not directory.is_dir():
        raise BackupError("Expected a directory.")
    result: dict[str, Path] = {}
    folded: set[str] = set()
    for path in directory.rglob("*"):
        no_links(path)
        name = path.relative_to(directory).as_posix()
        safe_relative(name)
        if name.casefold() in folded:
            raise BackupError("Case-colliding paths are not portable.")
        folded.add(name.casefold())
        if path.is_file():
            result[name] = path
        elif not path.is_dir():
            raise BackupError("Unsupported file type in backup.")
    return result


def sqlite_summary(path: Path, *, immutable: bool = False) -> dict:
    no_links(path)
    if not path.is_file():
        raise BackupError("SQLite snapshot is missing.")
    uri = path.resolve().as_uri() + "?mode=ro" + ("&immutable=1" if immutable else "")
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise BackupError("SQLite integrity_check failed.")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise BackupError("SQLite foreign_key_check failed.")
        names = [row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        counts = {name: db.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0]
                  for name in names}
        return {"integrity_check": "ok", "foreign_key_check": "ok", "tables": counts,
                "user_version": db.execute("PRAGMA user_version").fetchone()[0]}


def sqlite_snapshot(source: Path, destination: Path, *, immutable_source: bool = False) -> dict:
    """SQLite's backup API includes committed WAL content; never raw-copy live DBs."""
    no_links(source)
    no_links(destination)
    if not source.is_file() or destination.exists():
        raise BackupError("SQLite source must exist and destination must be new.")
    with destination.open("xb"):
        pass
    uri = source.resolve().as_uri() + "?mode=ro" + ("&immutable=1" if immutable_source else "")
    with closing(sqlite3.connect(uri, uri=True)) as src:
        with closing(sqlite3.connect(destination)) as dst:
            src.backup(dst)
            if dst.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
                raise BackupError("Could not seal the SQLite snapshot without WAL sidecars.")
    return sqlite_summary(destination, immutable=True)


def copy_sources(source: Path, destination: Path) -> None:
    entries = files_under(source)
    destination.mkdir()
    for name, path in entries.items():
        target = destination / safe_relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)


def archive_entries(path: Path, *, max_bytes: int = 20 * 1024**3) -> list[tarfile.TarInfo]:
    """Validate the entire archive before creating any restored files."""
    entries: list[tarfile.TarInfo] = []
    seen: set[str] = set()
    total = 0
    with tarfile.open(path, "r:gz") as archive:
        for entry in archive:
            name = entry.name.rstrip("/") if entry.isdir() else entry.name
            relative = safe_relative(name)
            if relative.parts[0] not in {"uploads", "data"}:
                raise BackupError("Unexpected root in LibreChat file archive.")
            if not (entry.isfile() or entry.isdir()) or entry.size < 0:
                raise BackupError("Links and special files are forbidden in file archives.")
            if name.casefold() in seen:
                raise BackupError("Duplicate archive path.")
            seen.add(name.casefold())
            total += entry.size
            if total > max_bytes or len(entries) >= 100000:
                raise BackupError("File archive exceeds the restore safety limit.")
            entries.append(entry)
    file_names = {item.name.casefold() for item in entries if item.isfile()}
    for entry in entries:
        if any(parent.as_posix().casefold() in file_names
               for parent in PurePosixPath(entry.name).parents if str(parent) != "."):
            raise BackupError("An archive file is also used as a directory.")
    if {PurePosixPath(item.name).parts[0] for item in entries} != {"uploads", "data"}:
        raise BackupError("Both LibreChat file volume roots must be present.")
    return entries


def extract_files(path: Path, destination: Path) -> dict:
    entries = archive_entries(path)
    no_links(destination)
    destination.mkdir()  # Never merge with an existing directory.
    count = size = 0
    with tarfile.open(path, "r:gz") as archive:
        for entry in entries:
            target = destination / safe_relative(entry.name.rstrip("/"))
            if entry.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(entry)
            if stream is None:
                raise BackupError("Unreadable archive entry.")
            with stream, target.open("xb") as out:
                shutil.copyfileobj(stream, out)
            if target.stat().st_size != entry.size:
                raise BackupError("Truncated archive entry.")
            count += 1
            size += entry.size
    return {"files": count, "bytes": size}


def write_manifest(directory: Path, metadata: dict) -> dict:
    entries = files_under(directory)
    if "manifest.json" in entries:
        raise BackupError("Manifest already exists.")
    result = {**metadata, "format": FORMAT, "files": {
        name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in sorted(entries.items()) if name != ".incomplete"
    }}
    with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def verify_bundle(directory: Path, manifest_sha256: str | None = None) -> dict:
    entries = files_under(directory)
    if ".incomplete" in entries or "manifest.json" not in entries:
        raise BackupError("Backup is incomplete or has no manifest.")
    manifest_path = entries.pop("manifest.json")
    digest = sha256(manifest_path)
    if manifest_sha256 is not None and digest != manifest_sha256.lower():
        raise BackupError("Manifest does not match the independently retained SHA-256.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise BackupError("Invalid backup manifest JSON.") from error
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise BackupError("Unsupported backup manifest.")
    records = manifest.get("files")
    if not isinstance(records, dict) or not REQUIRED_PAYLOADS <= records.keys():
        raise BackupError("Required backup payloads are missing from the manifest.")
    if not any(name.startswith("knowledge-source/") for name in records):
        raise BackupError("Knowledge source files are missing from the manifest.")
    for name, record in records.items():
        safe_relative(name)
        if not isinstance(record, dict) or not isinstance(record.get("bytes"), int) or record["bytes"] < 0:
            raise BackupError("Invalid backup file record.")
        if not isinstance(record.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", record["sha256"]):
            raise BackupError("Invalid backup file digest.")
    if "support.db-journal" in records:
        raise BackupError("A hot rollback journal is not a sealed SQLite snapshot.")
    if "support.db-wal" in records and records["support.db-wal"]["bytes"] != 0:
        raise BackupError("Nonempty SQLite WAL is not a sealed main-file snapshot.")
    # Older read-only validators could create an empty WAL plus SHM metadata.
    # Preserve those bytes/manifest hashes, but never consume their contents.
    # Any nonempty unverified WAL is rejected before SQLite is opened.
    ignored_sidecars = []
    wal = entries.get("support.db-wal")
    if wal is not None:
        if wal.stat().st_size != 0:
            raise BackupError("Unexpected nonempty SQLite WAL is not a verified backup payload.")
        ignored_sidecars.append("support.db-wal")
        if "support.db-wal" not in records:
            entries.pop("support.db-wal")
        if "support.db-shm" in entries:
            ignored_sidecars.append("support.db-shm")
            if "support.db-shm" not in records:
                entries.pop("support.db-shm")
    if set(records) != set(entries):
        raise BackupError("Backup contains missing or unexpected files.")
    for name, record in records.items():
        path = entries[name]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise BackupError("Backup payload failed its size or SHA-256 check.")
    # immutable is safe here only because this is a fully hashed backup file,
    # never the live database. It cannot read or create WAL/SHM sidecars.
    summary = sqlite_summary(directory / "support.db", immutable=True)
    if summary != manifest.get("sqlite"):
        raise BackupError("SQLite tables differ from the recorded backup summary.")
    archive_entries(directory / "librechat-files.tar.gz")
    return {"manifest": manifest, "manifest_sha256": digest, "verified_files": len(records),
            "ignored_empty_wal_metadata": ignored_sidecars}


def new_directory(root: Path, prefix: str) -> Path:
    no_links(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / (prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    target.mkdir(mode=0o700)
    return target


class Docker:
    def __init__(self, root: Path = ROOT):
        self.root = root

    def run(self, *args: str, output: Path | None = None, timeout: int = 300) -> str:
        """Do not echo captured Docker output: it can contain paths or DB values."""
        try:
            if output is not None:
                with output.open("xb") as stream:
                    subprocess.run(["docker", *args], cwd=self.root, stdout=stream,
                                   stderr=subprocess.PIPE, check=True, timeout=timeout)
                return ""
            result = subprocess.run(["docker", *args], cwd=self.root, capture_output=True,
                                    check=True, timeout=timeout)
            return result.stdout.decode("utf-8").strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, UnicodeError) as error:
            raise BackupError("Docker operation failed; inspect local container health/logs privately.") from error

    def inspect(self, container: str) -> dict:
        value = json.loads(self.run("inspect", container))
        if not isinstance(value, list) or len(value) != 1:
            raise BackupError("Expected exactly one deployment container.")
        return value[0]

    def deployment(self) -> dict[str, dict]:
        containers = {}
        for service in ["api", "support-agent", "mongodb"]:
            value = self.run("compose", "-f", "compose.support.yml", "ps", "-a", "-q", service)
            if not re.fullmatch("[0-9a-f]{12,64}", value):
                raise BackupError("Expected one existing api, support-agent and mongodb container.")
            details = self.inspect(value)
            labels = details["Config"].get("Labels", {})
            if labels.get("com.docker.compose.service") != service:
                raise BackupError("Container service identity mismatch.")
            containers[service] = details
        if not containers["mongodb"]["State"]["Running"]:
            raise BackupError("MongoDB must be running for a logical dump.")
        return containers


@contextmanager
def quiesced(docker: Docker, containers: dict, enabled: bool) -> Iterator[None]:
    stopped: list[str] = []
    try:
        if enabled:
            for service in ["api", "support-agent"]:
                details = containers[service]
                if details["State"]["Running"]:
                    # Record intent first: even if a stop times out, try to recover it.
                    stopped.append(details["Id"])
                    docker.run("stop", "--time", "30", details["Id"], timeout=60)
        yield
    finally:
        failures = []
        for container in reversed(stopped):
            try:
                docker.run("start", container, timeout=60)
            except BackupError:
                failures.append(container)
        if failures:
            raise BackupError("Backup ended but container restart failed; run the normal project Start command.")


SQLITE_STREAM = """
import os, pathlib, shutil, sqlite3, sys, tempfile
from contextlib import closing
source = pathlib.Path(sys.argv[1])
if not source.is_file(): raise RuntimeError('SQLite source is missing')
with tempfile.TemporaryDirectory() as tmp:
    if len(sys.argv) > 2 and sys.argv[2] == 'cold':
        # All application writers have stopped. Retain any uncheckpointed WAL,
        # then let SQLite rebuild private lock metadata on the scratch copy.
        frozen = pathlib.Path(tmp) / 'source.db'
        shutil.copyfile(source, frozen)
        wal = pathlib.Path(str(source) + '-wal')
        if wal.exists(): shutil.copyfile(wal, str(frozen) + '-wal')
        source = frozen
    target = pathlib.Path(tmp) / 'snapshot.db'
    with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as src:
        with closing(sqlite3.connect(target)) as dst:
            src.backup(dst)
            if dst.execute('PRAGMA journal_mode=DELETE').fetchone()[0] != 'delete':
                raise RuntimeError('Could not seal SQLite snapshot')
    with target.open('rb') as stream: shutil.copyfileobj(stream, sys.stdout.buffer)
"""

FILES_STREAM = """
import pathlib, sys, tarfile
roots = [pathlib.Path('/app/uploads'), pathlib.Path('/app/data')]
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz', dereference=True) as archive:
    for root in roots:
        if not root.is_dir() or root.is_symlink(): raise RuntimeError('Missing file volume')
        for path in [root, *sorted(root.rglob('*'))]:
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise RuntimeError('Unsupported link or special file')
            archive.add(path, arcname=path.relative_to('/app').as_posix(), recursive=False)
"""


def config_references(root: Path) -> dict:
    templates = ["compose.support.yml", "compose.retrieval.yml", "compose.operations.yml", "compose.debug.yml", "compose.public-evaluation.yml",
                 "extensions/support-agent/deploy/config.yaml", "extensions/support-agent/deploy/nginx.conf"]
    return {
        "templates_not_copied": [{"path": name, "sha256": sha256(root / name)}
                                 for name in templates if (root / name).is_file()],
        "private_files_not_copied": [
            {"path": ".env", "purpose": "Original CREDS_KEY/CREDS_IV decrypt LibreChat action credentials; model/JWT keys."},
            {"path": ".local/users.json", "purpose": "Support user identity, password/API key provisioning; retain current roles."},
            {"path": ".local/demo.json", "purpose": "Local bootstrap/login identity, if still used."},
            {"path": ".local/librechat-users.json", "purpose": "Per-user LibreChat account credentials and native identity mapping."},
            {"path": ".local/operations.json", "purpose": "OAuth client secret and monitoring token."},
            {"path": ".local/support-tls/server.key", "purpose": "Local support HTTPS private key; retain securely or regenerate and reconfigure trust."},
        ],
        "public_certificate_not_copied": {"path": ".local/support-tls/server.crt", "purpose": "Local HTTPS certificate; retain with its matching private key or regenerate and reconfigure trust."},
        "recovery_requirement": "Securely retain original private configuration separately; this bundle alone is not a turnkey disaster recovery package.",
    }


def create_backup(root: Path, backup_root: Path, *, pause_writes: bool = False,
                  docker: Docker | None = None) -> dict:
    docker = docker or Docker(root)
    containers = docker.deployment()
    support = containers["support-agent"]
    api = containers["api"]
    source = root / "extensions/support-agent/data"
    no_links(backup_root)
    if backup_root.resolve().is_relative_to(source.resolve()):
        raise BackupError("Backup destination cannot be inside the knowledge source directory.")
    directory = new_directory(backup_root, "backup-")
    marker = directory / ".incomplete"
    marker.write_text("Do not restore: backup creation has not completed.\n", encoding="utf-8")
    started = utc_now()
    env = dict(item.split("=", 1) for item in support["Config"].get("Env", []) if "=" in item)
    db_path = env.get("SUPPORT_DB_PATH", "/data/support.db")
    if not db_path.startswith("/") or ":" in db_path:
        raise BackupError("Expected an absolute Linux SQLite path in the support container.")
    image = support["Image"]  # Pinned local image ID; never pull a new image.
    with quiesced(docker, containers, pause_writes):
        if pause_writes or not support["State"]["Running"]:
            # Static db/WAL copied from read-only mounts into disposable scratch;
            # the final snapshot still uses SQLite's backup API.
            docker.run("run", "--rm", "--network", "none", "--volumes-from", support["Id"] + ":ro",
                       "--entrypoint", "python", image, "-c", SQLITE_STREAM, db_path, "cold",
                       output=directory / "support.db")
        else:
            # A live WAL database may need to create its lock/SHM metadata even
            # for a mode=ro reader. Run in the existing container; never copy live
            # db/WAL files or set immutable=1 on a changing database.
            docker.run("exec", support["Id"], "python", "-c", SQLITE_STREAM, db_path,
                       output=directory / "support.db")
        sqlite = sqlite_summary(directory / "support.db", immutable=True)
        docker.run("exec", containers["mongodb"]["Id"], "mongodump", "--db", "LibreChat",
                   "--archive", "--gzip", output=directory / "mongo.archive.gz")
        docker.run("run", "--rm", "--network", "none", "--user", "0:0", "--volumes-from", api["Id"] + ":ro",
                   "--entrypoint", "python", image, "-c", FILES_STREAM,
                   output=directory / "librechat-files.tar.gz")
        archive_entries(directory / "librechat-files.tar.gz")
        copy_sources(source, directory / "knowledge-source")
        captured = utc_now()
    manifest = write_manifest(directory, {
        "started_at": started, "captured_at": captured, "completed_at": utc_now(),
        "consistency": "application_writers_quiesced" if pause_writes else "online_independent_store_snapshots",
        "consistency_limit": "Not a distributed transaction. Online: only SQLite is a transactional snapshot; Mongo collections and file volumes may change during capture. Quiesce stops only api/support-agent writers, not external clients or Mongo background TTL expiry.",
        "sensitive": True,
        "privacy": "Contains business data, password/token hashes and encrypted action credentials. Local private storage only; do not publish or commit.",
        "storage_security": "Not encrypted by this tool; inherits host filesystem access controls.",
        "scope": ["support SQLite including imported document text/revisions/chunks", "LibreChat MongoDB", "LibreChat uploads and runtime data", "repository knowledge source documents"],
        "excluded": ["plaintext secret configuration", "logs", "rebuildable vector/model caches", "separate public benchmark service"],
        "sqlite": sqlite, "configuration": config_references(root),
        "images": {name: info["Image"] for name, info in containers.items()},
    })
    marker.unlink()  # This exact newly created marker is the only deletion in create.
    try:
        verified = verify_bundle(directory)
    except Exception:
        marker.write_text("Do not restore: final verification failed.\n", encoding="utf-8")
        raise
    return {"backup": str(directory), "consistency": manifest["consistency"],
            "manifest_sha256": verified["manifest_sha256"], "verified_files": verified["verified_files"],
            "sqlite": sqlite, "secret_configuration_included": False}


def mongo_drill(docker: Docker, bundle: Path, image: str) -> dict:
    """Restore to an ephemeral container with no host port or network, then remove it."""
    name = "support-restore-" + uuid.uuid4().hex
    mount = "type=bind,source=" + str(bundle.resolve()) + ",target=/backup,readonly"
    if "," in str(bundle.resolve()):
        raise BackupError("Docker bind mount paths cannot contain commas.")
    # The image must already exist locally; --pull=never prevents network downloads.
    created = False
    try:
        docker.run("run", "--detach", "--rm", "--pull=never", "--name", name,
                   "--label", DRILL_LABEL + "=true", "--network", "none", "--memory", "512m",
                   "--mount", mount, image, "mongod", "--bind_ip", "127.0.0.1",
                   "--wiredTigerCacheSizeGB", "0.25", timeout=60)
        created = True
        for attempt in range(30):
            try:
                docker.run("exec", name, "mongosh", "--quiet", "--eval", "db.adminCommand('ping').ok", timeout=10)
                break
            except BackupError:
                if attempt == 29:
                    raise BackupError("Isolated MongoDB did not become ready.")
                time.sleep(1)
        docker.run("exec", name, "mongorestore", "--gzip", "--archive=/backup/mongo.archive.gz",
                   "--stopOnError", "--nsInclude=LibreChat.*")
        counts = json.loads(docker.run("exec", name, "mongosh", "--quiet", "LibreChat", "--eval",
            "JSON.stringify(Object.fromEntries(db.getCollectionNames().sort().map(n=>[n,db.getCollection(n).countDocuments({})])))"))
        if not isinstance(counts, dict) or any(not isinstance(v, int) or v < 0 for v in counts.values()):
            raise BackupError("Unexpected Mongo restore validation result.")
        return {"status": "restored", "isolated": True, "network": "none", "collections": counts}
    finally:
        if created:
            details = docker.inspect(name)
            if details["Config"].get("Labels", {}).get(DRILL_LABEL) != "true" or details["Name"] != "/" + name:
                raise BackupError("Refused cleanup: restore container ownership mismatch.")
            docker.run("stop", "--time", "20", name, timeout=40)


def restore_drill(bundle: Path, restore_root: Path, *, skip_mongo: bool = False,
                  manifest_sha256: str | None = None, docker: Docker | None = None) -> dict:
    verified = verify_bundle(bundle, manifest_sha256)
    if restore_root.resolve().is_relative_to(bundle.resolve()):
        raise BackupError("Restore destination cannot be inside its source bundle.")
    directory = new_directory(restore_root, "drill-")
    result = {"started_at": utc_now(), "restore_directory": str(directory),
              "manifest_sha256": verified["manifest_sha256"], "production_modified": False,
              "ignored_empty_wal_metadata": verified["ignored_empty_wal_metadata"]}
    try:
        result["sqlite"] = sqlite_snapshot(bundle / "support.db", directory / "support.db", immutable_source=True)
        if result["sqlite"] != verified["manifest"]["sqlite"]:
            raise BackupError("Restored SQLite summary differs from backup.")
        result["librechat_files"] = extract_files(bundle / "librechat-files.tar.gz", directory / "librechat")
        copy_sources(bundle / "knowledge-source", directory / "knowledge-source")
        result["mongo"] = {"status": "skipped", "reason": "Explicit --skip-mongo; not a full restore drill."}
        if not skip_mongo:
            image = verified["manifest"].get("images", {}).get("mongodb")
            if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
                raise BackupError("Missing pinned local Mongo image ID.")
            result["mongo"] = mongo_drill(docker or Docker(), bundle, image)
        result["status"] = "partial_pass" if skip_mongo else "passed"
    except Exception:
        result["status"] = "failed"
        raise
    finally:
        result["completed_at"] = utc_now()
        with (directory / "restore-report.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return result


def retention_plan(root: Path, keep: int) -> dict:
    """List bounded, verified candidates only. No automatic deletion is implemented."""
    if keep < 1:
        raise BackupError("Keep at least one backup.")
    no_links(root)
    candidates = []
    skipped = []
    if root.is_dir():
        for directory in root.iterdir():
            if directory.is_dir() and re.fullmatch(r"backup-\d{8}T\d{6}Z-[0-9a-f]{8}", directory.name):
                no_links(directory)
                if (directory / ".incomplete").exists() or not (directory / "manifest.json").is_file():
                    skipped.append({"name": directory.name, "reason": "incomplete; never a removal candidate"})
                    continue
                verify_bundle(directory)
                candidates.append(directory.name)
    ordered = sorted(candidates, reverse=True)
    return {"root": str(root.resolve()), "keep": ordered[:keep], "removal_candidates": ordered[keep:],
            "skipped": skipped, "deleted": 0,
            "notice": "Plan only. Review destinations and recoverability before any manual cleanup."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create", help="Create a private local bundle; default online.")
    create.add_argument("--backup-root", type=Path, default=ROOT / ".local/backups")
    create.add_argument("--quiesce", action="store_true", help="Temporarily stop only api/support-agent writes.")
    verify = sub.add_parser("verify", help="Verify manifest, payload hashes, SQLite and archive paths.")
    verify.add_argument("bundle", type=Path)
    verify.add_argument("--manifest-sha256")
    drill = sub.add_parser("drill", help="Restore to a new local directory and isolated temporary MongoDB.")
    drill.add_argument("bundle", type=Path)
    drill.add_argument("--restore-root", type=Path, default=ROOT / ".local/restore-drills")
    drill.add_argument("--manifest-sha256")
    drill.add_argument("--skip-mongo", action="store_true")
    retention = sub.add_parser("retention-plan", help="Show retention candidates; never delete.")
    retention.add_argument("--backup-root", type=Path, default=ROOT / ".local/backups")
    retention.add_argument("--keep", type=int, default=7)
    args = parser.parse_args()
    try:
        if args.action == "create":
            result = create_backup(ROOT, args.backup_root, pause_writes=args.quiesce)
        elif args.action == "verify":
            verified = verify_bundle(args.bundle, args.manifest_sha256)
            result = {"status": "verified", "manifest_sha256": verified["manifest_sha256"],
                      "verified_files": verified["verified_files"], "sqlite": verified["manifest"]["sqlite"],
                      "ignored_empty_wal_metadata": verified["ignored_empty_wal_metadata"]}
        elif args.action == "drill":
            result = restore_drill(args.bundle, args.restore_root, skip_mongo=args.skip_mongo,
                                   manifest_sha256=args.manifest_sha256)
        else:
            result = retention_plan(args.backup_root, args.keep)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (BackupError, OSError, sqlite3.DatabaseError, tarfile.TarError, ValueError) as error:
        # OS/database exception bodies may contain sensitive paths or data.
        print("Backup operation failed: " + (str(error) if isinstance(error, BackupError)
                                           else type(error).__name__ + "; inspect the local input privately."), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
