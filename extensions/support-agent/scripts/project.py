"""VS Code entry point. Uses Python without changing Windows execution policy."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[3]
SERVICE = ROOT / "extensions/support-agent"
PUBLIC = ROOT / ".local/public-benchmark"
PUBLIC_DATA = PUBLIC / "mtrag"
PUBLIC_MODELS = ROOT / ".local/models/public-retrieval"
PUBLIC_ACTIONS = ["PreparePublicData", "PreparePublicModels", "BuildPublicIndex", "StartPublic", "StopPublic", "SetupPublicAgent", "PublicAgentDryRun"]
OPERATIONS_ACTIONS = ["OperationsStart", "StartDebug", "OpenApps", "Backup", "Monitor", "MultiUser"]


def run(*args: str, cwd: Path = ROOT) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def run_json(*args: str) -> dict:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    try:
        value = json.loads(result.stdout)
    except ValueError:
        raise RuntimeError("Local setup returned an invalid result; inspect its command privately.") from None
    if not isinstance(value, dict):
        raise RuntimeError("Local setup returned an unexpected result.")
    return value


def operations_compose(*, require_retrieval: bool = False) -> list[str]:
    manifest = ROOT / ".local/models/support-retrieval/manifest.json"
    if require_retrieval and not manifest.is_file():
        raise RuntimeError("Run PrepareRetrieval first to download and verify the pinned CPU models.")
    command = ["docker", "compose", "-f", "compose.support.yml"]
    if manifest.is_file():
        command.extend(["-f", "compose.retrieval.yml"])
    command.extend(["-f", "compose.operations.yml"])
    return command


def tls_trusted(powershell: str) -> bool:
    # Root is passed as a PowerShell single-quoted literal, never interpolated code.
    escaped_root = str(ROOT).replace("'", "''")
    script = (
        "$ErrorActionPreference='Stop'; "
        f"$certPath=Join-Path '{escaped_root}' '.local/support-tls/server.crt'; "
        "$cert=[System.Security.Cryptography.X509Certificates.X509Certificate2]::CreateFromPem([System.IO.File]::ReadAllText($certPath)); "
        "$valid=$cert.NotBefore.ToUniversalTime() -le [DateTime]::UtcNow -and $cert.NotAfter.ToUniversalTime() -gt [DateTime]::UtcNow; "
        "$store=[System.Security.Cryptography.X509Certificates.X509Store]::new('Root','CurrentUser'); "
        "$store.Open([System.Security.Cryptography.X509Certificates.OpenFlags]::ReadOnly); "
        "try { $matches=$store.Certificates.Find([System.Security.Cryptography.X509Certificates.X509FindType]::FindByThumbprint,$cert.Thumbprint,$false); "
        "@{trusted=($matches.Count -gt 0);valid=$valid} | ConvertTo-Json } finally {$store.Close()}"
    )
    state = run_json(powershell, "-NoProfile", "-Command", script)
    if state.get("valid") is not True:
        raise RuntimeError("Local TLS certificate is expired or not yet valid. Deliberately replace the key/certificate pair and update trust; existing files were preserved.")
    return state.get("trusted") is True


def prepare_local_tls() -> None:
    powershell = shutil.which("pwsh")
    if not powershell:
        raise RuntimeError("PowerShell 7 (pwsh) is required to prepare local HTTPS and CurrentUser certificate trust.")
    command = [powershell, "-NoProfile", "-File", str(SERVICE / "scripts/local_tls.ps1"), "-Root", str(ROOT)]
    try:
        run_json(*command)
        for name in ["server.crt", "server.key"]:
            if not (ROOT / ".local/support-tls" / name).is_file():
                raise RuntimeError("Local TLS files are missing; local_tls.ps1 did not produce a complete key/certificate pair.")
        if tls_trusted(powershell):
            print("Local HTTPS certificate is valid and already trusted by the current Windows user.")
            return
        print("Local OAuth uses HTTPS. Windows may ask to trust this local server certificate for the current user; approve the matching LibreChat Support Local certificate to continue.", flush=True)
        run_json(*command, "-Trust")
        if not tls_trusted(powershell):
            raise RuntimeError("Local certificate trust was not completed. Rerun Start and complete the Windows certificate prompt.")
    except subprocess.CalledProcessError:
        raise RuntimeError("Local TLS preparation failed. Check for a missing/mismatched or expired server.crt/server.key pair and complete CurrentUser trust; no existing pair is automatically replaced.") from None


def require_operations_ready() -> None:
    try:
        with urlopen("http://127.0.0.1:8090/ready", timeout=5) as response:
            ready = json.load(response)
        with urlopen("http://127.0.0.1:8090/openapi.json", timeout=5) as response:
            spec = json.load(response)
        if ready.get("status") != "ready" or spec.get("servers", [{}])[0].get("url") != "https://support-tls:8443":
            raise ValueError("operations deployment is not active")
        # Never dump the container environment. Check only the required flags.
        state = run_json(*operations_compose(), "exec", "-T", "support-agent", "python", "-c",
            "import json,os; print(json.dumps({'operations': os.getenv('SUPPORT_ALLOW_LEGACY_TOOL_KEYS') == 'false' and bool(os.getenv('SUPPORT_OPERATIONS_PATH')) and os.getenv('SUPPORT_SECURE_COOKIES') == 'true'}))")
        if state.get("operations") is not True:
            raise ValueError("legacy credentials are still enabled")
    except (OSError, ValueError, AttributeError, IndexError, subprocess.CalledProcessError):
        raise RuntimeError("Operations deployment is not ready. Run OperationsStart first; OAuth/TLS must be enabled and legacy shared tool keys disabled.") from None


def bootstrap_operations() -> dict:
    require_operations_ready()
    result = run_json(sys.executable, str(SERVICE / "integration/bootstrap.py"))
    if result.get("status") != "ready" or result.get("authentication") != "per_user_oauth":
        raise RuntimeError("Agent bootstrap did not confirm per-user OAuth.")
    if result.get("support_restart_required") is True:
        run(*operations_compose(), "restart", "--no-deps", "support-agent")
        deadline = time.monotonic() + 180
        while True:
            try:
                require_operations_ready()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Support service did not become ready after its OAuth callback update.") from None
                time.sleep(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def start_operations(*, require_retrieval: bool = False) -> None:
    compose = operations_compose(require_retrieval=require_retrieval)
    run(sys.executable, str(SERVICE / "scripts/init.py"))
    prepare_local_tls()
    run(*compose, "up", "-d", "--build", "--wait", "--wait-timeout", "240",
        "mongodb", "support-agent", "support-tls", "api")
    bootstrap_operations()
    print("LibreChat: http://localhost:3080 | Approval console: https://localhost:8443 | Per-user OAuth enabled; no model calls.")


def start_debug() -> None:
    if not (ROOT / "compose.debug.yml").is_file():
        raise RuntimeError("compose.debug.yml is missing; prepare the OAuth-compatible debug overlay before StartDebug.")
    start_operations()  # One initialization/TLS/bootstrap flow, retaining OAuth.
    run(*operations_compose(), "-f", "compose.debug.yml", "up", "-d", "--no-deps", "--wait",
        "--wait-timeout", "180", "support-agent")
    require_operations_ready()
    print("Debug mode ready: attach VS Code to 127.0.0.1:5678. OAuth/TLS remains enabled; Start returns to deployment mode without debugger/reload. No model calls.")


def open_apps() -> None:
    urls = ["http://localhost:3080", "https://localhost:8443/console"]
    opened = [webbrowser.open(url, new=2) for url in urls]
    if not all(opened):
        raise RuntimeError("Could not launch both pages. Open http://localhost:3080 and https://localhost:8443/console in the same system-browser profile manually.")
    print("Requested both pages in the default system browser. Complete LibreChat login, OAuth authorization and callback in that same browser profile; no model calls.")


def ensure_public_config() -> Path:
    """Create the separate tool credential only when missing; never rotate it on start."""
    path = PUBLIC / "config.json"
    PUBLIC.mkdir(parents=True, exist_ok=True)
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(config, dict):
        raise RuntimeError("Public config.json must be a JSON object.")
    if "tool_token" in config:
        if not isinstance(config["tool_token"], str) or len(config["tool_token"]) < 32:
            raise RuntimeError("Existing public tool_token is invalid; it has not been replaced.")
        return path
    config["tool_token"] = secrets.token_urlsafe(48)
    partial = path.with_suffix(".json.tmp")
    with partial.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    partial.replace(path)
    return path


def public_agent_mode() -> str:
    state = PUBLIC / "agents.json"
    if not state.is_file():
        return "lexical"
    value = json.loads(state.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Public agents.json must be a JSON object.")
    mode = value.get("mode", "lexical")
    if mode not in {"lexical", "hybrid", "rerank"}:
        raise RuntimeError("Existing public Agent mode is invalid; it has not been reset.")
    return mode


def ensure_public_selection() -> Path:
    sys.path.insert(0, str(SERVICE))
    from evaluation.public_data import select_dev8
    expected = select_dev8(PUBLIC_DATA)
    path = PUBLIC_DATA / "dev8.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("Existing dev8 selection differs from the fixed development pilot.")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(expected, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return path


def public_models_ready() -> bool:
    """An unchanged download keeps its manifest hash and existing vector index valid."""
    manifest = PUBLIC_MODELS / "manifest.json"
    if not manifest.is_file():
        return False
    sys.path.insert(0, str(SERVICE))
    from scripts.prepare_retrieval import checksum, verified
    lock = json.loads((SERVICE / "evaluation/public_models.json").read_text(encoding="utf-8"))
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    if set(saved.get("models", {})) != set(lock["models"]):
        raise RuntimeError("Public model manifest differs from the pinned model roles.")
    for role, model in lock["models"].items():
        recorded = saved["models"][role]
        if any(recorded.get(key) != value for key, value in model.items() if key != "files"):
            raise RuntimeError("Existing public model revision differs from its lock.")
        if set(recorded.get("files", {})) != set(model["files"]):
            raise RuntimeError("Existing public model file list differs from its lock.")
        for name, entry in model["files"].items():
            path = PUBLIC_MODELS / role / name
            if not verified(path, entry) or recorded["files"][name].get("sha256") != checksum(path):
                raise RuntimeError("Existing public model file failed its pinned integrity check.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["Setup", "Start", "Stop", "Status", "Test", "Evaluate", "KnowledgeEvaluate", "PrepareRetrieval", "StartRetrieval", "RetrievalDev", "RetrievalRegression", "EvidenceDev", "EvidenceRegression", "Bootstrap", "PrepareEval", "AgentSmoke", "AgentBaseline", "AgentRepair", "AgentFollowup", "AgentVerify", "StopEval", *PUBLIC_ACTIONS, *OPERATIONS_ACTIONS])
    args = parser.parse_args()
    python = sys.executable
    compose = ["docker", "compose", "-f", "compose.support.yml"]
    try:
        if args.action == "Setup":
            if sys.version_info < (3, 12):
                raise RuntimeError("Use Python 3.12 or newer to create the project .venv.")
            run(python, "-m", "pip", "install", "-r", str(SERVICE / "requirements.txt"), "-r", str(SERVICE / "tests/requirements.txt"))
            run(python, str(SERVICE / "scripts/init.py"))
        elif args.action in {"Start", "OperationsStart"}:
            start_operations()
        elif args.action == "StartDebug":
            start_debug()
        elif args.action == "OpenApps":
            open_apps()
        elif args.action == "Stop":
            run(*operations_compose(), "stop", "api", "support-tls", "support-agent", "mongodb")
        elif args.action == "Status":
            run(*operations_compose(), "ps")
        elif args.action == "Backup":
            run(python, str(SERVICE / "scripts/backup.py"), "create")
        elif args.action == "Monitor":
            run(python, str(SERVICE / "scripts/monitor.py"), "--docker")
        elif args.action == "MultiUser":
            require_operations_ready()
            run(python, str(SERVICE / "integration/multiuser.py"), "--apply")
        elif args.action == "Test":
            run(python, "-m", "pytest", "tests", "integration", "-q", cwd=SERVICE)
        elif args.action == "Evaluate":
            run(python, str(SERVICE / "evaluation/run.py"))
        elif args.action == "KnowledgeEvaluate":
            folder = ROOT / ".local/knowledge-evaluation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            runner = str(SERVICE / "evaluation/knowledge_run.py")
            run(python, runner, "--mode", "baseline", "--split", "all", "--include-legacy", "--output", str(folder / "baseline"))
            run(python, runner, "--mode", "candidate", "--split", "all", "--include-legacy", "--baseline-report", str(folder / "baseline/report.json"), "--output", str(folder / "candidate"))
            print(f"Knowledge comparison (no model calls): {folder / 'candidate/report.html'}")
        elif args.action == "PrepareRetrieval":
            run(python, "-m", "pip", "install", "-r", str(SERVICE / "requirements.hybrid.txt"))
            run(python, str(SERVICE / "scripts/prepare_retrieval.py"))
        elif args.action == "PreparePublicData":
            run(python, "-m", "evaluation.public_data", "--root", str(PUBLIC_DATA), "--prepare", "--download-missing", cwd=SERVICE)
            ensure_public_selection()
            print(f"Pinned IBM Cloud data and dev8 selection: {PUBLIC_DATA}")
        elif args.action == "PreparePublicModels":
            run(python, "-m", "pip", "install", "-r", str(SERVICE / "requirements.hybrid.txt"))
            if public_models_ready():
                print("Public models verified; existing manifest and vector-index binding preserved.")
            else:
                run(python, str(SERVICE / "scripts/prepare_retrieval.py"), "--lock", str(SERVICE / "evaluation/public_models.json"),
                    "--output", str(PUBLIC_MODELS))
        elif args.action == "BuildPublicIndex":
            if not public_models_ready():
                raise RuntimeError("Run PreparePublicModels before building the public index.")
            run(python, str(SERVICE / "scripts/prepare_public.py"), "--root", str(PUBLIC_DATA), "--models", str(PUBLIC_MODELS),
                "--vectors", "--workers", "4", "--threads", "2")
        elif args.action == "StartPublic":
            if not (PUBLIC_DATA / "index/manifest.json").is_file():
                raise RuntimeError("Run PreparePublicData and BuildPublicIndex before starting the public service.")
            ensure_public_config()
            public_compose = [*compose, "-f", "compose.retrieval.yml", "-f", "compose.public-evaluation.yml"]
            run(*public_compose, "up", "-d", "--no-deps", "--wait", "--wait-timeout", "180", "support-public")
            run(*compose, "restart", "api")
            print("Public knowledge: http://localhost:8192 | Docker tools: http://support-public:8092")
            print("LibreChat API restarted to load the configured public tool domain. Existing public Agent mode is unchanged.")
        elif args.action == "StopPublic":
            run(*compose, "-f", "compose.public-evaluation.yml", "stop", "support-public")
            print("Public service stopped; corpus, indexes and normal business data are preserved.")
        elif args.action == "SetupPublicAgent":
            ensure_public_config()
            run(python, str(SERVICE / "evaluation/public_agent_setup.py"), "--apply", "--mode", public_agent_mode(),
                "--credentials", str(PUBLIC / "config.json"), "--state", str(PUBLIC / "agents.json"),
                "--service-url", "http://localhost:8192")
        elif args.action == "PublicAgentDryRun":
            selection = ensure_public_selection()
            run(python, str(SERVICE / "evaluation/public_agent.py"), "--split", "dev", "--stage", "answer",
                "--data-root", str(PUBLIC_DATA), "--selection", str(selection),
                "--state", str(PUBLIC / "agents.json"), "--credentials", str(PUBLIC / "config.json"),
                "--service-url", "http://localhost:8192")
        elif args.action == "StartRetrieval":
            start_operations(require_retrieval=True)
            print("Retrieval modes: https://localhost:8443/console/knowledge | default: lexical")
        elif args.action in ("RetrievalDev", "RetrievalRegression"):
            folder = ROOT / ".local/hybrid-evaluation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            split = "dev" if args.action == "RetrievalDev" else "test"
            run(python, str(SERVICE / "evaluation/hybrid_run.py"), "--split", split, "--output", str(folder))
            print(f"Local CPU retrieval comparison (no paid generation): {folder / 'report.html'}")
        elif args.action in ("EvidenceDev", "EvidenceRegression"):
            folder = ROOT / ".local/evidence-evaluation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            split = "dev" if args.action == "EvidenceDev" else "test"
            run(python, str(SERVICE / "evaluation/evidence_run.py"), "--stage", "expanded", "--split", split, "--output", str(folder))
            print(f"Evidence policy comparison (same retrieved chunks, no paid generation): {folder / 'report.html'}")
        elif args.action == "Bootstrap":
            bootstrap_operations()
        elif args.action == "PrepareEval":
            run(python, str(SERVICE / "evaluation/setup.py"))
        elif args.action == "AgentSmoke":
            run(python, str(SERVICE / "evaluation/agent.py"), "--dataset", "dev", "--limit", "3", "--max-budget-cny", "0.5", "--run")
        elif args.action == "AgentBaseline":
            run(python, str(SERVICE / "evaluation/agent.py"), "--dataset", "test", "--max-budget-cny", "5", "--run")
        elif args.action == "AgentRepair":
            run(python, str(SERVICE / "evaluation/agent.py"), "--dataset", "repair", "--max-budget-cny", "2", "--run")
        elif args.action == "AgentFollowup":
            run(python, str(SERVICE / "evaluation/agent.py"), "--dataset", "followup", "--max-budget-cny", "1", "--run")
        elif args.action == "AgentVerify":
            run(python, str(SERVICE / "evaluation/agent.py"), "--dataset", "verify", "--max-budget-cny", "0.5", "--run")
        elif args.action == "StopEval":
            run(*compose, "-f", "compose.evaluation.yml", "--profile", "evaluation", "stop", "support-eval")
    except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            print(f"Project task failed with exit code {error.returncode}.", file=sys.stderr)
        else:
            print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
