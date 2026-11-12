from dataclasses import dataclass
from pathlib import Path
import os
from typing import Literal


RetrievalMode = Literal["lexical", "hybrid", "rerank"]
_PARENTS = Path(__file__).resolve().parents
_PROJECT_ROOT = _PARENTS[3] if len(_PARENTS) > 3 else Path.cwd()


@dataclass(frozen=True)
class Settings:
    db_path: Path
    credentials_path: Path
    public_base_url: str = "http://localhost:8090"
    tool_base_url: str = "http://support-agent:8090"
    draft_ttl_seconds: int = 1800
    session_ttl_seconds: int = 28800
    secure_cookies: bool = False
    seed_path: Path = Path(__file__).resolve().parent.parent / "data" / "knowledge.json"
    retrieval_mode: RetrievalMode = "lexical"
    retrieval_enabled: bool = False
    models_path: Path = _PROJECT_ROOT / ".local/models/support-retrieval"
    vector_cache_path: Path = _PROJECT_ROOT / ".local/retrieval-vectors"
    retrieval_threads: int = 2
    operations_path: Path | None = None
    allow_legacy_tool_keys: bool = True

    def __post_init__(self):
        if self.retrieval_mode not in {"lexical", "hybrid", "rerank"}:
            raise ValueError("SUPPORT_RETRIEVAL_MODE must be lexical, hybrid or rerank.")
        if self.retrieval_threads < 1:
            raise ValueError("SUPPORT_RETRIEVAL_THREADS must be positive.")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=Path(os.environ.get("SUPPORT_DB_PATH", "./.local/support.db")),
            credentials_path=Path(os.environ.get("SUPPORT_CONFIG_PATH", "./.local/users.json")),
            public_base_url=os.environ.get("SUPPORT_PUBLIC_BASE_URL", "http://localhost:8090").rstrip("/"),
            tool_base_url=os.environ.get("SUPPORT_TOOL_BASE_URL", "http://support-agent:8090").rstrip("/"),
            draft_ttl_seconds=int(os.environ.get("SUPPORT_DRAFT_TTL_SECONDS", "1800")),
            session_ttl_seconds=int(os.environ.get("SUPPORT_SESSION_TTL_SECONDS", "28800")),
            secure_cookies=os.environ.get("SUPPORT_SECURE_COOKIES", "false").lower() == "true",
            retrieval_mode=os.environ.get("SUPPORT_RETRIEVAL_MODE", "lexical"),
            retrieval_enabled=os.environ.get("SUPPORT_RETRIEVAL_ENABLED", "false").lower() == "true",
            models_path=Path(os.environ.get("SUPPORT_MODELS_PATH", str(_PROJECT_ROOT / ".local/models/support-retrieval"))),
            vector_cache_path=Path(os.environ.get("SUPPORT_VECTOR_CACHE_PATH", str(_PROJECT_ROOT / ".local/retrieval-vectors"))),
            retrieval_threads=int(os.environ.get("SUPPORT_RETRIEVAL_THREADS", "2")),
            operations_path=Path(os.environ["SUPPORT_OPERATIONS_PATH"]) if os.environ.get("SUPPORT_OPERATIONS_PATH") else None,
            allow_legacy_tool_keys=os.environ.get("SUPPORT_ALLOW_LEGACY_TOOL_KEYS", "true").lower() == "true",
        )
