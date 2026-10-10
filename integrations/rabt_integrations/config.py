"""Backend settings, read from the environment (.env is loaded if present). Secrets never live in the repo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _client_secret_file(data_dir: Path) -> Path:
    """RABT_GOOGLE_CLIENT_SECRET_FILE, else secrets/google_client_secret.json, else the one
    client_secret*.json that Google's download button produced in the secrets folder."""
    if os.environ.get("RABT_GOOGLE_CLIENT_SECRET_FILE"):
        return Path(os.environ["RABT_GOOGLE_CLIENT_SECRET_FILE"]).expanduser()
    secrets_dir = data_dir / "secrets"
    default = secrets_dir / "google_client_secret.json"
    downloads = sorted(secrets_dir.glob("client_secret*.json")) if secrets_dir.is_dir() else []
    return downloads[0] if not default.is_file() and len(downloads) == 1 else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    ai_mode: str  # "real" or "mock"
    ai_base_url: str
    internal_api_key: str
    session_secret: str
    google_client_secret_file: Path
    timezone: str
    host: str
    port: int

    @property
    def secrets_dir(self) -> Path:
        return self.data_dir / "secrets"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "integrations.sqlite3"


def load_settings() -> Settings:
    _load_dotenv(Path.cwd() / ".env")
    # Separate from the AI engine's RABT_DATA_DIR so OAuth tokens never sit next to its files, and
    # outside %LOCALAPPDATA%, which MSIX-packaged apps (e.g. the Claude desktop app) redirect.
    data_dir = Path(os.environ.get("RABT_BACKEND_DATA_DIR") or Path.home() / ".rabt-backend").expanduser()
    return Settings(
        data_dir=data_dir,
        ai_mode=os.environ.get("RABT_AI_MODE", "mock").lower(),
        ai_base_url=os.environ.get("RABT_AI_BASE_URL", "http://127.0.0.1:8765").rstrip("/"),
        internal_api_key=os.environ.get("RABT_INTERNAL_API_KEY", ""),
        session_secret=os.environ.get("RABT_SESSION_SECRET", ""),
        google_client_secret_file=_client_secret_file(data_dir),
        timezone=os.environ.get("RABT_DEFAULT_TIMEZONE", "Asia/Dubai"),
        host=os.environ.get("RABT_BACKEND_HOST", "127.0.0.1"),
        port=int(os.environ.get("RABT_BACKEND_PORT", "8800")),
    )
