"""Backend settings, read from the environment (.env is loaded if present). Secrets never live in the repo."""

from __future__ import annotations

import os
import sys
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


def _default_data_dir() -> Path:
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "Rabt"
    return Path.home() / ".local" / "share" / "rabt"


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
    data_dir = Path(os.environ.get("RABT_DATA_DIR") or _default_data_dir())
    return Settings(
        data_dir=data_dir,
        ai_mode=os.environ.get("RABT_AI_MODE", "mock").lower(),
        ai_base_url=os.environ.get("RABT_AI_BASE_URL", "http://127.0.0.1:8765").rstrip("/"),
        internal_api_key=os.environ.get("RABT_INTERNAL_API_KEY", ""),
        session_secret=os.environ.get("RABT_SESSION_SECRET", ""),
        google_client_secret_file=Path(
            os.environ.get("RABT_GOOGLE_CLIENT_SECRET_FILE")
            or data_dir / "secrets" / "google_client_secret.json"
        ),
        timezone=os.environ.get("RABT_DEFAULT_TIMEZONE", "Asia/Dubai"),
        host=os.environ.get("RABT_BACKEND_HOST", "127.0.0.1"),
        port=int(os.environ.get("RABT_BACKEND_PORT", "8800")),
    )
