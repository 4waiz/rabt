"""Runtime settings: environment variables prefixed with ``RABT_`` (optionally from ``.env``)."""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rabt_shared import MAX_AUDIO_DURATION_S, MAX_AUDIO_UPLOAD_BYTES

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent

# Hosted inference providers. Rabt's runtime must never send prompts or audio to any of them.
HOSTED_INFERENCE_DOMAINS = (
    "openai.com",
    "anthropic.com",
    "claude.ai",
    "googleapis.com",
    "groq.com",
    "openrouter.ai",
    "together.xyz",
    "together.ai",
    "mistral.ai",
    "cohere.ai",
    "cohere.com",
    "deepseek.com",
    "ollama.com",
    "ollama.ai",
    "fireworks.ai",
    "perplexity.ai",
    "x.ai",
    "huggingface.co",
    "replicate.com",
    "azure.com",
    "amazonaws.com",
    "elevenlabs.io",
)


class HostedEndpointError(ValueError):
    """Raised when a configured inference endpoint is not on this machine or private network."""


def default_data_dir() -> Path:
    """Per-user data directory, deliberately outside the (possibly cloud-synced) repository.

    Windows uses ``%USERPROFILE%\\.rabt`` (like Ollama's ``.ollama``) rather than AppData:
    processes started from MSIX-packaged apps (e.g. the Claude desktop app) have AppData
    writes silently redirected to a private per-app folder, which would split Rabt's data
    between "launched from an app" and "launched from a terminal".
    """
    if sys.platform == "win32":
        return Path.home() / ".rabt"
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local" / "share") / "rabt"


def ensure_local_url(url: str) -> str:
    """Accept only loopback / private-network URLs and refuse hosted inference services."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise HostedEndpointError(f"invalid inference URL {url!r}")
    host = parts.hostname.lower().rstrip(".")
    if any(host == domain or host.endswith("." + domain) for domain in HOSTED_INFERENCE_DOMAINS):
        raise HostedEndpointError(f"{host} is a hosted inference service; Rabt only runs inference locally")
    if host == "localhost":
        return url
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, parts.port or 80, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise HostedEndpointError(f"cannot resolve inference host {host!r}") from exc
        addresses = [ipaddress.ip_address(info[4][0]) for info in infos]
    public = [str(a) for a in addresses if a.is_global or a.is_multicast or a.is_unspecified]
    if public:
        raise HostedEndpointError(
            f"inference host {host!r} resolves to non-private address(es) {public}; "
            "use Ollama on this machine or your private network"
        )
    return url


def is_cloud_model_name(model: str) -> bool:
    """Ollama 'cloud' models (e.g. ``gpt-oss:120b-cloud``) run on ollama.com, not locally."""
    name = model.lower()
    return name.endswith("-cloud") or name.endswith(":cloud") or ":cloud-" in name


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RABT_",
        env_file=(str(REPO_ROOT / ".env"), ".env"),
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `RABT_DATA_DIR=` must mean "default", not the current directory
        extra="ignore",
    )

    # --- service
    internal_api_key: SecretStr = Field(min_length=24)
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    data_dir: Path = Field(default_factory=default_data_dir)
    model_dir: Path | None = None  # default: <data_dir>/models
    default_timezone: str = "Asia/Dubai"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    identities_file: Path = PACKAGE_DIR / "defaults" / "identities.toml"
    retention_days: int = Field(default=30, ge=1, le=3650)

    # --- local LLM (Ollama)
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3.5:9b"
    llm_think: bool = False
    llm_num_ctx: int = Field(default=16384, ge=2048, le=262144)
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    llm_top_k: int = Field(default=20, ge=1, le=200)
    llm_presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_seed: int | None = None
    llm_keep_alive: str = "30m"
    llm_timeout_s: float = Field(default=180.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # --- agent loop bounds
    agent_max_llm_calls: int = Field(default=4, ge=1, le=10)
    agent_max_proposals_per_turn: int = Field(default=6, ge=1, le=20)
    agent_max_tool_calls_per_step: int = Field(default=3, ge=1, le=8)
    agent_history_char_budget: int = Field(default=24_000, ge=2_000)

    # --- speech
    stt_model: str = "large-v3-turbo"
    stt_device: Literal["auto", "cuda", "cpu"] = "auto"
    stt_compute_type: str = "auto"
    stt_beam_size: int = Field(default=5, ge=1, le=10)
    stt_language: str = "en"
    tts_engine: Literal["kokoro"] = "kokoro"

    # --- request limits
    max_json_body_bytes: int = 64 * 1024
    max_audio_upload_bytes: int = MAX_AUDIO_UPLOAD_BYTES
    max_audio_duration_s: int = MAX_AUDIO_DURATION_S

    @field_validator("llm_base_url")
    @classmethod
    def _llm_must_be_local(cls, value: str) -> str:
        return ensure_local_url(value)

    @field_validator("llm_model")
    @classmethod
    def _no_cloud_models(cls, value: str) -> str:
        if is_cloud_model_name(value):
            raise ValueError(f"{value!r} is an Ollama cloud model; choose a locally pulled model")
        return value

    @field_validator("default_timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rabt.db"

    @property
    def models_dir(self) -> Path:
        return self.model_dir or self.data_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"
