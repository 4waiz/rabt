"""Local-only inference guarantees, logging redaction and configuration safety (14)."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import make_settings
from pydantic import ValidationError

from rabt_ai.config import HostedEndpointError, Settings, ensure_local_url, is_cloud_model_name
from rabt_ai.errors import RabtError
from rabt_ai.llm.ollama import OllamaClient
from rabt_ai.logging_setup import redact

AI_SRC = Path(__file__).resolve().parents[2] / "ai-engine" / "rabt_ai"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://[::1]:11434",
        "http://192.168.1.20:11434",
        "http://10.0.0.5",
    ],
)
def test_local_and_private_endpoints_are_allowed(url: str) -> None:
    assert ensure_local_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "https://api.anthropic.com",
        "https://generativelanguage.googleapis.com",
        "https://api.groq.com/openai/v1",
        "https://ollama.com",
        "http://8.8.8.8:11434",
        "http://[2001:4860:4860::8888]:11434",
        "ftp://127.0.0.1",
    ],
)
def test_hosted_or_public_endpoints_are_refused(url: str) -> None:
    with pytest.raises(HostedEndpointError):
        ensure_local_url(url)


def test_settings_refuse_hosted_urls_and_cloud_models(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        make_settings(tmp_path, llm_base_url="https://api.openai.com/v1")
    for model in ("gpt-oss:120b-cloud", "qwen3-coder:480b-cloud", "deepseek-v3.1:671b-cloud"):
        assert is_cloud_model_name(model)
        with pytest.raises(ValidationError):
            make_settings(tmp_path, llm_model=model)
    assert not is_cloud_model_name("qwen3.5:9b")


def test_empty_env_values_fall_back_to_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RABT_DATA_DIR", "")
    monkeypatch.setenv("RABT_INTERNAL_API_KEY", "k" * 32)
    settings = Settings(_env_file=None)
    assert settings.data_dir != Path(".") and settings.data_dir.is_absolute()


def test_service_key_must_be_long(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        make_settings(tmp_path, internal_api_key="short")


def _client(tmp_path: Path, handler: Any) -> OllamaClient:
    return OllamaClient(make_settings(tmp_path), transport=httpx.MockTransport(handler))


def test_ollama_client_ignores_environment_proxies(tmp_path: Path) -> None:
    client = OllamaClient(make_settings(tmp_path))
    assert client._http._trust_env is False  # noqa: SLF001 - the property under test
    asyncio.run(client.aclose())


def test_remote_ollama_models_are_refused(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(
                200, json={"remote_host": "https://ollama.com:443", "remote_model": "gpt-oss:120b"}
            )
        raise AssertionError(f"chat must not be called: {request.url}")

    client = _client(tmp_path, handler)
    with pytest.raises(RabtError) as exc:
        asyncio.run(client.chat([{"role": "user", "content": "hi"}], []))
    assert "remotely" in exc.value.message


def test_ollama_client_retries_malformed_tool_calls_and_parses_arguments(tmp_path: Path) -> None:
    attempts = {"chat": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "tools"]})
        attempts["chat"] += 1
        payload = json.loads(request.content)
        assert (
            payload["options"]["num_ctx"] == 16384
            and payload["think"] is False
            and payload["stream"] is False
        )
        if attempts["chat"] == 1:
            return httpx.Response(500, json={"error": "failed to parse tool call"})
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "index": 0,
                        "name": "find_mutual_free_time",
                        "arguments": '{"date": "2026-10-11"}',
                    },
                }
            ],
        }
        return httpx.Response(
            200, json={"model": "qwen3.5:9b", "message": message, "done_reason": "stop", "eval_count": 9}
        )

    result = asyncio.run(
        _client(tmp_path, handler).chat([{"role": "user", "content": "hi"}], [{"type": "function"}])
    )
    assert attempts["chat"] == 2
    assert result.tool_calls[0].name == "find_mutual_free_time"
    assert result.tool_calls[0].arguments == {"date": "2026-10-11"}


def test_ollama_down_maps_to_llm_unavailable(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(RabtError) as exc:
        asyncio.run(_client(tmp_path, handler).chat([{"role": "user", "content": "hi"}], []))
    assert exc.value.code == "llm_unavailable"


def test_engine_source_has_no_hosted_inference_clients() -> None:
    forbidden_imports = re.compile(
        r"^\s*(from|import)\s+(openai|anthropic|google\.generativeai|google\.genai|groq|cohere|mistralai|elevenlabs|litellm)\b",
        re.MULTILINE,
    )
    for path in AI_SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not forbidden_imports.search(text), f"hosted inference client imported in {path}"
        if path.name != "config.py":  # config.py holds the deny-list itself
            assert "api.openai.com" not in text and "api.anthropic.com" not in text, path


def test_logs_redact_secrets_and_personal_data() -> None:
    line = (
        "Authorization: Bearer abcDEF1234567890xyz token=supersecretvalue1 key sk-proj-ABCDEFGHIJKLMNOPQRST "
        "mail huda@example.com phone +971 50 123 4567 ghp_abcdefghijklmnopqrstuvwxyz0123"
    )
    out = redact(line)
    for secret in (
        "abcDEF1234567890xyz",
        "supersecretvalue1",
        "sk-proj-ABCDEFGHIJKLMNOPQRST",
        "huda@example.com",
        "50 123 4567",
        "ghp_abcdefghijklmnopqrstuvwxyz0123",
    ):
        assert secret not in out
