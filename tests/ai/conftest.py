"""Shared fixtures for AI engine tests.

Unit tests drive the real agent loop with ``FakeLlm`` (a scripted stand-in for the model).
Tests marked ``live`` use the real local Ollama model; ``speech`` tests use the downloaded
Whisper/Kokoro models. Neither kind ever talks to a hosted service.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from rabt_ai.api import create_app
from rabt_ai.config import Settings
from rabt_ai.llm import ChatResult, ToolCall
from rabt_ai.services import Engine
from rabt_shared import ComponentState, LlmStatus

TEST_KEY = "test-service-key-0123456789abcdefghijkl"
SENT_AT = "2026-10-10T20:15:00+04:00"  # Saturday evening in Dubai; "tomorrow" = Sun 2026-10-11


def reply(text: str) -> ChatResult:
    return ChatResult(content=text, thinking="", tool_calls=[], model="fake-llm")


def call(name: str, content: str = "", **arguments: Any) -> ChatResult:
    return ChatResult(
        content=content, thinking="", tool_calls=[ToolCall(name=name, arguments=arguments)], model="fake-llm"
    )


def calls(*items: tuple[str, dict[str, Any]]) -> ChatResult:
    return ChatResult(
        content="",
        thinking="",
        tool_calls=[ToolCall(name=n, arguments=a) for n, a in items],
        model="fake-llm",
    )


class FakeLlm:
    """Scripted model. Each script item is a ChatResult, an Exception, or fn(messages, tools)."""

    model_name = "fake-llm"

    def __init__(self, script: list[Any] | None = None):
        self.script = list(script or [])
        self.calls: list[dict[str, Any]] = []

    def push(self, *items: Any) -> None:
        self.script.extend(items)

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatResult:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)})
        if not self.script:
            raise AssertionError("FakeLlm script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(messages, tools) if callable(item) else item

    async def llm_status(self) -> LlmStatus:
        return LlmStatus(
            model=self.model_name, state=ComponentState.READY, reachable=True, model_available=True
        )

    def last_messages(self) -> list[dict[str, Any]]:
        return self.calls[-1]["messages"]

    def all_text(self, index: int = -1) -> str:
        return "\n".join(str(m.get("content", "")) for m in self.calls[index]["messages"])


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "internal_api_key": TEST_KEY,
        "data_dir": tmp_path / "data",
        "llm_model": "qwen3.5:9b",
        "llm_base_url": "http://127.0.0.1:11434",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def fake_llm() -> FakeLlm:
    return FakeLlm()


@pytest.fixture
def engine(settings: Settings, fake_llm: FakeLlm) -> Iterator[Engine]:
    eng = Engine.build(settings, llm=fake_llm)
    yield eng
    eng.storage.close()


@pytest.fixture
def client(engine: Engine) -> Iterator[TestClient]:
    app = create_app(engine=engine, warm_up=False)
    with TestClient(app, headers={"Authorization": f"Bearer {TEST_KEY}"}) as c:
        yield c


@pytest.fixture
def respond_body() -> Callable[..., dict[str, Any]]:
    def build(message: str, **overrides: Any) -> dict[str, Any]:
        body = {
            "request_id": str(uuid4()),
            "conversation_id": overrides.pop("conversation_id", None) or str(uuid4()),
            "account_id": "awaiz",
            "role": "tutor",
            "channel": "web",
            "message": message,
            "timezone": "Asia/Dubai",
            "sent_at": SENT_AT,
        }
        body.update(overrides)
        return body

    return build


def tool_result_body(
    response: dict[str, Any], proposal: dict[str, Any], account_id: str = "awaiz", **extra: Any
) -> dict[str, Any]:
    body = {
        "request_id": response["request_id"],
        "conversation_id": response["conversation_id"],
        "account_id": account_id,
        "proposal_id": proposal["proposal_id"],
        "tool": proposal["tool"],
    }
    body.update(extra)
    return body


def slots_result(*pairs: tuple[str, str]) -> dict[str, Any]:
    return {"slots": [{"start": s, "end": e} for s, e in pairs]}


def iso(text: str) -> datetime:
    return datetime.fromisoformat(text)
