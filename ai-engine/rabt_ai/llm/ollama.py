"""Async client for a local Ollama server (``POST /api/chat`` with native tool calling).

Safety properties:
* the base URL must be loopback/private (checked in config and again here);
* environment proxies are ignored (``trust_env=False``) so traffic cannot be rerouted;
* Ollama "cloud" models, whose prompts are forwarded to ollama.com, are refused.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from rabt_ai.config import Settings, ensure_local_url, is_cloud_model_name
from rabt_ai.errors import RabtError
from rabt_ai.logging_setup import log_event
from rabt_shared import ComponentState, ErrorCode, LlmStatus

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None
    arguments_error: str | None = None


@dataclass
class ChatResult:
    content: str
    thinking: str
    tool_calls: list[ToolCall]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_ms: int = 0
    load_ms: int = 0
    done_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class ChatModel(Protocol):
    """What the agent needs from a language model. Implemented by OllamaClient and test fakes."""

    model_name: str

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatResult: ...


def _error_text(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("error", resp.text))
    except (ValueError, AttributeError):
        return resp.text


def parse_chat_response(body: dict[str, Any], wall_ms: int) -> ChatResult:
    msg = body.get("message") or {}
    calls: list[ToolCall] = []
    for raw in msg.get("tool_calls") or []:
        fn = raw.get("function") or {}
        args: Any = fn.get("arguments")
        error = None
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args, error = {}, "arguments were not valid JSON"
        if args is None:
            args = {}
        if not isinstance(args, dict):
            args, error = {}, "arguments must be a JSON object"
        calls.append(
            ToolCall(name=str(fn.get("name") or ""), arguments=args, id=raw.get("id"), arguments_error=error)
        )
    return ChatResult(
        content=msg.get("content") or "",
        thinking=msg.get("thinking") or "",
        tool_calls=calls,
        model=str(body.get("model") or ""),
        prompt_tokens=int(body.get("prompt_eval_count") or 0),
        completion_tokens=int(body.get("eval_count") or 0),
        wall_ms=wall_ms,
        load_ms=int((body.get("load_duration") or 0) / 1e6),
        done_reason=body.get("done_reason"),
        raw=body,
    )


class OllamaClient:
    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None):
        self._settings = settings
        self.model_name = settings.llm_model
        self.base_url = ensure_local_url(settings.llm_base_url)
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=5.0),
            trust_env=False,
            transport=transport,
        )
        self._local_model_verified = False

    async def aclose(self) -> None:
        await self._http.aclose()

    def options(self) -> dict[str, Any]:
        s = self._settings
        opts: dict[str, Any] = {
            # Always send num_ctx: Ollama's VRAM-based default is only 4096 on a 16 GB GPU,
            # and changing it between calls forces a model reload.
            "num_ctx": s.llm_num_ctx,
            "temperature": s.llm_temperature,
            "top_p": s.llm_top_p,
            "top_k": s.llm_top_k,
            "presence_penalty": s.llm_presence_penalty,
        }
        if s.llm_seed is not None:
            opts["seed"] = s.llm_seed
        return opts

    # ----------------------------------------------------------------- metadata

    async def _get_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            resp = await self._http.request(method, path, timeout=15.0, **kwargs)
        except httpx.TransportError as exc:
            raise RabtError(
                ErrorCode.LLM_UNAVAILABLE, f"cannot reach Ollama at {self.base_url} ({type(exc).__name__})"
            ) from exc
        if resp.status_code == 404:
            raise RabtError(
                ErrorCode.LLM_UNAVAILABLE,
                f"model {self.model_name!r} is not available locally; run: ollama pull {self.model_name}",
            )
        if resp.status_code >= 400:
            raise RabtError(
                ErrorCode.LLM_UNAVAILABLE, f"Ollama error {resp.status_code}: {_error_text(resp)[:300]}"
            )
        return resp.json()

    async def version(self) -> str:
        return str((await self._get_json("GET", "/api/version")).get("version", "unknown"))

    async def show(self) -> dict[str, Any]:
        return await self._get_json("POST", "/api/show", json={"model": self.model_name})

    async def running_models(self) -> list[dict[str, Any]]:
        return list((await self._get_json("GET", "/api/ps")).get("models") or [])

    async def ensure_local_model(self) -> dict[str, Any]:
        """Refuse remote/cloud models: their prompts would leave this machine."""
        show = await self.show()
        if show.get("remote_host") or show.get("remote_model") or is_cloud_model_name(self.model_name):
            raise RabtError(
                ErrorCode.LLM_UNAVAILABLE,
                f"{self.model_name} is served remotely by Ollama; Rabt only uses models running on this machine",
            )
        self._local_model_verified = True
        return show

    async def llm_status(self) -> LlmStatus:
        """Report reachability, local availability and actual GPU residency (from /api/ps)."""
        try:
            show = await self.ensure_local_model()
        except RabtError as exc:
            reachable = "cannot reach" not in exc.message
            return LlmStatus(
                model=self.model_name,
                state=ComponentState.UNAVAILABLE,
                reachable=reachable,
                model_available=False,
                detail=exc.message,
            )
        capabilities = [str(c) for c in show.get("capabilities") or []]
        running = await self.running_models()
        entry = next((m for m in running if self.model_name in (m.get("name"), m.get("model"))), None)
        if entry is None:
            return LlmStatus(
                model=self.model_name,
                state=ComponentState.NOT_LOADED,
                reachable=True,
                model_available=True,
                capabilities=capabilities,
                detail="model is pulled; it loads on the first request",
            )
        size, size_vram = int(entry.get("size") or 0), int(entry.get("size_vram") or 0)
        detail = None
        if size and size_vram < size:
            detail = f"{100 * size_vram / size:.0f}% of the model is in GPU memory; the rest runs on CPU"
        return LlmStatus(
            model=self.model_name,
            state=ComponentState.READY,
            reachable=True,
            model_available=True,
            on_gpu=size_vram > 0,
            vram_bytes=size_vram,
            context_length=entry.get("context_length"),
            capabilities=capabilities,
            detail=detail,
        )

    async def warm_up(self) -> None:
        """Load the model into memory with the same options real requests use (avoids a reload)."""
        await self.ensure_local_model()
        payload = {
            "model": self.model_name,
            "messages": [],
            "keep_alive": self._settings.llm_keep_alive,
            "options": self.options(),
        }
        await self._http.post("/api/chat", json=payload)

    # ----------------------------------------------------------------- chat

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatResult:
        if not self._local_model_verified:
            await self.ensure_local_model()
        options = self.options()
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": self._settings.llm_think,
            "keep_alive": self._settings.llm_keep_alive,
            "options": options,
        }
        if tools:
            payload["tools"] = tools

        attempts = self._settings.llm_max_retries + 1
        last_error = ""
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                resp = await self._http.post("/api/chat", json=payload)
            except httpx.TimeoutException as exc:
                raise RabtError(
                    ErrorCode.LLM_TIMEOUT,
                    f"the local model did not answer within {self._settings.llm_timeout_s:.0f}s",
                ) from exc
            except httpx.TransportError as exc:
                raise RabtError(
                    ErrorCode.LLM_UNAVAILABLE,
                    f"cannot reach Ollama at {self.base_url} ({type(exc).__name__})",
                ) from exc
            wall_ms = int((time.perf_counter() - started) * 1000)

            if resp.status_code == 404:
                raise RabtError(
                    ErrorCode.LLM_UNAVAILABLE,
                    f"model {self.model_name!r} is not available locally; run: ollama pull {self.model_name}",
                )
            if resp.status_code >= 500:
                # Ollama returns 500 when the model emits malformed tool-call markup; resample.
                last_error = _error_text(resp)
                log_event(
                    log,
                    "llm_retry",
                    logging.WARNING,
                    attempt=attempt,
                    status=resp.status_code,
                    error=last_error[:200],
                )
                payload["options"] = {**options, "seed": random.randint(1, 2**31 - 1)}
                await asyncio.sleep(0.25 * attempt)
                continue
            if resp.status_code >= 400:
                raise RabtError(
                    ErrorCode.LLM_UNAVAILABLE,
                    f"Ollama rejected the request ({resp.status_code}): {_error_text(resp)[:300]}",
                )
            result = parse_chat_response(resp.json(), wall_ms)
            log_event(
                log,
                "llm_call",
                model=self.model_name,
                wall_ms=wall_ms,
                load_ms=result.load_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                tool_calls=[c.name for c in result.tool_calls],
                done_reason=result.done_reason,
            )
            return result

        raise RabtError(
            ErrorCode.LLM_UNAVAILABLE,
            f"the local model failed {attempts} times; last error: {last_error[:300]}",
        )
