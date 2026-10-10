"""HTTP API (FastAPI) implementing docs/API_CONTRACT.md."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Body, Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException

from rabt_ai import __version__
from rabt_ai.config import Settings
from rabt_ai.errors import RabtError
from rabt_ai.llm import OllamaClient
from rabt_ai.logging_setup import configure_logging, log_event
from rabt_ai.services import Engine
from rabt_shared import (
    CONTRACT_VERSION,
    ERROR_HTTP_STATUS,
    AgentRespondRequest,
    AgentResponse,
    AgentStatusResponse,
    ErrorCode,
    ErrorInfo,
    ErrorResponse,
    HealthResponse,
    SpeakRequest,
    ToolResultRequest,
    TranscriptionResponse,
)

log = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

_bearer = HTTPBearer(auto_error=False, description="Shared service token (RABT_INTERNAL_API_KEY).")


def _envelope(code: ErrorCode, message: str, request_id: str | None, details: Any = None) -> dict[str, Any]:
    return ErrorResponse(
        error=ErrorInfo(code=code, message=message, request_id=request_id, details=details)
    ).model_dump(mode="json")


def _error_response(
    request: Request,
    code: ErrorCode,
    message: str,
    details: Any = None,
    *,
    status: int | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    rid = request_id or getattr(request.state, "request_id", None)
    headers = {"X-Request-Id": rid} if rid else {}
    if code == ErrorCode.UNAUTHORIZED:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        _envelope(code, message, rid, details), status_code=status or ERROR_HTTP_STATUS[code], headers=headers
    )


class GuardMiddleware:
    """Adds contract headers and enforces request body size limits before parsing."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-rabt-contract-version", CONTRACT_VERSION.encode()))
                message["headers"] = headers
            await send(message)

        engine: Engine | None = getattr(scope["app"].state, "engine", None)
        settings = engine.settings if engine else None
        if settings is None:
            await self.app(scope, receive, send_with_headers)
            return
        is_upload = scope.get("path") == "/v1/audio/transcribe"
        limit = settings.max_audio_upload_bytes + 64 * 1024 if is_upload else settings.max_json_body_bytes
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            body = json.dumps(
                _envelope(ErrorCode.PAYLOAD_TOO_LARGE, f"request body exceeds {limit} bytes", None)
            ).encode()
            await send_with_headers(
                {
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send_with_headers({"type": "http.response.body", "body": body})
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise StarletteHTTPException(
                        status_code=413, detail=f"request body exceeds {limit} bytes"
                    )
            return message

        await self.app(scope, limited_receive, send_with_headers)


def _engine(request: Request) -> Engine:
    return request.app.state.engine


async def require_service(
    request: Request, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]
) -> None:
    expected = _engine(request).settings.internal_api_key.get_secret_value().encode()
    supplied = (credentials.credentials if credentials else "").encode()
    if (
        not credentials
        or credentials.scheme.lower() != "bearer"
        or not hmac.compare_digest(supplied, expected)
    ):
        raise RabtError(ErrorCode.UNAUTHORIZED, "missing or invalid service token")


async def _retention_loop(engine: Engine) -> None:
    while True:
        await asyncio.sleep(3600)
        removed = engine.storage.purge_older_than(engine.settings.retention_days)
        if removed:
            log_event(log, "retention_purge", conversations=removed)


async def _warm_up(engine: Engine) -> None:
    if not isinstance(engine.llm, OllamaClient):
        return
    try:
        await engine.llm.warm_up()
        log_event(log, "llm_warm", model=engine.llm.model_name)
    except Exception as exc:  # the status endpoint reports the problem; startup must not fail
        log_event(log, "llm_warm_failed", logging.WARNING, error=str(exc)[:300])


def create_app(
    settings: Settings | None = None, *, engine: Engine | None = None, warm_up: bool = True
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        eng = engine or Engine.build(settings or Settings())
        if engine is None:
            configure_logging(eng.settings.log_level)
        app.state.engine = eng
        removed = eng.storage.purge_older_than(eng.settings.retention_days)
        log_event(
            log,
            "engine_started",
            version=__version__,
            contract=CONTRACT_VERSION,
            model=eng.llm.model_name,
            host=eng.settings.host,
            port=eng.settings.port,
            purged=removed,
        )
        tasks = [asyncio.create_task(_retention_loop(eng))]
        if warm_up:
            tasks.append(asyncio.create_task(_warm_up(eng)))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await eng.aclose()

    app = FastAPI(
        title="Rabt AI Engine",
        version=__version__,
        summary="Local LLM agent, speech-to-text and text-to-speech for Rabt.",
        description="Contract: docs/API_CONTRACT.md (version " + CONTRACT_VERSION + "). "
        "All inference runs locally. The engine proposes actions; the integration backend executes them.",
        lifespan=lifespan,
    )
    app.add_middleware(GuardMiddleware)

    # ------------------------------------------------------------------ errors

    @app.exception_handler(RabtError)
    async def _rabt_error(request: Request, exc: RabtError) -> JSONResponse:
        if exc.http_status >= 500:
            log_event(log, "request_failed", logging.WARNING, code=exc.code.value, error=exc.message[:300])
        return _error_response(request, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "loc": [str(p) for p in err.get("loc", ())],
                "msg": str(err.get("msg", "")),
                "type": str(err.get("type", "")),
            }
            for err in exc.errors()[:20]
        ]
        in_result = request.url.path.endswith("/tool-result") and any(
            "result" in d["loc"][1:3] for d in details
        )
        code = ErrorCode.INVALID_TOOL_RESULT if in_result else ErrorCode.VALIDATION_ERROR
        body = exc.body if isinstance(exc.body, dict) else {}
        rid = body.get("request_id") if isinstance(body.get("request_id"), str) else None
        return _error_response(request, code, "Request body failed validation.", details, request_id=rid)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {
            401: ErrorCode.UNAUTHORIZED,
            404: ErrorCode.NOT_FOUND,
            405: ErrorCode.NOT_FOUND,
            413: ErrorCode.PAYLOAD_TOO_LARGE,
            415: ErrorCode.UNSUPPORTED_MEDIA_TYPE,
        }.get(
            exc.status_code, ErrorCode.VALIDATION_ERROR if exc.status_code < 500 else ErrorCode.INTERNAL_ERROR
        )
        return _error_response(request, code, str(exc.detail), status=exc.status_code)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", extra={"fields": {"path": request.url.path}})
        return _error_response(request, ErrorCode.INTERNAL_ERROR, "internal error; see engine logs")

    # ------------------------------------------------------------------ routes

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse(service="rabt-ai-engine", version=__version__)

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_service)])

    @v1.get("/agent/status", response_model=AgentStatusResponse, tags=["agent"])
    async def agent_status(request: Request) -> AgentStatusResponse:
        return await _engine(request).status()

    @v1.post("/agent/respond", response_model=AgentResponse, tags=["agent"])
    async def agent_respond(body: AgentRespondRequest, request: Request, response: Response) -> AgentResponse:
        request.state.request_id = str(body.request_id)
        response.headers["X-Request-Id"] = str(body.request_id)
        return await _engine(request).agent.respond(body)

    @v1.post("/agent/tool-result", response_model=AgentResponse, tags=["agent"])
    async def agent_tool_result(
        body: Annotated[ToolResultRequest, Body()], request: Request, response: Response
    ) -> AgentResponse:
        request.state.request_id = str(body.request_id)
        response.headers["X-Request-Id"] = str(body.request_id)
        return await _engine(request).agent.tool_result(body)

    @v1.post("/audio/transcribe", response_model=TranscriptionResponse, tags=["audio"])
    async def audio_transcribe(
        request: Request,
        response: Response,
        file: Annotated[UploadFile, File(description="Audio: wav, mp3, ogg/opus, webm, m4a, flac")],
        account_id: Annotated[str, Form(pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=64)],
        request_id: Annotated[UUID | None, Form()] = None,
        language: Annotated[str | None, Form(pattern=r"^[a-z]{2,3}$")] = None,
    ) -> TranscriptionResponse:
        engine = _engine(request)
        if request_id:
            request.state.request_id = str(request_id)
            response.headers["X-Request-Id"] = str(request_id)
        engine.registry.require(account_id)
        content_type = (file.content_type or "").split(";")[0].strip().lower()
        if content_type and not (
            content_type.startswith(("audio/", "video/")) or content_type == "application/octet-stream"
        ):
            raise RabtError(ErrorCode.UNSUPPORTED_MEDIA_TYPE, f"unsupported content type {content_type!r}")
        data = await file.read(engine.settings.max_audio_upload_bytes + 1)
        if len(data) > engine.settings.max_audio_upload_bytes:
            raise RabtError(ErrorCode.PAYLOAD_TOO_LARGE, "audio file exceeds the upload limit")
        result = await engine.stt.transcribe(data, language)
        return TranscriptionResponse(request_id=request_id, **result)

    @v1.post(
        "/audio/speak",
        response_class=Response,
        tags=["audio"],
        responses={
            200: {
                "description": "Synthesized speech (AI-generated).",
                "content": {"audio/wav": {}, "audio/ogg": {}},
            }
        },
    )
    async def audio_speak(body: SpeakRequest, request: Request) -> Response:
        engine = _engine(request)
        if body.request_id:
            request.state.request_id = str(body.request_id)
        profile = engine.registry.voice_profile_for(body.account_id, body.voice_profile)
        synth = await engine.tts.synthesize(profile, body.text, body.speed, body.format)
        headers = {
            "X-Rabt-AI-Generated": "true",
            "X-Rabt-Voice-Profile": profile.profile_id,
            "X-Rabt-TTS-Model": f"{synth.model}:{synth.voice}",
            "X-Rabt-Latency-Ms": str(synth.latency_ms),
            "X-Rabt-Audio-Duration-S": f"{synth.duration_s:.3f}",
            "Content-Disposition": f'inline; filename="rabt-{profile.profile_id}.{"ogg" if synth.mime_type == "audio/ogg" else "wav"}"',
        }
        if body.request_id:
            headers["X-Request-Id"] = str(body.request_id)
        return Response(content=synth.audio, media_type=synth.mime_type, headers=headers)

    app.include_router(v1)
    return app
