"""FastAPI app: dashboard login, chat, approvals, lesson workflow, Google status, voice proxy."""

from __future__ import annotations

import logging
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from rabt_integrations.accounts import ACCOUNTS, get_account
from rabt_integrations.ai_client import AIError, MockAIClient, RealAIClient
from rabt_integrations.config import Settings, load_settings
from rabt_integrations.google_api import GoogleApiError, GoogleGateway, GoogleNotConnected
from rabt_integrations.service import PolicyError, Service
from rabt_integrations.store import Store

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
MAX_BODY = 64 * 1024
MAX_AUDIO = 25 * 1024 * 1024


class LoginBody(BaseModel):
    account_id: str = Field(max_length=64)
    pin: str = Field(max_length=128)


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=4096)


class LessonBody(BaseModel):
    action: str = Field(max_length=20)
    start: str | None = Field(default=None, max_length=64)


class ShareBody(BaseModel):
    enabled: bool


class SpeakBody(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def create_app(settings: Settings | None = None, google=None, ai=None, store: Store | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = store or Store(settings.db_path)
    google = google or GoogleGateway(settings)
    if ai is None:
        ai = (
            RealAIClient(settings.ai_base_url, settings.internal_api_key)
            if settings.ai_mode == "real"
            else MockAIClient()
        )
    svc = Service(store, google, ai, settings.timezone)
    app = FastAPI(title="Rabt integrations")
    app.state.svc = svc
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret or secrets.token_urlsafe(32),
        same_site="strict",
        https_only=False,
    )

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        limit = MAX_AUDIO if request.url.path == "/api/voice/transcribe" else MAX_BODY
        if int(request.headers.get("content-length") or 0) > limit:
            return JSONResponse({"error": {"code": "payload_too_large"}}, status_code=413)
        return await call_next(request)

    def current(request: Request) -> str:
        acct = request.session.get("account_id")
        if acct not in ACCOUNTS:
            raise HTTPException(401, "Log in first.")
        return acct

    def guard(fn, *args):
        try:
            return fn(*args)
        except PolicyError as exc:
            status = 404 if exc.code == "not_found" else 409
            raise HTTPException(status, {"code": exc.code, "message": str(exc)}) from exc
        except AIError as exc:
            raise HTTPException(
                502 if exc.status < 500 else 503, {"code": exc.code, "message": str(exc)}
            ) from exc

    @app.get("/")
    def index():
        return FileResponse(FRONTEND / "index.html")

    @app.post("/api/login")
    def login(body: LoginBody, request: Request):
        acct = ACCOUNTS.get(body.account_id)
        if not acct or not acct.check_pin(body.pin):
            store.audit(body.account_id[:64] or "?", "login.failed")
            raise HTTPException(401, "Wrong account or PIN.")
        request.session.clear()
        request.session["account_id"] = acct.account_id
        store.audit(acct.account_id, "login.ok")
        return {"account_id": acct.account_id}

    @app.post("/api/logout")
    def logout(request: Request):
        request.session.clear()
        return {"ok": True}

    @app.get("/api/me")
    def me(account_id: str = Depends(current)):
        a = get_account(account_id)
        return {
            "account_id": a.account_id,
            "display_name": a.display_name,
            "role": a.role.value,
            "voice_profile": a.voice_profile,
            "shares_freebusy": svc.shares_freebusy(account_id),
            "lesson": svc.lesson_view(account_id),
        }

    @app.get("/api/status")
    def status(account_id: str = Depends(current)):
        accounts = {}
        for aid in ACCOUNTS:
            accounts[aid] = {
                "google_connected": google.connected(aid),
                # Only your own Google address is shown; for others just connected yes/no.
                "google_email": google.token_email(aid) if aid == account_id else None,
                "shares_freebusy": svc.shares_freebusy(aid),
            }
        return {"ai_mode": ai.mode, "ai": ai.health(), "accounts": accounts}

    @app.post("/api/settings/share-freebusy")
    def share(body: ShareBody, account_id: str = Depends(current)):
        store.set_setting(account_id, "share_freebusy", "1" if body.enabled else "0")
        store.audit(account_id, "consent.freebusy", enabled=body.enabled)
        return {"shares_freebusy": body.enabled}

    @app.post("/api/google/test")
    def google_test(account_id: str = Depends(current)):
        """Smoke test with the caller's own token: 3 recent emails (metadata) + busy blocks next 7 days."""
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        try:
            msgs = google.list_recent(account_id, 3, None, False)
            now = datetime.now(ZoneInfo(settings.timezone))
            busy = google.busy(account_id, now, now + timedelta(days=7), settings.timezone)
        except GoogleNotConnected as exc:
            raise HTTPException(409, {"code": "google_not_connected", "message": str(exc)}) from exc
        except GoogleApiError as exc:
            raise HTTPException(502, {"code": exc.code, "message": str(exc)}) from exc
        store.audit(account_id, "google.test", messages=len(msgs), busy=len(busy))
        return {
            "email": google.token_email(account_id),
            "messages": [{k: m[k] for k in ("sender", "subject", "received_at")} for m in msgs],
            "busy_blocks_next_7_days": [{"start": s.isoformat(), "end": e.isoformat()} for s, e in busy[:5]],
        }

    @app.post("/api/chat")
    def chat(body: ChatBody, account_id: str = Depends(current)):
        return guard(svc.chat, account_id, body.message)

    @app.post("/api/chat/reset")
    def reset(account_id: str = Depends(current)):
        store.conversation_id(account_id, reset=True)
        return {"ok": True}

    @app.get("/api/proposals")
    def proposals(account_id: str = Depends(current)):
        return store.list_proposals(account_id)

    @app.post("/api/proposals/{proposal_id}/approve")
    def approve(proposal_id: str, account_id: str = Depends(current)):
        return guard(svc.approve, account_id, proposal_id)

    @app.post("/api/proposals/{proposal_id}/reject")
    def reject(proposal_id: str, account_id: str = Depends(current)):
        return guard(svc.reject, account_id, proposal_id)

    @app.get("/api/lesson")
    def lesson(account_id: str = Depends(current)):
        return svc.lesson_view(account_id)

    @app.post("/api/lesson/{lesson_id}")
    def lesson_action(lesson_id: str, body: LessonBody, account_id: str = Depends(current)):
        return guard(svc.lesson_action, account_id, lesson_id, body.action, body.start)

    @app.get("/api/activity")
    def activity(account_id: str = Depends(current)):
        return store.audit_log(account_id)

    @app.post("/api/voice/transcribe")
    async def transcribe(file: UploadFile = File(...), account_id: str = Depends(current)):
        data = await file.read()
        if len(data) > MAX_AUDIO:
            raise HTTPException(413, "Audio too large.")
        return guard(
            ai.transcribe, account_id, file.filename or "audio.webm", data, file.content_type or "audio/webm"
        )

    @app.post("/api/voice/speak")
    def speak(body: SpeakBody, account_id: str = Depends(current)):
        audio, ctype, meta = guard(ai.speak, account_id, body.text)
        return Response(audio, media_type=ctype, headers={**meta, "X-Rabt-AI-Generated": "true"})

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level="INFO")
    settings = load_settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
