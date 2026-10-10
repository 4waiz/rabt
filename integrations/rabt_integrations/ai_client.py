"""Client for Awaiz's local AI engine (docs/API_CONTRACT.md), plus a clearly labelled mock.

The mock is rule-based (no LLM) and only exists so the backend can be developed and tested while
the engine is unavailable. Its responses carry model="MOCK (no LLM)" and are validated against the
same rabt_shared models as the real engine's.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from rabt_integrations.accounts import ACCOUNTS, other_account
from rabt_shared.agent import AgentResponse, TurnStatus


class AIError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


class RealAIClient:
    mode = "real"

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        self.base_url = base_url
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> AgentResponse:
        try:
            r = httpx.post(self.base_url + path, json=body, headers=self._headers, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise AIError(503, "ai_unreachable", f"AI engine unreachable at {self.base_url}") from exc
        if r.status_code >= 400:
            try:
                err = r.json()["error"]
            except Exception:
                err = {"code": "ai_error", "message": r.text[:300]}
            raise AIError(r.status_code, err.get("code", "ai_error"), err.get("message", ""))
        return AgentResponse.model_validate(r.json())

    def respond(self, req: dict) -> AgentResponse:
        return self._post("/v1/agent/respond", req)

    def tool_result(self, req: dict) -> AgentResponse:
        return self._post("/v1/agent/tool-result", req)

    def health(self) -> dict:
        try:
            r = httpx.get(self.base_url + "/health", timeout=5)
            return {"reachable": r.status_code == 200, **(r.json() if r.status_code == 200 else {})}
        except httpx.HTTPError:
            return {"reachable": False}

    def transcribe(self, account_id: str, filename: str, data: bytes, content_type: str) -> dict:
        r = httpx.post(
            self.base_url + "/v1/audio/transcribe",
            headers=self._headers,
            files={"file": (filename, data, content_type)},
            data={"account_id": account_id},
            timeout=self.timeout,
        )
        if r.status_code >= 400:
            raise AIError(r.status_code, "stt_error", r.text[:300])
        return r.json()

    def speak(self, account_id: str, text: str) -> tuple[bytes, str, dict]:
        r = httpx.post(
            self.base_url + "/v1/audio/speak",
            headers=self._headers,
            json={"account_id": account_id, "text": text[:2000]},
            timeout=self.timeout,
        )
        if r.status_code >= 400:
            raise AIError(r.status_code, "tts_error", r.text[:300])
        meta = {k: v for k, v in r.headers.items() if k.lower().startswith("x-rabt")}
        return r.content, r.headers.get("content-type", "audio/wav"), meta


_SLOT_TEXT = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}) to (\S+?[+-]\d{2}:\d{2})")


def _fmt(dt: datetime) -> str:
    return dt.strftime("%a %d %b, %H:%M")


class MockAIClient:
    """Deterministic stand-in for the AI engine. Not an LLM; never presented as one."""

    mode = "mock"
    MODEL = "MOCK (no LLM)"

    def __init__(self) -> None:
        self._pending: dict[str, set[str]] = {}
        self._proposals: dict[str, dict] = {}
        self._notes: dict[str, list[str]] = {}

    def health(self) -> dict:
        return {"reachable": True, "service": "mock"}

    def transcribe(self, *a: Any, **k: Any) -> dict:
        raise AIError(503, "stt_unavailable", "Speech input needs Awaiz's AI engine (mock mode).")

    def speak(self, *a: Any, **k: Any):
        raise AIError(503, "tts_unavailable", "Voice output needs Awaiz's AI engine (mock mode).")

    def _resp(self, req: dict, reply: str, proposals: list[dict] | None = None) -> AgentResponse:
        proposals = proposals or []
        rid = req["request_id"]
        for p in proposals:
            self._pending.setdefault(rid, set()).add(p["proposal_id"])
            self._proposals[p["proposal_id"]] = p
        pending = sorted(self._pending.get(rid, set()))
        if proposals:
            status = TurnStatus.ACTION_REQUIRED
        elif pending:
            status, reply = TurnStatus.AWAITING_RESULTS, ""
        else:
            status = TurnStatus.COMPLETED
        return AgentResponse.model_validate(
            {
                "request_id": rid,
                "conversation_id": req["conversation_id"],
                "status": status,
                "reply_text": reply,
                "action_proposals": proposals,
                "pending_proposal_ids": pending,
                "model": self.MODEL,
            }
        )

    def respond(self, req: dict) -> AgentResponse:
        msg = req["message"].lower()
        me = req["account_id"]
        other = other_account(me).account_id
        tz = ZoneInfo(req.get("timezone") or "Asia/Dubai")
        sent = datetime.fromisoformat(req["sent_at"]) if req.get("sent_at") else datetime.now(tz)
        sent = sent.astimezone(tz)
        m = _SLOT_TEXT.search(req["message"])
        if m and any(w in msg for w in ("book", "confirm", "yes")):
            start, end = datetime.fromisoformat(m.group(1)), datetime.fromisoformat(m.group(2))
            who = ACCOUNTS[other].display_name
            return self._resp(
                req,
                f"Here is the booking for {_fmt(start)}-{end:%H:%M}. Please review and approve.",
                [
                    {
                        "proposal_id": str(uuid.uuid4()),
                        "tool": "calendar.create_event",
                        "args": {
                            "title": "Tuition: Awaiz & Huda",
                            "start": start.isoformat(),
                            "end": end.isoformat(),
                            "timezone": str(tz),
                            "attendee_account_ids": [other],
                            "description": "Tuition session arranged by Rabt.",
                        },
                        "requires_approval": True,
                        "summary": f"Create 'Tuition: Awaiz & Huda' on {_fmt(start)}-{end:%H:%M} "
                        f"and invite {who}.",
                    },
                    {
                        "proposal_id": str(uuid.uuid4()),
                        "tool": "gmail.create_draft",
                        "args": {
                            "to_account_ids": [other],
                            "subject": f"Tuition on {start:%a %d %b} at {start:%H:%M}",
                            "body": f"Hi {who.split()[0]},\n\nConfirming tuition on "
                            f"{_fmt(start)}-{end:%H:%M} (Asia/Dubai). "
                            "A calendar invitation is on its way.\n\nThanks!",
                        },
                        "requires_approval": True,
                        "summary": f"Draft (not send) an email to {who} confirming tuition on {_fmt(start)}.",
                    },
                ],
            )
        if any(w in msg for w in ("free", "tuition", "lesson", "available", "slot")):
            day = (sent + timedelta(days=1)).date() if "tomorrow" in msg else sent.date()
            if "today" not in msg and "tomorrow" not in msg:
                day = (sent + timedelta(days=1)).date()
            start = datetime(day.year, day.month, day.day, 17, 0, tzinfo=tz)
            end = start + timedelta(hours=4)
            return self._resp(
                req,
                "Let me check when you are both free.",
                [
                    {
                        "proposal_id": str(uuid.uuid4()),
                        "tool": "calendar.find_mutual_slot",
                        "args": {
                            "participant_account_ids": [me, other],
                            "window_start": start.isoformat(),
                            "window_end": end.isoformat(),
                            "duration_minutes": 60,
                            "timezone": str(tz),
                            "max_slots": 5,
                        },
                        "requires_approval": False,
                        "summary": f"Find a 60-minute slot when {me} and {other} are both free on "
                        f"{start:%a %d %b}, 17:00-21:00.",
                    }
                ],
            )
        if any(w in msg for w in ("email", "mail", "inbox")):
            return self._resp(
                req,
                "Checking your recent email.",
                [
                    {
                        "proposal_id": str(uuid.uuid4()),
                        "tool": "gmail.list_recent",
                        "args": {
                            "max_results": 5,
                            "query": "tuition OR lesson" if "tuition" in msg else None,
                        },
                        "requires_approval": False,
                        "summary": "List your 5 most recent emails (sender, subject, snippet).",
                    }
                ],
            )
        return self._resp(req, "I can check availability for tuition, read recent email, or book a session.")

    def tool_result(self, req: dict) -> AgentResponse:
        pid = req["proposal_id"]
        self._pending.get(req["request_id"], set()).discard(pid)
        status, result, tool = req["status"], req.get("result") or {}, req["tool"]
        if status == "rejected":
            reply = "Okay, I did not do that."
        elif status in ("failed", "denied"):
            reply = f"That did not happen: {req['error']['message']}"
        elif tool == "calendar.find_mutual_slot":
            slots = result.get("slots", [])
            if not slots:
                reply = "You are not both free in that window. Try another day?"
            else:
                times = ", ".join(datetime.fromisoformat(s["start"]).strftime("%H:%M") for s in slots[:4])
                reply = f"You are both free at {times}. I've sent these options to the tutor to propose one."
        elif tool == "calendar.create_event":
            reply = f"Booked and verified on Google Calendar (event {result['event_id'][:10]}...)."
        elif tool == "gmail.create_draft":
            reply = "The email draft is saved in your Gmail drafts. It has not been sent."
        elif tool == "gmail.list_recent":
            msgs = result.get("messages", [])
            reply = f"You have {len(msgs)} recent emails. " + "; ".join(
                f"{m['sender'][:40]}: {m['subject'][:60]}" for m in msgs[:3]
            )
        else:
            reply = "Done."
        notes = self._notes.setdefault(req["request_id"], [])
        notes.append(reply)
        if self._pending.get(req["request_id"]):
            return self._resp(req, "")
        return self._resp(req, " ".join(self._notes.pop(req["request_id"])))
