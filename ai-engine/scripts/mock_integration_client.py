"""MOCK integration backend for exercising the Rabt AI engine end to end.

Everything here is fake: an in-memory calendar with made-up busy times, fake event ids and
fake message delivery. Nothing is sent anywhere. It shows how the real integration backend
should drive the API (docs/API_CONTRACT.md) and is used by the live tests.

    python ai-engine/scripts/mock_integration_client.py                 # scripted demo
    python ai-engine/scripts/mock_integration_client.py --interactive   # chat as awaiz
    python ai-engine/scripts/mock_integration_client.py --reject        # decline approvals

Needs a running engine (`rabt-ai serve`) and RABT_INTERNAL_API_KEY in the environment or .env.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx

DUBAI = ZoneInfo("Asia/Dubai")
Interval = tuple[datetime, datetime]


def _dt(text: str) -> datetime:
    return datetime.fromisoformat(text)


class MockCalendar:
    """Fake calendars: busy blocks per account. Only free/busy ever leaves this class."""

    def __init__(self, busy: dict[str, list[Interval]] | None = None):
        self.busy: dict[str, list[Interval]] = {k: list(v) for k, v in (busy or {}).items()}
        self.events: list[dict[str, Any]] = []

    @classmethod
    def demo(cls, day: datetime) -> MockCalendar:
        """Awaiz busy 17:00-18:00, Huda busy 19:00-19:30 on ``day`` (Dubai time)."""
        at = lambda h, m=0: day.replace(hour=h, minute=m, second=0, microsecond=0)  # noqa: E731
        return cls({"awaiz": [(at(17), at(18))], "huda": [(at(19), at(19, 30))]})

    def find_mutual_slot(self, args: dict[str, Any]) -> dict[str, Any]:
        start, end = _dt(args["window_start"]), _dt(args["window_end"])
        need = timedelta(minutes=int(args["duration_minutes"]))
        tz = ZoneInfo(args.get("timezone") or "Asia/Dubai")
        busy = sorted(b for p in args["participant_account_ids"] for b in self.busy.get(p, []))
        windows: list[Interval] = []
        day = start.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
        while day < end:
            lo, hi = day, day + timedelta(days=1)
            if args.get("daily_start"):
                h, m = map(int, args["daily_start"].split(":"))
                lo = day.replace(hour=h, minute=m)
            if args.get("daily_end"):
                h, m = map(int, args["daily_end"].split(":"))
                hi = day.replace(hour=h, minute=m)
            lo, hi = max(lo, start), min(hi, end)
            if hi > lo:
                windows.append((lo, hi))
            day += timedelta(days=1)
        slots: list[dict[str, str]] = []
        for lo, hi in windows:
            cursor = lo
            for b_start, b_end in busy:
                if b_end <= cursor or b_start >= hi:
                    continue
                while cursor + need <= min(b_start, hi):
                    slots.append({"start": cursor.isoformat(), "end": (cursor + need).isoformat()})
                    cursor += need
                cursor = max(cursor, b_end)
            while cursor + need <= hi:
                slots.append({"start": cursor.isoformat(), "end": (cursor + need).isoformat()})
                cursor += need
        return {"slots": slots[: int(args.get("max_slots", 5))]}

    def create_event(self, organizer: str, args: dict[str, Any]) -> dict[str, Any]:
        start, end = _dt(args["start"]), _dt(args["end"])
        event_id = f"mock-evt-{len(self.events) + 1}"
        self.events.append({"event_id": event_id, "organizer": organizer, **args})
        for account in [organizer, *args.get("attendee_account_ids", [])]:
            self.busy.setdefault(account, []).append((start, end))
        return {
            "event_id": event_id,
            "start": args["start"],
            "end": args["end"],
            "invited_account_ids": args.get("attendee_account_ids", []),
            "html_link": None,
        }


class MockBackend:
    """Plays the integration backend: executes proposals against fake services."""

    def __init__(
        self,
        http: httpx.Client,
        calendar: MockCalendar,
        approve: Callable[[dict[str, Any]], bool] = lambda proposal: True,
        inbox: dict[str, list[dict[str, Any]]] | None = None,
        say: Callable[[str], None] = lambda line: None,
    ):
        self.http = http
        self.calendar = calendar
        self.approve = approve
        self.inbox = inbox or {}
        self.say = say
        self.sent_messages: list[dict[str, Any]] = []
        self.executed: list[dict[str, Any]] = []

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self.http.post(path, json=body)
        if resp.status_code != 200:
            raise RuntimeError(f"{path} -> HTTP {resp.status_code}: {resp.text}")
        return resp.json()

    def _execute(self, account_id: str, proposal: dict[str, Any]) -> dict[str, Any]:
        tool, args = proposal["tool"], proposal["args"]
        if proposal["requires_approval"]:
            self.say(f"  [approval prompt] {proposal['summary']}")
            if not self.approve(proposal):
                self.say("  [approval] DECLINED by user")
                return {"status": "rejected"}
            self.say("  [approval] APPROVED by user")
        self.executed.append(proposal)
        now = datetime.now(DUBAI).isoformat()
        if tool == "calendar.find_mutual_slot":
            return {"status": "succeeded", "result": self.calendar.find_mutual_slot(args)}
        if tool == "calendar.create_event":
            return {"status": "succeeded", "result": self.calendar.create_event(account_id, args)}
        if tool == "messaging.send_message":
            self.sent_messages.append({"from": account_id, **args})
            return {
                "status": "succeeded",
                "result": {
                    "message_id": f"mock-msg-{len(self.sent_messages)}",
                    "channel": args.get("channel") or "whatsapp",
                    "sent_at": now,
                },
            }
        if tool == "gmail.list_recent":
            return {
                "status": "succeeded",
                "result": {"messages": self.inbox.get(account_id, [])[: args["max_results"]]},
            }
        if tool == "gmail.create_draft":
            return {"status": "succeeded", "result": {"draft_id": f"mock-draft-{uuid4().hex[:8]}"}}
        return {
            "status": "denied",
            "error": {"code": "unsupported_tool", "message": f"mock backend cannot run {tool}"},
        }

    def send(
        self,
        account_id: str,
        role: str,
        conversation_id: str,
        message: str,
        *,
        sent_at: str | None = None,
        channel: str = "web",
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Send one user message and run every resulting proposal. Returns (final response, all responses)."""
        body = {
            "request_id": str(uuid4()),
            "conversation_id": conversation_id,
            "account_id": account_id,
            "role": role,
            "channel": channel,
            "message": message,
            "timezone": "Asia/Dubai",
        }
        if sent_at:
            body["sent_at"] = sent_at
        response = self._post("/v1/agent/respond", body)
        history = [response]
        while response["status"] == "action_required":
            self.say(f"  rabt (interim): {response['reply_text']}")
            for proposal in response["action_proposals"]:
                self.say(f"  -> proposal {proposal['tool']} (approval: {proposal['requires_approval']})")
                outcome = self._execute(account_id, proposal)
                result_body = {
                    "request_id": response["request_id"],
                    "conversation_id": conversation_id,
                    "account_id": account_id,
                    "proposal_id": proposal["proposal_id"],
                    "tool": proposal["tool"],
                    **outcome,
                }
                next_response = self._post("/v1/agent/tool-result", result_body)
                history.append(next_response)
            response = next_response
        return response, history


def _load_env(repo_root: Path) -> None:
    env = repo_root / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--reject", action="store_true", help="decline every approval prompt")
    parser.add_argument("--account", default="awaiz")
    args = parser.parse_args()

    _load_env(Path(__file__).resolve().parents[2])
    base_url = os.environ.get("RABT_AI_BASE_URL", "http://127.0.0.1:8765")
    key = os.environ.get("RABT_INTERNAL_API_KEY")
    if not key:
        print("RABT_INTERNAL_API_KEY is not set (.env)", file=sys.stderr)
        return 2
    role = {"awaiz": "tutor", "huda": "student"}.get(args.account, "tutor")
    tomorrow = (datetime.now(DUBAI) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    print(
        "MOCK backend: fake calendar (awaiz busy 17:00-18:00, huda busy 19:00-19:30 tomorrow); nothing is really sent.\n"
    )

    def approve(proposal: dict[str, Any]) -> bool:
        if args.reject:
            return False
        if args.interactive:
            return input(f"Approve? {proposal['summary']} [y/N] ").strip().lower().startswith("y")
        return True

    with httpx.Client(
        base_url=base_url, headers={"Authorization": f"Bearer {key}"}, timeout=300, trust_env=False
    ) as http:
        backend = MockBackend(http, MockCalendar.demo(tomorrow), approve=approve, say=print)
        conversation = str(uuid4())
        script = [
            "Hey Rabt, ask Huda when she's free tomorrow evening, find a suitable time for tuition, and prepare the appointment.",
            "Yes, go with that.",
        ]
        turns = iter(script) if not args.interactive else iter(lambda: input(f"\n{args.account}> "), "")
        for text in turns:
            if not args.interactive:
                print(f"\n{args.account}> {text}")
            started = time.perf_counter()
            final, _ = backend.send(args.account, role, conversation, text)
            print(f"rabt> {final['reply_text']}   [{final['model']}, {time.perf_counter() - started:.1f}s]")
        print(f"\nMock calendar events created: {backend.calendar.events or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
