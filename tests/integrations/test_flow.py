"""Integration-backend tests. All use the MOCK AI and a FAKE Google gateway (no real APIs)."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from rabt_integrations.ai_client import MockAIClient
from rabt_integrations.app import create_app
from rabt_integrations.config import load_settings
from rabt_integrations.slots import find_slots, free_windows
from rabt_integrations.store import Store

TZ = ZoneInfo("Asia/Dubai")


class FakeGoogle:
    def __init__(self):
        self.busy_map: dict[str, list] = {"awaiz": [], "huda": []}
        self.events: dict[tuple[str, str], dict] = {}
        self.drafts: list[tuple[str, list[str]]] = []
        self.inserts = 0

    def connected(self, a):
        return True

    def token_email(self, a):
        return f"{a}@example.com"

    def list_recent(self, a, n, q, unread):
        return [
            {
                "message_id": f"{a}-1",
                "sender": f"friend@{a}.test",
                "subject": f"hello {a}",
                "snippet": "x",
                "received_at": "2026-10-10T10:00:00+00:00",
                "unread": False,
            }
        ]

    def create_draft(self, a, to, subject, body):
        self.drafts.append((a, to))
        return f"draft-{len(self.drafts)}"

    def busy(self, a, start, end, tz):
        return [(s, e) for s, e in self.busy_map[a] if s < end and e > start]

    def create_event(self, a, event_id, body):
        self.inserts += 1
        self.events.setdefault((a, event_id), {"id": event_id, **body, "htmlLink": "https://cal/x"})
        return self.events[(a, event_id)]

    def get_event(self, a, event_id):
        return self.events[(a, event_id)]


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("RABT_PIN_AWAIZ", "1111")
    monkeypatch.setenv("RABT_PIN_HUDA", "2222")
    monkeypatch.setenv("RABT_GOOGLE_EMAIL_AWAIZ", "awaiz@example.com")
    monkeypatch.setenv("RABT_GOOGLE_EMAIL_HUDA", "huda@example.com")
    monkeypatch.setenv("RABT_DATA_DIR", str(tmp_path))
    google = FakeGoogle()
    app = create_app(load_settings(), google=google, ai=MockAIClient(), store=Store(":memory:"))
    return app, google


def client(app, who, pin):
    c = TestClient(app)
    assert c.post("/api/login", json={"account_id": who, "pin": pin}).status_code == 200
    return c


def test_slot_algorithm_respects_busy_and_boundaries():
    s = datetime(2026, 10, 11, 17, tzinfo=TZ)
    busy = [(s, s + timedelta(minutes=45)), (s + timedelta(hours=2), s + timedelta(hours=3))]
    assert free_windows(s, s + timedelta(hours=4), busy) == [
        (s + timedelta(minutes=45), s + timedelta(hours=2)),
        (s + timedelta(hours=3), s + timedelta(hours=4)),
    ]
    slots = find_slots(s, s + timedelta(hours=4), busy, timedelta(hours=1))
    assert slots[0][0] == s + timedelta(hours=1)  # 17:45 rounded up to the 18:00 grid
    assert all(not (a < e and b > st) for a, b in slots for st, e in busy)


def test_auth_required_and_wrong_pin(env):
    app, _ = env
    c = TestClient(app)
    assert c.get("/api/me").status_code == 401
    assert c.post("/api/login", json={"account_id": "huda", "pin": "1111"}).status_code == 401
    assert c.post("/api/chat", json={"message": "hi"}).status_code == 401


def test_freebusy_requires_consent(env):
    app, _ = env
    huda = client(app, "huda", "2222")
    r = huda.post("/api/chat", json={"message": "Can we do tuition tomorrow evening?"}).json()
    assert "did not happen" in r["replies"][-1] and r["lesson"] is None
    acts = [a["action"] for a in huda.get("/api/activity").json()]
    assert "tool.denied" in acts


def full_flow_until_booking(app, google):
    huda, awaiz = client(app, "huda", "2222"), client(app, "awaiz", "1111")
    awaiz.post("/api/settings/share-freebusy", json={"enabled": True})
    huda.post("/api/settings/share-freebusy", json={"enabled": True})
    r = huda.post(
        "/api/chat", json={"message": "Can we do tuition tomorrow evening? Check when we're both free."}
    )
    lesson = r.json()["lesson"]
    assert lesson["state"] == "REQUESTED" and lesson["options"]
    # tutor proposes the 2nd option, student counters with the 1st, tutor accepts
    opt = lesson["options"]
    lid = lesson["lesson_id"]
    assert (
        awaiz.post(f"/api/lesson/{lid}", json={"action": "propose", "start": opt[1]["start"]}).json()["state"]
        == "PROPOSED"
    )
    v = huda.post(f"/api/lesson/{lid}", json={"action": "counter", "start": opt[0]["start"]}).json()
    assert v["state"] == "COUNTERED" and not v["tutor_agreed"]
    v = awaiz.post(f"/api/lesson/{lid}", json={"action": "accept"}).json()
    assert v["state"] == "AWAITING_APPROVAL"
    msg = f"Yes, please book tuition with Awaiz from {opt[0]['start']} to {opt[0]['end']} and draft an email."
    pending = huda.post("/api/chat", json={"message": msg}).json()["pending"]
    return huda, awaiz, lid, {p["tool"]: p for p in pending}


def test_end_to_end_booking_with_mocks(env):
    app, google = env
    huda, awaiz, lid, props = full_flow_until_booking(app, google)
    ev = props["calendar.create_event"]
    # Awaiz cannot approve Huda's proposal (cross-account)
    assert awaiz.post(f"/api/proposals/{ev['proposal_id']}/approve").status_code == 404
    assert google.inserts == 0  # nothing created before approval
    r = huda.post(f"/api/proposals/{ev['proposal_id']}/approve")
    assert r.status_code == 200, r.text
    assert google.inserts == 1 and r.json()["replies"] == []  # draft still pending in this turn
    # duplicate approve is refused; event not created twice
    assert huda.post(f"/api/proposals/{ev['proposal_id']}/approve").status_code == 409
    assert google.inserts == 1
    d = huda.post(f"/api/proposals/{props['gmail.create_draft']['proposal_id']}/approve").json()
    assert google.drafts == [("huda", ["awaiz@example.com"])]
    assert "Booked" in d["replies"][-1] and "not been sent" in d["replies"][-1]
    assert huda.get("/api/lesson").json()["state"] == "EXECUTED"


def test_rejected_approval_creates_nothing(env):
    app, google = env
    huda, _, _, props = full_flow_until_booking(app, google)
    huda.post(f"/api/proposals/{props['calendar.create_event']['proposal_id']}/reject")
    r = huda.post(f"/api/proposals/{props['gmail.create_draft']['proposal_id']}/reject").json()
    assert google.inserts == 0 and not google.drafts and "did not" in r["replies"][-1]
    assert huda.get("/api/lesson").json()["state"] == "REJECTED"


def test_conflict_recheck_blocks_booking(env):
    app, google = env
    huda, _, _, props = full_flow_until_booking(app, google)
    ev = props["calendar.create_event"]
    s = datetime.fromisoformat(ev["args"]["start"])
    google.busy_map["awaiz"].append((s, s + timedelta(minutes=30)))  # someone booked meanwhile
    huda.post(f"/api/proposals/{ev['proposal_id']}/approve")
    r = huda.post(f"/api/proposals/{props['gmail.create_draft']['proposal_id']}/reject").json()
    assert google.inserts == 0 and "Someone became busy" in r["replies"][-1]
    assert huda.get("/api/lesson").json()["state"] == "FAILED"


def test_booking_blocked_without_tutor_agreement(env):
    app, google = env
    huda, awaiz = client(app, "huda", "2222"), client(app, "awaiz", "1111")
    for c in (huda, awaiz):
        c.post("/api/settings/share-freebusy", json={"enabled": True})
    lesson = huda.post("/api/chat", json={"message": "tuition tomorrow?"}).json()["lesson"]
    o = lesson["options"][0]
    pending = huda.post("/api/chat", json={"message": f"yes book from {o['start']} to {o['end']}"}).json()[
        "pending"
    ]
    ev = next(p for p in pending if p["tool"] == "calendar.create_event")
    r = huda.post(f"/api/proposals/{ev['proposal_id']}/approve")
    assert r.status_code == 409 and google.inserts == 0


def test_gmail_reads_only_own_mailbox(env):
    app, _ = env
    awaiz = client(app, "awaiz", "1111")
    r = awaiz.post("/api/chat", json={"message": "check my email"}).json()
    assert "hello awaiz" in r["replies"][-1] and "huda" not in r["replies"][-1]
