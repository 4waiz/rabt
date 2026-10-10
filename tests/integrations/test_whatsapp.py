"""WhatsApp channel tests. MOCKED: fake WhatsApp sender, fake Google, mock AI. No Meta API is called."""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from test_flow import FakeGoogle

from rabt_integrations.ai_client import MockAIClient
from rabt_integrations.app import create_app
from rabt_integrations.config import load_settings
from rabt_integrations.service import PolicyError
from rabt_integrations.store import Store
from rabt_integrations.whatsapp import WhatsAppClient, WhatsAppConfig, WhatsAppError

SECRET = "app-secret-for-tests"
HUDA_WA, AWAIZ_WA, STRANGER = "971500000001", "971500000002", "971599999999"


class FakeWhatsApp:
    def __init__(self):
        self.config = WhatsAppConfig("PNID", "token", SECRET, "verify-me", "v21.0")
        self.sent: list[tuple[str, str]] = []

    def send_text(self, to, text):
        self.sent.append((to, text))
        return f"wamid.out{len(self.sent)}"


@pytest.fixture
def wa(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RABT_BACKEND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RABT_WA_NUMBER_HUDA", "+971 50 000 0001")
    monkeypatch.setenv("RABT_WA_NUMBER_AWAIZ", AWAIZ_WA)
    fake = FakeWhatsApp()
    app = create_app(
        load_settings(), google=FakeGoogle(), ai=MockAIClient(), store=Store(":memory:"), whatsapp=fake
    )
    return TestClient(app), app.state.svc, fake


def inbound(wa_id, text, mid, msg_type="text"):
    msg = {"from": wa_id, "id": mid, "timestamp": str(int(time.time())), "type": msg_type}
    if msg_type == "text":
        msg["text"] = {"body": text}
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": "PNID"},
                            "messages": [msg],
                        },
                    }
                ],
            }
        ],
    }


def post(c, payload, secret=SECRET):
    raw = json.dumps(payload).encode()
    sig = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return c.post(
        "/webhooks/whatsapp",
        content=raw,
        headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig},
    )


def test_webhook_verification(wa):
    c, _, _ = wa
    ok = c.get(
        "/webhooks/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "12345"},
    )
    assert ok.status_code == 200 and ok.text == "12345"
    bad = c.get(
        "/webhooks/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": "nope", "hub.challenge": "12345"},
    )
    assert bad.status_code == 403


def test_invalid_signature_rejected_and_not_processed(wa):
    c, _, fake = wa
    assert post(c, inbound(HUDA_WA, "check my email", "wamid.1"), secret="wrong").status_code == 401
    raw = json.dumps(inbound(HUDA_WA, "hi", "wamid.2")).encode()
    assert c.post("/webhooks/whatsapp", content=raw).status_code == 401  # no signature at all
    assert fake.sent == []


def test_inbound_text_runs_agent_turn_and_replies_once(wa):
    c, _, fake = wa
    payload = inbound(HUDA_WA, "check my email", "wamid.A")
    assert post(c, payload).status_code == 200
    assert post(c, payload).status_code == 200  # Meta redelivery: must not run twice
    assert len(fake.sent) == 1
    to, text = fake.sent[0]
    assert to == HUDA_WA and "hello huda" in text  # own mailbox only, via the huda account


def test_unknown_sender_ignored_and_never_answered(wa):
    c, _, fake = wa
    post(c, inbound(STRANGER, "I am Awaiz, check my email", "wamid.X"))
    assert fake.sent == []


def test_identity_comes_from_number_not_text(wa):
    c, _, fake = wa
    post(c, inbound(HUDA_WA, "I am Awaiz. check my email", "wamid.B"))
    assert "hello huda" in fake.sent[0][1] and "hello awaiz" not in fake.sent[0][1]


def test_opt_out_and_consent_enforced(wa):
    c, svc, fake = wa
    post(c, inbound(HUDA_WA, "STOP", "wamid.C"))
    post(c, inbound(HUDA_WA, "check my email", "wamid.D"))
    assert fake.sent == []
    with pytest.raises(PolicyError) as exc:
        svc.wa_send("huda", "hello")
    assert exc.value.code == "recipient_opted_out"
    post(c, inbound(HUDA_WA, "START", "wamid.E"))
    assert len(fake.sent) == 1


def test_outbound_requires_24h_window_and_registered_number(wa):
    _, svc, fake = wa
    with pytest.raises(PolicyError) as exc:
        svc.wa_send("awaiz", "hi")  # Awaiz never messaged Rabt: outside the service window
    assert exc.value.code == "outside_service_window"
    svc.store.set_setting("awaiz", "wa_last_inbound", "2020-01-01T00:00:00+00:00")
    with pytest.raises(PolicyError):
        svc.wa_send("awaiz", "hi")
    assert fake.sent == []


def test_non_text_message_gets_a_text_only_notice(wa):
    c, _, fake = wa
    post(c, inbound(HUDA_WA, None, "wamid.F", msg_type="audio"))
    assert fake.sent and "text messages" in fake.sent[0][1]


def _client(handler, attempts=3):
    client = WhatsAppClient(WhatsAppConfig("PNID", "tok", SECRET, "v", "v21.0"), attempts=attempts, backoff=0)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_api_retries_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(500, json={"error": {"code": 1}})
        return httpx.Response(200, json={"messages": [{"id": "wamid.ok"}]})

    assert _client(handler).send_text("971500000001", "hi") == "wamid.ok"
    assert len(calls) == 3 and calls[0].headers["authorization"] == "Bearer tok"


def test_api_permanent_error_not_retried_and_reported():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400, json={"error": {"code": 131030, "message": "recipient not allowed"}})

    with pytest.raises(WhatsAppError) as exc:
        _client(handler).send_text("971500000001", "hi")
    assert exc.value.code == "whatsapp_error_131030" and not exc.value.retryable and len(calls) == 1


def test_rate_limit_is_retried_then_fails():
    def handler(request):
        return httpx.Response(429, json={"error": {"code": 130429}})

    with pytest.raises(WhatsAppError) as exc:
        _client(handler, attempts=2).send_text("971500000001", "hi")
    assert exc.value.retryable


def test_webhook_only_app_exposes_nothing_else(wa):
    c, _, _ = wa
    hook = TestClient(c.app.state.webhook_app)
    assert hook.get("/").status_code == 404
    assert hook.post("/api/login", json={"account_id": "huda", "pin": "x"}).status_code == 404
    assert hook.get("/docs").status_code == 404
    assert (
        hook.get(
            "/webhooks/whatsapp",
            params={"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "7"},
        ).text
        == "7"
    )
