"""WhatsApp Business Platform (Cloud API) channel.

Official Cloud API only: no WhatsApp Web automation. Inbound messages arrive on a signed webhook
(X-Hub-Signature-256 = HMAC-SHA256 of the raw body with the Meta App Secret). Senders are mapped
to registered accounts by their WhatsApp number (RABT_WA_NUMBER_<ID>); unknown numbers are ignored
and never answered. Outbound text is only sent to registered, opted-in accounts inside the 24-hour
customer-service window; anything else would need an approved template, which is not used here.
Message text is never logged.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

from rabt_integrations.accounts import ACCOUNTS

log = logging.getLogger(__name__)

SERVICE_WINDOW = timedelta(hours=24)
MAX_TEXT = 4096
RETRYABLE_META_CODES = {4, 80007, 130429, 131016, 131056}  # rate limits / temporary unavailability


class WhatsAppError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


@dataclass(frozen=True)
class WhatsAppConfig:
    phone_number_id: str
    access_token: str
    app_secret: str
    verify_token: str
    graph_version: str

    @classmethod
    def from_env(cls) -> WhatsAppConfig:
        return cls(
            phone_number_id=os.environ.get("RABT_WA_PHONE_NUMBER_ID", "").strip(),
            access_token=os.environ.get("RABT_WA_ACCESS_TOKEN", "").strip(),
            app_secret=os.environ.get("RABT_WA_APP_SECRET", "").strip(),
            verify_token=os.environ.get("RABT_WA_VERIFY_TOKEN", "").strip(),
            graph_version=os.environ.get("RABT_WA_GRAPH_VERSION", "v21.0").strip(),
        )

    @property
    def enabled(self) -> bool:
        return all([self.phone_number_id, self.access_token, self.app_secret, self.verify_token])


def number_of(account_id: str) -> str | None:
    """The account's WhatsApp number in international format, digits only (RABT_WA_NUMBER_<ID>)."""
    return _digits(os.environ.get(f"RABT_WA_NUMBER_{account_id.upper()}", "")) or None


def account_for_number(wa_id: str) -> str | None:
    wa_id = _digits(wa_id)
    return next((a for a in ACCOUNTS if wa_id and number_of(a) == wa_id), None)


def verify_signature(app_secret: str, raw_body: bytes, header: str | None) -> bool:
    if not app_secret or not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


@dataclass(frozen=True)
class InboundMessage:
    message_id: str
    wa_id: str
    type: str
    text: str | None
    timestamp: datetime


@dataclass(frozen=True)
class StatusUpdate:
    message_id: str
    status: str
    error_code: str | None


def parse_webhook(payload: dict, phone_number_id: str) -> tuple[list[InboundMessage], list[StatusUpdate]]:
    """Extract messages and delivery statuses addressed to our phone number only."""
    messages: list[InboundMessage] = []
    statuses: list[StatusUpdate] = []
    if payload.get("object") != "whatsapp_business_account":
        return messages, statuses
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            if change.get("field") != "messages":
                continue
            if (value.get("metadata") or {}).get("phone_number_id") != phone_number_id:
                continue
            for m in value.get("messages") or []:
                text = (m.get("text") or {}).get("body") if m.get("type") == "text" else None
                messages.append(
                    InboundMessage(
                        message_id=str(m.get("id", "")),
                        wa_id=_digits(str(m.get("from", ""))),
                        type=str(m.get("type", "unknown")),
                        text=text[:MAX_TEXT] if text else None,
                        timestamp=datetime.fromtimestamp(int(m.get("timestamp") or time.time()), UTC),
                    )
                )
            for s in value.get("statuses") or []:
                errors = s.get("errors") or []
                statuses.append(
                    StatusUpdate(
                        message_id=str(s.get("id", "")),
                        status=str(s.get("status", "")),
                        error_code=str(errors[0].get("code")) if errors else None,
                    )
                )
    return [m for m in messages if m.message_id and m.wa_id], [s for s in statuses if s.message_id]


class WhatsAppClient:
    """Sends text messages through the Cloud API with retries on rate limits and server errors."""

    def __init__(self, config: WhatsAppConfig, attempts: int = 3, backoff: float = 1.0):
        self.config = config
        self.attempts = attempts
        self.backoff = backoff
        self._http = httpx.Client(timeout=20, trust_env=False)

    def send_text(self, to_number: str, text: str) -> str:
        if not self.config.enabled:
            raise WhatsAppError("whatsapp_not_configured", "WhatsApp is not configured on this backend.")
        url = f"https://graph.facebook.com/{self.config.graph_version}/{self.config.phone_number_id}/messages"
        body = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": _digits(to_number),
            "type": "text",
            "text": {"preview_url": False, "body": text[:MAX_TEXT]},
        }
        headers = {"Authorization": f"Bearer {self.config.access_token}"}
        last: WhatsAppError | None = None
        for attempt in range(self.attempts):
            try:
                r = self._http.post(url, json=body, headers=headers)
            except httpx.HTTPError:
                last = WhatsAppError("whatsapp_unreachable", "Could not reach the WhatsApp API.", True)
            else:
                if r.status_code < 300:
                    return r.json()["messages"][0]["id"]
                try:
                    err = r.json().get("error", {})
                except (ValueError, json.JSONDecodeError):
                    err = {}
                code = err.get("code")
                retryable = r.status_code >= 500 or r.status_code == 429 or code in RETRYABLE_META_CODES
                # Log the provider's error code only, never the message text or the token.
                log.warning("whatsapp send failed: http=%s code=%s", r.status_code, code)
                last = WhatsAppError(
                    f"whatsapp_error_{code or r.status_code}", "WhatsApp rejected the message.", retryable
                )
                if not retryable:
                    raise last
            if attempt + 1 < self.attempts:
                time.sleep(self.backoff * 2**attempt)
        assert last is not None
        raise last
