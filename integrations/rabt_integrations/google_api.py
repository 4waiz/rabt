"""Real Google Gmail + Calendar access, one OAuth token per Rabt account.

Each person authorizes their own Google account (`python -m rabt_integrations.google_auth <account_id>`).
Tokens are stored per account under RABT_DATA_DIR/secrets, outside the repository. A token is only
used for the account it was issued to: the backend never reads one person's mailbox or calendar
with another person's credentials.
"""

from __future__ import annotations

import base64
import html
import json
import logging
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from rabt_integrations.config import Settings

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",  # list recent mail (metadata + snippet)
    "https://www.googleapis.com/auth/gmail.compose",  # create drafts; nothing is sent automatically
    "https://www.googleapis.com/auth/calendar.freebusy",  # busy intervals only, no titles
    "https://www.googleapis.com/auth/calendar.events",  # create the approved, invited event
]


class GoogleNotConnected(RuntimeError):
    """The account has not completed Google OAuth yet."""


class GoogleApiError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def token_path(settings: Settings, account_id: str) -> Path:
    return settings.secrets_dir / f"google_token_{account_id}.json"


class GoogleGateway:
    """Thin wrapper over the Google APIs. Tests substitute a fake with the same methods."""

    def __init__(self, settings: Settings):
        self.settings = settings

    # ------------------------------------------------------------------ credentials
    def _credentials(self, account_id: str):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        path = token_path(self.settings, account_id)
        if not path.is_file():
            raise GoogleNotConnected(f"{account_id} has not connected Google yet")
        creds = Credentials.from_authorized_user_file(str(path), SCOPES)
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                path.write_text(creds.to_json(), encoding="utf-8")
            else:
                raise GoogleNotConnected(f"{account_id}'s Google token is invalid; re-run google_auth")
        return creds

    def _service(self, account_id: str, api: str, version: str):
        from googleapiclient.discovery import build

        return build(api, version, credentials=self._credentials(account_id), cache_discovery=False)

    def _call(self, request):
        from googleapiclient.errors import HttpError

        try:
            return request.execute(num_retries=2)
        except HttpError as exc:
            status = getattr(exc.resp, "status", 0)
            raise GoogleApiError(
                f"google_http_{status}",
                f"Google API error {status}",
                retryable=status >= 500 or status == 429,
            ) from exc

    def connected(self, account_id: str) -> bool:
        return token_path(self.settings, account_id).is_file()

    def token_email(self, account_id: str) -> str | None:
        path = token_path(self.settings, account_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("rabt_email")

    def profile_email(self, account_id: str) -> str:
        svc = self._service(account_id, "gmail", "v1")
        return self._call(svc.users().getProfile(userId="me"))["emailAddress"].lower()

    # ------------------------------------------------------------------ gmail
    def list_recent(
        self, account_id: str, max_results: int, query: str | None, unread_only: bool
    ) -> list[dict]:
        svc = self._service(account_id, "gmail", "v1")
        q = " ".join(part for part in [query or "", "is:unread" if unread_only else ""] if part) or None
        listing = self._call(svc.users().messages().list(userId="me", maxResults=max_results, q=q))
        out = []
        for ref in listing.get("messages", [])[:max_results]:
            msg = self._call(
                svc.users()
                .messages()
                .get(userId="me", id=ref["id"], format="metadata", metadataHeaders=["From", "Subject"])
            )
            headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
            out.append(
                {
                    "message_id": msg["id"],
                    "thread_id": msg.get("threadId"),
                    "sender": (headers.get("from") or "(unknown)")[:320],
                    "subject": (headers.get("subject") or "")[:500],
                    "snippet": html.unescape(msg.get("snippet", ""))[:500],
                    "received_at": datetime.fromtimestamp(int(msg["internalDate"]) / 1000, UTC).isoformat(),
                    "unread": "UNREAD" in msg.get("labelIds", []),
                }
            )
        return out

    def create_draft(self, account_id: str, to: list[str], subject: str, body: str) -> str:
        svc = self._service(account_id, "gmail", "v1")
        mime = EmailMessage()
        mime["To"] = ", ".join(to)
        mime["Subject"] = subject
        mime.set_content(body)
        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
        draft = self._call(svc.users().drafts().create(userId="me", body={"message": {"raw": raw}}))
        return draft["id"]

    # ------------------------------------------------------------------ calendar
    def busy(
        self, account_id: str, start: datetime, end: datetime, tz: str
    ) -> list[tuple[datetime, datetime]]:
        """The account's own busy intervals. No event titles are requested or returned."""
        svc = self._service(account_id, "calendar", "v3")
        body = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "timeZone": tz,
            "items": [{"id": "primary"}],
        }
        resp = self._call(svc.freebusy().query(body=body))
        cal = resp.get("calendars", {}).get("primary", {})
        if cal.get("errors"):
            raise GoogleApiError("freebusy_error", f"free/busy unavailable for {account_id}")
        return [
            (
                datetime.fromisoformat(b["start"].replace("Z", "+00:00")),
                datetime.fromisoformat(b["end"].replace("Z", "+00:00")),
            )
            for b in cal.get("busy", [])
        ]

    def create_event(self, account_id: str, event_id: str, body: dict[str, Any]) -> dict:
        """Insert with a deterministic id so a retried approval can never create a duplicate."""
        from googleapiclient.errors import HttpError

        svc = self._service(account_id, "calendar", "v3")
        try:
            return (
                svc.events()
                .insert(calendarId="primary", body={**body, "id": event_id}, sendUpdates="all")
                .execute()
            )
        except HttpError as exc:
            if getattr(exc.resp, "status", 0) == 409:  # already created by an earlier attempt
                log.info("event %s already exists; returning the existing event", event_id)
                return self.get_event(account_id, event_id)
            raise GoogleApiError(f"google_http_{exc.resp.status}", "calendar insert failed") from exc

    def get_event(self, account_id: str, event_id: str) -> dict:
        svc = self._service(account_id, "calendar", "v3")
        return self._call(svc.events().get(calendarId="primary", eventId=event_id))
