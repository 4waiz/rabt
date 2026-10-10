"""Registered Rabt accounts. Identity comes from the dashboard login, never from message text."""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass

from rabt_shared.common import Role


@dataclass(frozen=True)
class Account:
    account_id: str
    display_name: str
    role: Role
    voice_profile: str

    @property
    def google_email(self) -> str | None:
        """The Google account this identity must authorize with (RABT_GOOGLE_EMAIL_<ID>)."""
        value = os.environ.get(f"RABT_GOOGLE_EMAIL_{self.account_id.upper()}", "").strip().lower()
        return value or None

    def check_pin(self, pin: str) -> bool:
        expected = os.environ.get(f"RABT_PIN_{self.account_id.upper()}", "")
        return bool(expected) and hmac.compare_digest(expected.encode(), pin.encode())


ACCOUNTS: dict[str, Account] = {
    "awaiz": Account("awaiz", "Awaiz Ahmed", Role.TUTOR, "awaiz-default"),
    "huda": Account("huda", "Huda Mueen", Role.STUDENT, "huda-default"),
}


def get_account(account_id: str) -> Account:
    try:
        return ACCOUNTS[account_id]
    except KeyError:
        raise KeyError(f"unknown account {account_id!r}") from None


def other_account(account_id: str) -> Account:
    return next(a for a in ACCOUNTS.values() if a.account_id != account_id)
