"""Primitive types and enums shared by every Rabt contract model."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints

DEFAULT_TIMEZONE = "Asia/Dubai"
MAX_MESSAGE_CHARS = 4096


class StrictModel(BaseModel):
    """Base for all contract models: unknown fields are rejected and strings are trimmed."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Role(StrEnum):
    TUTOR = "tutor"
    STUDENT = "student"


class Channel(StrEnum):
    WEB = "web"
    WHATSAPP = "whatsapp"
    DISCORD = "discord"


def _check_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except Exception as exc:  # ZoneInfoNotFoundError, ValueError, OSError for odd keys
        raise ValueError(f"unknown IANA timezone {value!r}") from exc
    return value


AccountId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$"),
]
"""Stable lowercase account identifier assigned by the integration backend, e.g. ``awaiz``."""

Timezone = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64),
    AfterValidator(_check_timezone),
]
"""IANA timezone name such as ``Asia/Dubai``."""

OpaqueId = Annotated[str, StringConstraints(min_length=1, max_length=256)]
"""Identifier issued by an external system (Gmail, Google Calendar, WhatsApp, ...)."""
