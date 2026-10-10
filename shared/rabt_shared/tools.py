"""Tool catalog: the only external actions the agent may propose.

The AI engine never executes these tools. It returns validated proposals; the
integration backend authorizes them, obtains human approval when required,
executes them and reports the outcome through ``POST /v1/agent/tool-result``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from rabt_shared.common import DEFAULT_TIMEZONE, AccountId, OpaqueId, StrictModel, Timezone


class ToolName(StrEnum):
    GMAIL_LIST_RECENT = "gmail.list_recent"
    GMAIL_CREATE_DRAFT = "gmail.create_draft"
    CALENDAR_FIND_MUTUAL_SLOT = "calendar.find_mutual_slot"
    CALENDAR_CREATE_EVENT = "calendar.create_event"
    MESSAGING_SEND_MESSAGE = "messaging.send_message"


class MessagingChannel(StrEnum):
    WHATSAPP = "whatsapp"
    DISCORD = "discord"


class UnknownToolError(ValueError):
    """Raised when a tool name is not in the allowlist."""


EmailAddress = Annotated[
    str,
    StringConstraints(
        min_length=3, max_length=254, pattern=r"^[^@\s<>\"',;]+@[^@\s<>\"',;]+\.[^@\s<>\"',;]+$"
    ),
]

LocalTime = Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
"""Wall-clock time ``HH:MM`` (24h) in the tool's ``timezone``."""

MAX_FREEBUSY_WINDOW = timedelta(days=14)
MAX_EVENT_DURATION = timedelta(hours=8)


def _require_unique(values: list[str], field: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicates")


# --------------------------------------------------------------------------- arguments


class ListRecentEmailArgs(StrictModel):
    """Read recent messages from the requester's own mailbox (metadata + short snippet only)."""

    max_results: int = Field(default=10, ge=1, le=25)
    query: str | None = Field(
        default=None,
        max_length=200,
        description="Gmail search syntax, e.g. 'from:huda newer_than:7d'.",
    )
    unread_only: bool = False


class CreateDraftArgs(StrictModel):
    """Create (not send) an email draft in the requester's own mailbox."""

    to_account_ids: list[AccountId] = Field(default_factory=list, max_length=10)
    to_emails: list[EmailAddress] = Field(default_factory=list, max_length=10)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def _check(self) -> CreateDraftArgs:
        if not self.to_account_ids and not self.to_emails:
            raise ValueError("at least one recipient (to_account_ids or to_emails) is required")
        _require_unique(self.to_account_ids, "to_account_ids")
        _require_unique([e.lower() for e in self.to_emails], "to_emails")
        return self


class FindMutualSlotArgs(StrictModel):
    """Find windows where every participant is free, using authorized free/busy data only."""

    participant_account_ids: list[AccountId] = Field(
        min_length=2, max_length=6, description="All participants, requester first."
    )
    window_start: AwareDatetime
    window_end: AwareDatetime
    duration_minutes: int = Field(ge=15, le=480)
    timezone: Timezone = DEFAULT_TIMEZONE
    max_slots: int = Field(default=5, ge=1, le=10)
    daily_start: LocalTime | None = Field(
        default=None,
        description="Optional (0.1.1): on every day of the window, only consider times from this local "
        "time (HH:MM in `timezone`). Used for searches like 'any evening this week'.",
    )
    daily_end: LocalTime | None = Field(
        default=None,
        description="Optional (0.1.1): on every day of the window, only consider times until this local time.",
    )

    @model_validator(mode="after")
    def _check(self) -> FindMutualSlotArgs:
        _require_unique(self.participant_account_ids, "participant_account_ids")
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        span = self.window_end - self.window_start
        if span > MAX_FREEBUSY_WINDOW:
            raise ValueError("search window must not exceed 14 days")
        if timedelta(minutes=self.duration_minutes) > span:
            raise ValueError("duration_minutes does not fit inside the search window")
        if self.daily_start and self.daily_end:
            start_min = int(self.daily_start[:2]) * 60 + int(self.daily_start[3:])
            end_min = int(self.daily_end[:2]) * 60 + int(self.daily_end[3:])
            if end_min <= start_min:
                raise ValueError("daily_end must be after daily_start")
            if self.duration_minutes > end_min - start_min:
                raise ValueError("duration_minutes does not fit inside the daily window")
        return self


class CreateEventArgs(StrictModel):
    """Create a calendar event organized by the requester and invite attendees."""

    title: str = Field(min_length=1, max_length=120)
    start: AwareDatetime
    end: AwareDatetime
    timezone: Timezone = DEFAULT_TIMEZONE
    attendee_account_ids: list[AccountId] = Field(default_factory=list, max_length=10)
    description: str | None = Field(default=None, max_length=2000)
    location: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _check(self) -> CreateEventArgs:
        _require_unique(self.attendee_account_ids, "attendee_account_ids")
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if self.end - self.start > MAX_EVENT_DURATION:
            raise ValueError("events longer than 8 hours are not supported")
        return self


class SendMessageArgs(StrictModel):
    """Send a chat message to a registered account."""

    recipient_account_id: AccountId
    channel: MessagingChannel | None = Field(
        default=None, description="Omit to let the backend use the recipient's preferred channel."
    )
    text: str = Field(min_length=1, max_length=2000)


# --------------------------------------------------------------------------- results


class TimeSlot(StrictModel):
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def _check(self) -> TimeSlot:
        if self.end <= self.start:
            raise ValueError("slot end must be after start")
        return self


class FindMutualSlotResult(StrictModel):
    """Free windows only. Never include event titles, attendees or other private details."""

    slots: list[TimeSlot] = Field(max_length=20)


class EmailSummary(StrictModel):
    message_id: OpaqueId
    thread_id: OpaqueId | None = None
    sender: str = Field(min_length=1, max_length=320)
    subject: str = Field(default="", max_length=500)
    snippet: str = Field(default="", max_length=500)
    received_at: AwareDatetime
    unread: bool = False


class ListRecentEmailResult(StrictModel):
    messages: list[EmailSummary] = Field(max_length=25)


class CreateDraftResult(StrictModel):
    draft_id: OpaqueId


class CreateEventResult(StrictModel):
    event_id: OpaqueId
    start: AwareDatetime
    end: AwareDatetime
    invited_account_ids: list[AccountId] = Field(default_factory=list)
    html_link: str | None = Field(default=None, max_length=2048)


class SendMessageResult(StrictModel):
    message_id: OpaqueId
    channel: MessagingChannel
    sent_at: AwareDatetime


# --------------------------------------------------------------------------- catalog


@dataclass(frozen=True)
class ToolSpec:
    name: ToolName
    description: str
    args_model: type[StrictModel]
    result_model: type[StrictModel]
    side_effects: bool

    @property
    def requires_approval(self) -> bool:
        """Side-effecting tools always need explicit human approval."""
        return self.side_effects


TOOL_SPECS: Mapping[ToolName, ToolSpec] = MappingProxyType(
    {
        ToolName.GMAIL_LIST_RECENT: ToolSpec(
            name=ToolName.GMAIL_LIST_RECENT,
            description="List recent emails in the requester's own mailbox (sender, subject, short snippet).",
            args_model=ListRecentEmailArgs,
            result_model=ListRecentEmailResult,
            side_effects=False,
        ),
        ToolName.GMAIL_CREATE_DRAFT: ToolSpec(
            name=ToolName.GMAIL_CREATE_DRAFT,
            description="Create an email draft in the requester's mailbox. Does not send it.",
            args_model=CreateDraftArgs,
            result_model=CreateDraftResult,
            side_effects=True,
        ),
        ToolName.CALENDAR_FIND_MUTUAL_SLOT: ToolSpec(
            name=ToolName.CALENDAR_FIND_MUTUAL_SLOT,
            description="Find time windows when all participants are free (free/busy only).",
            args_model=FindMutualSlotArgs,
            result_model=FindMutualSlotResult,
            side_effects=False,
        ),
        ToolName.CALENDAR_CREATE_EVENT: ToolSpec(
            name=ToolName.CALENDAR_CREATE_EVENT,
            description="Create a calendar event organized by the requester and invite attendees.",
            args_model=CreateEventArgs,
            result_model=CreateEventResult,
            side_effects=True,
        ),
        ToolName.MESSAGING_SEND_MESSAGE: ToolSpec(
            name=ToolName.MESSAGING_SEND_MESSAGE,
            description="Send a WhatsApp/Discord message to a registered account.",
            args_model=SendMessageArgs,
            result_model=SendMessageResult,
            side_effects=True,
        ),
    }
)


def get_tool_spec(name: str) -> ToolSpec:
    """Return the spec for an allowlisted tool or raise :class:`UnknownToolError`."""
    try:
        return TOOL_SPECS[ToolName(name)]
    except ValueError:
        raise UnknownToolError(f"tool {name!r} is not in the allowlist") from None
