"""LLM-facing tools and their translation into contract proposals.

The model sees small, forgiving schemas (contact ids, YYYY-MM-DD dates, HH:MM times).
The engine converts them into strict contract arguments (timezone-aware datetimes,
account ids), validates them with the shared Pydantic models and writes a deterministic
summary for approval prompts. The model never decides ``requires_approval``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rabt_ai.agent.dates import (
    PARTS_OF_DAY,
    DateContext,
    DateParseError,
    at,
    ceil_to_quarter,
    fmt_clock,
    fmt_day,
    fmt_window,
    parse_clock,
    parse_day,
)
from rabt_ai.identity import Contact, Identity
from rabt_ai.llm import ToolCall
from rabt_shared import TOOL_SPECS, ToolName
from rabt_shared.common import StrictModel
from rabt_shared.tools import (
    CreateDraftArgs,
    CreateEventArgs,
    FindMutualSlotArgs,
    ListRecentEmailArgs,
    SendMessageArgs,
)

_EMAIL = re.compile(r"^[^@\s<>\"',;]+@[^@\s<>\"',;]+\.[^@\s<>\"',;]+$")


class ToolCallRejected(Exception):
    """A model tool call that cannot become a proposal. The message is fed back to the model."""


@dataclass(frozen=True)
class BuildContext:
    identity: Identity
    dates: DateContext
    preferences: Mapping[str, Any]

    def default_duration(self) -> int:
        value = self.preferences.get("lesson_duration_minutes", 60)
        if isinstance(value, int | float) and 15 <= value <= 480:
            return int(value)
        return 60

    def lesson_title(self) -> str:
        value = self.preferences.get("lesson_title", "Tuition")
        return str(value)[:60] if value else "Tuition"


@dataclass(frozen=True)
class BuiltProposal:
    tool: ToolName
    args: StrictModel
    summary: str
    interim: str

    @property
    def requires_approval(self) -> bool:
        return TOOL_SPECS[self.tool].requires_approval


# ----------------------------------------------------------------------------- model inputs


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _as_list(value: Any) -> Any:
    return [value] if isinstance(value, str) else value


class FindFreeTimeInput(_Input):
    with_contacts: list[str] = Field(min_length=1, max_length=5)
    date: str
    end_date: str | None = None
    part_of_day: Literal["morning", "afternoon", "evening", "night", "any"] | None = None
    earliest: str | None = None
    latest: str | None = None
    duration_minutes: int | None = Field(default=None, ge=15, le=480)

    _list = field_validator("with_contacts", mode="before")(_as_list)


class CreateEventInput(_Input):
    date: str
    start_time: str
    attendees: list[str] = Field(min_length=1, max_length=10)
    duration_minutes: int | None = Field(default=None, ge=15, le=480)
    title: str | None = Field(default=None, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    location: str | None = Field(default=None, max_length=200)

    _list = field_validator("attendees", mode="before")(_as_list)


class SendMessageInput(_Input):
    to: str
    text: str = Field(min_length=1, max_length=2000)
    channel: Literal["whatsapp", "discord"] | None = None


class ListEmailsInput(_Input):
    max_results: int | None = Field(default=None, ge=1, le=25)
    query: str | None = Field(default=None, max_length=200)
    unread_only: bool | None = None


class CreateDraftInput(_Input):
    to: list[str] = Field(min_length=1, max_length=10)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=10_000)

    _list = field_validator("to", mode="before")(_as_list)


# ----------------------------------------------------------------------------- helpers


def _known(ctx: BuildContext) -> str:
    return ", ".join(f"{c.account_id} ({c.display_name})" for c in ctx.identity.contacts) or "none"


def _resolve_contacts(refs: list[str], ctx: BuildContext) -> list[Contact]:
    me = {ctx.identity.account_id, ctx.identity.display_name.lower(), "me", "myself", "i", "user"}
    found: list[Contact] = []
    for ref in refs:
        key = str(ref).strip()
        if key.lower() in me:
            continue
        contact = ctx.identity.find_contact(key)
        if contact is None:
            raise ToolCallRejected(
                f"{key!r} is not one of the user's contacts. Known contacts: {_known(ctx)}."
            )
        if contact not in found:
            found.append(contact)
    if not found:
        raise ToolCallRejected(
            f"name at least one contact other than the user. Known contacts: {_known(ctx)}."
        )
    return found


def _names(contacts: list[Contact]) -> str:
    names = [c.display_name for c in contacts]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _with_you(contacts: list[Contact]) -> str:
    verb = "both" if len(contacts) == 1 else "all"
    return f"you and {_names(contacts)} are {verb}"


def _minutes(lo: Any, hi: Any) -> int:
    return (hi.hour * 60 + hi.minute) - (lo.hour * 60 + lo.minute)


# ----------------------------------------------------------------------------- builders


def _build_find(raw: dict[str, Any], ctx: BuildContext) -> BuiltProposal:
    a = FindFreeTimeInput.model_validate(raw)
    contacts = _resolve_contacts(a.with_contacts, ctx)
    tz = ctx.dates.tz
    first = parse_day(a.date, ctx.dates)
    last = parse_day(a.end_date, ctx.dates) if a.end_date else first
    if last < first:
        raise ToolCallRejected("end_date is before date")
    if (last - first).days > 13:
        raise ToolCallRejected("search at most 14 days at a time")
    lo, hi = PARTS_OF_DAY[a.part_of_day or "any"]
    if a.earliest:
        lo = parse_clock(a.earliest)
    if a.latest:
        hi = parse_clock(a.latest)
    if _minutes(lo, hi) <= 0:
        raise ToolCallRejected("latest must be after earliest")
    duration = a.duration_minutes or ctx.default_duration()
    if duration > _minutes(lo, hi):
        raise ToolCallRejected(f"a {duration}-minute slot does not fit between {lo:%H:%M} and {hi:%H:%M}")

    window_start, window_end = at(first, lo, tz), at(last, hi, tz)
    now = ctx.dates.now
    if window_end - timedelta(minutes=duration) < now:
        raise ToolCallRejected(
            f"that time has already passed (it is now {fmt_day(now.date())} {fmt_clock(now.time())}); "
            "choose a later date or time"
        )
    if window_start < now:
        window_start = ceil_to_quarter(now)

    multi_day = last > first
    args = FindMutualSlotArgs(
        participant_account_ids=[ctx.identity.account_id, *[c.account_id for c in contacts]],
        window_start=window_start,
        window_end=window_end,
        duration_minutes=duration,
        timezone=ctx.dates.tz_name,
        max_slots=5,
        daily_start=f"{lo:%H:%M}" if multi_day else None,
        daily_end=f"{hi:%H:%M}" if multi_day else None,
    )
    when = (
        f"{fmt_day(first)} to {fmt_day(last)}, {lo:%H:%M}-{hi:%H:%M} each day"
        if multi_day
        else fmt_window(window_start, window_end, tz)
    )
    return BuiltProposal(
        tool=ToolName.CALENDAR_FIND_MUTUAL_SLOT,
        args=args,
        summary=f"Find a {duration}-minute slot when {_with_you(contacts)} free: {when} ({ctx.dates.tz_name}).",
        interim=f"Let me check when {_with_you(contacts)} free ({when}).",
    )


def _build_event(raw: dict[str, Any], ctx: BuildContext) -> BuiltProposal:
    a = CreateEventInput.model_validate(raw)
    contacts = _resolve_contacts(a.attendees, ctx)
    tz = ctx.dates.tz
    day = parse_day(a.date, ctx.dates)
    start = at(day, parse_clock(a.start_time), tz)
    duration = a.duration_minutes or ctx.default_duration()
    end = start + timedelta(minutes=duration)
    if start <= ctx.dates.now:
        raise ToolCallRejected("that start time is in the past; choose a future time")
    if (day - ctx.dates.today).days > 365:
        raise ToolCallRejected("events can be scheduled at most one year ahead")
    title = a.title or f"{ctx.lesson_title()} with {_names(contacts)}"
    args = CreateEventArgs(
        title=title,
        start=start,
        end=end,
        timezone=ctx.dates.tz_name,
        attendee_account_ids=[c.account_id for c in contacts],
        description=a.description or None,
        location=a.location or None,
    )
    summary = (
        f'Create calendar event "{title}" on {fmt_day(day)}, {start:%H:%M}-{end:%H:%M} '
        f"({ctx.dates.tz_name}) and invite {_names(contacts)}."
    )
    return BuiltProposal(ToolName.CALENDAR_CREATE_EVENT, args, summary, f"Ready for your approval: {summary}")


def _build_message(raw: dict[str, Any], ctx: BuildContext) -> BuiltProposal:
    a = SendMessageInput.model_validate(raw)
    contact = _resolve_contacts([a.to], ctx)[0]
    args = SendMessageArgs(recipient_account_id=contact.account_id, channel=a.channel, text=a.text)
    via = f" on {a.channel.capitalize()}" if a.channel else ""
    preview = a.text if len(a.text) <= 300 else a.text[:297].rstrip() + "..."
    summary = f'Send a message to {contact.display_name}{via}: "{preview}"'
    return BuiltProposal(
        ToolName.MESSAGING_SEND_MESSAGE, args, summary, f"Ready for your approval: {summary}"
    )


def _build_list_emails(raw: dict[str, Any], ctx: BuildContext) -> BuiltProposal:
    a = ListEmailsInput.model_validate(raw)
    args = ListRecentEmailArgs(
        max_results=a.max_results or 10, query=a.query or None, unread_only=bool(a.unread_only)
    )
    unread = " unread" if args.unread_only else ""
    match = f' matching "{args.query}"' if args.query else ""
    return BuiltProposal(
        ToolName.GMAIL_LIST_RECENT,
        args,
        f"Read the {args.max_results} most recent{unread} emails in your mailbox{match}.",
        "Let me check your recent emails.",
    )


def _build_draft(raw: dict[str, Any], ctx: BuildContext) -> BuiltProposal:
    a = CreateDraftInput.model_validate(raw)
    ids: list[str] = []
    emails: list[str] = []
    labels: list[str] = []
    for ref in a.to:
        contact = ctx.identity.find_contact(ref)
        if contact is not None:
            if contact.account_id not in ids:
                ids.append(contact.account_id)
                labels.append(contact.display_name)
        elif _EMAIL.fullmatch(ref.strip()):
            if ref.strip().lower() not in (e.lower() for e in emails):
                emails.append(ref.strip())
                labels.append(ref.strip())
        else:
            raise ToolCallRejected(f"{ref!r} is neither a contact ({_known(ctx)}) nor an email address")
    args = CreateDraftArgs(to_account_ids=ids, to_emails=emails, subject=a.subject, body=a.body)
    summary = f'Create an email draft (not sent) to {", ".join(labels)} with subject "{a.subject[:120]}".'
    return BuiltProposal(ToolName.GMAIL_CREATE_DRAFT, args, summary, f"Ready for your approval: {summary}")


# ----------------------------------------------------------------------------- definitions


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    # Ollama keeps only type/description/enum/items/properties/required; limits live in descriptions.
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _contact_enum(identity: Identity) -> list[str]:
    return [c.account_id for c in identity.contacts]


def _def_find(identity: Identity) -> dict[str, Any]:
    return _fn(
        "find_mutual_free_time",
        "Look up times when the user and the given contacts are all free (calendar free/busy only). "
        "Use it before suggesting or booking a lesson or meeting time.",
        {
            "with_contacts": {
                "type": "array",
                "items": {"type": "string", "enum": _contact_enum(identity)},
                "description": "account_ids of the other people (not the user).",
            },
            "date": {"type": "string", "description": "First day to search, YYYY-MM-DD from the date list."},
            "end_date": {
                "type": "string",
                "description": "Optional last day for a multi-day search (at most 14 days), YYYY-MM-DD.",
            },
            "part_of_day": {
                "type": "string",
                "enum": ["morning", "afternoon", "evening", "night", "any"],
                "description": "Part of the day to search; 'any' if the user did not say.",
            },
            "earliest": {"type": "string", "description": "Optional earliest start time, HH:MM 24h."},
            "latest": {"type": "string", "description": "Optional latest end time, HH:MM 24h."},
            "duration_minutes": {
                "type": "integer",
                "description": "Length in minutes (15-480). Omit to use the user's usual lesson length.",
            },
        },
        ["with_contacts", "date"],
    )


def _def_event(identity: Identity) -> dict[str, Any]:
    return _fn(
        "create_calendar_event",
        "Prepare a calendar event with contacts. Nothing is created until the user approves it in the app. "
        "Only call this after the user agreed to the exact day and time.",
        {
            "date": {"type": "string", "description": "YYYY-MM-DD from the date list."},
            "start_time": {"type": "string", "description": "Start time, HH:MM 24h."},
            "attendees": {
                "type": "array",
                "items": {"type": "string", "enum": _contact_enum(identity)},
                "description": "account_ids to invite.",
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Length in minutes (15-480). Omit to use the user's usual lesson length.",
            },
            "title": {"type": "string", "description": "Short title, e.g. 'Tuition with Huda'."},
            "description": {"type": "string", "description": "Optional notes for the event."},
            "location": {"type": "string", "description": "Optional place or meeting link."},
        },
        ["date", "start_time", "attendees"],
    )


def _def_message(identity: Identity) -> dict[str, Any]:
    return _fn(
        "send_chat_message",
        "Prepare a WhatsApp/Discord message to a contact. Nothing is sent until the user approves it. "
        "Use it to ask a contact something or to share details the user agreed to.",
        {
            "to": {"type": "string", "enum": _contact_enum(identity), "description": "Recipient account_id."},
            "text": {"type": "string", "description": "The message text (max 2000 characters)."},
            "channel": {
                "type": "string",
                "enum": ["whatsapp", "discord"],
                "description": "Optional; omit to use the contact's usual channel.",
            },
        },
        ["to", "text"],
    )


def _def_list_emails(identity: Identity) -> dict[str, Any]:
    return _fn(
        "list_recent_emails",
        "Read sender, subject and a short snippet of recent emails in the user's own mailbox.",
        {
            "max_results": {"type": "integer", "description": "How many emails, 1-25 (default 10)."},
            "query": {
                "type": "string",
                "description": "Optional Gmail search, e.g. 'from:huda newer_than:7d'.",
            },
            "unread_only": {"type": "boolean", "description": "Only unread emails."},
        },
        [],
    )


def _def_draft(identity: Identity) -> dict[str, Any]:
    return _fn(
        "create_email_draft",
        "Prepare an email draft in the user's mailbox. It is not sent, and it is only saved after the user approves.",
        {
            "to": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Recipients: contact account_ids or email addresses.",
            },
            "subject": {"type": "string", "description": "Email subject."},
            "body": {"type": "string", "description": "Email body."},
        },
        ["to", "subject", "body"],
    )


@dataclass(frozen=True)
class AgentTool:
    llm_name: str
    tool: ToolName
    needs_contacts: bool
    definition: Callable[[Identity], dict[str, Any]]
    build: Callable[[dict[str, Any], BuildContext], BuiltProposal]


AGENT_TOOLS: tuple[AgentTool, ...] = (
    AgentTool("find_mutual_free_time", ToolName.CALENDAR_FIND_MUTUAL_SLOT, True, _def_find, _build_find),
    AgentTool("create_calendar_event", ToolName.CALENDAR_CREATE_EVENT, True, _def_event, _build_event),
    AgentTool("send_chat_message", ToolName.MESSAGING_SEND_MESSAGE, True, _def_message, _build_message),
    AgentTool("list_recent_emails", ToolName.GMAIL_LIST_RECENT, False, _def_list_emails, _build_list_emails),
    AgentTool("create_email_draft", ToolName.GMAIL_CREATE_DRAFT, False, _def_draft, _build_draft),
)
_BY_NAME: dict[str, AgentTool] = {t.llm_name: t for t in AGENT_TOOLS} | {t.tool.value: t for t in AGENT_TOOLS}


def available_tools(identity: Identity) -> list[AgentTool]:
    return [
        t
        for t in AGENT_TOOLS
        if t.tool in identity.allowed_tools and (identity.contacts or not t.needs_contacts)
    ]


def tool_definitions(identity: Identity) -> list[dict[str, Any]]:
    return [t.definition(identity) for t in available_tools(identity)]


def _format_validation(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:3]:
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "__root__")
        msg = str(err.get("msg", "invalid")).removeprefix("Value error, ")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)


def build_proposal(call: ToolCall, ctx: BuildContext) -> BuiltProposal:
    """Turn one model tool call into a validated proposal, or raise ToolCallRejected."""
    allowed = available_tools(ctx.identity)
    tool = _BY_NAME.get(call.name)
    if tool is None or tool not in allowed:
        names = ", ".join(t.llm_name for t in allowed) or "none"
        raise ToolCallRejected(f"unknown or unavailable tool {call.name!r}. Available tools: {names}.")
    if call.arguments_error:
        raise ToolCallRejected(f"{call.name}: {call.arguments_error}")
    args = {k: v for k, v in call.arguments.items() if v is not None and v != ""}
    try:
        return tool.build(args, ctx)
    except ValidationError as exc:
        raise ToolCallRejected(f"invalid arguments for {call.name}: {_format_validation(exc)}") from None
    except DateParseError as exc:
        raise ToolCallRejected(f"invalid arguments for {call.name}: {exc}") from None
