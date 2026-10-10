"""Deterministic date/time handling in the user's timezone.

Small models are unreliable at calendar arithmetic, so the prompt lists concrete upcoming
dates and the engine (not the model) turns "tomorrow evening" into exact datetimes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

PARTS_OF_DAY: dict[str, tuple[time, time]] = {
    "morning": (time(8, 0), time(12, 0)),
    "afternoon": (time(12, 0), time(17, 0)),
    "evening": (time(17, 0), time(21, 0)),
    "night": (time(20, 0), time(23, 0)),
    "any": (time(8, 0), time(22, 0)),
}

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}


class DateParseError(ValueError):
    """Raised with a message that is shown to the model so it can correct itself."""


@dataclass(frozen=True)
class DateContext:
    now: datetime  # timezone-aware, expressed in the user's timezone
    tz_name: str

    @classmethod
    def create(cls, anchor: datetime, tz_name: str) -> DateContext:
        return cls(now=anchor.astimezone(ZoneInfo(tz_name)), tz_name=tz_name)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.tz_name)

    @property
    def today(self) -> date:
        return self.now.date()

    def utc_offset(self) -> str:
        offset = self.now.utcoffset() or timedelta(0)
        minutes = int(offset.total_seconds() // 60)
        sign = "+" if minutes >= 0 else "-"
        return f"UTC{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"

    def describe(self, days: int = 14) -> str:
        lines = [
            f"- Now: {self.now:%A %d %B %Y}, {fmt_clock(self.now.time())} ({self.tz_name}, {self.utc_offset()})"
        ]
        for i in range(days):
            day = self.today + timedelta(days=i)
            label = "Today" if i == 0 else "Tomorrow" if i == 1 else f"{day:%A}"
            suffix = " (next week)" if i >= 7 else ""
            lines.append(f"- {label}: {day:%A} {day.isoformat()}{suffix}")
        parts = ", ".join(
            f"{name} {lo:%H:%M}-{hi:%H:%M}" for name, (lo, hi) in PARTS_OF_DAY.items() if name != "any"
        )
        lines.append(f"- Parts of the day: {parts}.")
        lines.append(
            "- In tool calls, write dates as YYYY-MM-DD copied from the list above and times as HH:MM (24h)."
        )
        return "\n".join(lines)


def fmt_clock(t: time) -> str:
    """12-hour clock for replies, e.g. 6:00 PM."""
    hour = t.hour % 12 or 12
    return f"{hour}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"


def fmt_day(d: date) -> str:
    return f"{d:%a} {d.day} {d:%b %Y}"


def fmt_window(start: datetime, end: datetime, tz: ZoneInfo) -> str:
    s, e = start.astimezone(tz), end.astimezone(tz)
    if s.date() == e.date():
        return f"{fmt_day(s.date())}, {s:%H:%M}-{e:%H:%M}"
    return f"{fmt_day(s.date())} {s:%H:%M} - {fmt_day(e.date())} {e:%H:%M}"


def parse_day(text: str, ctx: DateContext) -> date:
    raw = text.strip().lower()
    t = re.sub(r"\s+", " ", raw.replace(",", " ")).strip()
    if not t:
        raise DateParseError("date is empty; use YYYY-MM-DD from the date list")
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", t):
        try:
            y, m, d = (int(p) for p in t.split("-"))
            return date(y, m, d)
        except ValueError as exc:
            raise DateParseError(f"{text!r} is not a real date") from exc
    if t in ("today", "tonight", "this evening", "this afternoon", "this morning"):
        return ctx.today
    if t == "tomorrow":
        return ctx.today + timedelta(days=1)
    if t in ("day after tomorrow", "the day after tomorrow"):
        return ctx.today + timedelta(days=2)
    m = re.fullmatch(r"in (\d{1,2}) days?", t)
    if m:
        return ctx.today + timedelta(days=int(m.group(1)))
    m = re.fullmatch(r"(?:(this|next|coming) )?([a-z]+)", t)
    if m and m.group(2) in WEEKDAYS:
        delta = (WEEKDAYS.index(m.group(2)) - ctx.today.weekday()) % 7
        if m.group(1) in ("next", "coming") and delta == 0:
            delta = 7
        return ctx.today + timedelta(days=delta)
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)? ([a-z]+)(?: (\d{4}))?", t) or re.fullmatch(
        r"([a-z]+) (\d{1,2})(?:st|nd|rd|th)?(?: (\d{4}))?", t
    )
    if m:
        a, b, year = m.groups()
        day_s, month_s = (a, b) if a.isdigit() else (b, a)
        month = _MONTHS.get(month_s)
        if month:
            y = int(year) if year else ctx.today.year
            try:
                result = date(y, month, int(day_s))
            except ValueError as exc:
                raise DateParseError(f"{text!r} is not a real date") from exc
            if not year and result < ctx.today:
                result = result.replace(year=y + 1)
            return result
    raise DateParseError(f"cannot understand date {text!r}; use YYYY-MM-DD from the date list")


def parse_clock(text: str) -> time:
    t = text.strip().lower().replace(".", ":").replace(" ", "")
    if t in ("noon", "midday"):
        return time(12, 0)
    if t == "midnight":
        return time(0, 0)
    m = re.fullmatch(r"(\d{1,2})(?::?(\d{2}))?(am|pm)?", t)
    if not m:
        raise DateParseError(f"cannot understand time {text!r}; use HH:MM (24h), e.g. 18:00")
    hour, minute, meridiem = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if meridiem:
        if not 1 <= hour <= 12:
            raise DateParseError(f"invalid time {text!r}")
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    if hour > 23 or minute > 59:
        raise DateParseError(f"invalid time {text!r}")
    return time(hour, minute)


def at(d: date, t: time, tz: ZoneInfo) -> datetime:
    return datetime.combine(d, t, tzinfo=tz)


def ceil_to_quarter(dt: datetime) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    extra = (-dt.minute) % 15
    return dt + timedelta(minutes=extra)
