"""Interval-based free/busy maths. Availability is computed here, never guessed by the LLM."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

Interval = tuple[datetime, datetime]


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted((s, e) for s, e in intervals if e > s)
    merged: list[Interval] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def free_windows(window_start: datetime, window_end: datetime, busy: Iterable[Interval]) -> list[Interval]:
    """Gaps inside [window_start, window_end) not covered by any busy interval."""
    free: list[Interval] = []
    cursor = window_start
    for start, end in merge_intervals(busy):
        if end <= cursor or start >= window_end:
            continue
        if start > cursor:
            free.append((cursor, min(start, window_end)))
        cursor = max(cursor, end)
        if cursor >= window_end:
            break
    if cursor < window_end:
        free.append((cursor, window_end))
    return free


def _ceil_to(dt: datetime, step: timedelta) -> datetime:
    floor = dt.replace(minute=0, second=0, microsecond=0)
    while floor < dt:
        floor += step
    return floor


def find_slots(
    window_start: datetime,
    window_end: datetime,
    busy: Iterable[Interval],
    duration: timedelta,
    max_slots: int = 5,
    step: timedelta = timedelta(minutes=30),
) -> list[Interval]:
    """Candidate start times on a `step` grid where `duration` fits entirely in free time."""
    slots: list[Interval] = []
    for free_start, free_end in free_windows(window_start, window_end, busy):
        start = _ceil_to(free_start, step)
        while start + duration <= free_end and len(slots) < max_slots:
            slots.append((start, start + duration))
            start += duration if duration > step else step
        if len(slots) >= max_slots:
            break
    return slots


def overlaps(slot: Interval, busy: Iterable[Interval]) -> bool:
    return any(s < slot[1] and e > slot[0] for s, e in busy)
