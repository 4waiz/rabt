"""Persistent SQLite state: proposals, approvals, the shared lesson workflow and the audit log."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (account_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS settings (account_id TEXT, key TEXT, value TEXT, PRIMARY KEY (account_id, key));
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY, request_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
    account_id TEXT NOT NULL, tool TEXT NOT NULL, args TEXT NOT NULL, summary TEXT NOT NULL,
    requires_approval INTEGER NOT NULL, status TEXT NOT NULL, args_hash TEXT NOT NULL,
    approved_by TEXT, approved_hash TEXT, result TEXT, error TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lessons (
    lesson_id TEXT PRIMARY KEY, student_id TEXT NOT NULL, tutor_id TEXT NOT NULL, state TEXT NOT NULL,
    version INTEGER NOT NULL, options TEXT NOT NULL, slot_start TEXT, slot_end TEXT, timezone TEXT NOT NULL,
    tutor_approved_version INTEGER, student_approved_version INTEGER, last_action_by TEXT,
    event_id TEXT, event_link TEXT, note TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, account_id TEXT NOT NULL,
    action TEXT NOT NULL, detail TEXT NOT NULL
);
"""

# Lesson states (the appointment-coordination state machine).
REQUESTED = "REQUESTED"  # student asked, mutual free slots found, waiting for the tutor
PROPOSED = "PROPOSED"  # tutor proposed a slot, waiting for the student
COUNTERED = "COUNTERED"  # student asked for a different slot, waiting for the tutor
AWAITING_APPROVAL = "AWAITING_APPROVAL"  # both agreed on the same version; booking needs explicit approval
APPROVED = "APPROVED"
EXECUTED = "EXECUTED"
REJECTED = "REJECTED"
CANCELLED = "CANCELLED"
FAILED = "FAILED"
OPEN_STATES = (REQUESTED, PROPOSED, COUNTERED, AWAITING_APPROVAL, APPROVED)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._db.executescript(SCHEMA)

    def _q(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    # ------------------------------------------------------------------ conversations / settings
    def conversation_id(self, account_id: str, reset: bool = False) -> str:
        rows = self._q("SELECT conversation_id FROM conversations WHERE account_id=?", (account_id,))
        if rows and not reset:
            return rows[0]["conversation_id"]
        cid = str(uuid.uuid4())
        self._q("INSERT OR REPLACE INTO conversations VALUES (?, ?)", (account_id, cid))
        return cid

    def get_setting(self, account_id: str, key: str, default: str = "") -> str:
        rows = self._q("SELECT value FROM settings WHERE account_id=? AND key=?", (account_id, key))
        return rows[0]["value"] if rows else default

    def set_setting(self, account_id: str, key: str, value: str) -> None:
        self._q("INSERT OR REPLACE INTO settings VALUES (?, ?, ?)", (account_id, key, value))

    # ------------------------------------------------------------------ audit
    def audit(self, account_id: str, action: str, **detail: Any) -> None:
        self._q(
            "INSERT INTO audit (at, account_id, action, detail) VALUES (?, ?, ?, ?)",
            (now_iso(), account_id, action, json.dumps(detail, default=str)),
        )

    def audit_log(self, account_id: str, limit: int = 100) -> list[dict]:
        rows = self._q("SELECT * FROM audit WHERE account_id=? ORDER BY id DESC LIMIT ?", (account_id, limit))
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]

    # ------------------------------------------------------------------ proposals
    def add_proposal(self, p: dict) -> None:
        ts = now_iso()
        self._q(
            "INSERT OR IGNORE INTO proposals (proposal_id, request_id, conversation_id, account_id, tool, args,"
            " summary, requires_approval, status, args_hash, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                p["proposal_id"],
                p["request_id"],
                p["conversation_id"],
                p["account_id"],
                p["tool"],
                json.dumps(p["args"]),
                p["summary"],
                int(p["requires_approval"]),
                "pending",
                p["args_hash"],
                ts,
                ts,
            ),
        )

    def get_proposal(self, proposal_id: str) -> dict | None:
        rows = self._q("SELECT * FROM proposals WHERE proposal_id=?", (proposal_id,))
        return self._proposal(rows[0]) if rows else None

    def list_proposals(self, account_id: str, limit: int = 50) -> list[dict]:
        rows = self._q(
            "SELECT * FROM proposals WHERE account_id=? ORDER BY created_at DESC LIMIT ?", (account_id, limit)
        )
        return [self._proposal(r) for r in rows]

    def claim_proposal(self, proposal_id: str, from_status: str, to_status: str, **fields: Any) -> bool:
        """Atomic status transition: only one caller wins, so a double-click cannot execute twice."""
        sets = ", ".join(f"{k}=?" for k in fields)
        sql = f"UPDATE proposals SET status=?, updated_at=?{', ' + sets if sets else ''} WHERE proposal_id=? AND status=?"
        values = [
            json.dumps(v) if k in ("result", "error") and v is not None else v for k, v in fields.items()
        ]
        with self._lock:
            cur = self._db.execute(sql, (to_status, now_iso(), *values, proposal_id, from_status))
            return cur.rowcount == 1

    def cancel_proposals(self, ids: list[str]) -> None:
        for pid in ids:
            self.claim_proposal(pid, "pending", "cancelled")

    @staticmethod
    def _proposal(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["args"] = json.loads(d["args"])
        d["requires_approval"] = bool(d["requires_approval"])
        for k in ("result", "error"):
            d[k] = json.loads(d[k]) if d[k] else None
        return d

    # ------------------------------------------------------------------ lessons
    def open_lesson(self, student_id: str, tutor_id: str) -> dict | None:
        marks = ",".join("?" * len(OPEN_STATES))
        rows = self._q(
            f"SELECT * FROM lessons WHERE student_id=? AND tutor_id=? AND state IN ({marks})"
            " ORDER BY created_at DESC LIMIT 1",
            (student_id, tutor_id, *OPEN_STATES),
        )
        return self._lesson(rows[0]) if rows else None

    def latest_lesson(self, account_id: str) -> dict | None:
        rows = self._q(
            "SELECT * FROM lessons WHERE student_id=? OR tutor_id=? ORDER BY updated_at DESC LIMIT 1",
            (account_id, account_id),
        )
        return self._lesson(rows[0]) if rows else None

    def get_lesson(self, lesson_id: str) -> dict | None:
        rows = self._q("SELECT * FROM lessons WHERE lesson_id=?", (lesson_id,))
        return self._lesson(rows[0]) if rows else None

    def create_lesson(self, student_id: str, tutor_id: str, options: list[dict], tz: str, by: str) -> dict:
        for old in self._q(
            "SELECT lesson_id FROM lessons WHERE student_id=? AND tutor_id=? AND state IN (%s)"
            % ",".join("?" * len(OPEN_STATES)),
            (student_id, tutor_id, *OPEN_STATES),
        ):
            self.update_lesson(old["lesson_id"], state=CANCELLED, note="superseded by a new request")
        ts = now_iso()
        lid = str(uuid.uuid4())
        self._q(
            "INSERT INTO lessons (lesson_id, student_id, tutor_id, state, version, options, timezone,"
            " last_action_by, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (lid, student_id, tutor_id, REQUESTED, 1, json.dumps(options), tz, by, ts, ts),
        )
        return self.get_lesson(lid)  # type: ignore[return-value]

    def update_lesson(self, lesson_id: str, **fields: Any) -> dict:
        if "options" in fields:
            fields["options"] = json.dumps(fields["options"])
        sets = ", ".join(f"{k}=?" for k in fields)
        self._q(
            f"UPDATE lessons SET {sets}, updated_at=? WHERE lesson_id=?",
            (*fields.values(), now_iso(), lesson_id),
        )
        return self.get_lesson(lesson_id)  # type: ignore[return-value]

    @staticmethod
    def _lesson(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["options"] = json.loads(d["options"])
        return d
