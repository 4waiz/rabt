"""Local SQLite memory: conversations, messages, proposals, tool results, idempotency, preferences.

Every read and write that touches user data is scoped by ``account_id`` as well as by its
primary key, so a bug elsewhere cannot leak one account's memory into another's.
Never stored here: OAuth credentials, full mailbox dumps, raw audio.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from rabt_ai.errors import RabtError
from rabt_ai.identity import IdentityRegistry
from rabt_shared import ErrorCode

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    display_name TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(account_id),
    channel TEXT NOT NULL,
    timezone TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_conversations_account ON conversations(account_id, updated_at);
CREATE TABLE IF NOT EXISTS turns (
    request_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    account_id TEXT NOT NULL,
    anchor_time TEXT NOT NULL,
    timezone TEXT NOT NULL,
    channel TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    account_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    tool_calls TEXT,
    proposal_id TEXT,
    tool_name TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_conversation ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    account_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    args TEXT NOT NULL,
    requires_approval INTEGER NOT NULL,
    summary TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending','succeeded','failed','rejected','denied','cancelled')),
    result TEXT,
    error TEXT,
    resolution_hash TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_proposals_conversation ON proposals(conversation_id, status);
CREATE TABLE IF NOT EXISTS idempotency (
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    account_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    payload_hash TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (scope, key)
);
CREATE TABLE IF NOT EXISTS preferences (
    account_id TEXT NOT NULL REFERENCES accounts(account_id),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (account_id, key)
);
CREATE TABLE IF NOT EXISTS voice_profiles (
    profile_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(account_id),
    engine TEXT NOT NULL,
    voice TEXT NOT NULL,
    speed REAL NOT NULL,
    lang TEXT NOT NULL,
    description TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


@dataclass(frozen=True)
class ConversationRow:
    conversation_id: str
    account_id: str
    channel: str
    timezone: str


@dataclass(frozen=True)
class TurnRow:
    request_id: str
    conversation_id: str
    account_id: str
    anchor_time: datetime
    timezone: str
    channel: str


@dataclass(frozen=True)
class MessageRow:
    id: int
    request_id: str
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None
    proposal_id: str | None
    tool_name: str | None


@dataclass(frozen=True)
class ProposalRow:
    proposal_id: str
    conversation_id: str
    account_id: str
    request_id: str
    tool: str
    args: dict[str, Any]
    requires_approval: bool
    summary: str
    status: str
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    resolution_hash: str | None = None


class Storage:
    def __init__(self, path: Path | str):
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            if self._path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------------ registry seed

    def sync_registry(self, registry: IdentityRegistry) -> None:
        now = _iso(utcnow())
        with self._tx() as db:
            for ident in registry.identities.values():
                db.execute(
                    "INSERT INTO accounts(account_id, role, display_name, updated_at) VALUES (?,?,?,?) "
                    "ON CONFLICT(account_id) DO UPDATE SET role=excluded.role, "
                    "display_name=excluded.display_name, updated_at=excluded.updated_at",
                    (ident.account_id, ident.role.value, ident.display_name, now),
                )
                for key, value in ident.preferences.items():
                    db.execute(
                        "INSERT OR IGNORE INTO preferences(account_id, key, value, source, updated_at) "
                        "VALUES (?,?,?,?,?)",
                        (ident.account_id, key, json.dumps(value), "config", now),
                    )
            for vp in registry.voice_profiles.values():
                db.execute(
                    "INSERT INTO voice_profiles(profile_id, account_id, engine, voice, speed, lang, description, "
                    "updated_at) VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(profile_id) DO NOTHING",
                    (
                        vp.profile_id,
                        vp.account_id,
                        vp.engine,
                        vp.voice,
                        vp.speed,
                        vp.lang,
                        vp.description,
                        now,
                    ),
                )

    # ------------------------------------------------------------------ conversations + turns

    def get_or_create_conversation(
        self, conversation_id: str, account_id: str, channel: str, timezone: str
    ) -> ConversationRow:
        now = _iso(utcnow())
        with self._tx() as db:
            row = db.execute(
                "SELECT conversation_id, account_id, channel, timezone FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO conversations(conversation_id, account_id, channel, timezone, created_at, "
                    "updated_at) VALUES (?,?,?,?,?,?)",
                    (conversation_id, account_id, channel, timezone, now, now),
                )
                return ConversationRow(conversation_id, account_id, channel, timezone)
            if row["account_id"] != account_id:
                raise RabtError(
                    ErrorCode.CONVERSATION_FORBIDDEN,
                    "conversation_id belongs to a different account",
                )
            db.execute(
                "UPDATE conversations SET channel=?, timezone=?, updated_at=? WHERE conversation_id=?",
                (channel, timezone, now, conversation_id),
            )
            return ConversationRow(conversation_id, account_id, channel, timezone)

    def get_conversation(self, conversation_id: str, account_id: str) -> ConversationRow | None:
        rows = self._query(
            "SELECT conversation_id, account_id, channel, timezone FROM conversations "
            "WHERE conversation_id=? AND account_id=?",
            (conversation_id, account_id),
        )
        return ConversationRow(**dict(rows[0])) if rows else None

    def create_turn(self, turn: TurnRow) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT INTO turns(request_id, conversation_id, account_id, anchor_time, timezone, channel, "
                "created_at) VALUES (?,?,?,?,?,?,?)",
                (
                    turn.request_id,
                    turn.conversation_id,
                    turn.account_id,
                    _iso(turn.anchor_time),
                    turn.timezone,
                    turn.channel,
                    _iso(utcnow()),
                ),
            )

    def get_turn(self, request_id: str, account_id: str) -> TurnRow | None:
        rows = self._query(
            "SELECT request_id, conversation_id, account_id, anchor_time, timezone, channel FROM turns "
            "WHERE request_id=? AND account_id=?",
            (request_id, account_id),
        )
        if not rows:
            return None
        row = dict(rows[0])
        row["anchor_time"] = datetime.fromisoformat(row["anchor_time"])
        return TurnRow(**row)

    def request_id_in_use(self, request_id: str) -> bool:
        return bool(self._query("SELECT 1 FROM turns WHERE request_id=?", (request_id,)))

    # ------------------------------------------------------------------ messages

    def add_message(
        self,
        *,
        conversation_id: str,
        account_id: str,
        request_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        proposal_id: str | None = None,
        tool_name: str | None = None,
    ) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT INTO messages(conversation_id, account_id, request_id, role, content, tool_calls, "
                "proposal_id, tool_name, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    conversation_id,
                    account_id,
                    request_id,
                    role,
                    content,
                    json.dumps(tool_calls) if tool_calls else None,
                    proposal_id,
                    tool_name,
                    _iso(utcnow()),
                ),
            )
            db.execute(
                "UPDATE conversations SET updated_at=? WHERE conversation_id=? AND account_id=?",
                (_iso(utcnow()), conversation_id, account_id),
            )

    def load_messages(self, conversation_id: str, account_id: str) -> list[MessageRow]:
        rows = self._query(
            "SELECT id, request_id, role, content, tool_calls, proposal_id, tool_name FROM messages "
            "WHERE conversation_id=? AND account_id=? ORDER BY id",
            (conversation_id, account_id),
        )
        return [
            MessageRow(
                id=r["id"],
                request_id=r["request_id"],
                role=r["role"],
                content=r["content"],
                tool_calls=json.loads(r["tool_calls"]) if r["tool_calls"] else None,
                proposal_id=r["proposal_id"],
                tool_name=r["tool_name"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ proposals

    def create_proposal(
        self,
        *,
        proposal_id: str,
        conversation_id: str,
        account_id: str,
        request_id: str,
        tool: str,
        args: dict[str, Any],
        requires_approval: bool,
        summary: str,
    ) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT INTO proposals(proposal_id, conversation_id, account_id, request_id, tool, args, "
                "requires_approval, summary, status, created_at) VALUES (?,?,?,?,?,?,?,?,'pending',?)",
                (
                    proposal_id,
                    conversation_id,
                    account_id,
                    request_id,
                    tool,
                    json.dumps(args),
                    int(requires_approval),
                    summary,
                    _iso(utcnow()),
                ),
            )

    @staticmethod
    def _proposal(row: sqlite3.Row) -> ProposalRow:
        return ProposalRow(
            proposal_id=row["proposal_id"],
            conversation_id=row["conversation_id"],
            account_id=row["account_id"],
            request_id=row["request_id"],
            tool=row["tool"],
            args=json.loads(row["args"]),
            requires_approval=bool(row["requires_approval"]),
            summary=row["summary"],
            status=row["status"],
            result=json.loads(row["result"]) if row["result"] else None,
            error=json.loads(row["error"]) if row["error"] else None,
            resolution_hash=row["resolution_hash"],
        )

    def get_proposal(self, proposal_id: str, account_id: str) -> ProposalRow | None:
        rows = self._query(
            "SELECT * FROM proposals WHERE proposal_id=? AND account_id=?", (proposal_id, account_id)
        )
        return self._proposal(rows[0]) if rows else None

    def proposals_for_turn(self, conversation_id: str, account_id: str, request_id: str) -> list[ProposalRow]:
        rows = self._query(
            "SELECT * FROM proposals WHERE conversation_id=? AND account_id=? AND request_id=? ORDER BY created_at",
            (conversation_id, account_id, request_id),
        )
        return [self._proposal(r) for r in rows]

    def pending_proposals(self, conversation_id: str, account_id: str) -> list[ProposalRow]:
        rows = self._query(
            "SELECT * FROM proposals WHERE conversation_id=? AND account_id=? AND status='pending' "
            "ORDER BY created_at",
            (conversation_id, account_id),
        )
        return [self._proposal(r) for r in rows]

    def resolve_proposal(
        self,
        proposal_id: str,
        account_id: str,
        status: str,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        resolution_hash: str | None = None,
    ) -> bool:
        """Move a pending proposal to a final state. Returns False if it was no longer pending."""
        with self._tx() as db:
            cur = db.execute(
                "UPDATE proposals SET status=?, result=?, error=?, resolution_hash=?, resolved_at=? "
                "WHERE proposal_id=? AND account_id=? AND status='pending'",
                (
                    status,
                    json.dumps(result) if result is not None else None,
                    json.dumps(error) if error is not None else None,
                    resolution_hash,
                    _iso(utcnow()),
                    proposal_id,
                    account_id,
                ),
            )
            return cur.rowcount == 1

    def has_succeeded_action(self, conversation_id: str, account_id: str, tools: tuple[str, ...]) -> bool:
        placeholders = ",".join("?" for _ in tools)
        rows = self._query(
            f"SELECT 1 FROM proposals WHERE conversation_id=? AND account_id=? AND status='succeeded' "  # noqa: S608
            f"AND tool IN ({placeholders}) LIMIT 1",
            (conversation_id, account_id, *tools),
        )
        return bool(rows)

    def delete_turn(self, request_id: str, account_id: str) -> None:
        """Undo a turn whose processing failed so the backend can retry the same request_id."""
        with self._tx() as db:
            db.execute(
                "DELETE FROM messages WHERE request_id=? AND account_id=? AND role != 'tool'",
                (request_id, account_id),
            )
            db.execute("DELETE FROM proposals WHERE request_id=? AND account_id=?", (request_id, account_id))
            db.execute("DELETE FROM turns WHERE request_id=? AND account_id=?", (request_id, account_id))

    # ------------------------------------------------------------------ idempotency

    def get_idempotent(self, scope: str, key: str) -> tuple[str, str, dict[str, Any]] | None:
        rows = self._query(
            "SELECT account_id, payload_hash, response FROM idempotency WHERE scope=? AND key=?", (scope, key)
        )
        if not rows:
            return None
        return rows[0]["account_id"], rows[0]["payload_hash"], json.loads(rows[0]["response"])

    def put_idempotent(
        self,
        scope: str,
        key: str,
        account_id: str,
        conversation_id: str,
        payload_hash: str,
        response: dict[str, Any],
    ) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO idempotency(scope, key, account_id, conversation_id, payload_hash, response, "
                "created_at) VALUES (?,?,?,?,?,?,?)",
                (scope, key, account_id, conversation_id, payload_hash, json.dumps(response), _iso(utcnow())),
            )

    # ------------------------------------------------------------------ preferences + voices

    def get_preferences(self, account_id: str) -> dict[str, Any]:
        rows = self._query(
            "SELECT key, value FROM preferences WHERE account_id=? ORDER BY key", (account_id,)
        )
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def set_preference(self, account_id: str, key: str, value: Any, source: str = "explicit") -> None:
        with self._tx() as db:
            db.execute(
                "INSERT INTO preferences(account_id, key, value, source, updated_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(account_id, key) DO UPDATE SET value=excluded.value, source=excluded.source, "
                "updated_at=excluded.updated_at",
                (account_id, key, json.dumps(value), source, _iso(utcnow())),
            )

    def get_voice_settings(self, profile_id: str, account_id: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT engine, voice, speed, lang FROM voice_profiles WHERE profile_id=? AND account_id=?",
            (profile_id, account_id),
        )
        return dict(rows[0]) if rows else None

    def update_voice_settings(
        self, profile_id: str, account_id: str, *, voice: str | None = None, speed: float | None = None
    ) -> bool:
        with self._tx() as db:
            cur = db.execute(
                "UPDATE voice_profiles SET voice=COALESCE(?, voice), speed=COALESCE(?, speed), updated_at=? "
                "WHERE profile_id=? AND account_id=?",
                (voice, speed, _iso(utcnow()), profile_id, account_id),
            )
            return cur.rowcount == 1

    # ------------------------------------------------------------------ retention

    def purge_older_than(self, days: int) -> int:
        cutoff = _iso(utcnow() - timedelta(days=days))
        with self._tx() as db:
            return db.execute("DELETE FROM conversations WHERE updated_at < ?", (cutoff,)).rowcount

    def purge_account(self, account_id: str) -> int:
        with self._tx() as db:
            return db.execute("DELETE FROM conversations WHERE account_id=?", (account_id,)).rowcount

    def purge_conversation(self, conversation_id: str, account_id: str) -> int:
        with self._tx() as db:
            return db.execute(
                "DELETE FROM conversations WHERE conversation_id=? AND account_id=?",
                (conversation_id, account_id),
            ).rowcount

    def purge_all(self) -> int:
        with self._tx() as db:
            return db.execute("DELETE FROM conversations").rowcount

    def stats(self) -> dict[str, int]:
        out = {}
        for table in ("conversations", "messages", "proposals", "turns", "idempotency"):
            out[table] = self._query(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]  # noqa: S608 (fixed names)
        return out
