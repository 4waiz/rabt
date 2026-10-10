"""Engine exceptions that map 1:1 onto the contract's error envelope."""

from __future__ import annotations

from typing import Any

from rabt_shared import ERROR_HTTP_STATUS, ErrorCode


class RabtError(Exception):
    """An error with a contract error code; the API layer renders it as the standard envelope."""

    def __init__(self, code: ErrorCode, message: str, *, details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    @property
    def http_status(self) -> int:
        return ERROR_HTTP_STATUS[self.code]
