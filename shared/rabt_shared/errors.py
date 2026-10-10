"""Uniform error envelope returned by every endpoint."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from rabt_shared.common import StrictModel


class ErrorCode(StrEnum):
    UNAUTHORIZED = "unauthorized"
    UNKNOWN_ACCOUNT = "unknown_account"
    ROLE_MISMATCH = "role_mismatch"
    CONVERSATION_FORBIDDEN = "conversation_forbidden"
    VOICE_PROFILE_FORBIDDEN = "voice_profile_forbidden"
    VALIDATION_ERROR = "validation_error"
    INVALID_TOOL_RESULT = "invalid_tool_result"
    NOT_FOUND = "not_found"
    PROPOSAL_MISMATCH = "proposal_mismatch"
    PROPOSAL_NOT_PENDING = "proposal_not_pending"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    AUDIO_DECODE_ERROR = "audio_decode_error"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_TIMEOUT = "llm_timeout"
    STT_UNAVAILABLE = "stt_unavailable"
    TTS_UNAVAILABLE = "tts_unavailable"
    INTERNAL_ERROR = "internal_error"


ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.UNKNOWN_ACCOUNT: 403,
    ErrorCode.ROLE_MISMATCH: 403,
    ErrorCode.CONVERSATION_FORBIDDEN: 403,
    ErrorCode.VOICE_PROFILE_FORBIDDEN: 403,
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.INVALID_TOOL_RESULT: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.PROPOSAL_MISMATCH: 409,
    ErrorCode.PROPOSAL_NOT_PENDING: 409,
    ErrorCode.IDEMPOTENCY_CONFLICT: 409,
    ErrorCode.PAYLOAD_TOO_LARGE: 413,
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: 415,
    ErrorCode.AUDIO_DECODE_ERROR: 422,
    ErrorCode.LLM_UNAVAILABLE: 503,
    ErrorCode.LLM_TIMEOUT: 504,
    ErrorCode.STT_UNAVAILABLE: 503,
    ErrorCode.TTS_UNAVAILABLE: 503,
    ErrorCode.INTERNAL_ERROR: 500,
}


class ErrorInfo(StrictModel):
    code: ErrorCode
    message: str
    request_id: str | None = None
    details: list[dict[str, Any]] | None = None


class ErrorResponse(StrictModel):
    error: ErrorInfo
