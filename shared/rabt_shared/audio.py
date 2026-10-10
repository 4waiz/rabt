"""Audio endpoints: ``POST /v1/audio/transcribe`` and ``POST /v1/audio/speak``."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, StringConstraints

from rabt_shared.common import AccountId, StrictModel

MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_AUDIO_DURATION_S = 300
MAX_SPEAK_CHARS = 2000


class AudioFormat(StrEnum):
    WAV = "wav"  # 16-bit PCM mono
    OGG_OPUS = "ogg_opus"  # WhatsApp voice-note compatible


AUDIO_MIME_TYPES: dict[AudioFormat, str] = {
    AudioFormat.WAV: "audio/wav",
    AudioFormat.OGG_OPUS: "audio/ogg",
}

VoiceProfileId = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
]


class SpeakRequest(StrictModel):
    """Text-to-speech. The response body is audio bytes, not JSON."""

    request_id: UUID | None = None
    account_id: AccountId
    text: str = Field(min_length=1, max_length=MAX_SPEAK_CHARS)
    voice_profile: VoiceProfileId | None = Field(
        default=None,
        description="Defaults to the account's own profile. Must be authorized for account_id.",
    )
    format: AudioFormat = AudioFormat.WAV
    speed: float | None = Field(default=None, ge=0.5, le=2.0, description="Overrides the profile speed.")


class TranscriptSegment(StrictModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    text: str


class TranscriptionResponse(StrictModel):
    request_id: UUID | None = None
    text: str
    language: str
    language_probability: float | None = Field(default=None, ge=0, le=1)
    audio_duration_s: float = Field(ge=0)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    model: str
    device: Literal["cuda", "cpu"]
    compute_type: str
    latency_ms: int = Field(ge=0)
    mode: Literal["local"] = "local"


SPEAK_RESPONSE_HEADERS = (
    "X-Rabt-AI-Generated",  # always "true": the audio is synthetic
    "X-Rabt-Voice-Profile",
    "X-Rabt-TTS-Model",
    "X-Rabt-Latency-Ms",
    "X-Rabt-Audio-Duration-S",
)
