"""Local text-to-speech with Kokoro-82M (kokoro-onnx, CPU).

Voices are stock Kokoro voices chosen per identity. Every output is labelled as
AI-generated (WAV INFO / OGG comment, and the ``X-Rabt-AI-Generated`` response header).
Personal voice cloning is not implemented and must require documented consent.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rabt_ai.config import Settings
from rabt_ai.errors import RabtError
from rabt_ai.identity import IdentityRegistry, VoiceProfile
from rabt_ai.logging_setup import log_event
from rabt_ai.speech.audio_io import encode_ogg_opus, encode_wav
from rabt_ai.storage import Storage
from rabt_shared import AUDIO_MIME_TYPES, AudioFormat, ComponentState, ErrorCode, TtsStatus

log = logging.getLogger(__name__)

KOKORO_MODEL_FILE = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = "voices-v1.0.bin"
KOKORO_LABEL = "kokoro-82m-v1.0"


def kokoro_dir(settings: Settings) -> Path:
    return settings.models_dir / "kokoro"


@dataclass(frozen=True)
class Synthesis:
    audio: bytes
    mime_type: str
    sample_rate: int
    duration_s: float
    latency_ms: int
    voice: str
    model: str


_MARKDOWN = re.compile(r"[*_`#>|]+")


def speakable(text: str) -> str:
    """Strip markdown symbols that a voice would otherwise read out."""
    return re.sub(r"\s+", " ", _MARKDOWN.sub(" ", text)).strip()


class TextToSpeech:
    def __init__(self, settings: Settings, registry: IdentityRegistry, storage: Storage):
        self._settings = settings
        self._registry = registry
        self._storage = storage
        self._kokoro: Any = None
        self._error: str | None = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    def status(self) -> TtsStatus:
        if self._kokoro is not None:
            state = ComponentState.READY
        elif self._error:
            state = ComponentState.UNAVAILABLE
        else:
            state = ComponentState.NOT_LOADED
        return TtsStatus(
            engine="kokoro-onnx",
            model=KOKORO_LABEL,
            state=state,
            voice_profiles=sorted(self._registry.voice_profiles),
            detail=self._error,
        )

    def load(self) -> None:
        with self._load_lock:
            if self._kokoro is not None:
                return
            model = kokoro_dir(self._settings) / KOKORO_MODEL_FILE
            voices = kokoro_dir(self._settings) / KOKORO_VOICES_FILE
            if not model.is_file() or not voices.is_file():
                self._error = f"Kokoro model files missing in {model.parent}; run: rabt-ai setup-models"
                raise RabtError(ErrorCode.TTS_UNAVAILABLE, self._error)
            started = time.perf_counter()
            try:
                from kokoro_onnx import Kokoro

                kokoro = Kokoro(str(model), str(voices))
                available = set(kokoro.get_voices())
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {str(exc)[:300]}"
                raise RabtError(
                    ErrorCode.TTS_UNAVAILABLE, f"text-to-speech unavailable: {self._error}"
                ) from exc
            missing = sorted({p.voice for p in self._registry.voice_profiles.values()} - available)
            if missing:
                self._error = f"voice(s) not in {KOKORO_VOICES_FILE}: {missing}"
                raise RabtError(ErrorCode.TTS_UNAVAILABLE, self._error)
            self._kokoro, self._error = kokoro, None
            log_event(
                log, "tts_loaded", model=KOKORO_LABEL, load_ms=int((time.perf_counter() - started) * 1000)
            )

    async def synthesize(
        self, profile: VoiceProfile, text: str, speed: float | None, fmt: AudioFormat
    ) -> Synthesis:
        if self._kokoro is None:
            await asyncio.to_thread(self.load)
        stored = self._storage.get_voice_settings(profile.profile_id, profile.account_id) or {}
        voice = stored.get("voice") or profile.voice
        rate = speed or stored.get("speed") or profile.speed
        clean = speakable(text)
        if not clean:
            raise RabtError(ErrorCode.VALIDATION_ERROR, "nothing speakable in text")
        comment = f"AI-generated speech by Rabt ({KOKORO_LABEL}, voice {voice}, profile {profile.profile_id})"

        def run() -> tuple[bytes, int, float, float]:
            with self._run_lock:
                started = time.perf_counter()
                samples, sample_rate = self._kokoro.create(
                    clean, voice=voice, speed=float(rate), lang=profile.lang
                )
                seconds = time.perf_counter() - started
            if fmt == AudioFormat.OGG_OPUS:
                audio = encode_ogg_opus(samples, sample_rate, comment)
            else:
                audio = encode_wav(samples, sample_rate, comment)
            return audio, sample_rate, len(samples) / sample_rate, seconds

        audio, sample_rate, duration, seconds = await asyncio.to_thread(run)
        latency_ms = int(seconds * 1000)
        log_event(
            log,
            "tts_synthesized",
            profile=profile.profile_id,
            voice=voice,
            chars=len(clean),
            audio_s=round(duration, 2),
            latency_ms=latency_ms,
            rtf=round(seconds / max(duration, 1e-6), 3),
        )
        return Synthesis(
            audio=audio,
            mime_type=AUDIO_MIME_TYPES[fmt],
            sample_rate=sample_rate,
            duration_s=round(duration, 3),
            latency_ms=latency_ms,
            voice=voice,
            model=KOKORO_LABEL,
        )
