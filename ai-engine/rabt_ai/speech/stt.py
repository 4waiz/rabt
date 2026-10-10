"""Offline speech-to-text with faster-whisper (CTranslate2), GPU first with CPU fallback.

Notes for RTX 50-series on Windows (see docs/AI_HANDOFF.md):
* float16 on CUDA; INT8 on sm_120 is unreliable in some CTranslate2 builds.
* temperature fallback is disabled: it is linked to a known Windows crash on unload.
* the model is loaded once and kept for the life of the process.
* models are loaded with ``local_files_only`` - download them once with ``rabt-ai setup-models``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from rabt_ai.config import Settings
from rabt_ai.errors import RabtError
from rabt_ai.logging_setup import log_event
from rabt_ai.speech.audio_io import decode_to_mono_16k
from rabt_ai.speech.cuda_libs import prepare_cuda_libraries
from rabt_shared import ComponentState, ErrorCode, SttStatus

log = logging.getLogger(__name__)


def whisper_cache_dir(settings: Settings) -> Path:
    return settings.models_dir / "whisper"


class SpeechToText:
    def __init__(self, settings: Settings, vocabulary: list[str] | None = None):
        self._settings = settings
        # Names Whisper has never seen ("Rabt", contacts) are biased in via hotwords.
        self._hotwords = " ".join(dict.fromkeys(["Rabt", *(vocabulary or [])]))
        self._model: Any = None
        self._device: str | None = None
        self._compute_type: str | None = None
        self._error: str | None = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def label(self) -> str:
        return f"faster-whisper:{self._settings.stt_model}"

    def status(self) -> SttStatus:
        if self._model is not None:
            state = ComponentState.READY
        elif self._error:
            state = ComponentState.UNAVAILABLE
        else:
            state = ComponentState.NOT_LOADED
        return SttStatus(
            engine="faster-whisper",
            model=self._settings.stt_model,
            state=state,
            device=self._device,
            compute_type=self._compute_type,
            detail=self._error,
        )

    def _candidates(self) -> list[tuple[str, str]]:
        import ctranslate2

        want, compute = self._settings.stt_device, self._settings.stt_compute_type
        out: list[tuple[str, str]] = []
        if want in ("auto", "cuda") and ctranslate2.get_cuda_device_count() > 0:
            out.append(("cuda", "float16" if compute == "auto" else compute))
        if want in ("auto", "cpu"):
            out.append(("cpu", "int8" if compute == "auto" else compute))
        return out

    def load(self) -> None:
        """Blocking. Loads the model on the best working device and runs a warm-up pass."""
        with self._load_lock:
            if self._model is not None:
                return
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            prepare_cuda_libraries()
            from faster_whisper import WhisperModel

            errors: list[str] = []
            for device, compute in self._candidates():
                started = time.perf_counter()
                try:
                    model = WhisperModel(
                        self._settings.stt_model,
                        device=device,
                        compute_type=compute,
                        download_root=str(whisper_cache_dir(self._settings)),
                        local_files_only=True,
                    )
                    # Warm-up: first CUDA use JIT-compiles kernels for new GPU architectures.
                    segments, _ = model.transcribe(
                        np.zeros(16_000, dtype=np.float32), language="en", beam_size=1, temperature=0.0
                    )
                    list(segments)
                except Exception as exc:
                    errors.append(f"{device}/{compute}: {type(exc).__name__}: {str(exc)[:200]}")
                    log_event(
                        log,
                        "stt_device_failed",
                        logging.WARNING,
                        device=device,
                        compute_type=compute,
                        error=errors[-1],
                    )
                    continue
                self._model, self._device, self._compute_type, self._error = model, device, compute, None
                log_event(
                    log,
                    "stt_loaded",
                    model=self._settings.stt_model,
                    device=device,
                    compute_type=compute,
                    load_ms=int((time.perf_counter() - started) * 1000),
                )
                return
            self._error = "; ".join(errors) or "no usable device"
            if "not found" in self._error.lower() or "local_files_only" in self._error.lower():
                self._error += " (download models with: rabt-ai setup-models)"
            raise RabtError(ErrorCode.STT_UNAVAILABLE, f"speech recognition unavailable: {self._error[:400]}")

    async def transcribe(self, data: bytes, language: str | None = None) -> dict[str, Any]:
        audio, duration = await asyncio.to_thread(
            decode_to_mono_16k, data, self._settings.max_audio_duration_s
        )
        if duration < 0.1:
            raise RabtError(ErrorCode.AUDIO_DECODE_ERROR, "the audio is empty or shorter than 0.1 s")
        if self._model is None:
            await asyncio.to_thread(self.load)
        lang = language or self._settings.stt_language

        def run() -> tuple[list[Any], Any, float]:
            with self._run_lock:
                started = time.perf_counter()
                segments, info = self._model.transcribe(
                    audio,
                    language=lang,
                    beam_size=self._settings.stt_beam_size,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=duration > 30,
                    hotwords=self._hotwords,
                )
                return list(segments), info, time.perf_counter() - started

        segments, info, seconds = await asyncio.to_thread(run)
        latency_ms = int(seconds * 1000)
        log_event(
            log,
            "stt_transcribed",
            device=self._device,
            audio_s=round(duration, 2),
            latency_ms=latency_ms,
            rtf=round(seconds / duration, 3),
        )
        return {
            "text": " ".join(s.text.strip() for s in segments).strip(),
            "language": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "audio_duration_s": round(duration, 3),
            "segments": [
                {"start_s": round(float(s.start), 2), "end_s": round(float(s.end), 2), "text": s.text.strip()}
                for s in segments
            ],
            "model": self.label,
            "device": self._device,
            "compute_type": self._compute_type,
            "latency_ms": latency_ms,
        }
