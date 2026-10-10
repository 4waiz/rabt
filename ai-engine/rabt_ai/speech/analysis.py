"""Small numpy-only audio measurements used to check and compare voices."""

from __future__ import annotations

import io
import wave

import numpy as np


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV bytes -> (float32 mono samples, sample rate)."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM")
        rate, channels = wav.getframerate(), wav.getnchannels()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, rate


def rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return 20 * np.log10(max(rms, 1e-9))


def median_pitch_hz(samples: np.ndarray, sample_rate: int) -> float | None:
    """Median fundamental frequency over voiced frames (autocorrelation, 70-400 Hz)."""
    frame = int(0.04 * sample_rate)
    hop = frame // 2
    lo_lag, hi_lag = int(sample_rate / 400), int(sample_rate / 70)
    if samples.size < frame * 4:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame)[::hop]
    energy = np.sqrt(np.mean(np.square(frames), axis=1))
    threshold = 0.3 * np.percentile(energy, 95)
    pitches = []
    for f in frames[energy > threshold]:
        x = f - f.mean()
        ac = np.correlate(x, x, mode="full")[frame - 1 :]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lag = int(np.argmax(ac[lo_lag:hi_lag])) + lo_lag
        if ac[lag] > 0.5:
            pitches.append(sample_rate / lag)
    return float(np.median(pitches)) if len(pitches) >= 5 else None
