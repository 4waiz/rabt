"""Audio decoding (any container FFmpeg understands) and encoding (WAV, OGG/Opus).

Decoding is done here with PyAV rather than faster-whisper's helper (incompatible with
PyAV 19), and enforces the duration limit while decoding, so a tiny upload cannot expand
into hours of audio.
"""

from __future__ import annotations

import io
import struct

import numpy as np

from rabt_ai.errors import RabtError
from rabt_shared import ErrorCode

STT_SAMPLE_RATE = 16_000


def decode_to_mono_16k(data: bytes, max_duration_s: float) -> tuple[np.ndarray, float]:
    import av

    max_samples = int(max_duration_s * STT_SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    total = 0
    try:
        container = av.open(io.BytesIO(data), mode="r")
    except Exception as exc:  # FFmpeg reports unknown/corrupt containers in many ways
        raise RabtError(ErrorCode.AUDIO_DECODE_ERROR, f"could not read audio: {type(exc).__name__}") from exc
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RabtError(ErrorCode.UNSUPPORTED_MEDIA_TYPE, "the upload contains no audio stream")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=STT_SAMPLE_RATE)
        frames = container.decode(stream)
        for frame in [*frames, None]:
            for out in resampler.resample(frame):
                arr = out.to_ndarray().reshape(-1)
                total += arr.size
                if total > max_samples:
                    raise RabtError(
                        ErrorCode.PAYLOAD_TOO_LARGE, f"audio is longer than the {max_duration_s:.0f} s limit"
                    )
                chunks.append(arr)
    except RabtError:
        raise
    except Exception as exc:
        raise RabtError(
            ErrorCode.AUDIO_DECODE_ERROR, f"could not decode audio: {type(exc).__name__}"
        ) from exc
    finally:
        container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32), 0.0
    audio = np.concatenate(chunks).astype(np.float32) / 32768.0
    return audio, audio.size / STT_SAMPLE_RATE


def _to_int16(samples: np.ndarray) -> np.ndarray:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")


def encode_wav(samples: np.ndarray, sample_rate: int, comment: str) -> bytes:
    """16-bit PCM mono WAV with a RIFF INFO chunk marking the audio as AI-generated."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        pad = b"\x00" if len(payload) % 2 else b""
        return tag + struct.pack("<I", len(payload)) + payload + pad

    pcm = _to_int16(samples).tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    info = b"INFO" + chunk(b"ISFT", b"Rabt\x00") + chunk(b"ICMT", comment.encode("utf-8") + b"\x00")
    body = b"WAVE" + chunk(b"fmt ", fmt) + chunk(b"LIST", info) + chunk(b"data", pcm)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def encode_ogg_opus(samples: np.ndarray, sample_rate: int, comment: str) -> bytes:
    """OGG/Opus mono (WhatsApp voice-note compatible). ``sample_rate`` must be an Opus rate."""
    import av

    if sample_rate not in (8000, 12000, 16000, 24000, 48000):
        raise ValueError(f"{sample_rate} Hz is not a valid Opus sample rate")
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="ogg") as container:
        container.metadata["comment"] = comment
        stream = container.add_stream("libopus", rate=sample_rate, layout="mono")
        stream.bit_rate = 32_000
        frame = av.AudioFrame.from_ndarray(_to_int16(samples).reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = sample_rate
        fifo = av.AudioFifo()
        fifo.write(frame)
        frame_size = stream.codec_context.frame_size or sample_rate // 50
        while fifo.samples >= frame_size:
            for packet in stream.encode(fifo.read(frame_size)):
                container.mux(packet)
        if fifo.samples:
            tail = fifo.read()
            padded = np.zeros((1, frame_size), dtype="<i2")
            padded[0, : tail.samples] = tail.to_ndarray().reshape(-1)
            last = av.AudioFrame.from_ndarray(padded, format="s16", layout="mono")
            last.sample_rate = sample_rate
            for packet in stream.encode(last):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buf.getvalue()
