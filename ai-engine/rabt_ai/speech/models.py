"""One-time model downloads, run explicitly with ``rabt-ai setup-models``.

The running service never downloads anything: STT loads with ``local_files_only`` and TTS
reads files from the data directory. Kokoro files are verified against the SHA-256 digests
GitHub publishes for the release assets.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import httpx

from rabt_ai.config import Settings
from rabt_ai.speech.stt import whisper_cache_dir
from rabt_ai.speech.tts import KOKORO_MODEL_FILE, KOKORO_VOICES_FILE, kokoro_dir

KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
# Sizes and SHA-256 digests published by GitHub for the release assets (pinned, so setup needs no API call).
KOKORO_ASSETS = {
    KOKORO_MODEL_FILE: (325_505_369, "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a"),
    KOKORO_VOICES_FILE: (28_214_398, "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_whisper(settings: Settings, say: Callable[[str], None] = print) -> Path:
    os.environ.pop("HF_HUB_OFFLINE", None)
    from faster_whisper.utils import download_model

    target = whisper_cache_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    say(f"Downloading faster-whisper model {settings.stt_model!r} into {target} ...")
    path = Path(download_model(settings.stt_model, cache_dir=str(target)))
    say(f"  ok: {path}")
    return path


def download_kokoro(settings: Settings, say: Callable[[str], None] = print) -> list[Path]:
    dest = kokoro_dir(settings)
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0, connect=15.0)) as http:
        for name, (size, digest) in KOKORO_ASSETS.items():
            target = dest / name
            if target.is_file() and target.stat().st_size == size and _sha256(target) == digest:
                say(f"  {name}: already present and verified")
                out.append(target)
                continue
            url = f"{KOKORO_RELEASE}/{name}"
            say(f"Downloading {name} ({size / 1e6:.0f} MB) from {url} ...")
            partial = target.with_suffix(target.suffix + ".part")
            hasher = hashlib.sha256()
            with http.stream("GET", url) as resp, partial.open("wb") as fh:
                resp.raise_for_status()
                for block in resp.iter_bytes(1 << 20):
                    fh.write(block)
                    hasher.update(block)
            if partial.stat().st_size != size or hasher.hexdigest() != digest:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"{name}: size or SHA-256 mismatch; download rejected")
            partial.replace(target)
            say(f"  ok: {target} (SHA-256 verified)")
            out.append(target)
    return out
