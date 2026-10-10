"""Speak the same sentence with every Rabt voice profile, then transcribe it back.

Runs the local models directly (no server needed). Files go to <data_dir>/voice-samples/.

    python ai-engine/scripts/test_voices.py
    python ai-engine/scripts/test_voices.py --play --sentence "Your lesson is at six."
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from rabt_ai.config import Settings
from rabt_ai.services import Engine
from rabt_ai.speech.analysis import median_pitch_hz, read_wav, rms_dbfs
from rabt_shared import AudioFormat

DEFAULT_SENTENCE = (
    "Hi, this is Rabt. You and Huda are both free tomorrow at six PM. Should I book the tuition?"
)


async def run(sentence: str, play: bool) -> int:
    engine = Engine.build(Settings())
    out_dir = engine.settings.data_dir / "voice-samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sentence: {sentence!r}\n")
    print(
        f"{'profile':<16}{'voice':<12}{'audio s':>8}{'synth ms':>10}{'RTF':>7}{'pitch Hz':>10}{'dBFS':>7}  transcript"
    )
    try:
        for profile in engine.registry.voice_profiles.values():
            synth = await engine.tts.synthesize(profile, sentence, None, AudioFormat.WAV)
            path = out_dir / f"{profile.profile_id}.wav"
            path.write_bytes(synth.audio)
            samples, rate = read_wav(synth.audio)
            pitch = median_pitch_hz(samples, rate)
            started = time.perf_counter()
            heard = await engine.stt.transcribe(synth.audio)
            print(
                f"{profile.profile_id:<16}{synth.voice:<12}{synth.duration_s:>8.2f}{synth.latency_ms:>10}"
                f"{synth.latency_ms / 1000 / synth.duration_s:>7.2f}{(pitch or 0):>10.0f}{rms_dbfs(samples):>7.1f}  "
                f"{heard['text']!r} ({heard['device']}, {(time.perf_counter() - started) * 1000:.0f} ms)"
            )
            if play and sys.platform == "win32":
                import winsound

                winsound.PlaySound(str(path), winsound.SND_FILENAME)
        print(f"\nWAV files (marked AI-generated): {out_dir}")
    finally:
        await engine.aclose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE)
    parser.add_argument("--play", action="store_true", help="play each sample (Windows)")
    args = parser.parse_args()
    return asyncio.run(run(args.sentence, args.play))


if __name__ == "__main__":
    raise SystemExit(main())
