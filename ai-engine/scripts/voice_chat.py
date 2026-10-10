"""Talk to Rabt: microphone -> local STT -> agent -> local TTS -> speakers.

Push-to-talk: press Enter to start speaking and Enter again to stop. Tools run on the MOCK
backend (fake calendar from mock_integration_client.py), so nothing real is sent or booked.
This is the same pipeline a future phone/telephony adapter would use.

    uv sync --group cuda --group voice        # sounddevice for the microphone
    rabt-ai serve                             # in another terminal
    python ai-engine/scripts/voice_chat.py --account awaiz
    python ai-engine/scripts/voice_chat.py --list-devices
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
import wave
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from mock_integration_client import DUBAI, MockBackend, MockCalendar, _load_env  # noqa: E402

RATE = 16_000


def record_until_enter(device: int | None) -> bytes:
    import numpy as np
    import sounddevice as sd

    chunks: list[np.ndarray] = []
    stop = threading.Event()

    def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", device=device, callback=callback):
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        print("  recording... (Enter to stop)")
        while not stop.wait(0.1):
            pass
    pcm = np.concatenate(chunks).reshape(-1) if chunks else np.zeros(0, dtype="int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def play_wav(data: bytes) -> None:
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(data, winsound.SND_MEMORY)
        return
    import numpy as np
    import sounddevice as sd

    with wave.open(io.BytesIO(data)) as wav:
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        sd.play(pcm, wav.getframerate(), blocking=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--account", default="awaiz", choices=["awaiz", "huda"])
    parser.add_argument("--device", type=int, help="input device index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--save", action="store_true", help="keep your recordings in <data_dir>/recordings")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    _load_env(Path(__file__).resolve().parents[2])
    key = os.environ.get("RABT_INTERNAL_API_KEY")
    if not key:
        print("RABT_INTERNAL_API_KEY is not set (.env)", file=sys.stderr)
        return 2
    base_url = os.environ.get("RABT_AI_BASE_URL", "http://127.0.0.1:8765")
    role = {"awaiz": "tutor", "huda": "student"}[args.account]
    save_dir = Path.home() / ".rabt" / "recordings"
    tomorrow = (datetime.now(DUBAI) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    with httpx.Client(
        base_url=base_url, headers={"Authorization": f"Bearer {key}"}, timeout=300, trust_env=False
    ) as http:
        backend = MockBackend(
            http,
            MockCalendar.demo(tomorrow),
            approve=lambda p: input(f"  Approve? {p['summary']} [y/N] ").lower().startswith("y"),
        )
        conversation = str(uuid4())
        print(f"Talking as {args.account}. MOCK calendar; nothing real is sent. Ctrl+C to quit.")
        while True:
            input("\nPress Enter and speak...")
            audio = record_until_enter(args.device)
            if args.save:
                save_dir.mkdir(parents=True, exist_ok=True)
                (save_dir / f"{datetime.now():%Y%m%d-%H%M%S}.wav").write_bytes(audio)
            t0 = time.perf_counter()
            heard = http.post(
                "/v1/audio/transcribe",
                files={"file": ("mic.wav", audio, "audio/wav")},
                data={"account_id": args.account},
            ).json()
            t1 = time.perf_counter()
            text = heard.get("text", "").strip()
            print(f"  you said: {text!r}  [STT {heard.get('latency_ms')} ms on {heard.get('device')}]")
            if not text:
                continue
            final, _ = backend.send(args.account, role, conversation, text)
            t2 = time.perf_counter()
            print(f"  rabt: {final['reply_text']}  [agent {t2 - t1:.1f}s]")
            speech = http.post(
                "/v1/audio/speak", json={"account_id": args.account, "text": final["reply_text"][:2000]}
            )
            t3 = time.perf_counter()
            print(f"  [TTS {speech.headers.get('x-rabt-latency-ms')} ms; total {t3 - t0:.1f}s; AI voice]")
            play_wav(speech.content)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
