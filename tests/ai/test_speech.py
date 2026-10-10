"""Speech with the real local models. Run: pytest -m speech

9: transcription works on real audio; 10: both local voices produce playable, distinct output.
Optional human recordings: put `name.wav` + `name.txt` (reference text) in tests/ai/assets/.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import TEST_KEY, FakeLlm, make_settings
from fastapi.testclient import TestClient

from rabt_ai.api import create_app
from rabt_ai.config import default_data_dir
from rabt_ai.gpu import query_gpu
from rabt_ai.services import Engine
from rabt_ai.speech.analysis import median_pitch_hz, read_wav, rms_dbfs

pytestmark = pytest.mark.speech
MODELS = default_data_dir() / "models"
ASSETS = Path(__file__).parent / "assets"
SENTENCE = "Hey Rabt, ask Huda when she's free tomorrow evening and prepare a tuition appointment."

if not (MODELS / "kokoro" / "kokoro-v1.0.onnx").is_file() or not (MODELS / "whisper").is_dir():
    pytest.skip("speech models missing; run: rabt-ai setup-models", allow_module_level=True)


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    settings = make_settings(tmp_path_factory.mktemp("speech"), model_dir=MODELS)
    engine = Engine.build(settings, llm=FakeLlm())
    with TestClient(
        create_app(engine=engine, warm_up=False), headers={"Authorization": f"Bearer {TEST_KEY}"}
    ) as c:
        yield c


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("6 p.m.", "six pm")).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = _words(reference), _words(hypothesis)
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(len(ref), 1)


def _speak(client: TestClient, account: str, fmt: str = "wav") -> tuple[bytes, dict[str, str]]:
    resp = client.post("/v1/audio/speak", json={"account_id": account, "text": SENTENCE, "format": fmt})
    assert resp.status_code == 200, resp.text
    return resp.content, dict(resp.headers)


def test_both_voices_produce_playable_and_distinct_audio(client: TestClient) -> None:
    pitches = {}
    for account in ("awaiz", "huda"):
        audio, headers = _speak(client, account)
        assert headers["content-type"] == "audio/wav"
        assert headers["x-rabt-ai-generated"] == "true"
        assert (
            audio[:4] == b"RIFF" and audio[8:12] == b"WAVE" and b"AI-generated speech by Rabt" in audio[:400]
        )
        samples, rate = read_wav(audio)
        duration = samples.size / rate
        pitches[account] = median_pitch_hz(samples, rate)
        print(
            f"\n{account}: {headers['x-rabt-tts-model']} {duration:.2f}s audio, {rms_dbfs(samples):.1f} dBFS, "
            f"median pitch {pitches[account]:.0f} Hz, synthesized in {headers['x-rabt-latency-ms']} ms"
        )
        assert rate == 24_000 and 3.0 < duration < 12.0
        assert rms_dbfs(samples) > -35, "audio is (near) silent"
    assert pitches["awaiz"] and pitches["huda"]
    assert pitches["huda"] - pitches["awaiz"] > 40, f"voices are not clearly different: {pitches}"


def test_ogg_opus_output_for_whatsapp_decodes(client: TestClient) -> None:
    import av

    audio, headers = _speak(client, "huda", fmt="ogg_opus")
    assert headers["content-type"] == "audio/ogg" and audio[:4] == b"OggS"
    with av.open(io.BytesIO(audio)) as container:
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "opus"
        samples = sum(frame.samples for frame in container.decode(stream))
    assert 3.0 < samples / 48_000 < 12.0  # Opus decodes at 48 kHz


@pytest.mark.parametrize(("speaker", "fmt"), [("awaiz", "wav"), ("huda", "ogg_opus")])
def test_transcription_of_spoken_audio(client: TestClient, speaker: str, fmt: str) -> None:
    audio, _ = _speak(client, speaker, fmt=fmt)
    mime = "audio/ogg" if fmt == "ogg_opus" else "audio/wav"
    resp = client.post(
        "/v1/audio/transcribe", files={"file": (f"clip.{fmt}", audio, mime)}, data={"account_id": "awaiz"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    wer = word_error_rate(SENTENCE, body["text"])
    print(
        f"\n[{speaker}/{fmt}] {body['text']!r} WER={wer:.2f} device={body['device']} "
        f"compute={body['compute_type']} latency={body['latency_ms']}ms audio={body['audio_duration_s']}s"
    )
    assert wer <= 0.15
    assert body["mode"] == "local" and body["language"] == "en"
    if query_gpu().available:
        assert body["device"] == "cuda", "GPU present but speech recognition fell back to CPU"


@pytest.mark.parametrize("wav", sorted(ASSETS.glob("*.wav")) or [None], ids=lambda p: p.name if p else "none")
def test_transcription_of_human_recordings(client: TestClient, wav: Path | None) -> None:
    if wav is None or not wav.with_suffix(".txt").is_file():
        pytest.skip("no human recordings with reference transcripts in tests/ai/assets")
    reference = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
    resp = client.post(
        "/v1/audio/transcribe",
        files={"file": (wav.name, wav.read_bytes(), "audio/wav")},
        data={"account_id": "awaiz"},
    )
    assert resp.status_code == 200, resp.text
    wer = word_error_rate(reference, resp.json()["text"])
    print(f"\n{wav.name}: {resp.json()['text']!r} WER={wer:.2f} latency={resp.json()['latency_ms']}ms")
    assert wer <= 0.25


def test_status_reports_loaded_speech_models(client: TestClient) -> None:
    status = client.get("/v1/agent/status").json()
    assert status["stt"]["state"] == "ready" and status["tts"]["state"] == "ready"
    assert status["tts"]["voice_profiles"] == ["awaiz-default", "huda-default"]
