# rabt-ai-engine

Rabt's local AI service. It runs a tool-calling agent on a local Ollama model, offline speech recognition (faster-whisper), local voices (Kokoro), and SQLite conversation memory. It exposes the API in [`docs/API_CONTRACT.md`](../docs/API_CONTRACT.md). It only **proposes** actions; the integration backend executes them.

## Requirements

- Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running on `127.0.0.1:11434` with the model pulled: `ollama pull qwen3.5:9b`
- Optional NVIDIA GPU. On Windows, GPU speech recognition uses the `cuda` dependency group (cuBLAS 12.8+).

## Setup

```powershell
# from the repo root; on OneDrive-synced clones keep the venv outside the repo
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.rabt\venv"
uv sync --group cuda                      # drop --group cuda on machines without an NVIDIA GPU
copy .env.example .env                    # then set RABT_INTERNAL_API_KEY (uv run rabt-ai gen-key)
uv run rabt-ai setup-models               # one-time download: Whisper large-v3-turbo + Kokoro (~2 GB)
uv run rabt-ai status                     # LLM / GPU / speech readiness
uv run rabt-ai serve                      # http://127.0.0.1:8765  (docs at /docs)
```

## Useful commands

| Command | What it does |
|---|---|
| `rabt-ai serve` | Run the API (binds `RABT_HOST`, default `127.0.0.1`) |
| `rabt-ai status` | JSON status: model, GPU residency, STT/TTS state |
| `rabt-ai say awaiz "Hello"` | Speak with a voice profile and write a WAV |
| `rabt-ai transcribe clip.ogg` | Transcribe a file |
| `rabt-ai purge --all` / `--account huda` / `--older-than-days 7` | Delete local conversation data |
| `rabt-ai export-openapi` | Regenerate `docs/openapi.json` |
| `python ai-engine/scripts/test_voices.py` | Both voices, same sentence → WAV files + timings |
| `python ai-engine/scripts/mock_integration_client.py` | End-to-end scheduling flow against a running engine, using a **mock** calendar |

## Configuration (`RABT_*` environment variables or `.env`)

| Variable | Default | Notes |
|---|---|---|
| `RABT_INTERNAL_API_KEY` | — (required) | Shared bearer token; at least 24 characters |
| `RABT_HOST` / `RABT_PORT` | `127.0.0.1` / `8765` | Never bind publicly |
| `RABT_DATA_DIR` | `%USERPROFILE%\.rabt` (Linux: `~/.local/share/rabt`) | SQLite DB, models, logs. Keep it out of synced folders. Not AppData: MSIX-packaged apps redirect AppData writes into a private sandbox. |
| `RABT_LLM_BASE_URL` | `http://127.0.0.1:11434` | Must be loopback or private; hosted APIs are refused |
| `RABT_LLM_MODEL` | `qwen3.5:9b` | Any locally pulled tool-capable model; `*-cloud` models are refused |
| `RABT_LLM_THINK` | `false` | Qwen reasoning mode: slower, sometimes more careful |
| `RABT_LLM_NUM_CTX` | `16384` | Always sent explicitly (Ollama's default on 16 GB GPUs is 4096) |
| `RABT_LLM_TEMPERATURE` / `TOP_P` / `TOP_K` / `PRESENCE_PENALTY` / `SEED` | `0.3` / `0.8` / `20` / `0` / unset | Sampling |
| `RABT_AGENT_MAX_LLM_CALLS` | `4` | Model calls per request (bounded loop) |
| `RABT_AGENT_MAX_PROPOSALS_PER_TURN` | `6` | Actions one user message can produce |
| `RABT_STT_MODEL` / `RABT_STT_DEVICE` / `RABT_STT_COMPUTE_TYPE` | `large-v3-turbo` / `auto` / `auto` | `auto` = CUDA float16, falling back to CPU int8 |
| `RABT_RETENTION_DAYS` | `30` | Conversations idle longer than this are deleted |

Identities, contacts, permissions and voice profiles live in [`rabt_ai/defaults/identities.toml`](rabt_ai/defaults/identities.toml). Override the file with `RABT_IDENTITIES_FILE`.

## Layout

```
rabt_ai/
  api.py              FastAPI app: auth, limits, error envelope, the six endpoints
  services.py         Engine container + status
  config.py           Settings + local-only endpoint guard
  identity.py         Identities, contacts, permissions, voice profiles
  storage.py          SQLite memory (account-scoped), idempotency, retention
  llm/ollama.py       Ollama client (tool calling, retries, cloud-model refusal)
  agent/
    orchestrator.py   Bounded agent loop, proposals, tool-result resumption, cancellation
    tools.py          Model-facing tools -> validated contract proposals + approval summaries
    prompts.py        System prompt, history, tool-result rendering (untrusted-data wrapping)
    dates.py          Deterministic date handling (Asia/Dubai by default)
    guards.py         Injection hygiene, completion-claim detection
  speech/
    stt.py            faster-whisper (CUDA float16 -> CPU int8 fallback)
    tts.py            Kokoro voices, AI-generated labelling
    audio_io.py       Decode any container; encode WAV / OGG-Opus
    cuda_libs.py      cuBLAS DLL loading on Windows
    models.py         One-time model downloads (verified)
```
