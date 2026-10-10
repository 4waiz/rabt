"""``rabt-ai`` command line: run the service and manage local models and data."""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import sys
from pathlib import Path

from rabt_ai.config import REPO_ROOT


def _settings():  # imported lazily so `gen-key` / `export-openapi` work without a .env
    from rabt_ai.config import Settings

    return Settings()


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from rabt_ai.api import create_app

    settings = _settings()
    host = args.host or settings.host
    if host in ("0.0.0.0", "::"):
        print(
            "WARNING: listening on all interfaces. Never expose the AI engine to the internet.",
            file=sys.stderr,
        )
    uvicorn.run(
        create_app(settings), host=host, port=args.port or settings.port, log_config=None, access_log=False
    )
    return 0


def cmd_setup_models(args: argparse.Namespace) -> int:
    from rabt_ai.speech.models import download_kokoro, download_whisper

    settings = _settings()
    if not args.skip_whisper:
        download_whisper(settings)
    if not args.skip_kokoro:
        download_kokoro(settings)
    print("Models ready. The service runs fully offline from here.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from rabt_ai.services import Engine

    async def run() -> dict:
        engine = Engine.build(_settings())
        try:
            return (await engine.status()).model_dump(mode="json")
        finally:
            await engine.aclose()

    print(json.dumps(asyncio.run(run()), indent=2))
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    from rabt_ai.storage import Storage

    settings = _settings()
    storage = Storage(settings.db_path)
    if args.all:
        removed = storage.purge_all()
    elif args.conversation:
        if not args.account:
            print("--conversation requires --account", file=sys.stderr)
            return 2
        removed = storage.purge_conversation(args.conversation, args.account)
    elif args.account:
        removed = storage.purge_account(args.account)
    else:
        removed = storage.purge_older_than(args.older_than_days or settings.retention_days)
    storage.close()
    print(f"Deleted {removed} conversation(s) and their messages, proposals and tool results.")
    return 0


def cmd_export_openapi(args: argparse.Namespace) -> int:
    from rabt_ai.api import create_app

    spec = create_app(warm_up=False).openapi()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {out}")
    return 0


def cmd_gen_key(args: argparse.Namespace) -> int:
    print(secrets.token_urlsafe(48))
    return 0


def cmd_say(args: argparse.Namespace) -> int:
    from rabt_ai.services import Engine
    from rabt_shared import AudioFormat

    async def run() -> int:
        engine = Engine.build(_settings())
        try:
            profile = engine.registry.voice_profile_for(args.account, args.profile)
            synth = await engine.tts.synthesize(profile, args.text, args.speed, AudioFormat(args.format))
        finally:
            await engine.aclose()
        Path(args.out).write_bytes(synth.audio)
        print(
            f"{args.out}: {synth.duration_s:.2f}s audio, voice {synth.voice}, synthesized in {synth.latency_ms} ms"
        )
        return 0

    return asyncio.run(run())


def cmd_transcribe(args: argparse.Namespace) -> int:
    from rabt_ai.services import Engine

    async def run() -> int:
        engine = Engine.build(_settings())
        try:
            result = await engine.stt.transcribe(Path(args.path).read_bytes(), args.language)
        finally:
            await engine.aclose()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    return asyncio.run(run())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rabt-ai", description="Rabt local AI engine")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("serve", help="run the HTTP API (default)")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("setup-models", help="download Whisper and Kokoro models (one-time, needs internet)")
    p.add_argument("--skip-whisper", action="store_true")
    p.add_argument("--skip-kokoro", action="store_true")
    p.set_defaults(func=cmd_setup_models)

    p = sub.add_parser("status", help="print model / GPU / speech status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("purge", help="delete local conversation data")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="delete every conversation")
    g.add_argument(
        "--older-than-days", type=int, help="delete conversations idle for N days (default: retention)"
    )
    p.add_argument("--account", help="delete one account's conversations")
    p.add_argument("--conversation", help="delete one conversation (needs --account)")
    p.set_defaults(func=cmd_purge)

    p = sub.add_parser("export-openapi", help="write the OpenAPI spec")
    p.add_argument("--out", default=str(REPO_ROOT / "docs" / "openapi.json"))
    p.set_defaults(func=cmd_export_openapi)

    p = sub.add_parser("gen-key", help="print a random RABT_INTERNAL_API_KEY")
    p.set_defaults(func=cmd_gen_key)

    p = sub.add_parser("say", help="synthesize speech to a file")
    p.add_argument("account")
    p.add_argument("text")
    p.add_argument("--out", default="rabt-say.wav")
    p.add_argument("--profile")
    p.add_argument("--speed", type=float)
    p.add_argument("--format", choices=["wav", "ogg_opus"], default="wav")
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("transcribe", help="transcribe an audio file")
    p.add_argument("path")
    p.add_argument("--language")
    p.set_defaults(func=cmd_transcribe)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        args = parser.parse_args(["serve", *(argv or [])])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
