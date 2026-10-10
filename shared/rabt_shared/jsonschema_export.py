"""Regenerate ``shared/jsonschema/`` from the Pydantic models.

python -m rabt_shared.jsonschema_export          # write files
python -m rabt_shared.jsonschema_export --check  # exit 1 if committed files are stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter

from rabt_shared.agent import AgentRespondRequest, AgentResponse, ToolResultRequest
from rabt_shared.audio import SpeakRequest, TranscriptionResponse
from rabt_shared.errors import ErrorResponse
from rabt_shared.status import AgentStatusResponse, HealthResponse
from rabt_shared.tools import TOOL_SPECS
from rabt_shared.version import CONTRACT_VERSION

JsonMode = Literal["validation", "serialization"]

# schema name -> (model, mode). Requests use validation mode, responses serialization mode.
SCHEMA_MODELS: dict[str, tuple[Any, JsonMode]] = {
    "agent_respond_request": (AgentRespondRequest, "validation"),
    "agent_response": (AgentResponse, "serialization"),
    "tool_result_request": (ToolResultRequest, "validation"),
    "speak_request": (SpeakRequest, "validation"),
    "transcription_response": (TranscriptionResponse, "serialization"),
    "agent_status_response": (AgentStatusResponse, "serialization"),
    "health_response": (HealthResponse, "serialization"),
    "error_response": (ErrorResponse, "serialization"),
}


def _dump(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def build_schemas() -> dict[str, str]:
    """Return ``{filename: json_text}`` for every exported schema plus the tool catalog."""
    files: dict[str, str] = {}
    for name, (model, mode) in SCHEMA_MODELS.items():
        schema = TypeAdapter(model).json_schema(mode=mode)
        schema.setdefault("title", "".join(part.title() for part in name.split("_")))
        files[f"{name}.schema.json"] = _dump(
            {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}
        )

    catalog = {
        "contract_version": CONTRACT_VERSION,
        "tools": [
            {
                "name": spec.name.value,
                "description": spec.description,
                "side_effects": spec.side_effects,
                "requires_approval": spec.requires_approval,
                "args_schema": spec.args_model.model_json_schema(mode="validation"),
                "result_schema": spec.result_model.model_json_schema(mode="validation"),
            }
            for spec in TOOL_SPECS.values()
        ],
    }
    files["tool_catalog.json"] = _dump(catalog)
    return files


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "jsonschema"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if files on disk are out of date")
    parser.add_argument("--out", type=Path, default=default_output_dir())
    args = parser.parse_args(argv)

    files = build_schemas()
    if args.check:
        stale = [
            name
            for name, text in files.items()
            if not (args.out / name).is_file() or (args.out / name).read_text(encoding="utf-8") != text
        ]
        if stale:
            print("Out-of-date JSON Schemas: " + ", ".join(stale), file=sys.stderr)
            print("Run: python -m rabt_shared.jsonschema_export", file=sys.stderr)
            return 1
        print(f"{len(files)} schema files are up to date.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {len(files)} files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
