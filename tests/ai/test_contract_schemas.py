"""Contract tests for the shared schemas in ``rabt_shared``."""

from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from rabt_shared import (
    ACTION_PROPOSAL_ADAPTER,
    PROPOSAL_MODELS,
    TOOL_RESULT_ADAPTER,
    TOOL_RESULT_MODELS,
    TOOL_SPECS,
    AgentRespondRequest,
    AgentResponse,
    ToolName,
    UnknownToolError,
    get_tool_spec,
)
from rabt_shared.jsonschema_export import SCHEMA_MODELS, build_schemas, default_output_dir

FIXTURES = Path(__file__).resolve().parents[2] / "shared" / "fixtures"


def _respond(**overrides: Any) -> dict[str, Any]:
    body = {
        "request_id": str(uuid4()),
        "conversation_id": str(uuid4()),
        "account_id": "awaiz",
        "role": "tutor",
        "channel": "web",
        "message": "Book tuition with Huda tomorrow evening",
    }
    body.update(overrides)
    return body


def _find_slot_proposal(**arg_overrides: Any) -> dict[str, Any]:
    args = {
        "participant_account_ids": ["awaiz", "huda"],
        "window_start": "2026-10-11T17:00:00+04:00",
        "window_end": "2026-10-11T21:00:00+04:00",
        "duration_minutes": 60,
    }
    args.update(arg_overrides)
    return {
        "proposal_id": str(uuid4()),
        "tool": "calendar.find_mutual_slot",
        "args": args,
        "summary": "Find a slot",
    }


def _create_event_proposal(**overrides: Any) -> dict[str, Any]:
    proposal = {
        "proposal_id": str(uuid4()),
        "tool": "calendar.create_event",
        "args": {
            "title": "Tuition with Huda",
            "start": "2026-10-11T18:00:00+04:00",
            "end": "2026-10-11T19:00:00+04:00",
            "attendee_account_ids": ["huda"],
        },
        "requires_approval": True,
        "summary": "Create event",
    }
    proposal.update(overrides)
    return proposal


def _tool_result(**overrides: Any) -> dict[str, Any]:
    body = {
        "request_id": str(uuid4()),
        "conversation_id": str(uuid4()),
        "account_id": "awaiz",
        "proposal_id": str(uuid4()),
        "tool": "calendar.find_mutual_slot",
        "status": "succeeded",
        "result": {"slots": [{"start": "2026-10-11T18:00:00+04:00", "end": "2026-10-11T19:00:00+04:00"}]},
    }
    body.update(overrides)
    return body


# --------------------------------------------------------------------------- respond request


def test_respond_request_defaults_timezone_to_dubai() -> None:
    req = AgentRespondRequest.model_validate(_respond())
    assert req.timezone == "Asia/Dubai"
    assert req.sent_at is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"role": "admin"},
        {"channel": "sms"},
        {"timezone": "Mars/Olympus"},
        {"account_id": "Awaiz"},  # account ids are lowercase slugs
        {"account_id": "awaiz; drop table"},
        {"message": ""},
        {"message": "x" * 4097},
        {"request_id": "not-a-uuid"},
        {"sent_at": "2026-10-10T20:15:00"},  # naive timestamps are rejected
        {"is_admin": True},  # unknown fields are rejected
    ],
)
def test_respond_request_rejects_invalid_input(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AgentRespondRequest.model_validate(_respond(**overrides))


# --------------------------------------------------------------------------- proposals


def test_valid_find_slot_proposal_defaults() -> None:
    proposal = ACTION_PROPOSAL_ADAPTER.validate_python(_find_slot_proposal())
    assert proposal.requires_approval is False
    assert proposal.args.timezone == "Asia/Dubai"
    assert proposal.args.max_slots == 5


def test_unknown_tool_name_is_rejected() -> None:
    bad = _find_slot_proposal()
    bad["tool"] = "shell.exec"
    with pytest.raises(ValidationError):
        ACTION_PROPOSAL_ADAPTER.validate_python(bad)
    with pytest.raises(UnknownToolError):
        get_tool_spec("shell.exec")


@pytest.mark.parametrize(
    "arg_overrides",
    [
        {"window_end": "2026-10-11T16:00:00+04:00"},  # ends before it starts
        {"window_end": "2026-10-30T17:00:00+04:00"},  # longer than 14 days
        {"participant_account_ids": ["awaiz", "awaiz"]},
        {"participant_account_ids": ["awaiz"]},
        {"duration_minutes": 5},
        {"duration_minutes": 300},  # does not fit in a 4h window
        {"window_start": "2026-10-11T17:00:00"},  # naive
        {"timezone": "Not/AZone"},
        {"command": "rm -rf /"},  # unknown argument
    ],
)
def test_invalid_tool_arguments_are_rejected(arg_overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ACTION_PROPOSAL_ADAPTER.validate_python(_find_slot_proposal(**arg_overrides))


@pytest.mark.parametrize("tool", [t for t, spec in TOOL_SPECS.items() if spec.side_effects])
def test_side_effect_tools_cannot_skip_approval(tool: ToolName) -> None:
    field = PROPOSAL_MODELS[tool].model_fields["requires_approval"]
    assert typing.get_args(field.annotation) == (True,), f"{tool} must be Literal[True]"


def test_create_event_proposal_without_approval_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ACTION_PROPOSAL_ADAPTER.validate_python(_create_event_proposal(requires_approval=False))


def test_create_event_rejects_end_before_start() -> None:
    bad = _create_event_proposal()
    bad["args"]["end"] = "2026-10-11T17:00:00+04:00"
    with pytest.raises(ValidationError):
        ACTION_PROPOSAL_ADAPTER.validate_python(bad)


def test_catalog_proposals_and_results_cover_every_tool() -> None:
    assert set(TOOL_SPECS) == set(ToolName) == set(PROPOSAL_MODELS) == set(TOOL_RESULT_MODELS)
    for tool in ToolName:
        assert typing.get_args(PROPOSAL_MODELS[tool].model_fields["tool"].annotation) == (tool.value,)
        assert typing.get_args(TOOL_RESULT_MODELS[tool].model_fields["tool"].annotation) == (tool.value,)
        assert TOOL_SPECS[tool].requires_approval == TOOL_SPECS[tool].side_effects


# --------------------------------------------------------------------------- tool results


@pytest.mark.parametrize(
    "overrides",
    [
        {"result": None},  # succeeded needs a result
        {"error": {"code": "x", "message": "y"}},  # succeeded must not carry an error
        {"status": "failed"},  # failed needs an error and no result
        {"status": "denied", "result": None},  # denied needs an error
        {"status": "rejected"},  # rejected must not carry a result
        {"status": "approved"},  # the engine never accepts an 'approved' status
        {"tool": "calendar.create_event"},  # result shape must match the tool
    ],
)
def test_tool_result_status_and_payload_must_agree(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TOOL_RESULT_ADAPTER.validate_python(_tool_result(**overrides))


def test_tool_result_variants_that_are_valid() -> None:
    TOOL_RESULT_ADAPTER.validate_python(_tool_result(status="rejected", result=None))
    TOOL_RESULT_ADAPTER.validate_python(
        _tool_result(status="failed", result=None, error={"code": "calendar_timeout", "message": "timed out"})
    )


def test_free_busy_result_cannot_leak_private_event_details() -> None:
    leaky = _tool_result(
        result={
            "slots": [
                {
                    "start": "2026-10-11T18:00:00+04:00",
                    "end": "2026-10-11T19:00:00+04:00",
                    "title": "Doctor appointment",
                }
            ]
        }
    )
    with pytest.raises(ValidationError):
        TOOL_RESULT_ADAPTER.validate_python(leaky)


# --------------------------------------------------------------------------- response


def test_agent_response_status_invariants() -> None:
    base = {
        "request_id": str(uuid4()),
        "conversation_id": str(uuid4()),
        "model": "qwen3.5:9b",
    }
    proposal = _find_slot_proposal()
    AgentResponse.model_validate(
        {
            **base,
            "status": "action_required",
            "reply_text": "Checking",
            "action_proposals": [proposal],
            "pending_proposal_ids": [proposal["proposal_id"]],
        }
    )
    invalid = [
        {"status": "completed", "reply_text": ""},
        {"status": "completed", "reply_text": "ok", "action_proposals": [proposal]},
        {"status": "action_required", "reply_text": "x"},
        {"status": "action_required", "reply_text": "x", "action_proposals": [proposal]},
        {"status": "awaiting_results", "reply_text": ""},
        {"status": "completed", "reply_text": "ok", "mode": "cloud"},
    ]
    for extra in invalid:
        with pytest.raises(ValidationError):
            AgentResponse.model_validate({**base, **extra})


# --------------------------------------------------------------------------- fixtures + schemas


def _manifest() -> dict[str, str]:
    return json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("filename", sorted(_manifest()))
def test_fixture_validates_against_its_schema(filename: str) -> None:
    schema_name = _manifest()[filename]
    model, _mode = SCHEMA_MODELS[schema_name]
    payload = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    TypeAdapter(model).validate_python(payload)


def test_every_fixture_is_listed_in_manifest() -> None:
    on_disk = {p.name for p in FIXTURES.glob("*.json")} - {"manifest.json"}
    assert on_disk == set(_manifest())


def test_committed_json_schemas_are_up_to_date() -> None:
    out = default_output_dir()
    for name, text in build_schemas().items():
        path = out / name
        assert path.is_file(), f"missing {path}; run python -m rabt_shared.jsonschema_export"
        assert path.read_text(encoding="utf-8") == text, (
            f"{name} is stale; run python -m rabt_shared.jsonschema_export"
        )
