"""HTTP contract details: auth, error envelope, limits, headers, OpenAPI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conftest import TEST_KEY, FakeLlm, reply
from fastapi.testclient import TestClient

from rabt_ai.api import create_app
from rabt_shared import CONTRACT_VERSION, AgentStatusResponse, ErrorResponse

REPO = Path(__file__).resolve().parents[2]


def test_health_is_public_and_minimal(client: TestClient) -> None:
    resp = client.get("/health", headers={"Authorization": ""})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "rabt-ai-engine", "version": "0.1.0"}
    assert resp.headers["X-Rabt-Contract-Version"] == CONTRACT_VERSION


def test_v1_requires_the_service_token(client: TestClient, respond_body: Any) -> None:
    for headers in (
        {"Authorization": ""},
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": f"Basic {TEST_KEY}"},
    ):
        resp = client.post("/v1/agent/respond", json=respond_body("hi"), headers=headers)
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"] == "Bearer"
        assert ErrorResponse.model_validate(resp.json()).error.code == "unauthorized"
    assert client.get("/v1/agent/status", headers={"Authorization": ""}).status_code == 401


def test_validation_errors_use_the_envelope(client: TestClient, respond_body: Any) -> None:
    body = respond_body("hi", timezone="Mars/Olympus", extra_field=True)
    resp = client.post("/v1/agent/respond", json=body)
    assert resp.status_code == 422
    err = ErrorResponse.model_validate(resp.json()).error
    assert err.code == "validation_error" and err.request_id == body["request_id"]
    locs = {tuple(d["loc"]) for d in err.details or []}
    assert ("body", "timezone") in locs and ("body", "extra_field") in locs
    assert resp.headers["X-Request-Id"] == body["request_id"]


def test_oversized_body_is_rejected_before_parsing(client: TestClient, respond_body: Any) -> None:
    resp = client.post("/v1/agent/respond", json=respond_body("x" * 70_000))
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


def test_respond_sets_request_headers(client: TestClient, fake_llm: FakeLlm, respond_body: Any) -> None:
    fake_llm.push(reply("Hello!"))
    body = respond_body("hello")
    resp = client.post("/v1/agent/respond", json=body)
    assert resp.status_code == 200
    assert resp.headers["X-Request-Id"] == body["request_id"]
    assert resp.headers["X-Rabt-Contract-Version"] == CONTRACT_VERSION


def test_status_endpoint_matches_contract(client: TestClient) -> None:
    resp = client.get("/v1/agent/status")
    assert resp.status_code == 200
    status = AgentStatusResponse.model_validate(resp.json())
    assert status.mode == "local" and status.contract_version == CONTRACT_VERSION


def test_speak_refuses_someone_elses_voice(client: TestClient) -> None:
    resp = client.post(
        "/v1/audio/speak", json={"account_id": "huda", "text": "hello", "voice_profile": "awaiz-default"}
    )
    assert resp.status_code == 403 and resp.json()["error"]["code"] == "voice_profile_forbidden"
    resp = client.post("/v1/audio/speak", json={"account_id": "nobody", "text": "hello"})
    assert resp.status_code == 403 and resp.json()["error"]["code"] == "unknown_account"


def test_transcribe_rejects_non_audio(client: TestClient) -> None:
    files = {"file": ("notes.txt", b"definitely not audio", "text/plain")}
    resp = client.post("/v1/audio/transcribe", files=files, data={"account_id": "awaiz"})
    assert resp.status_code == 415 and resp.json()["error"]["code"] == "unsupported_media_type"
    files = {"file": ("clip.wav", b"RIFF....garbage", "audio/wav")}
    resp = client.post("/v1/audio/transcribe", files=files, data={"account_id": "awaiz"})
    assert resp.status_code == 422 and resp.json()["error"]["code"] == "audio_decode_error"
    resp = client.post("/v1/audio/transcribe", files=files, data={"account_id": "Robert'); DROP TABLE"})
    assert resp.status_code == 422


def test_openapi_lists_every_endpoint_with_bearer_auth() -> None:
    spec = create_app(warm_up=False).openapi()
    for path in (
        "/health",
        "/v1/agent/status",
        "/v1/agent/respond",
        "/v1/agent/tool-result",
        "/v1/audio/transcribe",
        "/v1/audio/speak",
    ):
        assert path in spec["paths"], path
    assert "HTTPBearer" in spec["components"]["securitySchemes"]


def test_committed_openapi_snapshot_is_current() -> None:
    spec = create_app(warm_up=False).openapi()
    committed = json.loads((REPO / "docs" / "openapi.json").read_text(encoding="utf-8"))
    assert committed == spec, "docs/openapi.json is stale; run: rabt-ai export-openapi"
