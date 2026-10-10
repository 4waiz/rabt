"""``GET /health`` and ``GET /v1/agent/status``."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from rabt_shared.common import StrictModel


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str


class ComponentState(StrEnum):
    READY = "ready"  # loaded and usable now
    NOT_LOADED = "not_loaded"  # usable, loads lazily on first request
    UNAVAILABLE = "unavailable"  # cannot be used; see `detail`


class GpuStatus(StrictModel):
    available: bool
    name: str | None = None
    driver_version: str | None = None
    total_vram_mb: int | None = None
    used_vram_mb: int | None = None
    compute_capability: str | None = None


class LlmStatus(StrictModel):
    provider: Literal["ollama"] = "ollama"
    model: str
    state: ComponentState
    reachable: bool
    model_available: bool
    on_gpu: bool | None = Field(default=None, description="True when the loaded weights are in VRAM.")
    vram_bytes: int | None = None
    context_length: int | None = None
    capabilities: list[str] = Field(default_factory=list)
    detail: str | None = None


class SttStatus(StrictModel):
    engine: str
    model: str
    state: ComponentState
    device: str | None = None
    compute_type: str | None = None
    detail: str | None = None


class TtsStatus(StrictModel):
    engine: str
    model: str
    state: ComponentState
    voice_profiles: list[str] = Field(default_factory=list)
    detail: str | None = None


class AgentStatusResponse(StrictModel):
    status: Literal["ok", "degraded"]
    mode: Literal["local"] = "local"
    contract_version: str
    engine_version: str
    llm: LlmStatus
    stt: SttStatus
    tts: TtsStatus
    gpu: GpuStatus
