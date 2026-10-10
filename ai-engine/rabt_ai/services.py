"""Wires the engine's components together."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from rabt_ai import __version__
from rabt_ai.agent.orchestrator import AgentService
from rabt_ai.config import Settings
from rabt_ai.gpu import query_gpu
from rabt_ai.identity import IdentityRegistry
from rabt_ai.llm import ChatModel, OllamaClient
from rabt_ai.speech.stt import SpeechToText
from rabt_ai.speech.tts import TextToSpeech
from rabt_ai.storage import Storage
from rabt_shared import CONTRACT_VERSION, AgentStatusResponse, ComponentState, LlmStatus


@dataclass
class Engine:
    settings: Settings
    registry: IdentityRegistry
    storage: Storage
    llm: ChatModel
    agent: AgentService
    stt: SpeechToText
    tts: TextToSpeech

    @classmethod
    def build(cls, settings: Settings, *, llm: ChatModel | None = None) -> Engine:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        registry = IdentityRegistry.load(settings.identities_file)
        storage = Storage(settings.db_path)
        storage.sync_registry(registry)
        model = llm or OllamaClient(settings)
        return cls(
            settings=settings,
            registry=registry,
            storage=storage,
            llm=model,
            agent=AgentService(settings, registry, storage, model),
            stt=SpeechToText(settings),
            tts=TextToSpeech(settings, registry, storage),
        )

    async def llm_status(self) -> LlmStatus:
        reporter: Any = getattr(self.llm, "llm_status", None)
        if reporter is None:
            return LlmStatus(
                model=self.llm.model_name,
                state=ComponentState.UNAVAILABLE,
                reachable=False,
                model_available=False,
                detail="language model does not report status",
            )
        return await reporter()

    async def status(self) -> AgentStatusResponse:
        llm, gpu = await asyncio.gather(self.llm_status(), asyncio.to_thread(query_gpu))
        stt, tts = self.stt.status(), self.tts.status()
        healthy = llm.state != ComponentState.UNAVAILABLE and ComponentState.UNAVAILABLE not in (
            stt.state,
            tts.state,
        )
        return AgentStatusResponse(
            status="ok" if healthy else "degraded",
            contract_version=CONTRACT_VERSION,
            engine_version=__version__,
            llm=llm,
            stt=stt,
            tts=tts,
            gpu=gpu,
        )

    async def aclose(self) -> None:
        closer = getattr(self.llm, "aclose", None)
        if closer is not None:
            await closer()
        self.storage.close()
