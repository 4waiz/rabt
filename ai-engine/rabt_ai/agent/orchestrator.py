"""The bounded agent loop behind ``/v1/agent/respond`` and ``/v1/agent/tool-result``.

One user message is one *turn*. Within a turn the model may request tools; each valid
request becomes a stored proposal that is returned to the integration backend. The loop
resumes only when the backend reports the outcome of every proposal in the batch.

Hard limits: model calls per step, tool calls per model response and proposals per turn.
The model can never approve, execute or fabricate the result of an action.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import weakref
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from rabt_ai.agent.dates import DateContext
from rabt_ai.agent.guards import claims_completion, clean_reply, has_leftover_markup
from rabt_ai.agent.prompts import build_messages, build_system_prompt, render_tool_result
from rabt_ai.agent.tools import (
    BuildContext,
    BuiltProposal,
    ToolCallRejected,
    build_proposal,
    tool_definitions,
)
from rabt_ai.config import Settings
from rabt_ai.errors import RabtError
from rabt_ai.identity import Identity, IdentityRegistry
from rabt_ai.llm import ChatModel, ChatResult, ToolCall
from rabt_ai.logging_setup import log_event
from rabt_ai.storage import ConversationRow, Storage, TurnRow, utcnow
from rabt_shared import (
    PROPOSAL_MODELS,
    TOOL_SPECS,
    AgentRespondRequest,
    AgentResponse,
    Channel,
    ErrorCode,
    ResponseMetrics,
    ToolResultRequest,
    TurnStatus,
)

log = logging.getLogger(__name__)

SIDE_EFFECT_TOOLS = tuple(t.value for t, spec in TOOL_SPECS.items() if spec.side_effects)

NUDGE_EMPTY = (
    "(Automatic message from the Rabt system, not from the user: your last reply was empty or malformed. "
    "Reply to the user in plain text, or call one of the tools.)"
)
CLAIM_CORRECTION = (
    "(Automatic check from the Rabt system, not from the user: your reply says an action was completed, "
    "but no tool result shows that it succeeded. Nothing has been sent, booked or created. Rewrite the reply "
    "without claiming it is done. If the user wants the action, call the right tool so it can be prepared "
    "for their approval.)"
)
SAFE_NOT_DONE = "Nothing has been sent or booked yet. Would you like me to prepare it for your approval?"
LOOP_EXHAUSTED = "I couldn't complete that reliably. Could you rephrase it or give me a bit more detail?"


def payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class _Metrics:
    started: float
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, result: ChatResult) -> None:
        self.llm_calls += 1
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens

    def freeze(self) -> ResponseMetrics:
        return ResponseMetrics(
            total_ms=int((time.perf_counter() - self.started) * 1000),
            llm_calls=self.llm_calls,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )


@dataclass
class _StepCheck:
    built: list[tuple[ToolCall, BuiltProposal]] = field(default_factory=list)
    feedback: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


class AgentService:
    def __init__(self, settings: Settings, registry: IdentityRegistry, storage: Storage, llm: ChatModel):
        self._settings = settings
        self._registry = registry
        self._storage = storage
        self._llm = llm
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    def _lock(self, conversation_id: str) -> asyncio.Lock:
        lock = self._locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[conversation_id] = lock
        return lock

    def _replay(
        self, scope: str, key: str, account_id: str, digest: str, conflict: ErrorCode
    ) -> AgentResponse | None:
        stored = self._storage.get_idempotent(scope, key)
        if stored is None:
            return None
        owner, stored_digest, response = stored
        if owner != account_id or stored_digest != digest:
            raise RabtError(conflict, f"{scope} {key} was already processed with a different payload")
        return AgentResponse.model_validate(response)

    # ------------------------------------------------------------------------ respond

    async def respond(self, req: AgentRespondRequest) -> AgentResponse:
        started = time.perf_counter()
        identity = self._registry.require(req.account_id, req.role)
        digest = payload_hash(req.model_dump(mode="json"))
        request_id, conversation_id = str(req.request_id), str(req.conversation_id)

        async with self._lock(conversation_id):
            replay = self._replay(
                "respond", request_id, req.account_id, digest, ErrorCode.IDEMPOTENCY_CONFLICT
            )
            if replay is not None:
                return replay
            if self._storage.request_id_in_use(request_id):
                raise RabtError(ErrorCode.IDEMPOTENCY_CONFLICT, "request_id is already used by another turn")

            conv = self._storage.get_or_create_conversation(
                conversation_id, identity.account_id, req.channel.value, req.timezone
            )
            turn = TurnRow(
                request_id=request_id,
                conversation_id=conversation_id,
                account_id=identity.account_id,
                anchor_time=req.sent_at or utcnow(),
                timezone=req.timezone,
                channel=req.channel.value,
            )
            self._storage.create_turn(turn)
            cancelled = self._cancel_pending(conv, identity)
            self._storage.add_message(
                conversation_id=conversation_id,
                account_id=identity.account_id,
                request_id=request_id,
                role="user",
                content=req.message,
            )
            log_event(
                log,
                "turn_started",
                request_id=request_id,
                conversation_id=conversation_id,
                account_id=identity.account_id,
                channel=req.channel.value,
                message_chars=len(req.message),
                cancelled=len(cancelled),
            )
            try:
                response = await self._run(identity, conv, turn, cancelled, _Metrics(started))
            except BaseException:
                # Leave no half-processed turn behind: the backend may retry with the same request_id.
                self._storage.delete_turn(request_id, identity.account_id)
                raise
            self._storage.put_idempotent(
                "respond",
                request_id,
                identity.account_id,
                conversation_id,
                digest,
                response.model_dump(mode="json"),
            )
            return response

    def _cancel_pending(self, conv: ConversationRow, identity: Identity) -> list[UUID]:
        cancelled: list[UUID] = []
        dates = DateContext.create(utcnow(), conv.timezone)
        for proposal in self._storage.pending_proposals(conv.conversation_id, identity.account_id):
            error = {"code": "superseded", "message": "the user sent a new message first"}
            if self._storage.resolve_proposal(
                proposal.proposal_id, identity.account_id, "cancelled", None, error
            ):
                self._storage.add_message(
                    conversation_id=conv.conversation_id,
                    account_id=identity.account_id,
                    request_id=proposal.request_id,
                    role="tool",
                    content=render_tool_result(proposal, "cancelled", None, error, dates),
                    proposal_id=proposal.proposal_id,
                    tool_name=proposal.tool,
                )
                cancelled.append(UUID(proposal.proposal_id))
        return cancelled

    # ------------------------------------------------------------------------ tool-result

    async def tool_result(self, req: ToolResultRequest) -> AgentResponse:
        started = time.perf_counter()
        identity = self._registry.require(req.account_id)
        digest = payload_hash(req.model_dump(mode="json"))
        proposal_id = str(req.proposal_id)

        async with self._lock(str(req.conversation_id)):
            proposal = self._storage.get_proposal(proposal_id, identity.account_id)
            if proposal is None:
                raise RabtError(ErrorCode.NOT_FOUND, "proposal not found for this account")
            mismatched = [
                name
                for name, ours, theirs in (
                    ("conversation_id", proposal.conversation_id, str(req.conversation_id)),
                    ("request_id", proposal.request_id, str(req.request_id)),
                    ("tool", proposal.tool, req.tool),
                )
                if ours != theirs
            ]
            if mismatched:
                raise RabtError(
                    ErrorCode.PROPOSAL_MISMATCH,
                    f"{', '.join(mismatched)} do(es) not match the stored proposal",
                    details=[{"field": name} for name in mismatched],
                )

            conv = self._storage.get_conversation(proposal.conversation_id, identity.account_id)
            turn = self._storage.get_turn(proposal.request_id, identity.account_id)
            if conv is None or turn is None:
                raise RabtError(ErrorCode.NOT_FOUND, "the conversation for this proposal no longer exists")

            if proposal.status != "pending":
                replay = self._replay(
                    "tool_result", proposal_id, identity.account_id, digest, ErrorCode.PROPOSAL_NOT_PENDING
                )
                if replay is not None:
                    return replay
                if proposal.resolution_hash != digest:
                    raise RabtError(ErrorCode.PROPOSAL_NOT_PENDING, f"proposal is already {proposal.status}")
                # Same result re-sent after a failed continuation: fall through and continue the turn.
            else:
                result = req.result.model_dump(mode="json") if req.result is not None else None
                error = req.error.model_dump(mode="json") if req.error is not None else None
                if not self._storage.resolve_proposal(
                    proposal_id, identity.account_id, req.status.value, result, error, digest
                ):
                    raise RabtError(ErrorCode.PROPOSAL_NOT_PENDING, "proposal was resolved concurrently")
                dates = DateContext.create(turn.anchor_time, turn.timezone)
                self._storage.add_message(
                    conversation_id=conv.conversation_id,
                    account_id=identity.account_id,
                    request_id=proposal.request_id,
                    role="tool",
                    content=render_tool_result(proposal, req.status.value, result, error, dates),
                    proposal_id=proposal_id,
                    tool_name=proposal.tool,
                )
                log_event(
                    log,
                    "tool_result",
                    request_id=proposal.request_id,
                    proposal_id=proposal_id,
                    tool=proposal.tool,
                    status=req.status.value,
                )

            pending = [
                UUID(p.proposal_id)
                for p in self._storage.proposals_for_turn(
                    conv.conversation_id, identity.account_id, turn.request_id
                )
                if p.status == "pending"
            ]
            metrics = _Metrics(started)
            if pending:
                response = AgentResponse(
                    request_id=req.request_id,
                    conversation_id=req.conversation_id,
                    status=TurnStatus.AWAITING_RESULTS,
                    reply_text="",
                    pending_proposal_ids=pending,
                    model=self.model_name,
                    metrics=metrics.freeze(),
                )
            else:
                response = await self._run(identity, conv, turn, [], metrics)
            self._storage.put_idempotent(
                "tool_result",
                proposal_id,
                identity.account_id,
                conv.conversation_id,
                digest,
                response.model_dump(mode="json"),
            )
            return response

    # ------------------------------------------------------------------------ loop

    async def _run(
        self,
        identity: Identity,
        conv: ConversationRow,
        turn: TurnRow,
        cancelled: list[UUID],
        metrics: _Metrics,
    ) -> AgentResponse:
        s = self._settings
        dates = DateContext.create(turn.anchor_time, turn.timezone)
        preferences = self._storage.get_preferences(identity.account_id)
        ctx = BuildContext(identity=identity, dates=dates, preferences=preferences)
        system = build_system_prompt(identity, dates, Channel(turn.channel), preferences)
        history = self._storage.load_messages(conv.conversation_id, identity.account_id)
        messages = build_messages(system, history, s.agent_history_char_budget)
        tools = tool_definitions(identity)
        used = len(
            self._storage.proposals_for_turn(conv.conversation_id, identity.account_id, turn.request_id)
        )

        scratch: list[dict[str, Any]] = []  # corrections live only inside this loop
        claim_corrected = False
        for _ in range(s.agent_max_llm_calls):
            result = await self._llm.chat(messages + scratch, tools)
            metrics.add(result)

            if result.tool_calls:
                check = self._check_calls(result.tool_calls, ctx, used)
                if check.rejected:
                    log_event(
                        log,
                        "tool_calls_rejected",
                        logging.WARNING,
                        request_id=turn.request_id,
                        reasons=check.rejected,
                    )
                    scratch.append(_assistant_with_calls(result))
                    scratch.extend(check.feedback)
                    continue
                return self._emit(identity, conv, turn, result, check.built, cancelled, metrics)

            reply = clean_reply(result.content)
            if not reply or has_leftover_markup(reply):
                scratch.append({"role": "user", "content": NUDGE_EMPTY})
                continue
            if claims_completion(reply) and not self._storage.has_succeeded_action(
                conv.conversation_id, identity.account_id, SIDE_EFFECT_TOOLS
            ):
                log_event(log, "completion_claim_blocked", logging.WARNING, request_id=turn.request_id)
                if not claim_corrected:
                    claim_corrected = True
                    scratch += [
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": CLAIM_CORRECTION},
                    ]
                    continue
                reply = SAFE_NOT_DONE
            return self._finish(identity, conv, turn, reply, cancelled, metrics)

        log_event(log, "agent_loop_exhausted", logging.WARNING, request_id=turn.request_id)
        return self._finish(identity, conv, turn, LOOP_EXHAUSTED, cancelled, metrics)

    def _check_calls(self, calls: list[ToolCall], ctx: BuildContext, used: int) -> _StepCheck:
        s = self._settings
        check = _StepCheck()
        errors: dict[int, str] = {}
        if len(calls) > s.agent_max_tool_calls_per_step:
            errors = {
                i: f"too many tool calls at once (max {s.agent_max_tool_calls_per_step})"
                for i in range(len(calls))
            }
        else:
            for i, call in enumerate(calls):
                try:
                    check.built.append((call, build_proposal(call, ctx)))
                except ToolCallRejected as exc:
                    errors[i] = str(exc)
            if not errors:
                lookups = [bp for _, bp in check.built if not bp.requires_approval]
                actions = [i for i, (_, bp) in enumerate(check.built) if bp.requires_approval]
                if lookups and actions:
                    # Never prepare an action in the same step as the lookup it should depend on.
                    errors = {i: "wait for the lookup results before preparing this action" for i in actions}
                elif used + len(check.built) > s.agent_max_proposals_per_turn:
                    errors = {
                        i: f"limit of {s.agent_max_proposals_per_turn} actions for one request reached; answer the user"
                        for i in range(len(calls))
                    }
        if errors:
            check.rejected = list(errors.values())
            check.built = []
            for i, call in enumerate(calls):
                msg = errors.get(
                    i,
                    "not submitted because another call in the same step was invalid; call it again if still needed",
                )
                check.feedback.append({"role": "tool", "tool_name": call.name, "content": f"error: {msg}"})
        return check

    def _emit(
        self,
        identity: Identity,
        conv: ConversationRow,
        turn: TurnRow,
        result: ChatResult,
        built: list[tuple[ToolCall, BuiltProposal]],
        cancelled: list[UUID],
        metrics: _Metrics,
    ) -> AgentResponse:
        proposals = []
        records = []
        for call, bp in built:
            proposal_id = str(uuid4())
            args = bp.args.model_dump(mode="json")
            self._storage.create_proposal(
                proposal_id=proposal_id,
                conversation_id=conv.conversation_id,
                account_id=identity.account_id,
                request_id=turn.request_id,
                tool=bp.tool.value,
                args=args,
                requires_approval=bp.requires_approval,
                summary=bp.summary,
            )
            proposals.append(
                PROPOSAL_MODELS[bp.tool].model_validate(
                    {
                        "proposal_id": proposal_id,
                        "tool": bp.tool.value,
                        "args": args,
                        "requires_approval": bp.requires_approval,
                        "summary": bp.summary,
                    }
                )
            )
            records.append({"name": call.name, "arguments": call.arguments, "proposal_id": proposal_id})
        self._storage.add_message(
            conversation_id=conv.conversation_id,
            account_id=identity.account_id,
            request_id=turn.request_id,
            role="assistant",
            content=clean_reply(result.content),
            tool_calls=records,
        )
        pending = [
            UUID(p.proposal_id)
            for p in self._storage.proposals_for_turn(
                conv.conversation_id, identity.account_id, turn.request_id
            )
            if p.status == "pending"
        ]
        reply = " ".join(dict.fromkeys(bp.interim for _, bp in built))
        log_event(
            log,
            "proposals_emitted",
            request_id=turn.request_id,
            tools=[bp.tool.value for _, bp in built],
            requires_approval=[bp.requires_approval for _, bp in built],
        )
        return AgentResponse(
            request_id=UUID(turn.request_id),
            conversation_id=UUID(conv.conversation_id),
            status=TurnStatus.ACTION_REQUIRED,
            reply_text=reply,
            action_proposals=proposals,
            pending_proposal_ids=pending,
            cancelled_proposal_ids=cancelled,
            model=self.model_name,
            metrics=metrics.freeze(),
        )

    def _finish(
        self,
        identity: Identity,
        conv: ConversationRow,
        turn: TurnRow,
        reply: str,
        cancelled: list[UUID],
        metrics: _Metrics,
    ) -> AgentResponse:
        self._storage.add_message(
            conversation_id=conv.conversation_id,
            account_id=identity.account_id,
            request_id=turn.request_id,
            role="assistant",
            content=reply,
        )
        frozen = metrics.freeze()
        log_event(
            log,
            "turn_completed",
            request_id=turn.request_id,
            reply_chars=len(reply),
            llm_calls=frozen.llm_calls,
            total_ms=frozen.total_ms,
        )
        return AgentResponse(
            request_id=UUID(turn.request_id),
            conversation_id=UUID(conv.conversation_id),
            status=TurnStatus.COMPLETED,
            reply_text=reply[:8000],
            cancelled_proposal_ids=cancelled,
            model=self.model_name,
            metrics=frozen,
        )


def _assistant_with_calls(result: ChatResult) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": clean_reply(result.content),
        "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in result.tool_calls],
    }
