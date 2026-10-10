"""Agent endpoints: ``POST /v1/agent/respond`` and ``POST /v1/agent/tool-result``."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, TypeAdapter, model_validator

from rabt_shared.common import (
    DEFAULT_TIMEZONE,
    MAX_MESSAGE_CHARS,
    AccountId,
    Channel,
    Role,
    StrictModel,
    Timezone,
)
from rabt_shared.tools import (
    CreateDraftArgs,
    CreateDraftResult,
    CreateEventArgs,
    CreateEventResult,
    FindMutualSlotArgs,
    FindMutualSlotResult,
    ListRecentEmailArgs,
    ListRecentEmailResult,
    SendMessageArgs,
    SendMessageResult,
    ToolName,
)

# --------------------------------------------------------------------------- respond


class AgentRespondRequest(StrictModel):
    """One user turn, sent by the authenticated integration backend."""

    request_id: UUID = Field(
        description="Unique per user turn. Re-sending the same body replays the stored response."
    )
    conversation_id: UUID = Field(description="Chosen by the backend; bound to account_id on first use.")
    account_id: AccountId = Field(
        description="Authenticated by the integration backend. Never derived from message text."
    )
    role: Role
    channel: Channel
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    timezone: Timezone = DEFAULT_TIMEZONE
    sent_at: AwareDatetime | None = Field(
        default=None,
        description="When the user sent the message. Defaults to the time the engine received it.",
    )


# --------------------------------------------------------------------------- proposals

ProposalSummary = Annotated[
    str,
    Field(
        min_length=1,
        max_length=500,
        description="Engine-generated description of exactly what would be executed. "
        "Show this text in approval prompts.",
    ),
]


class GmailListRecentProposal(StrictModel):
    proposal_id: UUID
    tool: Literal["gmail.list_recent"]
    args: ListRecentEmailArgs
    requires_approval: bool = False
    summary: ProposalSummary


class GmailCreateDraftProposal(StrictModel):
    proposal_id: UUID
    tool: Literal["gmail.create_draft"]
    args: CreateDraftArgs
    requires_approval: Literal[True] = True
    summary: ProposalSummary


class CalendarFindMutualSlotProposal(StrictModel):
    proposal_id: UUID
    tool: Literal["calendar.find_mutual_slot"]
    args: FindMutualSlotArgs
    requires_approval: bool = False
    summary: ProposalSummary


class CalendarCreateEventProposal(StrictModel):
    proposal_id: UUID
    tool: Literal["calendar.create_event"]
    args: CreateEventArgs
    requires_approval: Literal[True] = True
    summary: ProposalSummary


class MessagingSendMessageProposal(StrictModel):
    proposal_id: UUID
    tool: Literal["messaging.send_message"]
    args: SendMessageArgs
    requires_approval: Literal[True] = True
    summary: ProposalSummary


ActionProposal = Annotated[
    GmailListRecentProposal
    | GmailCreateDraftProposal
    | CalendarFindMutualSlotProposal
    | CalendarCreateEventProposal
    | MessagingSendMessageProposal,
    Field(discriminator="tool"),
]

PROPOSAL_MODELS: Mapping[ToolName, type[StrictModel]] = MappingProxyType(
    {
        ToolName.GMAIL_LIST_RECENT: GmailListRecentProposal,
        ToolName.GMAIL_CREATE_DRAFT: GmailCreateDraftProposal,
        ToolName.CALENDAR_FIND_MUTUAL_SLOT: CalendarFindMutualSlotProposal,
        ToolName.CALENDAR_CREATE_EVENT: CalendarCreateEventProposal,
        ToolName.MESSAGING_SEND_MESSAGE: MessagingSendMessageProposal,
    }
)

# --------------------------------------------------------------------------- tool results


class ToolExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"  # executed; `result` holds the structured output
    FAILED = "failed"  # execution attempted and failed; `error` explains why
    REJECTED = "rejected"  # the user declined the approval prompt; nothing was executed
    DENIED = "denied"  # backend policy refused the action; nothing was executed


class ToolError(StrictModel):
    code: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_.]*$")
    message: str = Field(min_length=1, max_length=500)
    retryable: bool = False


class _ToolResultBase(StrictModel):
    request_id: UUID = Field(description="request_id of the turn that produced the proposal.")
    conversation_id: UUID
    account_id: AccountId
    proposal_id: UUID
    status: ToolExecutionStatus
    error: ToolError | None = None

    @model_validator(mode="after")
    def _check_payload(self) -> Any:
        result = getattr(self, "result", None)
        if self.status == ToolExecutionStatus.SUCCEEDED:
            if result is None:
                raise ValueError("result is required when status is 'succeeded'")
            if self.error is not None:
                raise ValueError("error must be omitted when status is 'succeeded'")
            return self
        if result is not None:
            raise ValueError(f"result must be omitted when status is '{self.status}'")
        if self.status in (ToolExecutionStatus.FAILED, ToolExecutionStatus.DENIED) and self.error is None:
            raise ValueError(f"error is required when status is '{self.status}'")
        return self


class GmailListRecentToolResult(_ToolResultBase):
    tool: Literal["gmail.list_recent"]
    result: ListRecentEmailResult | None = None


class GmailCreateDraftToolResult(_ToolResultBase):
    tool: Literal["gmail.create_draft"]
    result: CreateDraftResult | None = None


class CalendarFindMutualSlotToolResult(_ToolResultBase):
    tool: Literal["calendar.find_mutual_slot"]
    result: FindMutualSlotResult | None = None


class CalendarCreateEventToolResult(_ToolResultBase):
    tool: Literal["calendar.create_event"]
    result: CreateEventResult | None = None


class MessagingSendMessageToolResult(_ToolResultBase):
    tool: Literal["messaging.send_message"]
    result: SendMessageResult | None = None


ToolResultRequest = Annotated[
    GmailListRecentToolResult
    | GmailCreateDraftToolResult
    | CalendarFindMutualSlotToolResult
    | CalendarCreateEventToolResult
    | MessagingSendMessageToolResult,
    Field(discriminator="tool"),
]

TOOL_RESULT_MODELS: Mapping[ToolName, type[_ToolResultBase]] = MappingProxyType(
    {
        ToolName.GMAIL_LIST_RECENT: GmailListRecentToolResult,
        ToolName.GMAIL_CREATE_DRAFT: GmailCreateDraftToolResult,
        ToolName.CALENDAR_FIND_MUTUAL_SLOT: CalendarFindMutualSlotToolResult,
        ToolName.CALENDAR_CREATE_EVENT: CalendarCreateEventToolResult,
        ToolName.MESSAGING_SEND_MESSAGE: MessagingSendMessageToolResult,
    }
)

# --------------------------------------------------------------------------- response


class TurnStatus(StrEnum):
    COMPLETED = "completed"  # reply_text is final for this turn; nothing to execute
    ACTION_REQUIRED = "action_required"  # execute / seek approval for every action proposal
    AWAITING_RESULTS = "awaiting_results"  # other proposals of this turn are still unresolved


class ResponseMetrics(StrictModel):
    total_ms: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class AgentResponse(StrictModel):
    """Returned by both agent endpoints."""

    request_id: UUID
    conversation_id: UUID
    status: TurnStatus
    reply_text: str = Field(max_length=8000)
    action_proposals: list[ActionProposal] = Field(default_factory=list, max_length=8)
    pending_proposal_ids: list[UUID] = Field(
        default_factory=list, description="Every unresolved proposal of this turn."
    )
    cancelled_proposal_ids: list[UUID] = Field(
        default_factory=list,
        description="Proposals withdrawn by the engine. Do not execute them; withdraw approval prompts.",
    )
    model: str = Field(min_length=1, max_length=200, description="Local model that produced the reply.")
    mode: Literal["local"] = "local"
    metrics: ResponseMetrics | None = None

    @model_validator(mode="after")
    def _check_status(self) -> AgentResponse:
        if self.status == TurnStatus.AWAITING_RESULTS:
            if self.action_proposals:
                raise ValueError("awaiting_results carries no new action proposals")
            if not self.pending_proposal_ids:
                raise ValueError("awaiting_results requires pending_proposal_ids")
            return self
        if not self.reply_text:
            raise ValueError(f"reply_text is required when status is '{self.status}'")
        if self.status == TurnStatus.COMPLETED:
            if self.action_proposals or self.pending_proposal_ids:
                raise ValueError("a completed turn has no action or pending proposals")
            return self
        if not self.action_proposals:
            raise ValueError("action_required needs at least one action proposal")
        missing = {p.proposal_id for p in self.action_proposals} - set(self.pending_proposal_ids)
        if missing:
            raise ValueError("every action proposal must also be listed in pending_proposal_ids")
        return self


ACTION_PROPOSAL_ADAPTER: TypeAdapter[Any] = TypeAdapter(ActionProposal)
TOOL_RESULT_ADAPTER: TypeAdapter[Any] = TypeAdapter(ToolResultRequest)
