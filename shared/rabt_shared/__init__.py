"""Rabt shared API contract between the AI engine and the integration backend.

The Pydantic models in this package are the source of truth. ``docs/API_CONTRACT.md``
explains them; ``shared/jsonschema/`` exports them for non-Python consumers.
"""

from rabt_shared.agent import (
    ACTION_PROPOSAL_ADAPTER,
    PROPOSAL_MODELS,
    TOOL_RESULT_ADAPTER,
    TOOL_RESULT_MODELS,
    ActionProposal,
    AgentRespondRequest,
    AgentResponse,
    ResponseMetrics,
    ToolError,
    ToolExecutionStatus,
    ToolResultRequest,
    TurnStatus,
)
from rabt_shared.audio import (
    AUDIO_MIME_TYPES,
    MAX_AUDIO_DURATION_S,
    MAX_AUDIO_UPLOAD_BYTES,
    AudioFormat,
    SpeakRequest,
    TranscriptionResponse,
    TranscriptSegment,
)
from rabt_shared.common import (
    DEFAULT_TIMEZONE,
    MAX_MESSAGE_CHARS,
    AccountId,
    Channel,
    Role,
    StrictModel,
    Timezone,
)
from rabt_shared.errors import ERROR_HTTP_STATUS, ErrorCode, ErrorInfo, ErrorResponse
from rabt_shared.status import (
    AgentStatusResponse,
    ComponentState,
    GpuStatus,
    HealthResponse,
    LlmStatus,
    SttStatus,
    TtsStatus,
)
from rabt_shared.tools import TOOL_SPECS, ToolName, ToolSpec, UnknownToolError, get_tool_spec
from rabt_shared.version import CONTRACT_VERSION

__all__ = [
    "ACTION_PROPOSAL_ADAPTER",
    "AUDIO_MIME_TYPES",
    "CONTRACT_VERSION",
    "DEFAULT_TIMEZONE",
    "ERROR_HTTP_STATUS",
    "MAX_AUDIO_DURATION_S",
    "MAX_AUDIO_UPLOAD_BYTES",
    "MAX_MESSAGE_CHARS",
    "PROPOSAL_MODELS",
    "TOOL_RESULT_ADAPTER",
    "TOOL_RESULT_MODELS",
    "TOOL_SPECS",
    "AccountId",
    "ActionProposal",
    "AgentRespondRequest",
    "AgentResponse",
    "AgentStatusResponse",
    "AudioFormat",
    "Channel",
    "ComponentState",
    "ErrorCode",
    "ErrorInfo",
    "ErrorResponse",
    "GpuStatus",
    "HealthResponse",
    "LlmStatus",
    "ResponseMetrics",
    "Role",
    "SpeakRequest",
    "StrictModel",
    "SttStatus",
    "Timezone",
    "ToolError",
    "ToolExecutionStatus",
    "ToolName",
    "ToolResultRequest",
    "ToolSpec",
    "TranscriptSegment",
    "TranscriptionResponse",
    "TtsStatus",
    "TurnStatus",
    "UnknownToolError",
    "get_tool_spec",
]
