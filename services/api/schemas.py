"""API request and response schemas.

These are the public-facing schemas that the frontend consumes.
They differ from the model service schemas to decouple the API
contract from model internals.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class PrecisionSettings(BaseModel):
    """User-facing precision controls (the "mixing board")."""

    logic: float = Field(default=1.0, gt=0, le=10.0)
    eq: float = Field(default=1.0, gt=0, le=10.0)


class ChatMessageRequest(BaseModel):
    """A message sent by the user."""

    content: str = Field(..., min_length=1, max_length=4096)
    precision: PrecisionSettings = Field(default_factory=PrecisionSettings)
    return_deliberation: bool = False


class DeliberationMetadata(BaseModel):
    """Deliberation info returned to the frontend."""

    steps: int
    module_precisions: dict[str, float]
    converged: bool
    final_entropy: float


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    deliberation: DeliberationMetadata | None = None


class ChatResponse(BaseModel):
    """Response to a chat message."""

    message: ChatMessage
    conversation_id: str


class ConversationSummary(BaseModel):
    """Brief info about a conversation for listing."""

    id: str
    title: str
    created_at: datetime
    message_count: int


class ConversationDetail(BaseModel):
    """Full conversation with all messages."""

    id: str
    title: str
    messages: list[ChatMessage]
    created_at: datetime
