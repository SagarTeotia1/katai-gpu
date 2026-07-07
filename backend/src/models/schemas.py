from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Message role: system | user | assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for /api/chat and /api/chat/stream."""

    messages: list[Message] = Field(..., min_length=1, description="Conversation history")
    max_tokens: int = Field(default=4096, ge=1, le=32768, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    stream: bool = Field(default=False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Response body for non-streaming /api/chat."""

    content: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model ID that generated the response")
    tokens_used: int = Field(..., description="Total tokens consumed (prompt + completion)")


class StreamChunk(BaseModel):
    """A single SSE payload chunk for /api/chat/stream."""

    content: str = Field(default="", description="Token text fragment")
    done: bool = Field(default=False, description="True on the final (empty) sentinel chunk")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    ollama_reachable: bool
    model_id: str
