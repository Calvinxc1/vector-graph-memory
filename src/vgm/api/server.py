"""FastAPI server with OpenAI-compatible API for Open WebUI integration."""

import os
from typing import Dict, List, Optional, Literal, cast
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from gremlin_python.driver import client as gremlin_client  # type: ignore[import-untyped]
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from dotenv import load_dotenv

from .. import __version__ as PACKAGE_VERSION
from ..MemoryAgent import MemoryAgent
from ..config import MemoryConfig, MemoryTriggerConfig, AuditConfig


# Load environment variables
load_dotenv()


# --- OpenAI-compatible API Models ---


class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""

    role: str  # system, user, assistant
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    user: Optional[str] = None  # Used as session_id


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "vector-graph-memory"


class ModelList(BaseModel):
    """List of available models."""

    object: str = "list"
    data: List[ModelInfo]


# --- Global state ---


class AppState:
    """Application state container."""

    agent: Optional[MemoryAgent] = None
    qdrant: Optional[QdrantClient] = None
    janus: Optional[gremlin_client.Client] = None
    session_proposals: Dict[str, List[str]] = {}  # session_id -> list of proposal_ids


state = AppState()


# --- Lifecycle management ---


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize and cleanup resources."""
    # Startup
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("🚀 Initializing Vector Graph Memory API...")

    # Connect to databases
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
    janusgraph_host = os.getenv("JANUSGRAPH_HOST", "localhost")
    janusgraph_port = int(os.getenv("JANUSGRAPH_PORT", "8182"))

    state.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
    state.janus = gremlin_client.Client(
        f"ws://{janusgraph_host}:{janusgraph_port}/gremlin", "g"
    )

    print(f"  ✓ Connected to Qdrant at {qdrant_host}:{qdrant_port}")
    print(f"  ✓ Connected to JanusGraph at {janusgraph_host}:{janusgraph_port}")

    # Initialize memory agent
    embedding_model = OpenAIEmbeddingModel("text-embedding-3-small")
    llm_model = os.getenv("LLM_MODEL", "openai:gpt-4o-mini")

    memory_config = MemoryConfig(
        use_case_description=os.getenv(
            "MEMORY_USE_CASE", "General purpose conversational memory"
        ),
        memory_threshold_description=os.getenv(
            "MEMORY_THRESHOLD",
            "Store important facts, decisions, and context from conversations",
        ),
        project_id=os.getenv("PROJECT_ID", "default"),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
    )

    # Parse trigger mode with type safety
    trigger_mode = os.getenv("TRIGGER_MODE", "ai_determined")
    valid_trigger_modes: tuple[Literal["phrase", "interval", "ai_determined"], ...] = (
        "phrase",
        "interval",
        "ai_determined",
    )
    if trigger_mode not in valid_trigger_modes:
        trigger_mode = "ai_determined"

    # Parse trigger interval with safety for empty strings
    trigger_interval_str = os.getenv("TRIGGER_INTERVAL", "0")
    trigger_interval = int(trigger_interval_str) if trigger_interval_str else 0

    trigger_config = MemoryTriggerConfig(
        mode=cast(Literal["phrase", "interval", "ai_determined"], trigger_mode),
        trigger_phrase=os.getenv("TRIGGER_PHRASE"),
        message_interval=trigger_interval or None,
    )

    # Parse audit backend with type safety
    audit_backend = os.getenv("AUDIT_BACKEND", "jsonl")
    valid_audit_backends: tuple[Literal["jsonl", "mongodb"], ...] = ("jsonl", "mongodb")
    if audit_backend not in valid_audit_backends:
        audit_backend = "jsonl"

    audit_config = AuditConfig(
        backend=cast(Literal["jsonl", "mongodb"], audit_backend),
        log_dir=os.getenv("AUDIT_LOG_DIR", "./logs"),
    )

    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful AI assistant with persistent memory capabilities.",
    )

    state.agent = MemoryAgent(
        qdrant_client=state.qdrant,
        janus_client=state.janus,
        embedding_model=embedding_model,
        llm_model=llm_model,
        system_prompt=system_prompt,
        memory_config=memory_config,
        trigger_config=trigger_config,
        audit_config=audit_config,
    )

    print("  ✓ Memory Agent initialized")
    print(f"    - Model: {llm_model}")
    print(f"    - Project: {memory_config.project_id}")
    print(f"    - Trigger: {trigger_config.mode}")
    print("✓ API ready on http://0.0.0.0:8000")

    yield

    # Shutdown
    print("\n🛑 Shutting down...")
    if state.janus:
        state.janus.close()
    print("✓ Cleanup complete")


# --- FastAPI app ---

app = FastAPI(
    title="Vector Graph Memory API",
    description="OpenAI-compatible API with persistent vector-graph memory",
    version=PACKAGE_VERSION,
    lifespan=lifespan,
)


# --- Endpoints ---


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "vector-graph-memory-api",
        "version": PACKAGE_VERSION,
    }


@app.get("/v1/models")
async def list_models() -> ModelList:
    """List available models (OpenAI compatible)."""
    return ModelList(
        data=[
            ModelInfo(
                id="vector-graph-memory",
                created=int(datetime.now().timestamp()),
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint."""
    if not state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # Extract messages
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    session_id = request.user or "default"

    # Build conversation context from message history
    # Format: "Previous context:\nUser: ...\nAssistant: ...\n\nCurrent question: ..."
    if len(request.messages) > 1:
        # Multi-turn conversation - include history for context
        context_parts = ["Here's our conversation so far:"]
        for msg in request.messages[:-1]:
            role = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role}: {msg.content}")
        context_parts.append(f"\nCurrent question: {request.messages[-1].content}")
        user_message = "\n".join(context_parts)
    else:
        # Single message
        user_message = request.messages[-1].content

    # Run agent
    try:
        result = state.agent.run(user_message, session_id=session_id)
        assistant_response = result.output

        # Track any new proposals for this session
        current_proposals = set(state.agent.pending_proposals.keys())
        session_proposals = set(state.session_proposals.get(session_id, []))
        new_proposals = current_proposals - session_proposals

        if new_proposals:
            state.session_proposals[session_id] = list(current_proposals)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Build response
    return ChatCompletionResponse(
        id=f"chatcmpl-{session_id}-{int(datetime.now().timestamp())}",
        created=int(datetime.now().timestamp()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=assistant_response),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=len(user_message.split()),  # Rough estimate
            completion_tokens=len(assistant_response.split()),
            total_tokens=len(user_message.split()) + len(assistant_response.split()),
        ),
    )


@app.post("/memory/confirm/{session_id}/{proposal_id}")
async def confirm_memory_proposal(
    session_id: str,
    proposal_id: str,
    action: str = "add_new",
    update_node_id: Optional[str] = None,
):
    """Confirm a memory proposal.

    Args:
        session_id: Session identifier
        proposal_id: Proposal identifier from agent
        action: 'add_new', 'update_existing', or 'cancel'
        update_node_id: Node ID to update (if action='update_existing')
    """
    if not state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    result = state.agent.confirm_memory_addition(
        proposal_id=proposal_id,
        action=action,
        update_node_id=update_node_id,
    )

    # Update session proposals
    if session_id in state.session_proposals:
        state.session_proposals[session_id] = list(state.agent.pending_proposals.keys())

    return {"status": "ok", "message": result}


@app.get("/memory/proposals/{session_id}")
async def get_pending_proposals(session_id: str):
    """Get pending memory proposals for a session."""
    if not state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session_proposal_ids = state.session_proposals.get(session_id, [])
    proposals = {}

    for proposal_id in session_proposal_ids:
        if proposal_id in state.agent.pending_proposals:
            proposals[proposal_id] = state.agent.pending_proposals[proposal_id]

    return {"session_id": session_id, "proposals": proposals}


@app.get("/memory/audit/{session_id}")
async def get_session_audit(session_id: str, limit: int = 50):
    """Get audit log for a session."""
    if not state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    entries = state.agent.get_audit_history(session_id=session_id, limit=limit)

    return {
        "session_id": session_id,
        "entries": [
            {
                "timestamp": entry.timestamp.isoformat(),
                "operation": entry.operation_type,
                "summary": entry.summary,
                "entities": entry.affected_entities,
            }
            for entry in entries
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
