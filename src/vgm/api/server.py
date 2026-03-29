"""FastAPI server with OpenAI-compatible API for Open WebUI integration."""

import asyncio
import os
import logging
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
from ..rag import (
    ConversationTurn,
    DEFAULT_EVAL_SOURCE_DIR,
    DEFAULT_EVAL_SUITE_PATH,
    DspyArtifactStore,
    DspyCompileManager,
    DspyModelIdentity,
    DspyRagEvalJudge,
    DspyRagSynthesizer,
    DspyRunLogger,
    RagContextBuilder,
    RubricRagEvaluator,
    build_evaluation_policy_key,
    build_dspy_lm,
    normalize_dspy_model_name,
)


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
    rag_context_builder: Optional[RagContextBuilder] = None
    rag_context_enabled: bool = False
    rag_synthesizer: Optional[DspyRagSynthesizer] = None
    rag_synthesis_enabled: bool = False
    rag_compile_manager: Optional[DspyCompileManager] = None
    rag_compile_task: Optional[asyncio.Task[None]] = None
    qdrant: Optional[QdrantClient] = None
    janus: Optional[gremlin_client.Client] = None
    session_proposals: Dict[str, List[str]] = {}  # session_id -> list of proposal_ids


state = AppState()
logger = logging.getLogger(__name__)


_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


# --- Lifecycle management ---


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize and cleanup resources."""
    # Startup
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
    state.rag_context_enabled = os.getenv("RAG_CONTEXT_ENABLED", "").lower() in _TRUTHY_ENV_VALUES
    state.rag_synthesis_enabled = os.getenv(
        "RAG_DSPY_SYNTHESIS_ENABLED", ""
    ).lower() in _TRUTHY_ENV_VALUES
    if state.rag_context_enabled or state.rag_synthesis_enabled:
        state.rag_context_builder = RagContextBuilder(
            store=state.agent.store,
            memory_config=memory_config,
        )
    if state.rag_synthesis_enabled:
        dspy_model_name = os.getenv("DSPY_MODEL_NAME") or normalize_dspy_model_name(llm_model)
        try:
            dspy_lm = build_dspy_lm(
                llm_model,
                model_name_override=os.getenv("DSPY_MODEL_NAME"),
                api_key=os.getenv("DSPY_API_KEY"),
                api_base=os.getenv("DSPY_API_BASE"),
                model_type=os.getenv("DSPY_MODEL_TYPE"),
            )
            state.rag_synthesizer = DspyRagSynthesizer.from_lm(dspy_lm)
            compile_enabled = os.getenv("RAG_DSPY_COMPILE_ENABLED", "").lower() in _TRUTHY_ENV_VALUES
            auto_compile_enabled = os.getenv(
                "RAG_DSPY_AUTO_COMPILE_ENABLED", ""
            ).lower() in _TRUTHY_ENV_VALUES
            if compile_enabled:
                try:
                    scoring_mode = os.getenv(
                        "RAG_DSPY_EVAL_SCORING_MODE", "deterministic"
                    ).lower()
                    if scoring_mode not in {"deterministic", "hybrid"}:
                        scoring_mode = "deterministic"
                    judge_model = os.getenv("RAG_DSPY_JUDGE_MODEL", llm_model)
                    judge_model_name = os.getenv("RAG_DSPY_JUDGE_MODEL_NAME") or normalize_dspy_model_name(
                        judge_model
                    )
                    judge_model_version = os.getenv("RAG_DSPY_JUDGE_MODEL_VERSION")
                    judge_lm = None
                    if scoring_mode == "hybrid":
                        judge_lm = build_dspy_lm(
                            judge_model,
                            model_name_override=os.getenv("RAG_DSPY_JUDGE_MODEL_NAME"),
                            api_key=os.getenv("RAG_DSPY_JUDGE_API_KEY")
                            or os.getenv("DSPY_API_KEY"),
                            api_base=os.getenv("RAG_DSPY_JUDGE_API_BASE")
                            or os.getenv("DSPY_API_BASE"),
                            model_type=os.getenv("RAG_DSPY_JUDGE_MODEL_TYPE")
                            or os.getenv("DSPY_MODEL_TYPE"),
                        )
                    evaluator = RubricRagEvaluator.from_suite(
                        suite_path=os.getenv(
                            "RAG_DSPY_EVAL_SUITE_PATH", str(DEFAULT_EVAL_SUITE_PATH)
                        ),
                        source_dir=os.getenv(
                            "RAG_DSPY_EVAL_SOURCE_DIR", str(DEFAULT_EVAL_SOURCE_DIR)
                        ),
                        use_case_description=memory_config.use_case_description,
                        project_id=memory_config.project_id,
                        judge=DspyRagEvalJudge.from_lm(judge_lm)
                        if judge_lm is not None
                        else None,
                        scoring_mode=cast(Literal["deterministic", "hybrid"], scoring_mode),
                    )
                    model_identity = DspyModelIdentity.from_model_name(
                        dspy_model_name,
                        model_version=os.getenv("DSPY_MODEL_VERSION"),
                        api_base=os.getenv("DSPY_API_BASE"),
                        model_type=os.getenv("DSPY_MODEL_TYPE"),
                        retrieval_schema_version=os.getenv(
                            "RAG_RETRIEVAL_SCHEMA_VERSION", "1"
                        ),
                        synthesis_program_version=os.getenv(
                            "RAG_DSPY_PROGRAM_VERSION", "1"
                        ),
                        eval_suite_id=evaluator.suite_id,
                        evaluation_policy_key=build_evaluation_policy_key(
                            scoring_mode,
                            judge_model_name=judge_model_name,
                            judge_model_version=judge_model_version,
                        ),
                    )
                    state.rag_compile_manager = DspyCompileManager(
                        lm=dspy_lm,
                        identity=model_identity,
                        evaluator=evaluator,
                        artifact_store=DspyArtifactStore(
                            base_dir=os.getenv(
                                "RAG_DSPY_CACHE_DIR", ".vgm/dspy_artifacts"
                            )
                        ),
                        auto_compile=auto_compile_enabled,
                        run_logger=DspyRunLogger(
                            base_dir=os.getenv("RAG_DSPY_RUN_LOG_DIR", ".vgm/dspy_runs")
                        ),
                    )
                    cached_synthesizer = state.rag_compile_manager.load_cached_synthesizer()
                    if cached_synthesizer is not None:
                        state.rag_synthesizer = cached_synthesizer
                except Exception:
                    logger.exception(
                        "[RAG] Failed to initialize DSPy compile manager; continuing with baseline synthesizer"
                    )
                    state.rag_compile_manager = None
        except Exception:
            logger.exception("[RAG] Failed to initialize baseline DSPy synthesizer")
            state.rag_synthesis_enabled = False
            state.rag_synthesizer = None
            state.rag_compile_manager = None

    print("  ✓ Memory Agent initialized")
    print(f"    - Model: {llm_model}")
    print(f"    - Project: {memory_config.project_id}")
    print(f"    - Trigger: {trigger_config.mode}")
    print(f"    - RAG context seam enabled: {state.rag_context_enabled}")
    print(f"    - DSPy synthesis enabled: {state.rag_synthesis_enabled}")
    print(f"    - DSPy compile manager enabled: {state.rag_compile_manager is not None}")
    print("✓ API ready on http://0.0.0.0:8000")

    yield

    # Shutdown
    print("\n🛑 Shutting down...")
    if state.rag_compile_task and not state.rag_compile_task.done():
        state.rag_compile_task.cancel()
    if state.janus:
        state.janus.close()
    print("✓ Cleanup complete")


# --- FastAPI app ---

app = FastAPI(
    title="Vector Graph Memory API",
    description="OpenAI-compatible API with persistent vector-graph memory",
    version=PACKAGE_VERSION,
    lifespan=lifespan,
    root_path=os.getenv("API_ROOT_PATH", ""),
)


async def _run_background_compile_job() -> None:
    """Compile a candidate DSPy program without blocking the request path."""

    compile_manager = state.rag_compile_manager
    if compile_manager is None:
        state.rag_compile_task = None
        return

    try:
        outcome = await asyncio.to_thread(compile_manager.compile_and_promote)
        logger.info(
            "[RAG] Background compile finished promoted=%s baseline_score=%.3f compiled_score=%.3f reason=%s",
            outcome.promoted,
            outcome.baseline_report.total_score,
            outcome.compiled_report.total_score,
            outcome.reason,
        )
        if outcome.promoted:
            compiled_synthesizer = compile_manager.load_cached_synthesizer()
            if compiled_synthesizer is not None:
                state.rag_synthesizer = compiled_synthesizer
    except Exception:
        logger.exception("[RAG] Background DSPy compile failed")
    finally:
        state.rag_compile_task = None


def _maybe_start_background_compile() -> None:
    """Queue one automatic compile attempt for the current runtime."""

    compile_manager = state.rag_compile_manager
    if compile_manager is None or state.rag_compile_task is not None:
        return
    if not compile_manager.begin_auto_compile():
        return
    state.rag_compile_task = asyncio.create_task(_run_background_compile_job())


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

    rag_context = None
    if (state.rag_context_enabled or state.rag_synthesis_enabled) and state.rag_context_builder:
        try:
            rag_messages = [
                ConversationTurn(role=msg.role, content=msg.content)
                for msg in request.messages
            ]
            rag_context = state.rag_context_builder.build_from_messages(
                messages=rag_messages,
                session_id=session_id,
            )
            logger.info(
                "[RAG] Built context for session=%s passages=%d history_turns=%d",
                session_id,
                len(rag_context.retrieved_passages),
                len(rag_context.conversation_history),
            )
        except Exception:
            logger.exception(
                "[RAG] Failed to build deterministic context for session=%s",
                session_id,
            )
            rag_context = None

    if state.rag_synthesis_enabled and state.rag_compile_manager:
        _maybe_start_background_compile()

    if state.rag_synthesis_enabled and state.rag_synthesizer and rag_context:
        try:
            rag_result = state.rag_synthesizer.synthesize(rag_context)
            logger.info(
                "[RAG] Synthesized answer for session=%s backend=%s cited_sources=%d abstain=%s",
                session_id,
                rag_result.backend,
                len(rag_result.cited_source_ids),
                rag_result.abstain,
            )
            if rag_result.answer:
                return ChatCompletionResponse(
                    id=f"chatcmpl-{session_id}-{int(datetime.now().timestamp())}",
                    created=int(datetime.now().timestamp()),
                    model=request.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=rag_result.answer),
                            finish_reason="stop",
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=len(rag_context.retrieval_query.split()),
                        completion_tokens=len(rag_result.answer.split()),
                        total_tokens=(
                            len(rag_context.retrieval_query.split())
                            + len(rag_result.answer.split())
                        ),
                    ),
                )
            logger.warning(
                "[RAG] Baseline DSPy synthesizer returned an empty answer for session=%s; falling back",
                session_id,
            )
        except Exception:
            logger.exception(
                "[RAG] DSPy synthesis failed for session=%s; falling back to MemoryAgent",
                session_id,
            )

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
