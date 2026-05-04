"""FastAPI server with OpenAI-compatible API for Open WebUI integration."""

import asyncio
import json
import os
import logging
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Literal, cast
from datetime import datetime
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from gremlin_python.driver import client as gremlin_client  # type: ignore[import-untyped]
from dotenv import load_dotenv

from .. import __version__ as PACKAGE_VERSION
from ..MemoryAgent import MemoryAgent, MemoryRunTrace
from ..config import MemoryConfig, MemoryTriggerConfig, AuditConfig
from ..model_provider import (
    build_chat_model_from_env,
    build_embedding_model_from_env,
    chat_model_name_from_env,
    embedding_model_name_from_env,
)
from ..rag import (
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
from ..rules import (
    LivePilotRulingEngine,
    LivePilotRulingInspection,
    RulesRulingRequest,
    RulesRulingResult,
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


class TraceCandidateSummary(BaseModel):
    """Compact case-candidate summary for UI traces and request logs."""

    question_id: str
    question_score: float
    evidence_score: int
    matched_reference_count: int


class RequestTraceLogEntry(BaseModel):
    """Persistent request-scoped diagnostic log entry."""

    request_id: str
    session_id: str
    model: str
    route: str
    stream: bool
    created_at: str
    trace: Dict[str, Any] = Field(default_factory=dict)


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
    rules_ruling_engine: Optional[LivePilotRulingEngine] = None
    qdrant: Optional[QdrantClient] = None
    janus: Optional[gremlin_client.Client] = None
    session_proposals: Dict[str, List[str]] = {}  # session_id -> list of proposal_ids


state = AppState()
logger = logging.getLogger(__name__)
_REQUEST_TRACE_WRITE_LOCK = Lock()

MEMORY_CHAT_MODEL_ID = "vector-graph-memory"
RULES_CHAT_MODEL_ID = "seti-rules-lawyer"
DEFAULT_API_TRACE_LOG_PATH = "./logs/api"
RULES_CASE_SELECTION_THRESHOLD = 0.14
_THINKING_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


_QDRANT_INSPECTOR_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Qdrant Inspector</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { color-scheme: light; --bg: #f5f3ef; --panel: #fffdf8; --ink: #1f2421; --muted: #66706b; --accent: #0d5c63; --line: #d8d3c9; }
    body { margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: linear-gradient(180deg, #f7f4ec 0%%, #eef3f2 100%%); color: var(--ink); }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    h1, h2 { margin: 0 0 12px; }
    p { color: var(--muted); }
    .grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 18px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.04); }
    label { display: block; font-size: 0.9rem; color: var(--muted); margin-bottom: 6px; }
    input, select, button { width: 100%%; box-sizing: border-box; font: inherit; padding: 10px 12px; border-radius: 10px; border: 1px solid var(--line); background: white; }
    button { background: var(--accent); color: white; border: none; cursor: pointer; font-weight: 600; }
    button.secondary { background: #e7efee; color: var(--accent); border: 1px solid #bdd5d2; }
    .actions { display: flex; gap: 10px; margin-top: 14px; }
    .actions a { flex: 1; text-decoration: none; }
    pre { background: #182022; color: #f4f7f6; padding: 14px; border-radius: 12px; overflow: auto; min-height: 220px; }
    table { width: 100%%; border-collapse: collapse; }
    th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
    code { font-family: "IBM Plex Mono", monospace; }
  </style>
</head>
<body>
  <main>
    <div class="panel">
      <h1>Qdrant Inspector</h1>
      <p>Use the raw Qdrant dashboard for the full vendor UI, or inspect live collections and points directly here.</p>
      <div class="actions">
        <a href="./ui/dashboard/" target="_blank" rel="noreferrer"><button type="button">Open Raw Qdrant Dashboard</button></a>
        <a href="./api/collections" target="_blank" rel="noreferrer"><button class="secondary" type="button">Open Collections JSON</button></a>
      </div>
    </div>
    <div class="grid" style="margin-top: 16px;">
      <section class="panel">
        <h2>Collections</h2>
        <table>
          <thead><tr><th>Name</th><th>Points</th><th>Vectors</th></tr></thead>
          <tbody id="collection-table"></tbody>
        </table>
      </section>
      <section class="panel">
        <h2>Scroll Points</h2>
        <label for="collection-select">Collection</label>
        <select id="collection-select"></select>
        <label for="point-limit" style="margin-top: 12px;">Limit</label>
        <input id="point-limit" type="number" min="1" max="100" value="10">
        <div class="actions">
          <button type="button" id="load-points">Load Points</button>
        </div>
      </section>
    </div>
    <section class="panel" style="margin-top: 16px;">
      <h2>Output</h2>
      <pre id="output">Loading collections...</pre>
    </section>
  </main>
  <script>
    const tableBody = document.getElementById("collection-table");
    const collectionSelect = document.getElementById("collection-select");
    const output = document.getElementById("output");

    async function loadCollections() {
      const response = await fetch("./api/collections");
      const data = await response.json();
      tableBody.innerHTML = "";
      collectionSelect.innerHTML = "";
      for (const item of data.collections) {
        const row = document.createElement("tr");
        row.innerHTML = `<td><code>${item.name}</code></td><td>${item.points_count ?? ""}</td><td>${item.vectors_count ?? ""}</td>`;
        tableBody.appendChild(row);
        const option = document.createElement("option");
        option.value = item.name;
        option.textContent = item.name;
        collectionSelect.appendChild(option);
      }
      output.textContent = JSON.stringify(data, null, 2);
    }

    async function loadPoints() {
      const collection = collectionSelect.value;
      const limit = Number(document.getElementById("point-limit").value || "10");
      const response = await fetch(`./api/collections/${encodeURIComponent(collection)}/points?limit=${limit}`);
      output.textContent = JSON.stringify(await response.json(), null, 2);
    }

    document.getElementById("load-points").addEventListener("click", loadPoints);
    loadCollections().catch((error) => {
      output.textContent = `Failed to load collections: ${error}`;
    });
  </script>
</body>
</html>
"""


_JANUS_INSPECTOR_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>JanusGraph Inspector</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { color-scheme: light; --bg: #f7f2eb; --panel: #fffdfa; --ink: #1e2328; --muted: #66707a; --accent: #aa3a2b; --line: #ddd4ca; }
    body { margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: radial-gradient(circle at top right, #f8e5da, transparent 30%%), linear-gradient(180deg, #f8f3ed 0%%, #eef2f0 100%%); color: var(--ink); }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    h1, h2 { margin: 0 0 12px; }
    p { color: var(--muted); }
    .grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 18px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.04); }
    label { display: block; font-size: 0.9rem; color: var(--muted); margin-bottom: 6px; }
    input, select, button { width: 100%%; box-sizing: border-box; font: inherit; padding: 10px 12px; border-radius: 10px; border: 1px solid var(--line); background: white; }
    button { background: var(--accent); color: white; border: none; cursor: pointer; font-weight: 600; }
    pre { background: #182022; color: #f4f7f6; padding: 14px; border-radius: 12px; overflow: auto; min-height: 240px; }
    .actions { display: flex; gap: 10px; margin-top: 14px; }
    .summary { display: grid; gap: 12px; grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .metric { border: 1px solid var(--line); border-radius: 12px; padding: 12px; background: #fff; }
    .metric strong { display: block; font-size: 1.4rem; margin-top: 6px; }
  </style>
</head>
<body>
  <main>
    <div class="panel">
      <h1>JanusGraph Inspector</h1>
      <p>This page queries the live Gremlin graph through the API process so you can inspect vertices, edges, and seed metadata without exposing raw Gremlin directly in the browser.</p>
    </div>
    <div class="panel" style="margin-top: 16px;">
      <h2>Summary</h2>
      <div class="summary">
        <div class="metric">Vertices<strong id="vertex-count">-</strong></div>
        <div class="metric">Edges<strong id="edge-count">-</strong></div>
        <div class="metric">Recent Sample<strong id="sample-count">10</strong></div>
      </div>
    </div>
    <div class="grid" style="margin-top: 16px;">
      <section class="panel">
        <h2>Vertices</h2>
        <label for="vertex-property">Property</label>
        <select id="vertex-property">
          <option value="node_id">node_id</option>
          <option value="logical_node_id">logical_node_id</option>
          <option value="seed_id">seed_id</option>
          <option value="project_id">project_id</option>
        </select>
        <label for="vertex-value" style="margin-top: 12px;">Value</label>
        <input id="vertex-value" type="text" placeholder="Exact match value">
        <label for="vertex-limit" style="margin-top: 12px;">Limit</label>
        <input id="vertex-limit" type="number" min="1" max="100" value="10">
        <div class="actions">
          <button type="button" id="sample-vertices">Sample Vertices</button>
          <button type="button" id="search-vertices">Search Vertices</button>
        </div>
      </section>
      <section class="panel">
        <h2>Edges</h2>
        <label for="edge-property">Property</label>
        <select id="edge-property">
          <option value="edge_id">edge_id</option>
          <option value="logical_edge_id">logical_edge_id</option>
          <option value="seed_id">seed_id</option>
          <option value="relationship_type">relationship_type</option>
          <option value="project_id">project_id</option>
        </select>
        <label for="edge-value" style="margin-top: 12px;">Value</label>
        <input id="edge-value" type="text" placeholder="Exact match value">
        <label for="edge-limit" style="margin-top: 12px;">Limit</label>
        <input id="edge-limit" type="number" min="1" max="100" value="10">
        <div class="actions">
          <button type="button" id="sample-edges">Sample Edges</button>
          <button type="button" id="search-edges">Search Edges</button>
        </div>
      </section>
    </div>
    <section class="panel" style="margin-top: 16px;">
      <h2>Output</h2>
      <pre id="output">Loading graph summary...</pre>
    </section>
  </main>
  <script>
    const output = document.getElementById("output");

    async function loadSummary() {
      const response = await fetch("./api/summary");
      const data = await response.json();
      document.getElementById("vertex-count").textContent = data.vertex_count;
      document.getElementById("edge-count").textContent = data.edge_count;
      output.textContent = JSON.stringify(data, null, 2);
    }

    async function sampleVertices() {
      const limit = Number(document.getElementById("vertex-limit").value || "10");
      const response = await fetch(`./api/vertices?limit=${limit}`);
      output.textContent = JSON.stringify(await response.json(), null, 2);
    }

    async function searchVertices() {
      const property = document.getElementById("vertex-property").value;
      const value = document.getElementById("vertex-value").value;
      const limit = Number(document.getElementById("vertex-limit").value || "10");
      const response = await fetch(`./api/vertices/search?property=${encodeURIComponent(property)}&value=${encodeURIComponent(value)}&limit=${limit}`);
      output.textContent = JSON.stringify(await response.json(), null, 2);
    }

    async function sampleEdges() {
      const limit = Number(document.getElementById("edge-limit").value || "10");
      const response = await fetch(`./api/edges?limit=${limit}`);
      output.textContent = JSON.stringify(await response.json(), null, 2);
    }

    async function searchEdges() {
      const property = document.getElementById("edge-property").value;
      const value = document.getElementById("edge-value").value;
      const limit = Number(document.getElementById("edge-limit").value || "10");
      const response = await fetch(`./api/edges/search?property=${encodeURIComponent(property)}&value=${encodeURIComponent(value)}&limit=${limit}`);
      output.textContent = JSON.stringify(await response.json(), null, 2);
    }

    document.getElementById("sample-vertices").addEventListener("click", sampleVertices);
    document.getElementById("search-vertices").addEventListener("click", searchVertices);
    document.getElementById("sample-edges").addEventListener("click", sampleEdges);
    document.getElementById("search-edges").addEventListener("click", searchEdges);
    loadSummary().catch((error) => {
      output.textContent = `Failed to load summary: ${error}`;
    });
  </script>
</body>
</html>
"""


_JANUS_VERTEX_PROPERTIES = {"node_id", "logical_node_id", "seed_id", "project_id"}
_JANUS_EDGE_PROPERTIES = {
    "edge_id",
    "logical_edge_id",
    "seed_id",
    "relationship_type",
    "project_id",
}


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
    logger.info("API request trace log path: %s", _request_trace_log_path())

    print("🚀 Initializing Vector Graph Memory API...")

    # Connect to databases
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_HTTP_PORT", "8111"))
    janusgraph_host = os.getenv("JANUSGRAPH_HOST", "localhost")
    janusgraph_port = int(os.getenv("JANUSGRAPH_PORT", "8182"))

    state.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
    state.janus = gremlin_client.Client(
        f"ws://{janusgraph_host}:{janusgraph_port}/gremlin", "g"
    )

    print(f"  ✓ Connected to Qdrant at {qdrant_host}:{qdrant_port}")
    print(f"  ✓ Connected to JanusGraph at {janusgraph_host}:{janusgraph_port}")

    # Initialize memory agent
    llm_model = chat_model_name_from_env()
    chat_model = build_chat_model_from_env(llm_model)
    embedding_model_name = embedding_model_name_from_env()
    embedding_model = build_embedding_model_from_env(embedding_model_name)

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
        llm_model=chat_model,
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
    state.rules_ruling_engine = LivePilotRulingEngine(
        state.agent.store,
        project_id=memory_config.project_id,
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
    print(f"    - Embeddings: {embedding_model_name}")
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


def _escape_gremlin_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _run_in_thread(func: Any) -> Any:
    result: list[Any] = []
    exception: Exception | None = None

    def run() -> None:
        nonlocal exception
        try:
            result.append(func())
        except Exception as exc:
            exception = exc

    import threading

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    if exception is not None:
        raise exception
    if not result:
        return None
    return result[0]


def _run_gremlin(query: str) -> list[Any]:
    if state.janus is None:
        raise HTTPException(status_code=503, detail="JanusGraph client not initialized")
    return _run_in_thread(lambda: state.janus.submit(query).all().result())


def _gremlin_count(query: str) -> int:
    results = _run_gremlin(query)
    if not results:
        return 0
    return int(results[0])


def _normalize_gremlin_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_gremlin_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        if len(value) == 1:
            return _normalize_gremlin_value(value[0])
        return [_normalize_gremlin_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_gremlin_value(item) for item in value]
    return value


def _normalize_gremlin_rows(rows: list[Any]) -> list[Any]:
    return [_normalize_gremlin_value(row) for row in rows]


def _format_rule_citation(citation: Any) -> str:
    label = citation.citation_label or citation.title
    locator = citation.locator or citation.citation_short
    if locator:
        return f"{label} ({locator})"
    return label


def _format_rules_ruling_for_chat(result: RulesRulingResult) -> str:
    lines = [result.ruling]

    if result.primary_citation:
        lines.extend(("", f"Primary authority: {_format_rule_citation(result.primary_citation)}"))

    if result.modifying_citations:
        lines.append("")
        lines.append("Modifiers:")
        lines.extend(f"- {_format_rule_citation(citation)}" for citation in result.modifying_citations)

    if result.supporting_citations:
        lines.append("")
        lines.append("Supporting authority:")
        lines.extend(f"- {_format_rule_citation(citation)}" for citation in result.supporting_citations)

    if result.precedence_order:
        lines.append("")
        lines.append("Precedence:")
        lines.extend(f"{entry.order}. {entry.summary}" for entry in result.precedence_order)

    if result.uncertainty:
        lines.extend(("", f"Uncertainty: {result.uncertainty}"))

    return "\n".join(lines)


def _request_trace_log_path() -> Path:
    configured = os.getenv("API_TRACE_LOG_PATH", DEFAULT_API_TRACE_LOG_PATH)
    return Path(os.path.expanduser(configured)).resolve()


def _append_request_trace_log(entry: RequestTraceLogEntry) -> str:
    base_dir = _request_trace_log_path()
    dated_dir = base_dir / "requests" / datetime.now().strftime("%Y-%m-%d")
    log_path = dated_dir / f"{entry.request_id}.json"
    dated_dir.mkdir(parents=True, exist_ok=True)
    with _REQUEST_TRACE_WRITE_LOCK:
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(entry.model_dump(mode="json"), handle, indent=2)
            handle.write("\n")
    return str(log_path)


def _seed_origin_label(
    inspection: LivePilotRulingInspection,
    *,
    requested_seed_id: str | None,
) -> str:
    if requested_seed_id:
        return "request override"
    if inspection.seed_inference.selected_seed_id:
        return "question inference"
    if inspection.evidence.seed_id:
        return "retrieved evidence majority"
    return "unresolved"


def _candidate_summaries(
    inspection: LivePilotRulingInspection,
    *,
    limit: int = 3,
) -> list[TraceCandidateSummary]:
    return [
        TraceCandidateSummary(
            question_id=candidate.question_id,
            question_score=candidate.question_score,
            evidence_score=candidate.evidence_score,
            matched_reference_count=len(candidate.matched_reference_ids),
        )
        for candidate in inspection.candidate_cases[:limit]
    ]


def _rules_abstain_kind(
    inspection: LivePilotRulingInspection,
    result: RulesRulingResult,
) -> str | None:
    if not result.abstain:
        return None
    if inspection.premise_screen.status != "valid":
        return "invalid_premise"
    if not inspection.evidence.nodes:
        return "no_evidence"
    if inspection.selected_seed_id is None:
        return "seed_unresolved"
    if inspection.candidate_cases and inspection.selected_case is None:
        return "near_miss"
    return "unsupported"


def _format_rules_trace_for_thinking(
    inspection: LivePilotRulingInspection,
    result: RulesRulingResult,
    *,
    request_id: str,
    log_path: str,
    requested_seed_id: str | None = None,
) -> str:
    lines = ["<think>", "Trace summary:"]

    lines.append(f"- Request: {request_id}")
    lines.append(f"- Log file: {log_path}")
    lines.append("- Route: seti-rules-lawyer -> live pilot ruling")

    if inspection.selected_seed_id:
        seed_line = (
            f"- Seed: {inspection.selected_seed_id}"
            f" via {_seed_origin_label(inspection, requested_seed_id=requested_seed_id)}"
        )
        if inspection.seed_inference.selected_score > 0:
            seed_line += f" (score {inspection.seed_inference.selected_score:.2f})"
        lines.append(seed_line)
    elif inspection.seed_inference.candidates:
        lines.append(
            "- Seed inference did not clear threshold; "
            f"top score was {inspection.seed_inference.candidates[0].score:.2f}."
        )

    if inspection.issue_inference.issue_type != "unsupported_unknown":
        lines.append(
            "- Question issue: "
            f"{inspection.issue_inference.issue_type} "
            f"(confidence {inspection.issue_inference.confidence:.2f})"
        )
    if inspection.premise_screen.status != "valid":
        lines.append(
            "- Premise screen: "
            f"{inspection.premise_screen.status} "
            f"(confidence {inspection.premise_screen.confidence:.2f})"
        )
        if inspection.premise_screen.reason:
            lines.append(f"- Premise reason: {inspection.premise_screen.reason}")

    if inspection.selected_case:
        lines.append(
            "- Matched case: "
            f"{inspection.selected_case.question_id} "
            f"(question {inspection.selected_case.question_score:.2f}, "
            f"evidence {inspection.selected_case.evidence_score}, "
            f"matched refs {len(inspection.selected_case.matched_reference_ids)})"
        )
    elif inspection.candidate_cases:
        top_case = inspection.candidate_cases[0]
        lines.append(
            f"- No supported case cleared threshold {RULES_CASE_SELECTION_THRESHOLD:.2f}; "
            f"top candidate was {top_case.question_id} "
            f"(question {top_case.question_score:.2f}, evidence {top_case.evidence_score}, "
            f"matched refs {len(top_case.matched_reference_ids)})."
        )
    else:
        lines.append("- No supported case candidates were available for this question.")

    lines.append(
        "- Retrieval: "
        f"{len(inspection.evidence.retrieved_node_ids)} initial nodes, "
        f"{len(inspection.evidence.expanded_node_ids)} expanded nodes, "
        f"{len(inspection.evidence.edges)} edges traversed."
    )

    candidate_summaries = _candidate_summaries(inspection)
    if candidate_summaries:
        candidate_line = "; ".join(
            f"{candidate.question_id} (q={candidate.question_score:.2f}, "
            f"e={candidate.evidence_score}, refs={candidate.matched_reference_count})"
            for candidate in candidate_summaries
        )
        lines.append(f"- Top candidates: {candidate_line}")

    if result.primary_citation:
        lines.append(f"- Primary authority: {_format_rule_citation(result.primary_citation)}")

    if result.modifying_citations:
        modifier_labels = ", ".join(
            _format_rule_citation(citation) for citation in result.modifying_citations
        )
        lines.append(f"- Modifiers considered: {modifier_labels}")

    if inspection.evidence.source_nodes:
        source_labels = ", ".join(
            _format_rule_citation(source)
            for source in inspection.evidence.source_nodes[:3]
        )
        lines.append(f"- Retrieved sources: {source_labels}")

    if result.precedence_order:
        lines.append(f"- Control rationale: {result.precedence_order[0].summary}")

    if inspection.selected_case_issue_mismatch_reason:
        lines.append(f"- Fit gate: {inspection.selected_case_issue_mismatch_reason}")

    abstain_kind = _rules_abstain_kind(inspection, result)
    if result.abstain and result.uncertainty:
        if abstain_kind is not None:
            lines.append(f"- Abstain kind: {abstain_kind}")
        lines.append(f"- Abstain reason: {result.uncertainty}")

    lines.append("</think>")
    return "\n".join(lines)


def _format_memory_trace_for_thinking(
    *,
    request_id: str,
    log_path: str,
    route: str,
    rag_context: Any | None,
    dspy_attempted: bool,
    dspy_used: bool,
    dspy_backend: str | None,
    dspy_failure: str | None,
    memory_trace: MemoryRunTrace | None,
    new_proposal_count: int,
    final_answer_source: str,
) -> str:
    lines = ["<think>", "Trace summary:"]
    lines.append(f"- Request: {request_id}")
    lines.append(f"- Log file: {log_path}")
    lines.append(f"- Route: {route}")

    if rag_context is not None:
        lines.append(
            "- RAG context: "
            f"{len(rag_context.retrieved_passages)} passages, "
            f"{len(rag_context.conversation_history)} history turns."
        )
    else:
        lines.append("- RAG context: not built.")

    if dspy_attempted:
        if dspy_used:
            lines.append(
                f"- DSPy synthesis: returned answer via {dspy_backend or 'unknown backend'}."
            )
        elif dspy_failure:
            lines.append(f"- DSPy synthesis: failed and fell back ({dspy_failure}).")
        else:
            lines.append("- DSPy synthesis: attempted but returned no answer; fell back.")
    else:
        lines.append("- DSPy synthesis: not attempted.")

    if memory_trace is not None:
        lines.append(
            "- Memory review trigger: "
            f"{memory_trace.trigger_mode}, activated={memory_trace.memory_check_triggered}."
        )
        if memory_trace.tool_calls:
            lines.append(
                "- Tools called: "
                + ", ".join(tool.tool_name for tool in memory_trace.tool_calls)
            )
            for tool in memory_trace.tool_calls[:3]:
                args_summary = ", ".join(f"{key}={value!r}" for key, value in tool.arguments.items())
                lines.append(
                    f"- Tool detail: {tool.tool_name}({args_summary}) -> {tool.result_summary}"
                )
        else:
            lines.append("- Tools called: none.")
    else:
        lines.append("- MemoryAgent trace: unavailable.")

    if new_proposal_count:
        lines.append(f"- Pending memory proposals added: {new_proposal_count}")

    lines.append(f"- Final answer source: {final_answer_source}")
    lines.append("</think>")
    return "\n".join(lines)


def _build_chat_completion_response(
    *,
    model: str,
    session_id: str,
    prompt_text: str,
    assistant_response: str,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{session_id}-{int(datetime.now().timestamp())}",
        created=int(datetime.now().timestamp()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=assistant_response),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=len(prompt_text.split()),
            completion_tokens=len(assistant_response.split()),
            total_tokens=len(prompt_text.split()) + len(assistant_response.split()),
        ),
    )


def _build_chat_completion_chunk(
    *,
    response_id: str,
    created: int,
    model: str,
    delta: dict[str, str],
    finish_reason: str | None = None,
) -> str:
    return json.dumps(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


def _split_stream_sections(assistant_response: str) -> list[str]:
    sections = assistant_response.split("\n\n")
    if not sections:
        return [assistant_response]
    streamed_sections = [sections[0]]
    streamed_sections.extend(f"\n\n{section}" for section in sections[1:])
    return streamed_sections


def _extract_model_thinking(assistant_response: str) -> tuple[str, list[str]]:
    """Remove complete model-supplied thinking blocks from visible response text."""

    thinking_blocks = [
        match.group(1).strip()
        for match in _THINKING_BLOCK_PATTERN.finditer(assistant_response)
        if match.group(1).strip()
    ]
    if not thinking_blocks:
        return assistant_response, []

    visible_response = _THINKING_BLOCK_PATTERN.sub("", assistant_response).strip()
    return visible_response, thinking_blocks


def _append_model_thinking_to_trace(
    trace_summary: str | None,
    thinking_blocks: list[str],
) -> str | None:
    """Append model-supplied thinking below diagnostic trace details."""

    if not thinking_blocks:
        return trace_summary

    model_thinking = "\n\nModel thinking:\n" + "\n\n".join(thinking_blocks)
    if trace_summary is None:
        return f"<think>{model_thinking}\n</think>"

    closing_tag_match = re.search(r"</think>\s*$", trace_summary, flags=re.IGNORECASE)
    if closing_tag_match is None:
        return f"{trace_summary}{model_thinking}"

    return (
        trace_summary[: closing_tag_match.start()]
        + model_thinking
        + "\n"
        + trace_summary[closing_tag_match.start() :]
    )


def _build_streaming_chat_response(
    *,
    model: str,
    session_id: str,
    trace_summary: str | None,
    assistant_response: str,
) -> StreamingResponse:
    response_id = f"chatcmpl-{session_id}-{int(datetime.now().timestamp())}"
    created = int(datetime.now().timestamp())
    assistant_response, thinking_blocks = _extract_model_thinking(assistant_response)
    trace_summary = _append_model_thinking_to_trace(trace_summary, thinking_blocks)
    sections = []
    if trace_summary:
        sections.append(trace_summary)
    sections.extend(_split_stream_sections(assistant_response))

    async def event_stream():
        yield f"data: {_build_chat_completion_chunk(response_id=response_id, created=created, model=model, delta={'role': 'assistant'})}\n\n"
        for section in sections:
            yield f"data: {_build_chat_completion_chunk(response_id=response_id, created=created, model=model, delta={'content': section})}\n\n"
        yield f"data: {_build_chat_completion_chunk(response_id=response_id, created=created, model=model, delta={}, finish_reason='stop')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
    created = int(datetime.now().timestamp())
    return ModelList(
        data=[
            ModelInfo(
                id=RULES_CHAT_MODEL_ID,
                created=created,
            ),
        ]
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    session_id = request.user or "default"
    request_id = f"{session_id}-{uuid4().hex[:8]}"
    latest_user_message = next(
        (message.content for message in reversed(request.messages) if message.role == "user"),
        None,
    )
    if latest_user_message is None:
        raise HTTPException(status_code=400, detail="No user message provided")

    if request.model == RULES_CHAT_MODEL_ID:
        if state.rules_ruling_engine is None:
            raise HTTPException(status_code=503, detail="Rules ruling engine not initialized")
        try:
            rules_request = RulesRulingRequest(question=latest_user_message)
            inspection = state.rules_ruling_engine.inspect_request(rules_request)
            result = state.rules_ruling_engine.answer(rules_request)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Rules ruling error: {exc}") from exc
        assistant_response = _format_rules_ruling_for_chat(result)
        log_path = _append_request_trace_log(
            RequestTraceLogEntry(
                request_id=request_id,
                session_id=session_id,
                model=request.model,
                route="seti-rules-lawyer",
                stream=bool(request.stream),
                created_at=datetime.now().isoformat(),
                trace={
                    "question": rules_request.question,
                    "seed_origin": _seed_origin_label(
                        inspection,
                        requested_seed_id=rules_request.seed_id,
                    ),
                    "selection_threshold": RULES_CASE_SELECTION_THRESHOLD,
                    "candidate_summaries": [
                        candidate.model_dump(mode="json")
                        for candidate in _candidate_summaries(inspection)
                    ],
                    "inspection": inspection.model_dump(mode="json"),
                    "result": result.model_dump(mode="json"),
                    "abstain_kind": _rules_abstain_kind(inspection, result),
                },
            )
        )
        if request.stream:
            return _build_streaming_chat_response(
                model=request.model,
                session_id=session_id,
                trace_summary=_format_rules_trace_for_thinking(
                    inspection,
                    result,
                    request_id=request_id,
                    log_path=log_path,
                    requested_seed_id=rules_request.seed_id,
                ),
                assistant_response=assistant_response,
            )
        return _build_chat_completion_response(
            model=request.model,
            session_id=session_id,
            prompt_text=latest_user_message,
            assistant_response=assistant_response,
        )

    if request.model == MEMORY_CHAT_MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{MEMORY_CHAT_MODEL_ID}' is temporarily disabled. "
                f"Use '{RULES_CHAT_MODEL_ID}' instead."
            ),
        )
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported model '{request.model}'. "
            f"Available model: {RULES_CHAT_MODEL_ID}"
        ),
    )


@app.post("/rules/pilot/ruling", response_model=RulesRulingResult)
async def pilot_rules_ruling(request: RulesRulingRequest) -> RulesRulingResult:
    """Return one structured ruling from the live SETI pilot graph."""

    if state.rules_ruling_engine is None:
        raise HTTPException(status_code=503, detail="Rules ruling engine not initialized")
    try:
        return state.rules_ruling_engine.answer(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rules ruling error: {exc}") from exc


@app.post("/rules/pilot/inspect", response_model=LivePilotRulingInspection)
async def pilot_rules_inspection(request: RulesRulingRequest) -> LivePilotRulingInspection:
    """Return the live inspection trace for one SETI pilot ruling request."""

    if state.rules_ruling_engine is None:
        raise HTTPException(status_code=503, detail="Rules ruling engine not initialized")
    try:
        return state.rules_ruling_engine.inspect_request(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rules inspection error: {exc}") from exc


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


@app.get("/inspect/qdrant/", response_class=HTMLResponse)
async def qdrant_inspector() -> HTMLResponse:
    """Serve a lightweight Qdrant inspection page."""

    return HTMLResponse(_QDRANT_INSPECTOR_HTML)


@app.get("/inspect/qdrant/api/collections")
async def qdrant_collection_summary():
    """List collections and their basic stats."""

    if state.qdrant is None:
        raise HTTPException(status_code=503, detail="Qdrant client not initialized")

    collections = []
    response = state.qdrant.get_collections()
    for item in response.collections:
        collection_name = item.name
        details = state.qdrant.get_collection(collection_name)
        collections.append(
            {
                "name": collection_name,
                "points_count": getattr(details, "points_count", None),
                "vectors_count": getattr(details, "vectors_count", None),
                "status": getattr(getattr(details, "status", None), "value", None)
                or str(getattr(details, "status", "")),
            }
        )

    return {"collections": collections}


@app.get("/inspect/qdrant/api/collections/{collection_name}/points")
async def qdrant_collection_points(collection_name: str, limit: int = Query(default=10, ge=1, le=100)):
    """Fetch a sample of points from one Qdrant collection."""

    if state.qdrant is None:
        raise HTTPException(status_code=503, detail="Qdrant client not initialized")

    points, next_page_offset = state.qdrant.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return {
        "collection_name": collection_name,
        "limit": limit,
        "next_page_offset": next_page_offset,
        "points": [
            {
                "id": point.id,
                "payload": point.payload,
            }
            for point in points
        ],
    }


@app.get("/inspect/janus/", response_class=HTMLResponse)
async def janus_inspector() -> HTMLResponse:
    """Serve a lightweight JanusGraph inspection page."""

    return HTMLResponse(_JANUS_INSPECTOR_HTML)


@app.get("/inspect/janus/api/summary")
async def janus_summary():
    """Return basic JanusGraph counts for inspector use."""

    return {
        "vertex_count": _gremlin_count("g.V().count()"),
        "edge_count": _gremlin_count("g.E().count()"),
    }


@app.get("/inspect/janus/api/vertices")
async def janus_vertices(limit: int = Query(default=10, ge=1, le=100)):
    """Return a sample of vertices with their properties."""

    rows = _run_gremlin(
        "g.V().limit(%d).project('id','label','properties')"
        ".by(id()).by(label()).by(valueMap())" % limit
    )
    return {"vertices": _normalize_gremlin_rows(rows), "limit": limit}


@app.get("/inspect/janus/api/vertices/search")
async def janus_vertices_search(
    property: str = Query(...),
    value: str = Query(..., min_length=1),
    limit: int = Query(default=10, ge=1, le=100),
):
    """Search vertices by one supported exact-match property."""

    if property not in _JANUS_VERTEX_PROPERTIES:
        raise HTTPException(status_code=400, detail=f"Unsupported vertex property: {property}")

    rows = _run_gremlin(
        "g.V().has('%s', '%s').limit(%d).project('id','label','properties')"
        ".by(id()).by(label()).by(valueMap())"
        % (_escape_gremlin_string(property), _escape_gremlin_string(value), limit)
    )
    return {
        "property": property,
        "value": value,
        "limit": limit,
        "vertices": _normalize_gremlin_rows(rows),
    }


@app.get("/inspect/janus/api/edges")
async def janus_edges(limit: int = Query(default=10, ge=1, le=100)):
    """Return a sample of edges with endpoints and properties."""

    rows = _run_gremlin(
        "g.E().limit(%d).project('id','label','out_node_id','in_node_id','properties')"
        ".by(id()).by(label()).by(outV().values('node_id')).by(inV().values('node_id')).by(valueMap())"
        % limit
    )
    return {"edges": _normalize_gremlin_rows(rows), "limit": limit}


@app.get("/inspect/janus/api/edges/search")
async def janus_edges_search(
    property: str = Query(...),
    value: str = Query(..., min_length=1),
    limit: int = Query(default=10, ge=1, le=100),
):
    """Search edges by one supported exact-match property."""

    if property not in _JANUS_EDGE_PROPERTIES:
        raise HTTPException(status_code=400, detail=f"Unsupported edge property: {property}")

    rows = _run_gremlin(
        "g.E().has('%s', '%s').limit(%d).project('id','label','out_node_id','in_node_id','properties')"
        ".by(id()).by(label()).by(outV().values('node_id')).by(inV().values('node_id')).by(valueMap())"
        % (_escape_gremlin_string(property), _escape_gremlin_string(value), limit)
    )
    return {
        "property": property,
        "value": value,
        "limit": limit,
        "edges": _normalize_gremlin_rows(rows),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
