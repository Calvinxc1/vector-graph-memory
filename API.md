# Vector Graph Memory API

This document describes the current implemented API behavior.

The API is still built on the Vector Graph Memory substrate, but it now includes
a narrow live `SETI` rules-lawyer pilot.

Current authoritative package version:

- version: `0.1.0`

## What The API Is Today

The current API provides:

- an OpenAI-compatible chat interface
- a live `SETI` pilot ruling path backed by retrieval, graph expansion, and the `RulesRulingResult` response schema
- memory proposal and confirmation endpoints
- audit-log access
- optional DSPy-backed grounded answer synthesis behind feature flags
- Open WebUI-friendly local deployment through Docker Compose

The API does not yet provide:

- a productionized rules-lawyer workflow beyond the bounded `SETI` pilot
- broad game coverage outside the imported pilot slices
- human review or release controls for accepted rulings

## Quick Start

### Option 1: Docker

Run the default stack with Docker Compose:

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env for either OpenAI or external Ollama provider settings

# 2. Start all services
docker compose up -d

# 3. Check logs
docker compose logs -f api
```

The default Docker entrypoint is the reverse proxy on `http://localhost:8052`.

Included services:

- Qdrant
- JanusGraph
- MongoDB container
- API server
- Open WebUI
- Nginx reverse proxy

Current limitation:

- MongoDB exists in `docker-compose.yml`, but MongoDB-backed audit logging is not yet fully wired through API startup. JSONL is the currently working audit path.

### Option 2: Local Development

Run the API locally while keeping databases in Docker:

```bash
# 1. Install dependencies
uv pip install -e ".[api]"

# 2. Configure environment
cp .env.example .env

# 3. Start only databases
docker compose up -d qdrant janusgraph

# 4. Initialize JanusGraph schema once
uv run python scripts/init_janusgraph_schema.py

# 5. Export provider settings in your shell
export OPENAI_API_KEY=sk-...

# 6. Start API locally
./start_api.sh
```

For an external Ollama service, export the provider and endpoint instead:

```bash
export LLM_PROVIDER=ollama
export EMBEDDING_PROVIDER=ollama
export OLLAMA_BASE_URL=https://ollama.example.com/v1
export OLLAMA_CHAT_MODEL=llama3.1:8b
export OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Note: `start_api.sh` checks exported shell variables and does not source `.env`.

## Open WebUI Integration

### Add As External Connection

1. Open Open WebUI.
2. Go to `Settings -> Connections`.
3. Add an `OpenAI API` connection with:
   - name: `Vector Graph Memory`
   - base URL: `http://localhost:8052/vgm-api/v1`
   - API key: any value for the current default setup
4. Save and select the exposed model:
   - `seti-rules-lawyer` for the live `SETI` pilot ruling path

### Usage

Through Open WebUI, the currently exposed chat path is the live `SETI` pilot ruling engine. The legacy `vector-graph-memory` chat model is temporarily disabled at the OpenAI-compatible API surface so the UI cannot accidentally route into the memory-oriented interface during rules-lawyer testing.

When `seti-rules-lawyer` is selected, Open WebUI routes the latest user message through retrieval plus graph expansion, asks the configured PydanticAI chat model for a `RulesRulingResult`, and renders that ruling with the streamed Thinking trace in the standard chat surface. The legacy adjudication draft/screener path is present but disabled by default.

## DSPy Grounded Synthesis Status

The repository includes a feature-flagged DSPy answer-synthesis path and compile/cache scaffold.

Current behavior:

- `RAG_CONTEXT_ENABLED=true` builds the deterministic `RagContext` seam for chat requests
- `RAG_DSPY_SYNTHESIS_ENABLED=true` routes answer synthesis through a baseline DSPy module
- `RAG_DSPY_COMPILE_ENABLED=true` enables a local compile manager that can load a promoted compiled artifact
- `RAG_DSPY_AUTO_COMPILE_ENABLED=true` allows one background compile attempt for an unseen exact model identity
- if DSPy synthesis fails, the API falls back to the existing `MemoryAgent` path

Important scope note:

- this is currently grounded-answer infrastructure
- it uses the local `SETI` rules-reference eval suite as an optimization target
- it does not by itself mean the repository already exposes a complete rules-lawyer API

See [dspy-rag-implementation.md](docs/plans/dspy-rag-implementation.md) for the staged implementation plan.

## API Endpoints

### OpenAI-Compatible Endpoints

#### `POST /v1/chat/completions`

Standard OpenAI chat completions endpoint. The selected `model` determines which backend handles the request:

- `seti-rules-lawyer`: live `SETI` pilot ruling path using retrieved graph evidence plus the `RulesRulingResult` schema
- `vector-graph-memory`: currently rejected as temporarily disabled

For `seti-rules-lawyer`, `stream: true` returns OpenAI-style SSE chunks for Open WebUI compatibility. The streamed output begins with a concise `<think>...</think>` diagnostic summary before the visible answer text:

- `seti-rules-lawyer` shows route, seed origin, premise-screen results for invalid scenarios, inferred question issue, retrieval counts, candidate diagnostics, authority selection, verifier output, abstain details, and the per-request trace file path for deeper inspection.
- If the configured model returns thinking through PydanticAI or emits complete `<think>...</think>` blocks, the API keeps that thinking out of the visible answer and appends it under `Model thinking:` below the diagnostic trace inside the same Thinking section.

The ruling or answer is still computed before streaming begins, so this improves inspection UX rather than first-token latency. Detailed per-request traces are written under `API_TRACE_LOG_PATH` (default: `./logs/api`) as one JSON file per request, partitioned by date, for example `./logs/api/requests/2026-05-01/<request-id>.json`.

Request:

```json
{
  "model": "seti-rules-lawyer",
  "messages": [
    {
      "role": "user",
      "content": "If another player already has an orbiter there, do I still get the cheaper landing cost?"
    }
  ],
  "user": "optional-session-id"
}
```

Response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "seti-rules-lawyer",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Yes. The FAQ says the discount applies if any orbiter is already at the planet, including an opponent's orbiter.\n\nPrimary authority: Core Rulebook, Land on Planet or Moon (p. 12)"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

#### `GET /v1/models`

List available models.

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "seti-rules-lawyer",
      "object": "model",
      "created": 1234567890,
      "owned_by": "vector-graph-memory"
    }
  ]
}
```

### Memory Management Endpoints

#### `GET /memory/proposals/{session_id}`

Get pending memory proposals for a session.

#### `POST /memory/confirm/{session_id}/{proposal_id}`

Confirm or reject a memory proposal.

Query parameters:

- `action`: `add_new`, `update_existing`, or `cancel`
- `update_node_id`: required if `action=update_existing`

#### `GET /memory/audit/{session_id}`

Get audit log entries for a session.

Current caveat:

- the route accepts `limit`, but session-scoped history does not currently enforce that limit

### Health Check

#### `GET /`

Response:

```json
{
  "status": "ok",
  "service": "vector-graph-memory-api",
  "version": "0.1.0"
}
```

## Configuration Reference

See `.env.example` for the full environment set.

Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Runtime chat/model provider: `openai` or external OpenAI-compatible `ollama` | `openai` |
| `LLM_MODEL` | PydanticAI model string | `openai:gpt-4o-mini` |
| `EMBEDDING_PROVIDER` | Embedding provider. If unset, follows `LLM_PROVIDER` | unset |
| `EMBEDDING_MODEL` | Embedding model for OpenAI-backed embeddings | `text-embedding-3-small` |
| `OLLAMA_BASE_URL` | External Ollama OpenAI-compatible `/v1` base URL | unset |
| `OLLAMA_CHAT_MODEL` | External Ollama chat model used when `LLM_PROVIDER=ollama` | `llama3.1:8b` |
| `OLLAMA_EMBEDDING_MODEL` | External Ollama embedding model used when `EMBEDDING_PROVIDER=ollama` | `nomic-embed-text` |
| `OLLAMA_API_KEY` | Optional API key for the external Ollama domain | unset |
| `PROJECT_ID` | Memory namespace | `default` |
| `MEMORY_USE_CASE` | Use case description | `General purpose memory` |
| `TRIGGER_MODE` | When to check memory | `ai_determined` |
| `SIMILARITY_THRESHOLD` | Duplicate detection threshold | `0.85` |
| `RAG_CONTEXT_ENABLED` | Build the deterministic RAG context seam on each chat request | `false` |
| `RAG_DSPY_SYNTHESIS_ENABLED` | Route answer synthesis through the baseline DSPy module | `false` |
| `RAG_DSPY_COMPILE_ENABLED` | Enable the local DSPy compile/cache manager | `false` |
| `RAG_DSPY_AUTO_COMPILE_ENABLED` | Queue one background compile attempt for unseen exact model configs | `false` |
| `RAG_DSPY_CACHE_DIR` | Local cache directory for promoted compiled DSPy artifacts | `.vgm/dspy_artifacts` |
| `RAG_DSPY_PROGRAM_VERSION` | Synthesis program version used for cache invalidation | `1` |
| `RAG_RETRIEVAL_SCHEMA_VERSION` | Retrieval schema version used for cache invalidation | `1` |
| `RAG_DSPY_EVAL_SUITE_PATH` | Local eval suite JSONL used by DSPy compilation | `tests/fixtures/rag_eval/seti_rules_reference_v1.jsonl` |
| `RAG_DSPY_EVAL_SOURCE_DIR` | Local extracted source documents used by the eval runner | `tests/fixtures/rag_eval/source_documents/extracted` |
| `RAG_DSPY_EVAL_SCORING_MODE` | Offline eval scoring mode: `deterministic` or `hybrid` | `deterministic` |
| `RAG_DSPY_JUDGE_MODEL` | Optional separate provider:model string for the hybrid eval judge | unset |
| `RAG_DSPY_JUDGE_MODEL_NAME` | Optional explicit DSPy judge model name override | unset |
| `RAG_DSPY_JUDGE_API_BASE` | Optional API base override for the judge model | unset |
| `RAG_DSPY_JUDGE_API_KEY` | Optional API key override for the judge model | unset |
| `RAG_DSPY_JUDGE_MODEL_TYPE` | Optional judge model type override such as `responses` | unset |
| `RAG_DSPY_JUDGE_MODEL_VERSION` | Optional exact judge model version tag for eval/cache identity | unset |
| `DSPY_MODEL_NAME` | Optional explicit DSPy model name override | unset |
| `DSPY_API_BASE` | Optional DSPy API base override for OpenAI-compatible endpoints | unset |
| `DSPY_API_KEY` | Optional DSPy API key override | unset |
| `DSPY_MODEL_TYPE` | Optional DSPy model type override such as `chat` | unset |
| `DSPY_MODEL_VERSION` | Optional exact model version tag for cache identity | unset |

MongoDB audit environment variables exist in `.env.example`, but they are not yet fully consumed by API startup code.

## Memory Trigger Modes

### `ai_determined`

- the server currently prompts memory review on every turn
- the model still decides whether to propose additions

### `phrase`

- trigger on a specific phrase
- requires explicit user request for memory storage

### `interval`

- check every N messages
- yields predictable review behavior

## Architecture Summary

The API server currently:

1. Initializes Qdrant, JanusGraph, and the `MemoryAgent`.
2. Receives OpenAI-compatible chat requests.
3. Uses the `user` field as the session identifier.
4. Rejects `vector-graph-memory` chat requests while that legacy interface is temporarily disabled.
5. Routes `seti-rules-lawyer` requests through live retrieval, graph expansion, and `RulesRulingResult` response generation.
6. Exposes proposal, confirmation, and audit endpoints for memory control.

## Known Gaps

- MongoDB audit logging is intended but not fully wired into API startup configuration.
- `start_api.sh` checks exported provider variables in the current shell and does not source `.env`.
- JanusGraph schema initialization is manual for local library and local API development.
- Session-scoped audit history does not currently apply the documented `limit` parameter.
- The OpenAI-compatible `seti-rules-lawyer` endpoint renders a `RulesRulingResult` into chat text; use `POST /rules/pilot/ruling` when the same typed ruling contract is needed directly.

## Development

### Run With Auto-Reload

```bash
uv run python -m uvicorn src.vgm.api.server:app --reload --host 0.0.0.0 --port 8000
```

### Basic API Checks

```bash
# Health check
curl http://localhost:8000/

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "seti-rules-lawyer",
    "messages": [{"role": "user", "content": "If another player already has an orbiter there, do I still get the cheaper landing cost?"}],
    "user": "test-session"
  }'

# Check proposals
curl http://localhost:8000/memory/proposals/test-session
```

### Interactive API Docs

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker Commands

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f api

# Rebuild API after code changes
docker compose up -d --build api

# Stop all services
docker compose down

# Stop and remove data
docker compose down -v
```
