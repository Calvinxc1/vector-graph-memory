# Vector Graph Memory API

This document describes the current implemented API behavior.

The API is still a memory-oriented interface over the Vector Graph Memory substrate. It is not yet a dedicated rules-lawyer API, even though the repository roadmap now includes that direction.

Current authoritative package version:

- version: `0.1.0`

## What The API Is Today

The current API provides:

- an OpenAI-compatible chat interface
- memory proposal and confirmation endpoints
- audit-log access
- optional DSPy-backed grounded answer synthesis behind feature flags
- Open WebUI-friendly local deployment through Docker Compose

The API does not yet provide:

- a formal rules-lawyer response schema
- game-specific ruling endpoints
- a productionized `SETI` adjudication path

## Quick Start

### Option 1: Docker

Run the default stack with Docker Compose:

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 2. Start all services
docker compose up -d

# 3. Check logs
docker compose logs -f api
```

The API will be available at `http://localhost:8000`.

Included services:

- Qdrant
- JanusGraph
- MongoDB container
- API server
- Open WebUI

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

# 5. Export OPENAI_API_KEY in your shell
export OPENAI_API_KEY=sk-...

# 6. Start API locally
./start_api.sh
```

Note: `start_api.sh` currently checks `OPENAI_API_KEY` from the shell environment and does not source `.env`.

## Open WebUI Integration

### Add As External Connection

1. Open Open WebUI.
2. Go to `Settings -> Connections`.
3. Add an `OpenAI API` connection with:
   - name: `Vector Graph Memory`
   - base URL: `http://localhost:8000/v1`
   - API key: any value for the current default setup
4. Save and select the `vector-graph-memory` model.

### Usage

Through Open WebUI, the current agent can:

- search memory for relevant context
- answer with either the baseline path or feature-flagged DSPy path
- propose storing information in memory
- record memory actions in the audit log

It is still a memory-oriented operator experience, not yet a game-ruling experience.

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

See [dspy-rag-implementation.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/dspy-rag-implementation.md) for the staged implementation plan.

## API Endpoints

### OpenAI-Compatible Endpoints

#### `POST /v1/chat/completions`

Standard OpenAI chat completions endpoint.

Request:

```json
{
  "model": "vector-graph-memory",
  "messages": [
    {"role": "user", "content": "What do you remember about my job search?"}
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
  "model": "vector-graph-memory",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Based on my memory, you applied to..."
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
      "id": "vector-graph-memory",
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
| `LLM_MODEL` | PydanticAI model string | `openai:gpt-4o-mini` |
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
4. Runs either the baseline answer path or the feature-flagged DSPy synthesis path.
5. Exposes proposal, confirmation, and audit endpoints for memory control.

## Known Gaps

- MongoDB audit logging is intended but not fully wired into API startup configuration.
- `start_api.sh` requires `OPENAI_API_KEY` to be exported in the current shell and does not source `.env`.
- JanusGraph schema initialization is manual for local library and local API development.
- Session-scoped audit history does not currently apply the documented `limit` parameter.
- The current API contract is not yet specialized for rules-lawyer output.

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
    "model": "vector-graph-memory",
    "messages": [{"role": "user", "content": "Hello!"}],
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
