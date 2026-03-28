# Vector Graph Memory API

OpenAI-compatible API server for integrating Vector Graph Memory with Open WebUI and other LLM frontends.

This document describes the current implemented API behavior. The repository is currently pre-1.0, and package version `0.1.0` is the authoritative version.

## Quick Start

### Option 1: Docker (Recommended)

Run the default stack with Docker Compose:

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env - at minimum set OPENAI_API_KEY

# 2. Start all services (databases + API)
docker compose up -d

# 3. Check logs
docker compose logs -f api
```

The API will be available at `http://localhost:8000`

**Included services:**
- Qdrant (vector database)
- JanusGraph (graph database)
- API server (FastAPI with memory agent)
- Open WebUI

**How it works:**
- All containers run on an internal Docker network (`vector-graph-network`)
- The API container connects to databases using container names (`qdrant`, `janusgraph`)
- The API port (8000) and Open WebUI port (3000 by default) are exposed to your host machine
- Database connections are handled automatically - no localhost configuration needed!

**Current limitation:**
- MongoDB is defined in `docker-compose.yml`, but MongoDB-backed audit logging is not fully wired through API startup configuration yet. JSONL is the currently working audit path.

### Option 2: Local Development

Run the API locally while databases run in Docker:

```bash
# 1. Install dependencies
pip install -e ".[api]"

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Start only databases
docker compose up -d qdrant janusgraph

# 4. Initialize JanusGraph schema once
python scripts/init_janusgraph_schema.py

# 5. Export OPENAI_API_KEY in your shell
export OPENAI_API_KEY=sk-...

# 6. Start API locally
./start_api.sh
```

The API will be available at `http://localhost:8000`

Note: `start_api.sh` currently checks `OPENAI_API_KEY` from the shell environment before starting the API. It does not source `.env` automatically.

## Open WebUI Integration

### Add as External Connection

1. Open your Open WebUI instance
2. Go to **Settings** → **Connections**
3. Add a new **OpenAI API** connection:
   - **Name**: Vector Graph Memory
   - **Base URL**: `http://localhost:8000/v1`
   - **API Key**: (any value, not validated)
4. Save and select the "vector-graph-memory" model

### Usage

Simply chat with the agent through Open WebUI. The agent will:

- Automatically search memory for relevant context
- Propose storing important information
- Track conversations in audit logs

**Memory Trigger Modes**:

- `ai_determined` (default): The server currently injects memory-review guidance on every turn, and the model decides whether to propose anything
- `phrase`: Trigger on specific phrase (e.g., "save this to memory")
- `interval`: Check every N messages

Configure via `TRIGGER_MODE` in `.env`

### DSPy RAG Synthesis Status

The repository now includes a baseline DSPy synthesis path and a first compile/cache scaffold behind feature flags, but the optimization stack is still intentionally narrow.

Current status:

- `RAG_CONTEXT_ENABLED=true` builds the deterministic `RagContext` seam for requests
- `RAG_DSPY_SYNTHESIS_ENABLED=true` routes answer synthesis through a baseline DSPy module
- `RAG_DSPY_COMPILE_ENABLED=true` enables a local compile manager that can load a promoted compiled artifact if one exists
- `RAG_DSPY_AUTO_COMPILE_ENABLED=true` allows one background compile attempt for an unseen exact model identity
- If DSPy synthesis fails, the API falls back to the existing `MemoryAgent` path
- Compilation currently uses the local SETI rules-reference eval suite and local ignored source documents
- Open WebUI feedback integration remains future work

See `docs/plans/dspy-rag-implementation.md` for the phased implementation plan.

## API Endpoints

### OpenAI-Compatible Endpoints

#### `POST /v1/chat/completions`

Standard OpenAI chat completions endpoint.

**Request:**
```json
{
  "model": "vector-graph-memory",
  "messages": [
    {"role": "user", "content": "What do you remember about my job search?"}
  ],
  "user": "optional-session-id"
}
```

**Response:**
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

**Response:**
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

**Response:**
```json
{
  "session_id": "user-123",
  "proposals": {
    "proposal-uuid-1": {
      "content": "Applied to Google for Senior SWE role",
      "entity_type": "job",
      "relationships": [],
      "similar_nodes": []
    }
  }
}
```

#### `POST /memory/confirm/{session_id}/{proposal_id}`

Confirm or reject a memory proposal.

**Query Parameters:**
- `action`: `add_new`, `update_existing`, or `cancel`
- `update_node_id`: Required if action is `update_existing`

**Response:**
```json
{
  "status": "ok",
  "message": "Successfully added job to memory with ID: abc-123"
}
```

#### `GET /memory/audit/{session_id}`

Get audit log for a session.

**Query Parameters:**
- `limit`: Maximum number of entries (default: 50) for non-session-scoped recent-history calls; session-scoped history does not currently enforce this limit

**Response:**
```json
{
  "session_id": "user-123",
  "entries": [
    {
      "timestamp": "2026-03-22T01:55:04",
      "operation": "add_node",
      "summary": "Added job: Applied to Google...",
      "entities": ["node-uuid-1"]
    }
  ]
}
```

### Health Check

#### `GET /`

API health check.

**Response:**
```json
{
  "status": "ok",
  "service": "vector-graph-memory-api",
  "version": "0.1.0"
}
```

## Configuration Reference

### Environment Variables

See `.env.example` for full configuration options.

**Key Settings:**

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_MODEL` | PydanticAI model string | `openai:gpt-4o-mini` |
| `PROJECT_ID` | Memory namespace | `default` |
| `MEMORY_USE_CASE` | Use case description | General purpose memory |
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

MongoDB audit environment variables are listed in `.env.example`, but they are not yet fully consumed by API startup code. JSONL is the currently functional audit backend.

### Memory Trigger Modes

1. **AI Determined** (`ai_determined`)
   - The server currently prompts memory review on every turn
   - The model still decides whether to propose additions
   - Set: `TRIGGER_MODE=ai_determined`

2. **Phrase-based** (`phrase`)
   - Trigger on specific phrase
   - User must explicitly request memory storage
   - Set: `TRIGGER_MODE=phrase` and `TRIGGER_PHRASE=save this to memory`

3. **Interval-based** (`interval`)
   - Check every N messages
   - Predictable behavior
   - Set: `TRIGGER_MODE=interval` and `TRIGGER_INTERVAL=5`

## Architecture

The API server:

1. **Initializes on startup**:
   - Connects to Qdrant and JanusGraph
   - Creates MemoryAgent instance
   - Loads configuration from environment

2. **Handles requests**:
   - Receives chat messages via OpenAI API
   - Runs agent with memory tools
   - Returns responses

3. **Manages sessions**:
   - Uses `user` field as session ID
   - Tracks pending proposals per session
   - Maintains conversation context

4. **Provides memory control**:
   - Endpoints to view/confirm proposals
   - Audit log access
   - Session management

## Known Gaps

- MongoDB audit logging is intended but not fully wired into API startup configuration.
- `start_api.sh` requires `OPENAI_API_KEY` to be exported in the current shell and does not source `.env`.
- JanusGraph schema initialization is manual for local library and local API development.
- Session-scoped audit history does not currently apply the documented `limit` parameter.

## Development

### Running with Auto-reload

```bash
python -m uvicorn src.vgm.api.server:app --reload --host 0.0.0.0 --port 8000
```

### Testing the API

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

### API Documentation

Interactive API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Docker Commands

### Start all services
```bash
docker compose up -d
```

### View logs
```bash
# All services
docker compose logs -f

# Just API
docker compose logs -f api

# Just databases
docker compose logs -f qdrant janusgraph
```

### Rebuild API after code changes
```bash
docker compose up -d --build api
```

### Stop all services
```bash
docker compose down
```

### Stop and remove data
```bash
docker compose down -v
```

### Restart just the API
```bash
docker compose restart api
```

## Troubleshooting

### Cannot connect to databases

Error: `Cannot connect to Qdrant/JanusGraph`

**Solution:**
```bash
docker compose up -d
# Wait 10-15 seconds for JanusGraph to initialize

# Check if services are healthy
docker compose ps
```

### API key not set

Error: `OPENAI_API_KEY not set`

**Solution:**
Add to `.env`:
```bash
OPENAI_API_KEY=sk-...
```

Then restart:
```bash
docker compose restart api
```

### Port already in use

Error: `Address already in use`

**Solution:**
Change port in `.env`:
```bash
API_PORT=8001
```

Then rebuild:
```bash
docker compose up -d
```

### API container won't start

**Check logs:**
```bash
docker compose logs api
```

**Common issues:**
- Missing `OPENAI_API_KEY` in `.env`
- Missing exported `OPENAI_API_KEY` in the current shell when using `./start_api.sh`
- Database services not ready (wait 15-20 seconds)
- Port conflict (change `API_PORT`)

**Force rebuild:**
```bash
docker compose up -d --build --force-recreate api
```

## Next Steps

- See `playground.ipynb` for usage examples
- Check `README.md` for overall project documentation
- Review `.env.example` for all configuration options
