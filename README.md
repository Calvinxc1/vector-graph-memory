# Vector Graph Memory

Vector Graph Memory is a hybrid vector-graph backend for AI agents that need persistent memory across long-running sessions and multi-step workflows.

It combines semantic retrieval from a vector store with relationship traversal in a graph database so an agent can both find relevant starting points and navigate connected context.

## What This Repo Is

This repository currently contains:

- A Python library for storing and retrieving memory in a combined Qdrant + JanusGraph backend
- An OpenAI-compatible REST API built with FastAPI
- A PydanticAI-based memory agent with proposal-and-confirmation workflows
- Docker-based local development infrastructure, including Open WebUI

The project started around professional networking and job-search tracking, but the underlying storage model is intended to be general-purpose.

## How It Works

The system splits responsibilities across two storage layers:

- Qdrant stores embeddings, full content, and node metadata
- JanusGraph stores node references and relationships for traversal

Typical query flow:

1. A user or agent submits a natural-language request.
2. Vector search finds semantically relevant nodes.
3. Graph traversal expands outward from those nodes.
4. The agent uses the resulting context to answer or propose memory operations.

The current data model is centered on:

- Nodes: content-bearing entities with embeddings
- Edges: typed relationships between nodes
- Audit entries: records of memory operations

## AI Usage

This repository is intentionally maintained as a fully vibe-coded, 100% AI-generated codebase under human direction.

All substantive code, documentation, and workflow changes may be produced by coding agents. Human oversight remains responsible for validation, acceptance, and release decisions.

Because of that operating model, the repo policy emphasizes explicit validation, skepticism toward existing implementation patterns, and stronger review discipline for agent-generated changes.

## Current State

This project is pre-1.0. The authoritative repository version is the package version in `pyproject.toml`, which is currently `0.1.0`.

The repo is usable, but it should be treated as an experimental implementation rather than a stable `1.0` release.

What works today:

- Hybrid vector-graph storage
- OpenAI-compatible chat API
- Memory proposal and confirmation flow
- JSONL audit logging
- Dockerized local stack
- Open WebUI integration

What is still incomplete or partially implemented:

- MongoDB audit logging is intended, but API startup does not yet wire MongoDB audit configuration end-to-end.
- Local API startup via `./start_api.sh` requires `OPENAI_API_KEY` to already be exported in the shell; the script does not currently source `.env`.
- `ai_determined` trigger mode currently injects memory-review guidance on every turn rather than selectively deciding when to review.
- `GET /memory/audit/{session_id}` accepts `limit`, but session-scoped audit queries do not currently enforce that limit.
- JanusGraph schema initialization is still manual for local library and local API usage outside the default Dockerized path.

## Architecture

- Type: Python library plus OpenAI-compatible REST API
- Runtime Python: 3.11+
- Current CI and local dev target: Python 3.14
- Vector database: Qdrant
- Graph database: JanusGraph
- Agent framework: PydanticAI
- API framework: FastAPI

## Development Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ runtime
- `uv` for local development workflows

### Quick Start: API Stack

This is the recommended path for trying the project.

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 2. Start the default stack
docker compose up -d

# 3. Check the API
curl http://localhost:8000/v1/models
```

This starts the default local stack, including:

- Qdrant
- JanusGraph
- MongoDB container
- FastAPI service
- Open WebUI

Endpoints:

- API: `http://localhost:8000`
- Open WebUI: `http://localhost:3000`

Note: MongoDB is part of the compose stack, but MongoDB-backed audit logging is not yet fully wired in the application runtime. JSONL is the currently working audit backend.

### Quick Start: Python Library

If you want to work with the library directly instead of the API:

```bash
# 1. Start only the backing databases
docker compose up -d qdrant janusgraph

# 2. Initialize JanusGraph schema once
python scripts/init_janusgraph_schema.py

# 3. Install the package
pip install -e .

# 4. Explore usage examples
jupyter notebook
```

Use `playground.ipynb` for direct library examples.

## Local API Development

To run the API outside Docker while keeping the databases in Docker:

```bash
# 1. Install API dependencies
pip install -e ".[api]"

# 2. Configure environment
cp .env.example .env

# 3. Start backing services
docker compose up -d qdrant janusgraph

# 4. Initialize JanusGraph schema once
python scripts/init_janusgraph_schema.py

# 5. Export your API key in the current shell
export OPENAI_API_KEY=sk-...

# 6. Start the API
./start_api.sh
```

Important caveat: `start_api.sh` currently validates `OPENAI_API_KEY` from the shell environment before startup. It does not automatically load `.env`.

## Open WebUI Integration

The easiest way to test the API interactively is through the included Open WebUI container started by the default compose stack.

Once the stack is running:

1. Open `http://localhost:3000`
2. Start chatting with the configured model
3. Let the agent propose memory additions
4. Confirm proposals through the API endpoints or notebook examples

If you want to use an external Open WebUI instance, configure an OpenAI-compatible connection with:

- Base URL: `http://localhost:8000/v1`
- API key: any value for the current default local setup
- Model: `vector-graph-memory`

If Open WebUI is on the same Docker network, use `http://api:8000/v1` instead.

## Notebooks And Docs

- `playground_api.ipynb`: examples using the REST API
- `playground.ipynb`: examples using the Python library directly
- `API.md`: API-specific setup, endpoints, and implementation caveats

## Roadmap

Near-term work still expected in this repo:

- Temporal tracking and recency scoring
- Memory consolidation and duplicate handling improvements
- Stronger graph traversal patterns
- Completing MongoDB audit support
- Better validation coverage and tests

## License

See [LICENSE](LICENSE) for details.
