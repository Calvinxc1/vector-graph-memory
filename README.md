# Vector Graph Memory

A hybrid vector-graph database backend for AI agents, providing long-term persistent memory and a single source of truth across extended agent runs.

## Overview

**Problem:** AI agents need scalable, persistent memory to maintain context across long-running sessions and complex multi-step tasks.

**Solution:** Vector Graph Memory combines vector embeddings for semantic search with graph databases for relationship tracking, enabling agents to efficiently discover and navigate interconnected information.

**Use Case:** Initially designed for tracking professional networking and job search efforts (jobs, companies, people, interactions), but built as a general-purpose knowledge graph system extensible to any domain.

## Architecture

- **Type:** Python library + OpenAI-compatible REST API
- **Python Version:** 3.11+
- **Vector Database:** Qdrant (semantic entry point discovery)
- **Graph Database:** JanusGraph (relationship storage and traversal)
- **AI Framework:** PydanticAI
- **API Framework:** FastAPI with OpenAI-compatible endpoints

## Core Concepts

### Hybrid Vector-Graph Approach

The system leverages the strengths of both database paradigms:

- **Graph Database:** Captures rich relationships between entities
  - Example: *Job X* located at *Company Y* who employs *Person Z* who I've had *Correspondence A* with
- **Vector Database:** Enables semantic search to find optimal entry points into the graph
  - Natural language queries mapped to relevant starting nodes via embeddings

### Query Flow

1. User/agent submits natural language query
2. Vector search identifies semantically relevant entry point(s) in the graph
3. Graph traversal explores relationships from entry points
4. Contextualized information returned to support agent response

### Data Model

- **Entities:** Graph nodes with vector embeddings on content
- **Relationships:** Typed edges connecting entities
- **Embeddings:** Vector representations enabling semantic similarity search

## Roadmap

### v1.0 - Core Primitives
- Generic entity and relationship storage
- Vector embedding generation and indexing
- Hybrid query interface (natural language → vectors → graph → context)
- Basic CRUD operations for entities and relationships

### v2.0+ - Advanced Features
- Temporal tracking (timestamps, recency scoring)
- Memory consolidation (duplicate/similar entity merging)
- Confidence scoring on relationships
- Export/backup functionality
- Multi-agent memory sharing
- Optional service wrapper (REST/GraphQL API)

## Development Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.14+
- uv (Python package manager)

### Quick Start

**Option 1: Using the API (Recommended)**

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env - at minimum set OPENAI_API_KEY

# 2. Start all services (databases + API)
docker compose up -d

# 3. Try the API
curl http://localhost:8000/v1/models
```

The API will be available at `http://localhost:8000` with OpenAI-compatible endpoints.

**Option 2: Using the Python Library**

```bash
# 1. Start databases only
docker compose up -d qdrant janusgraph

# 2. Install library
pip install -e .

# 3. Use in Python
# See playground.ipynb for examples
```

### Playground Notebooks

Two Jupyter notebooks are provided:

- **`playground_api.ipynb`** - Examples using the REST API (recommended)
- **`playground.ipynb`** - Examples using the Python library directly

Start Jupyter and open either notebook:

```bash
jupyter notebook
```

## Integration with Open WebUI

The Vector Graph Memory API is compatible with Open WebUI and other OpenAI-compatible clients.

### Using the Included Open WebUI Instance

The easiest way to test the API is with the included Open WebUI container:

```bash
# Start all services including Open WebUI
docker compose up -d

# Open WebUI will be available at http://localhost:3000
```

The Open WebUI instance is pre-configured to use the Vector Graph Memory API. Simply:

1. Navigate to `http://localhost:3000` in your browser
2. Start chatting - the agent has persistent memory capabilities
3. The agent will propose adding information to memory during conversations
4. Confirm memory proposals through the API endpoints (see `playground_api.ipynb` for examples)

### Using an External Open WebUI Instance

If you have your own Open WebUI installation:

1. Start the API: `docker compose up -d api`
2. In Open WebUI, add a new OpenAI API connection:
   - **Base URL:** `http://localhost:8000/v1` (or `http://api:8000/v1` if on the same Docker network)
   - **API Key:** (any value - authentication is disabled by default)
   - **Model:** `vector-graph-memory`

### Configuration

Customize Open WebUI port in `.env`:

```bash
WEBUI_PORT=3000  # Default port
```

See [API.md](API.md) for complete API documentation.

## Status

**Current Version:** v1.0 - Core functionality complete

**Features:**
- ✅ Hybrid vector-graph storage
- ✅ PydanticAI agent with memory tools
- ✅ OpenAI-compatible REST API
- ✅ Docker containerization
- ✅ Memory proposals and confirmations
- ✅ Audit logging (JSONL/MongoDB)
- ✅ Configurable memory triggers

**In Development:**
- Temporal tracking and recency scoring
- Memory consolidation
- Advanced graph traversal patterns

## License

See [LICENSE](LICENSE) file for details.
