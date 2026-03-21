# Vector Graph Memory

A hybrid vector-graph database backend for AI agents, providing long-term persistent memory and a single source of truth across extended agent runs.

## Overview

**Problem:** AI agents need scalable, persistent memory to maintain context across long-running sessions and complex multi-step tasks.

**Solution:** Vector Graph Memory combines vector embeddings for semantic search with graph databases for relationship tracking, enabling agents to efficiently discover and navigate interconnected information.

**Use Case:** Initially designed for tracking professional networking and job search efforts (jobs, companies, people, interactions), but built as a general-purpose knowledge graph system extensible to any domain.

## Architecture

- **Type:** Python library (v1), with potential service wrapper in future versions
- **Python Version:** 3.14+
- **Vector Database:** Qdrant (semantic entry point discovery)
- **Graph Database:** JanusGraph (relationship storage and traversal)
- **AI Framework:** PydanticAI

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

### Database Setup

Start the required databases using Docker Compose:

```bash
docker compose up -d
```

This will start:
- **Qdrant** on ports 6333 (HTTP) and 6334 (gRPC)
- **JanusGraph** on port 8182 (Gremlin Server)

To stop the databases:

```bash
docker compose down
```

To remove all data volumes:

```bash
docker compose down -v
```

### Environment Configuration

Copy the example environment file and configure as needed:

```bash
cp .env.example .env
```

## Status

**Current Phase:** Planning and design

This project is in active development. The README will be updated as implementation progresses.

## License

See [LICENSE](LICENSE) file for details.
