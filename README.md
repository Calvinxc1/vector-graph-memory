# Vector Graph Memory

Vector Graph Memory is a hybrid vector-plus-graph backend for AI systems that need retrieval over both semantic similarity and explicit relationships.

Today, this repository is still primarily an experimental memory system. The current strategic direction is to reuse that substrate for a rules-lawyer product focused on complex tabletop games, but that layer is still in planning and pilot work.

Current `SETI` pilot status:

- two manual `SETI` rule-graph seed slices now exist:
  - `landing_and_orbiter_interactions`
  - `free_action_timing_and_authority`
- the rules extraction path now uses strongly typed `PydanticAI` output as the required contract boundary
- the first live extraction path has reproduced the landing/orbiter seed under the current comparison rules
- seed materialization, import, and verification tooling exist for loading those slices into Qdrant plus JanusGraph
- a first live graph-backed ruling path exists for the frozen `SETI` pilot questions
- a typed ruling-eval suite now scores retrieval nodes, expanded evidence, seed inference, case selection, citation choice, modifier choice, and precedence assembly separately
- local inspection is available through direct database UIs:
  - Qdrant: `http://localhost:8111/dashboard/`
  - JanusGraph visualizer: `http://localhost:8112/`

## What Exists Today

The repository currently contains:

- a Python library for storing and retrieving nodes and typed edges in a combined Qdrant plus JanusGraph backend
- an OpenAI-compatible REST API built with FastAPI
- a PydanticAI-based memory agent with proposal and confirmation workflows
- Docker-based local development infrastructure, including Open WebUI
- a DSPy-backed grounded-answer synthesis path behind feature flags
- a local evaluation fixture built around `SETI` rules-reference cases for the DSPy synthesis path

The package metadata still reflects the currently implemented system:

- package name: `vector-graph-memory`
- package version: `0.1.0`
- package description: hybrid vector-graph backend for AI agents with persistent memory

## LLM Output Policy

This repository now treats strongly typed LLM output as a design requirement wherever the output is consumed by code.

- programmatic LLM output should go through `PydanticAI` with explicit `Pydantic` models
- freeform JSON or prose may be used for debugging, but it is not the preferred steady-state contract for application logic
- if an intermediate looser payload is unavoidable, normalize it into the typed model before validation, evaluation, or persistence

This matters especially for the rules-lawyer work, where extracted rules, citations, edges, and ruling structures must be machine-checkable before they are trusted.

## Current Product Position

There are two distinct layers in this repo and they should not be conflated.

Implemented now:

- hybrid storage and traversal substrate
- memory-oriented chat API
- memory proposal and confirmation flow
- JSONL audit logging
- Dockerized local stack with Open WebUI
- feature-flagged DSPy synthesis and compile scaffolding

Planned or in-progress direction:

- VGM Rules Lawyer, a retrieval-first adjudication system for board-game rules
- structured ruling outputs with citation chains and precedence order
- game-specific ingestion and graph construction, starting with a `SETI` pilot
- local-model validation for constrained rules-reference tasks
- multi-slice extraction evaluation and generalized rule-ingestion workflows beyond the current `SETI` pilot fixtures

If you are evaluating the repository as software, treat the memory substrate as the implemented product and the rules-lawyer work as a planned layer under active design.

## How The Substrate Works

The system splits responsibilities across two storage layers:

- Qdrant stores embeddings, content, and node metadata
- JanusGraph stores node references and typed relationships for traversal

Typical retrieval flow:

1. A user or agent submits a natural-language request.
2. Vector search finds semantically relevant nodes.
3. Graph traversal expands outward from those starting points.
4. The application uses the resulting context for answer synthesis or memory operations.

Current core data concepts:

- nodes: content-bearing entities with embeddings and metadata
- edges: typed relationships between nodes
- audit entries: records of memory operations

## Rules-Lawyer Direction

The near-term strategic direction is to adapt this substrate into a rules-lawyer system for complex tabletop games.

The intended product shape is:

- retrieve rules rather than generate unsupported answers
- treat citations as structural outputs attached from node metadata
- expose rule hierarchy and precedence order explicitly
- constrain the LLM to identifying, ranking, and formatting relevant rules
- deploy fully locally through Docker and Open WebUI on modest hardware

Current target game progression:

1. `SETI`
2. `Arkham Horror LCG`
3. `Stellar Horizons`
4. `Magic: The Gathering`

Roadmap and planning documents:

- [rules-lawyer-strategy.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/rules-lawyer-strategy.md)
- [rules-lawyer-roadmap.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/rules-lawyer-roadmap.md)
- [seti-pilot-next-steps.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/seti-pilot-next-steps.md)

## Current State And Known Gaps

This project is pre-1.0 and should be treated as experimental.

What works today:

- hybrid vector-graph storage
- OpenAI-compatible chat API
- memory proposal and confirmation flow
- JSONL audit logging
- Dockerized local stack
- Open WebUI integration
- feature-flagged DSPy synthesis seam and compile manager scaffold

What is incomplete or only partially implemented:

- MongoDB audit logging is intended, but API startup does not yet wire MongoDB audit configuration end to end.
- Local API startup via `./start_api.sh` requires `OPENAI_API_KEY` to already be exported in the shell and does not source `.env`.
- `ai_determined` trigger mode currently injects memory-review guidance on every turn rather than selectively deciding when to review.
- `GET /memory/audit/{session_id}` accepts `limit`, but session-scoped audit queries do not currently enforce that limit.
- JanusGraph schema initialization is still manual for local library and local API use outside the default Dockerized path.
- The rules-lawyer layer now has seed, extraction, and import scaffolding, but it is still not an implemented end-user product path.
- `scripts/verify_manual_seed.py` is now generic by default and only runs support-path checks for seed manifests that define them.

## Architecture

- type: Python library plus OpenAI-compatible REST API
- runtime Python: `3.11+`
- current local-dev and CI target: Python `3.14`
- vector database: Qdrant
- graph database: JanusGraph
- agent framework: PydanticAI
- API framework: FastAPI
- prompt-synthesis layer: DSPy-backed grounded answer synthesis behind feature flags

## Development Setup

### Prerequisites

- Docker and Docker Compose
- Python `3.11+`
- `uv` for local development workflows

### Quick Start: API Stack

This is the recommended path for trying the repository as it exists today.

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 2. Start the default stack
docker compose up -d

# 3. Check the proxied API
curl http://localhost:8052/vgm-api/v1/models
```

This starts:

- Qdrant
- JanusGraph
- MongoDB container
- FastAPI service
- Open WebUI
- Nginx reverse proxy

Endpoints:

- Open WebUI: `http://localhost:8052/`
- API: `http://localhost:8052/vgm-api/`
- Pilot ruling API: `POST /rules/pilot/ruling`
- Pilot ruling inspection API: `POST /rules/pilot/inspect`
- Qdrant UI: `http://localhost:8111/dashboard/`
- JanusGraph visualizer: `http://localhost:8112/`

Note: MongoDB is included in Compose, but JSONL is still the currently working audit backend.

### Quick Start: Python Library

If you want to work with the library directly instead of the API:

```bash
# 1. Start only the backing databases
docker compose up -d qdrant janusgraph

# 2. Initialize JanusGraph schema once
uv run python scripts/init_janusgraph_schema.py

# 3. Install the package
uv pip install -e .

# 4. Explore usage examples
uv run jupyter notebook
```

Use `playground.ipynb` for direct library examples.

### Local API Development

To run the API outside Docker while keeping the databases in Docker:

```bash
# 1. Install API dependencies
uv pip install -e ".[api]"

# 2. Configure environment
cp .env.example .env

# 3. Start backing services
docker compose up -d qdrant janusgraph

# 4. Initialize JanusGraph schema once
uv run python scripts/init_janusgraph_schema.py

# 5. Export your API key in the current shell
export OPENAI_API_KEY=sk-...

# 6. Start the API
./start_api.sh
```

Important caveat: `start_api.sh` does not currently source `.env`.

## Open WebUI Integration

The default Compose stack now serves Open WebUI and the API behind one reverse proxy entrypoint.

Once the stack is running:

1. Open `http://localhost:8052/`
2. Use Open WebUI at the root path
3. Reach the proxied API under `http://localhost:8052/vgm-api/`
4. Inspect Qdrant directly at `http://localhost:8111/dashboard/`
5. Inspect JanusGraph directly at `http://localhost:8112/`

For local host-run utilities such as `scripts/import_manual_seed.py` and
`scripts/verify_manual_seed.py`, the default direct backend targets are:

- Qdrant HTTP: `localhost:8111`
- JanusGraph Gremlin: `localhost:8182`

These defaults are intended to match the Docker Compose host port exposure.

For the live pilot ruling path, the main local utility scripts are:

- `uv run python scripts/run_pilot_ruling.py --question '...'`
- `uv run python scripts/run_pilot_ruling_eval.py`

The eval runner executes the tracked frozen suite at
`tests/fixtures/rag_eval/seti_rules_ruling_eval_v1.jsonl` and emits a typed JSON report.
The API now also exposes the same intermediate inspection trace through
`POST /rules/pilot/inspect` for debugging retrieval, seed selection, and case ranking.

If you want to use an external Open WebUI instance, configure:

- base URL: `http://localhost:8052/vgm-api/v1`
- API key: any value for the current default local setup
- model: `vector-graph-memory`

If Open WebUI is on the same Docker network and you want to bypass the proxy, use `http://api:8000/v1` instead.

## DSPy Synthesis Status

The repository includes a DSPy-backed synthesis seam behind feature flags.

What it currently does:

- builds a deterministic `RagContext`
- routes answer synthesis through a baseline DSPy module when enabled
- supports local compile and artifact caching for exact model identities
- evaluates against the tracked `SETI` rules-reference fixture

What it does not yet mean:

- the repo does not yet ship a complete rules-lawyer product flow
- the DSPy path is still grounded-answer infrastructure, not a full ruling engine
- successful eval runs do not by themselves validate the future game-specific ingestion pipeline

See [dspy-rag-implementation.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/dspy-rag-implementation.md) for the implementation plan.

## Docs

- [API.md](/home/jcherry/Documents/storage/git/vector-graph-memory/API.md): current API behavior, setup, and caveats
- [dspy-rag-implementation.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/dspy-rag-implementation.md): DSPy synthesis implementation plan
- [rules-lawyer-strategy.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/rules-lawyer-strategy.md): product strategy for the rules-lawyer direction
- [rules-lawyer-roadmap.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/rules-lawyer-roadmap.md): staged game roadmap
- [seti-pilot-next-steps.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/seti-pilot-next-steps.md): immediate `SETI` pilot plan

## Near-Term Development Priorities

Near-term work in this repo now falls into two buckets.

Substrate work:

- stronger graph traversal patterns
- memory consolidation and duplicate handling improvements
- completion of MongoDB audit support
- better validation coverage and tests

Rules-lawyer work:

- choose and freeze the first `SETI` subsystem
- define the pilot rule schema and ruling output contract
- build a manual ground-truth graph slice
- validate automated extraction against that slice
- test constrained outputs on local models

## AI Usage

This repository is intentionally maintained as a fully AI-generated codebase under human direction.

That operating model is specific to this repository. Human review remains responsible for validation, acceptance, and release decisions.

Because of that, existing code and documentation should be treated as potentially polished but wrong. Validation matters more than precedent here.

## License

See [LICENSE](/home/jcherry/Documents/storage/git/vector-graph-memory/LICENSE) for details.
