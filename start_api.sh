#!/bin/bash
# Start the Vector Graph Memory API server

set -e

echo "🚀 Starting Vector Graph Memory API..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your configuration and run again."
    exit 1
fi

# Check if databases are running
echo "Checking database connections..."

# Check Qdrant
QDRANT_HOST=${QDRANT_HOST:-localhost}
QDRANT_HTTP_PORT=${QDRANT_HTTP_PORT:-8111}

if ! curl -s "http://$QDRANT_HOST:$QDRANT_HTTP_PORT/healthz" > /dev/null; then
    echo "❌ Cannot connect to Qdrant at $QDRANT_HOST:$QDRANT_HTTP_PORT"
    echo "   Run: docker compose up -d"
    exit 1
fi
echo "✓ Qdrant is running"

# Check JanusGraph
JANUSGRAPH_HOST=${JANUSGRAPH_HOST:-localhost}
JANUSGRAPH_PORT=${JANUSGRAPH_PORT:-8182}

if ! timeout 2 bash -c "echo > /dev/tcp/$JANUSGRAPH_HOST/$JANUSGRAPH_PORT" 2>/dev/null; then
    echo "❌ Cannot connect to JanusGraph at $JANUSGRAPH_HOST:$JANUSGRAPH_PORT"
    echo "   Run: docker compose up -d"
    exit 1
fi
echo "✓ JanusGraph is running"

# Check provider credentials. This script reads exported shell variables; it
# does not parse .env because the example file contains unquoted values.
LLM_PROVIDER=${LLM_PROVIDER:-openai}
EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER:-$LLM_PROVIDER}

if [ "$LLM_PROVIDER" = "ollama" ] || [ "$EMBEDDING_PROVIDER" = "ollama" ]; then
    if [ -z "$OLLAMA_BASE_URL" ] && [ -z "$OLLAMA_CHAT_BASE_URL" ] && [ -z "$OLLAMA_EMBEDDING_BASE_URL" ]; then
        echo "❌ OLLAMA_BASE_URL must be exported for the external Ollama provider"
        exit 1
    fi
    echo "✓ External Ollama provider configured"
fi

if [ "$LLM_PROVIDER" = "openai" ] || [ "$EMBEDDING_PROVIDER" = "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "❌ OPENAI_API_KEY must be exported for OpenAI-backed chat or embeddings"
        exit 1
    fi
    echo "✓ OpenAI API key configured"
fi

# Start the API server
echo ""
echo "Starting API server..."
python -m uvicorn src.vgm.api.server:app \
    --host ${API_HOST:-0.0.0.0} \
    --port ${API_PORT:-8000} \
    --reload
