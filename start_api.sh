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
QDRANT_HTTP_PORT=${QDRANT_HTTP_PORT:-6333}

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

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set in .env file"
    exit 1
fi
echo "✓ OpenAI API key configured"

# Start the API server
echo ""
echo "Starting API server..."
python -m uvicorn src.vgm.api.server:app \
    --host ${API_HOST:-0.0.0.0} \
    --port ${API_PORT:-8000} \
    --reload
