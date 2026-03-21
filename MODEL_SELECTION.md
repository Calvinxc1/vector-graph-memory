# Model Selection Guide

This guide explains how to choose LLM and embedding models for Vector Graph Memory.

## Quick Start

```python
from pydantic_ai import Agent
from pydantic_ai.models import infer_model
from vector_graph_memory import VectorGraphMemory

# Choose your LLM for the agent
agent = Agent('openai:gpt-4', deps_type=DatabaseContext)

# Choose your embedding model for memory
embedding_model = infer_model('openai:text-embedding-3-small')
memory = VectorGraphMemory(
    qdrant_client=qdrant,
    janus_client=janus,
    embedding_model=embedding_model
)
```

## LLM Models

The LLM model is passed directly to `Agent()` from PydanticAI. This is the model that powers your AI agent's reasoning and tool use.

### Supported Providers

| Provider | Format | Example |
|----------|--------|---------|
| OpenAI | `openai:<model>` | `openai:gpt-4` |
| Anthropic | `anthropic:<model>` | `anthropic:claude-3-5-sonnet-20241022` |
| Google Gemini | `google-gla:<model>` | `google-gla:gemini-1.5-pro` |
| Google Vertex | `google-vertex:<model>` | `google-vertex:gemini-1.5-pro` |
| Groq | `groq:<model>` | `groq:llama-3.1-70b-versatile` |
| Ollama | `ollama:<model>` | `ollama:llama3` |

### Examples

```python
# Use OpenAI GPT-4
agent = Agent('openai:gpt-4')

# Use Anthropic Claude
agent = Agent('anthropic:claude-3-5-sonnet-20241022')

# Use local Ollama model
agent = Agent('ollama:llama3')
```

## Embedding Models

The embedding model is passed to `VectorGraphMemory()` to convert text into vectors for semantic search.

### Supported Providers

| Provider | Format | Example |
|----------|--------|---------|
| OpenAI | `openai:<model>` | `openai:text-embedding-3-small` |
| Google Gemini | `google-gla:<model>` | `google-gla:text-embedding-004` |
| Google Vertex | `google-vertex:<model>` | `google-vertex:text-embedding-004` |
| Cohere | `cohere:<model>` | `cohere:embed-english-v3.0` |
| VoyageAI | `voyageai:<model>` | `voyageai:voyage-2` |

### Choosing an Embedding Model

**For general use:**
- `openai:text-embedding-3-small` - Good balance of cost and performance (1536 dimensions)
- `openai:text-embedding-3-large` - Higher quality (3072 dimensions)

**For multilingual support:**
- `cohere:embed-multilingual-v3.0` - Supports 100+ languages

**For domain-specific tasks:**
- `voyageai:voyage-code-2` - Optimized for code
- `voyageai:voyage-finance-2` - Optimized for financial documents

### Examples

```python
from pydantic_ai.models import infer_model

# Use OpenAI small embedding model (recommended for most use cases)
embedding_model = infer_model('openai:text-embedding-3-small')

# Use Google's embedding model
embedding_model = infer_model('google-gla:text-embedding-004')

# Use Cohere for multilingual support
embedding_model = infer_model('cohere:embed-multilingual-v3.0')

# Pass to VectorGraphMemory
memory = VectorGraphMemory(
    qdrant_client=qdrant,
    janus_client=janus,
    embedding_model=embedding_model
)
```

## Complete Example

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import infer_model
from qdrant_client import QdrantClient
from gremlin_python.driver import client as gremlin_client
from vector_graph_memory import VectorGraphMemory

# Connect to databases
qdrant = QdrantClient(host='localhost', port=6333)
janus = gremlin_client.Client('ws://localhost:8182/gremlin', 'g')

# Choose embedding model
embedding_model = infer_model('openai:text-embedding-3-small')

# Initialize memory system
memory = VectorGraphMemory(
    qdrant_client=qdrant,
    janus_client=janus,
    embedding_model=embedding_model,
    collection_name="my_memory"
)

# Create agent with chosen LLM
agent = Agent('anthropic:claude-3-5-sonnet-20241022')

# Use different providers for LLM vs embeddings!
# For example: Claude for reasoning + OpenAI for embeddings
```

## API Keys

Set API keys as environment variables:

```bash
export OPENAI_API_KEY=your-api-key
export ANTHROPIC_API_KEY=your-api-key
export GOOGLE_API_KEY=your-api-key
export COHERE_API_KEY=your-api-key
export VOYAGEAI_API_KEY=your-api-key
```

Or in a `.env` file:

```bash
OPENAI_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-api-key
```

## Key Design Principles

1. **Models are API parameters, not configuration** - Pass model strings directly to `Agent()` and `infer_model()`
2. **Mix and match providers** - Use different providers for LLM vs embeddings (e.g., Claude + OpenAI)
3. **Drop-in replacement** - Change model strings to switch providers without code changes
4. **PydanticAI native** - Leverages PydanticAI's built-in model support

## Further Reading

- [PydanticAI Models Documentation](https://ai.pydantic.dev/models/overview/)
- [PydanticAI Embeddings Documentation](https://ai.pydantic.dev/embeddings/)
