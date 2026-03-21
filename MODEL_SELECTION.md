# Model Selection Guide

This guide explains how to configure LLM and embedding models for Vector Graph Memory.

## Quick Start

Configure models via environment variables in `.env`:

```bash
# LLM Model for the agent
LLM_MODEL=openai:gpt-4

# Embedding Model for vectorization
EMBEDDING_MODEL=openai:text-embedding-3-small
```

## LLM Models

Vector Graph Memory uses PydanticAI's agent system, which supports multiple LLM providers.

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

```bash
# Use OpenAI GPT-4
LLM_MODEL=openai:gpt-4

# Use Anthropic Claude
LLM_MODEL=anthropic:claude-3-5-sonnet-20241022

# Use local Ollama model
LLM_MODEL=ollama:llama3
```

## Embedding Models

Embedding models are used to convert text into vector representations for semantic search.

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

```bash
# Use OpenAI small embedding model (recommended for most use cases)
EMBEDDING_MODEL=openai:text-embedding-3-small

# Use Google's embedding model
EMBEDDING_MODEL=google-gla:text-embedding-004

# Use Cohere for multilingual support
EMBEDDING_MODEL=cohere:embed-multilingual-v3.0
```

## API Keys

Set API keys as environment variables:

```bash
# OpenAI
OPENAI_API_KEY=your-api-key

# Anthropic
ANTHROPIC_API_KEY=your-api-key

# Google (Gemini)
GOOGLE_API_KEY=your-api-key

# Cohere
COHERE_API_KEY=your-api-key

# VoyageAI
VOYAGEAI_API_KEY=your-api-key
```

## Complete Example

```bash
# .env file example
LLM_MODEL=anthropic:claude-3-5-sonnet-20241022
EMBEDDING_MODEL=openai:text-embedding-3-small

ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
```

## Notes

- **LLM and embedding models can use different providers** - For example, you can use Claude for the agent and OpenAI for embeddings
- **Model strings follow PydanticAI's format** - See [PydanticAI documentation](https://ai.pydantic.dev/models/overview/) for details
- **Embedding dimensions are detected automatically** - The system will create Qdrant collections with the correct vector size
- **Local models via Ollama** - You can use local models for both LLM and embeddings to avoid API costs

## Further Reading

- [PydanticAI Models Documentation](https://ai.pydantic.dev/models/overview/)
- [PydanticAI Embeddings Documentation](https://ai.pydantic.dev/embeddings/)
