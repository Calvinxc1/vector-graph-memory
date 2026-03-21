"""Embedding provider interface and implementations."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text string.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as a list of floats
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple text strings.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embeddings."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using their API."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        """Initialize OpenAI embedding provider.

        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required for OpenAIEmbeddingProvider")

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)
        self._dimension = self._get_dimension(model)

    @staticmethod
    def _get_dimension(model: str) -> int:
        """Get embedding dimension for a given model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(model, 1536)

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple text strings."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embeddings."""
        return self._dimension


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider using local models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize Sentence Transformer embedding provider.

        Args:
            model_name: HuggingFace model name for sentence transformers
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package required for SentenceTransformerEmbeddingProvider"
            )

        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        # sentence-transformers is synchronous, but we wrap in async
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple text strings."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embeddings."""
        return self._dimension
