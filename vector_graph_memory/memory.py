"""Core memory system integrating vector and graph databases."""

import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from gremlin_python.driver import client as gremlin_client
from pydantic_ai.models import EmbeddingModel


class VectorGraphMemory:
    """Hybrid vector-graph memory system."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        janus_client: gremlin_client.Client,
        embedding_model: EmbeddingModel,
        collection_name: str = "memory",
    ):
        """Initialize vector-graph memory system.

        Args:
            qdrant_client: Qdrant client instance
            janus_client: JanusGraph Gremlin client instance
            embedding_model: PydanticAI EmbeddingModel instance
            collection_name: Name of the Qdrant collection
        """
        self.qdrant = qdrant_client
        self.janus = janus_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure the Qdrant collection exists."""
        collections = self.qdrant.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            # Get embedding dimension from a test embedding
            import asyncio
            test_result = asyncio.run(self.embedding_model.embed(["test"]))
            embedding_dim = len(test_result.embeddings[0])

            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )

    async def add_entity(
        self,
        content: str,
        entity_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add an entity to both vector and graph databases.

        Args:
            content: The content/description of the entity
            entity_type: Type of entity (e.g., 'job', 'company', 'person')
            metadata: Additional metadata for the entity

        Returns:
            Entity ID (UUID)
        """
        entity_id = str(uuid.uuid4())
        metadata = metadata or {}

        # Generate embedding using PydanticAI
        embedding_result = await self.embedding_model.embed([content])
        embedding = embedding_result.embeddings[0]

        # Store in Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=entity_id,
                    vector=embedding,
                    payload={
                        "content": content,
                        "type": entity_type,
                        **metadata
                    }
                )
            ]
        )

        # Store in JanusGraph
        gremlin_query = (
            f"g.addV('{entity_type}')"
            f".property('id', '{entity_id}')"
            f".property('content', content)"
        )

        # Add metadata properties
        for key, value in metadata.items():
            # Escape single quotes in values
            escaped_value = str(value).replace("'", "\\'")
            gremlin_query += f".property('{key}', '{escaped_value}')"

        self.janus.submit(gremlin_query).all().result()

        return entity_id

    async def search_similar(
        self,
        query: str,
        limit: int = 5,
        entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for semantically similar entities.

        Args:
            query: Search query text
            limit: Maximum number of results
            entity_type: Optional filter by entity type

        Returns:
            List of matching entities with scores
        """
        # Generate query embedding using PydanticAI
        embedding_result = await self.embedding_model.embed([query])
        query_embedding = embedding_result.embeddings[0]

        # Search in Qdrant
        search_kwargs = {
            "collection_name": self.collection_name,
            "query_vector": query_embedding,
            "limit": limit
        }

        if entity_type:
            search_kwargs["query_filter"] = {
                "must": [{"key": "type", "match": {"value": entity_type}}]
            }

        results = self.qdrant.search(**search_kwargs)

        # Format results
        return [
            {
                "id": result.id,
                "score": result.score,
                "content": result.payload.get("content"),
                "type": result.payload.get("type"),
                "metadata": {
                    k: v for k, v in result.payload.items()
                    if k not in ["content", "type"]
                }
            }
            for result in results
        ]

    async def add_relationship(
        self,
        from_id: str,
        to_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a relationship between two entities in the graph.

        Args:
            from_id: Source entity ID
            to_id: Target entity ID
            relationship_type: Type of relationship (e.g., 'works_at', 'applied_to')
            properties: Optional properties for the relationship
        """
        properties = properties or {}

        # Build Gremlin query to add edge
        gremlin_query = (
            f"g.V().has('id', '{from_id}')"
            f".addE('{relationship_type}')"
            f".to(g.V().has('id', '{to_id}'))"
        )

        # Add properties to edge
        for key, value in properties.items():
            escaped_value = str(value).replace("'", "\\'")
            gremlin_query += f".property('{key}', '{escaped_value}')"

        self.janus.submit(gremlin_query).all().result()

    async def query_graph_traversal(
        self,
        start_id: str,
        gremlin_steps: str
    ) -> List[Dict[str, Any]]:
        """Execute a graph traversal starting from a specific entity.

        Args:
            start_id: Starting entity ID
            gremlin_steps: Gremlin traversal steps to execute

        Returns:
            List of results from the traversal
        """
        gremlin_query = f"g.V().has('id', '{start_id}').{gremlin_steps}"
        results = self.janus.submit(gremlin_query).all().result()
        return results
