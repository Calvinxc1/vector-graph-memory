"""Vector and graph database storage layer."""

import asyncio
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, cast
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from gremlin_python.driver import client as gremlin_client  # type: ignore[import-untyped]
from pydantic_ai import EmbeddingModel

from .schemas import NodeMetadata, EdgeMetadata, SimilarNode
from .config import VectorGraphConfig

T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run async code, handling both notebook and regular Python environments.

    This helper manages event loop differences between:
    - Regular Python (no running loop)
    - Jupyter notebooks (with running loop)
    - FastAPI/uvicorn (with uvloop that can't be patched)
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - we're in regular Python
        return asyncio.run(coro)
    else:
        # There's a running loop
        # Check if it's patchable (uvloop isn't, but asyncio default loop is)
        import nest_asyncio  # type: ignore[import-untyped]
        import threading

        loop_type = type(loop).__name__

        if loop_type == "Loop" and "uvloop" in loop.__class__.__module__:
            # uvloop can't be patched - run in new thread with new loop
            result: list[T] = []
            exception: Exception | None = None

            def run_in_thread() -> None:
                nonlocal result, exception
                try:
                    result.append(asyncio.run(coro))
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            if exception:
                raise exception
            if not result:
                raise RuntimeError("Async operation did not produce a result")
            return result[0]
        else:
            # Regular asyncio loop - can be patched for notebooks
            nest_asyncio.apply()
            return asyncio.run(coro)


def _run_in_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a synchronous function in a separate thread.

    This is needed for Gremlin client operations which internally use
    run_until_complete() and fail when called from an existing event loop
    (like FastAPI/uvicorn).
    """
    import threading

    result: list[T] = []
    exception: Exception | None = None

    def run() -> None:
        nonlocal result, exception
        try:
            result.append(func(*args, **kwargs))
        except Exception as e:
            exception = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    if exception:
        raise exception
    if not result:
        raise RuntimeError("Threaded operation did not produce a result")
    return result[0]


class VectorGraphStore:
    """Hybrid vector-graph memory storage system.

    Storage strategy:
    - Qdrant: Vectors + full content + metadata (source of truth for data)
    - JanusGraph: Node references + relationships only (source of truth for structure)
    """

    # Content size limit (1 MB soft limit, configurable)
    DEFAULT_MAX_CONTENT_SIZE = 1_000_000  # ~1 million characters

    def __init__(
        self,
        qdrant_client: QdrantClient,
        janus_client: gremlin_client.Client,
        embedding_model: EmbeddingModel,
        config: Optional[VectorGraphConfig] = None,
        max_content_size: Optional[int] = None,
    ):
        """Initialize vector-graph store.

        Args:
            qdrant_client: Qdrant client instance
            janus_client: JanusGraph Gremlin client instance
            embedding_model: PydanticAI EmbeddingModel instance
            config: Storage configuration (optional)
            max_content_size: Maximum content size in characters (default: 1MB)
        """
        self.qdrant = qdrant_client
        self.janus = janus_client
        self.embedding_model = embedding_model
        self.config = config or VectorGraphConfig()
        self.max_content_size = max_content_size or self.DEFAULT_MAX_CONTENT_SIZE

        self._ensure_janusgraph_schema()
        self._ensure_collection()

    def _ensure_janusgraph_schema(self) -> None:
        """Check if JanusGraph schema is initialized.

        Note: Schema initialization is skipped in this method to avoid asyncio
        event loop conflicts in Jupyter notebooks. Run scripts/init_janusgraph_schema.py
        once before using the library.
        """
        # Skip automatic schema initialization - must be done manually
        # This avoids asyncio event loop conflicts in Jupyter notebooks
        pass

    @staticmethod
    def _escape_gremlin_value(val: Any) -> str:
        """Escape values for safe use in Gremlin queries."""
        return str(val).replace("'", "\\'").replace('"', '\\"')

    def _ensure_collection(self) -> None:
        """Ensure the Qdrant collection exists."""
        collections = self.qdrant.get_collections().collections
        if not any(c.name == self.config.qdrant_collection for c in collections):
            # Get embedding dimension from a test embedding
            test_result = _run_async(
                self.embedding_model.embed(["test"], input_type="document")
            )
            embedding_dim = len(test_result.embeddings[0])

            self.qdrant.create_collection(
                collection_name=self.config.qdrant_collection,
                vectors_config=VectorParams(
                    size=embedding_dim, distance=Distance.COSINE
                ),
            )

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the embedding model.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        result = _run_async(self.embedding_model.embed([text], input_type="document"))
        return list(result.embeddings[0])

    def search_similar_nodes(
        self,
        content: str,
        threshold: Optional[float] = None,
        limit: int = 5,
        project_id: Optional[str] = None,
    ) -> List[SimilarNode]:
        """Search for similar existing nodes using Top-K retrieval.

        Args:
            content: Text to search for
            threshold: Optional minimum similarity score (0.0-1.0). If None, returns top K results.
            limit: Maximum number of results (default: 5)
            project_id: Filter by project (optional)

        Returns:
            List of top K similar nodes, optionally filtered by threshold
        """
        import logging

        logger = logging.getLogger(__name__)

        embedding = self._generate_embedding(content)

        # Build search parameters
        query_filter: Filter | None = None
        if project_id:
            query_filter = Filter(
                must=[
                    FieldCondition(key="project_id", match=MatchValue(value=project_id))
                ]
            )

        logger.info(f"[SEARCH] Query: '{content[:100]}...'")
        logger.info(f"[SEARCH] Project ID: {project_id}")
        logger.info(f"[SEARCH] Threshold: {threshold}, Limit: {limit}")

        response = self.qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,  # type: ignore[arg-type]  # list[float] is valid but not in type stubs
            limit=limit,
            score_threshold=threshold,  # None means no threshold - Top-K retrieval
            query_filter=query_filter,
        )

        logger.info(f"[SEARCH] Found {len(response.points)} results")
        for i, result in enumerate(response.points, 1):
            payload = cast(Dict[str, Any], result.payload or {})
            logger.info(
                f"[SEARCH]   {i}. [{payload.get('node_type')}] "
                f"score={result.score:.3f} content='{str(payload.get('content', ''))[:80]}...'"
            )

        return [
            SimilarNode(
                node_id=str(result.id),
                content=cast(str, (result.payload or {})["content"]),
                node_type=cast(str, (result.payload or {})["node_type"]),
                similarity_score=result.score,
                metadata={
                    k: v
                    for k, v in cast(Dict[str, Any], result.payload or {}).items()
                    if k not in ["content", "node_type"]
                },
            )
            for result in response.points
        ]

    def add_node(
        self,
        metadata: NodeMetadata,
    ) -> str:
        """Add node to both vector and graph databases.

        Args:
            metadata: Complete node metadata including content

        Returns:
            Node ID (UUID)

        Raises:
            ValueError: If content exceeds max_content_size
        """
        import logging

        logger = logging.getLogger(__name__)

        # Check content size
        if len(metadata.content) > self.max_content_size:
            raise ValueError(
                f"Content size ({len(metadata.content)} chars) exceeds maximum "
                f"({self.max_content_size} chars). Consider chunking large content."
            )

        # Generate embedding
        embedding = self._generate_embedding(metadata.content)

        # Store FULL DATA in Qdrant (source of truth for content)
        payload = {
            "content": metadata.content,
            "node_type": metadata.node_type,
            "created_at": metadata.created_at.isoformat(),
            "updated_at": metadata.updated_at.isoformat(),
            "source": metadata.source,
            "project_id": metadata.project_id,
            "embedding_model": metadata.embedding_model,
            **metadata.custom_metadata,
        }

        logger.info(f"[STORE] Adding node ID: {metadata.node_id}")
        logger.info(f"[STORE]   Type: {metadata.node_type}")
        logger.info(f"[STORE]   Project ID: {metadata.project_id}")
        logger.info(f"[STORE]   Content: '{metadata.content[:100]}...'")

        self.qdrant.upsert(
            collection_name=self.config.qdrant_collection,
            points=[
                PointStruct(
                    id=metadata.node_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

        logger.info("[STORE] Successfully added to Qdrant")

        # Store MINIMAL REFERENCE in JanusGraph (structure only)
        # Use generic "node" label, store actual type as a property
        # Use iterate() to avoid returning the vertex (avoids serialization issues)
        gremlin_query = (
            f"g.addV('node')"
            f".property('node_id', '{self._escape_gremlin_value(metadata.node_id)}')"
            f".property('node_type', '{self._escape_gremlin_value(metadata.node_type)}')"
            f".property('project_id', '{self._escape_gremlin_value(metadata.project_id)}')"
            f".property('created_at', '{metadata.created_at.isoformat()}')"
            f".property('updated_at', '{metadata.updated_at.isoformat()}')"
            f".iterate()"
        )
        _run_in_thread(lambda: self.janus.submit(gremlin_query).all().result())

        return metadata.node_id

    def update_node(
        self,
        node_id: str,
        content: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update existing node.

        Args:
            node_id: Node ID to update
            content: New content (will regenerate embedding if provided)
            custom_metadata: Metadata fields to update

        Raises:
            ValueError: If node not found or content exceeds max_content_size
        """
        # Get current node from Qdrant
        current = self.qdrant.retrieve(
            collection_name=self.config.qdrant_collection,
            ids=[node_id],
        )

        if not current:
            raise ValueError(f"Node {node_id} not found")

        current_payload = dict(cast(Dict[str, Any], current[0].payload or {}))
        updated_at = datetime.now(timezone.utc).isoformat()

        # Update payload
        if content:
            # Check content size
            if len(content) > self.max_content_size:
                raise ValueError(
                    f"Content size ({len(content)} chars) exceeds maximum "
                    f"({self.max_content_size} chars)"
                )

            current_payload["content"] = content
            current_payload["updated_at"] = updated_at
            # Regenerate embedding
            embedding = self._generate_embedding(content)
        else:
            # Keep existing embedding
            stored_vector = self.qdrant.retrieve(
                collection_name=self.config.qdrant_collection,
                ids=[node_id],
                with_vectors=True,
            )[0].vector
            if not isinstance(stored_vector, list):
                raise TypeError(f"Expected stored vector list for node {node_id}")
            if any(isinstance(value, list) for value in stored_vector):
                raise TypeError(f"Expected flat vector list for node {node_id}")
            embedding = [float(value) for value in cast(List[float], stored_vector)]

        if custom_metadata:
            current_payload.update(custom_metadata)
            current_payload["updated_at"] = updated_at

        # Update in Qdrant (content + vector)
        self.qdrant.upsert(
            collection_name=self.config.qdrant_collection,
            points=[
                PointStruct(
                    id=node_id,
                    vector=embedding,
                    payload=current_payload,
                )
            ],
        )

        # Update timestamp in JanusGraph (structure only)
        update_query = (
            f"g.V().has('node_id', '{self._escape_gremlin_value(node_id)}')"
            f".property('updated_at', '{updated_at}')"
        )

        _run_in_thread(lambda: self.janus.submit(update_query).all().result())

    def add_edge(
        self,
        metadata: EdgeMetadata,
    ) -> str:
        """Add edge to graph database.

        Args:
            metadata: Complete edge metadata

        Returns:
            Edge ID (UUID)
        """
        # Build Gremlin query to add edge.
        # Use generic "relationship" label and store actual type as a property.
        # The `to(...)` form requires an anonymous child traversal on JanusGraph,
        # so use `as()/addE()/to(select())` to keep the query server-compatible.
        gremlin_query = (
            f"g.V().has('node_id', '{self._escape_gremlin_value(metadata.from_node_id)}')"
            f".as('from_node')"
            f".V().has('node_id', '{self._escape_gremlin_value(metadata.to_node_id)}')"
            f".as('to_node')"
            f".addE('relationship').from('from_node').to('to_node')"
            f".property('edge_id', '{self._escape_gremlin_value(metadata.edge_id)}')"
            f".property('relationship_type', '{self._escape_gremlin_value(metadata.relationship_type)}')"
            f".property('description', '{self._escape_gremlin_value(metadata.description)}')"
            f".property('created_at', '{metadata.created_at.isoformat()}')"
            f".property('source', '{self._escape_gremlin_value(metadata.source)}')"
            f".property('project_id', '{self._escape_gremlin_value(metadata.project_id)}')"
        )
        if metadata.confidence is not None:
            gremlin_query += f".property('confidence', {float(metadata.confidence)}d)"
        for key, value in metadata.custom_metadata.items():
            gremlin_query += f".property('{self._escape_gremlin_value(key)}', '{self._escape_gremlin_value(value)}')"
        gremlin_query += ".iterate()"
        _run_in_thread(lambda: self.janus.submit(gremlin_query).all().result())

        return metadata.edge_id

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve node by ID from Qdrant.

        Args:
            node_id: Node ID

        Returns:
            Node data (including content) or None if not found
        """
        results = self.qdrant.retrieve(
            collection_name=self.config.qdrant_collection,
            ids=[node_id],
        )

        if not results:
            return None

        return cast(Dict[str, Any], results[0].payload or {})

    def get_nodes_batch(self, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Retrieve multiple nodes by ID from Qdrant.

        Args:
            node_ids: List of node IDs

        Returns:
            Dictionary mapping node_id to node data
        """
        results = self.qdrant.retrieve(
            collection_name=self.config.qdrant_collection,
            ids=node_ids,
        )

        return {
            str(result.id): cast(Dict[str, Any], result.payload or {})
            for result in results
        }

    def traverse_from_node(
        self,
        node_id: str,
        gremlin_steps: str,
    ) -> List[Dict[str, Any]]:
        """Execute a graph traversal starting from a specific node.

        Args:
            node_id: Starting node ID
            gremlin_steps: Gremlin traversal steps to execute

        Returns:
            List of traversal results with full node content

        Note:
            If traversal returns node IDs, this will batch-fetch full content from Qdrant
        """
        gremlin_query = (
            f"g.V().has('node_id', '{self._escape_gremlin_value(node_id)}')"
            f".{gremlin_steps}"
        )
        results = _run_in_thread(
            lambda: self.janus.submit(gremlin_query).all().result()
        )

        # If results contain node_ids, fetch full content from Qdrant
        # This is a simple heuristic - results might be node_ids or other data
        if results and all(isinstance(r, str) for r in results):
            # Looks like node IDs, batch fetch from Qdrant
            node_data = self.get_nodes_batch(results)
            return [node_data.get(node_id, {"node_id": node_id}) for node_id in results]

        return results
