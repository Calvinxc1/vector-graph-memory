"""Deterministic assembly of answer-time RAG context."""

from typing import Sequence

from ..VectorGraphStore import VectorGraphStore
from ..config import MemoryConfig
from .models import ConversationTurn, RagContext, RetrievedPassage


class RagContextBuilder:
    """Build the typed retrieval payload for the future synthesis layer.

    This intentionally stays conservative for the first seam:
    - retrieval uses vector search directly from the native store
    - graph facts are left empty until the schema is proven useful
    - history is preserved as typed turns instead of flattened prompt text
    """

    def __init__(
        self,
        store: VectorGraphStore,
        memory_config: MemoryConfig,
        retrieval_limit: int = 5,
        history_turn_limit: int = 6,
    ):
        self.store = store
        self.memory_config = memory_config
        self.retrieval_limit = retrieval_limit
        self.history_turn_limit = history_turn_limit

    def build_from_messages(
        self,
        messages: Sequence[ConversationTurn],
        session_id: str,
    ) -> RagContext:
        """Build a RAG context from chat messages."""
        if not messages:
            raise ValueError("Cannot build RAG context without messages")

        current_turn = messages[-1]
        history = [turn for turn in messages[:-1] if turn.content.strip()][
            -self.history_turn_limit :
        ]

        current_question = current_turn.content.strip()
        if not current_question:
            raise ValueError("Current question cannot be empty")

        retrieval_query = self._build_retrieval_query(history, current_question)
        similar_nodes = self.store.search_similar_nodes(
            content=retrieval_query,
            limit=self.retrieval_limit,
            project_id=self.memory_config.project_id,
        )

        passages = [
            RetrievedPassage(
                node_id=node.node_id,
                node_type=node.node_type,
                content=node.content,
                similarity_score=node.similarity_score,
                metadata=node.metadata,
            )
            for node in similar_nodes
        ]

        return RagContext(
            session_id=session_id,
            project_id=self.memory_config.project_id,
            use_case_description=self.memory_config.use_case_description,
            current_question=current_question,
            retrieval_query=retrieval_query,
            conversation_history=history,
            retrieved_passages=passages,
        )

    def _build_retrieval_query(
        self,
        history: Sequence[ConversationTurn],
        current_question: str,
    ) -> str:
        """Build a deterministic retrieval query from recent chat context."""
        if not history:
            return current_question

        recent_history = [
            f"{turn.role}: {turn.content.strip()}"
            for turn in history
            if turn.content.strip()
        ]

        return "\n".join(
            [
                "Recent conversation:",
                *recent_history,
                f"Current question: {current_question}",
            ]
        )
