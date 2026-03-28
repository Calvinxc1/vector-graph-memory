"""Typed models for answer-time RAG context."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """One prior chat turn preserved for RAG context assembly."""

    role: str
    content: str


class RetrievedPassage(BaseModel):
    """A vector-retrieved node exposed to the answer synthesizer."""

    node_id: str
    node_type: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphFact(BaseModel):
    """A graph-derived fact reserved for later synthesis phases."""

    source_node_id: str
    target_node_id: str
    relationship_type: str
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RagContext(BaseModel):
    """Structured retrieval payload for answer synthesis."""

    session_id: str
    project_id: str
    use_case_description: str
    current_question: str
    retrieval_query: str
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    retrieved_passages: List[RetrievedPassage] = Field(default_factory=list)
    graph_facts: List[GraphFact] = Field(default_factory=list)


class RagSynthesisResult(BaseModel):
    """Structured answer generated from a RAG context."""

    answer: str
    cited_source_ids: List[str] = Field(default_factory=list)
    abstain: bool = False
    backend: str
