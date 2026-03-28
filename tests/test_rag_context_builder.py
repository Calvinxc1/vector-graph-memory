"""Unit tests for deterministic RAG context assembly."""

from vgm.config import MemoryConfig
from vgm.rag import ConversationTurn, RagContextBuilder
from vgm.schemas import SimilarNode


class StubStore:
    """Minimal store stub for RAG context builder tests."""

    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search_similar_nodes(self, content, limit=5, project_id=None, threshold=None):
        self.calls.append(
            {
                "content": content,
                "limit": limit,
                "project_id": project_id,
                "threshold": threshold,
            }
        )
        return self.results


def build_memory_config() -> MemoryConfig:
    """Create a consistent memory config for tests."""
    return MemoryConfig(
        use_case_description="Track conversational memory for testing",
        memory_threshold_description="Store important facts only",
        project_id="test-project",
        similarity_threshold=0.85,
    )


def test_build_from_messages_uses_current_question_for_single_turn():
    store = StubStore()
    builder = RagContextBuilder(store=store, memory_config=build_memory_config())

    context = builder.build_from_messages(
        messages=[ConversationTurn(role="user", content="What do you know about Acme?")],
        session_id="session-1",
    )

    assert context.current_question == "What do you know about Acme?"
    assert context.retrieval_query == "What do you know about Acme?"
    assert context.conversation_history == []
    assert store.calls == [
        {
            "content": "What do you know about Acme?",
            "limit": 5,
            "project_id": "test-project",
            "threshold": None,
        }
    ]


def test_build_from_messages_preserves_recent_history_and_applies_limit():
    store = StubStore()
    builder = RagContextBuilder(
        store=store,
        memory_config=build_memory_config(),
        history_turn_limit=2,
    )

    context = builder.build_from_messages(
        messages=[
            ConversationTurn(role="user", content="First question"),
            ConversationTurn(role="assistant", content="First answer"),
            ConversationTurn(role="user", content="Second question"),
            ConversationTurn(role="assistant", content="Second answer"),
            ConversationTurn(role="user", content="Third question"),
        ],
        session_id="session-2",
    )

    assert [turn.content for turn in context.conversation_history] == [
        "Second question",
        "Second answer",
    ]
    assert context.retrieval_query == "\n".join(
        [
            "Recent conversation:",
            "user: Second question",
            "assistant: Second answer",
            "Current question: Third question",
        ]
    )


def test_build_from_messages_maps_similar_nodes_into_passages():
    store = StubStore(
        results=[
            SimilarNode(
                node_id="node-1",
                content="Acme is hiring platform engineers.",
                node_type="job",
                similarity_score=0.91,
                metadata={"source": "session-a"},
            ),
            SimilarNode(
                node_id="node-2",
                content="Acme recruiter contacted you last week.",
                node_type="interaction",
                similarity_score=0.84,
                metadata={"source": "session-b"},
            ),
        ]
    )
    builder = RagContextBuilder(store=store, memory_config=build_memory_config())

    context = builder.build_from_messages(
        messages=[ConversationTurn(role="user", content="What happened with Acme?")],
        session_id="session-3",
    )

    assert [passage.node_id for passage in context.retrieved_passages] == [
        "node-1",
        "node-2",
    ]
    assert [passage.node_type for passage in context.retrieved_passages] == [
        "job",
        "interaction",
    ]
    assert context.retrieved_passages[0].metadata == {"source": "session-a"}
    assert context.retrieved_passages[1].similarity_score == 0.84


def test_build_from_messages_rejects_empty_message_list():
    store = StubStore()
    builder = RagContextBuilder(store=store, memory_config=build_memory_config())

    try:
        builder.build_from_messages(messages=[], session_id="session-4")
    except ValueError as exc:
        assert str(exc) == "Cannot build RAG context without messages"
    else:
        raise AssertionError("Expected ValueError for empty message list")


def test_build_from_messages_rejects_blank_current_question():
    store = StubStore()
    builder = RagContextBuilder(store=store, memory_config=build_memory_config())

    try:
        builder.build_from_messages(
            messages=[
                ConversationTurn(role="user", content="Earlier question"),
                ConversationTurn(role="assistant", content="Earlier answer"),
                ConversationTurn(role="user", content="   "),
            ],
            session_id="session-5",
        )
    except ValueError as exc:
        assert str(exc) == "Current question cannot be empty"
    else:
        raise AssertionError("Expected ValueError for blank current question")
