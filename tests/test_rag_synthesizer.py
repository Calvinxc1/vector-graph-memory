"""Unit tests for the baseline DSPy RAG synthesizer."""

from types import SimpleNamespace

from vgm.rag import (
    ConversationTurn,
    DspyRagSynthesizer,
    GraphFact,
    RagContext,
    RetrievedPassage,
    normalize_dspy_model_name,
)


class FakePredictor:
    """Record synthesizer inputs and return a fixed prediction."""

    def __init__(self, prediction):
        self.prediction = prediction
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.prediction


def build_rag_context() -> RagContext:
    """Create a representative RAG context for synthesis tests."""
    return RagContext(
        session_id="session-1",
        project_id="project-1",
        use_case_description="Track job search memory and relationships",
        current_question="What do you remember about Acme?",
        retrieval_query="Recent conversation:\nuser: What do you remember about Acme?",
        conversation_history=[
            ConversationTurn(role="user", content="Who did I speak to at Acme?"),
            ConversationTurn(role="assistant", content="You spoke to a recruiter."),
        ],
        retrieved_passages=[
            RetrievedPassage(
                node_id="node-1",
                node_type="interaction",
                content="You spoke to an Acme recruiter named Jane.",
                similarity_score=0.91,
                metadata={"source": "session-a"},
            )
        ],
        graph_facts=[
            GraphFact(
                source_node_id="node-1",
                target_node_id="node-2",
                relationship_type="related_to",
                description="The recruiter interaction is related to the Acme application.",
            )
        ],
    )


def test_normalize_dspy_model_name_translates_provider_model_format():
    assert normalize_dspy_model_name("openai:gpt-4o-mini") == "openai/gpt-4o-mini"
    assert (
        normalize_dspy_model_name("anthropic:claude-sonnet-4-5")
        == "anthropic/claude-sonnet-4-5"
    )
    assert (
        normalize_dspy_model_name("openai/gpt-4o-mini")
        == "openai/gpt-4o-mini"
    )


def test_synthesize_formats_rag_context_and_maps_prediction():
    predictor = FakePredictor(
        SimpleNamespace(
            answer="You spoke to Jane at Acme and later applied there.",
            cited_source_ids=["node-1", "node-2"],
            abstain=False,
        )
    )
    synthesizer = DspyRagSynthesizer(predictor=predictor)

    result = synthesizer.synthesize(build_rag_context())

    assert result.answer == "You spoke to Jane at Acme and later applied there."
    assert result.cited_source_ids == ["node-1", "node-2"]
    assert result.abstain is False
    assert result.backend == "dspy-baseline"
    assert predictor.calls == [
        {
            "conversation_history": [
                "user: Who did I speak to at Acme?",
                "assistant: You spoke to a recruiter.",
            ],
            "question": "What do you remember about Acme?",
            "passages": [
                "[source_id=node-1] [node_type=interaction] [similarity=0.910] You spoke to an Acme recruiter named Jane."
            ],
            "graph_facts": [
                "[source_id=node-1] [target_id=node-2] [relationship=related_to] The recruiter interaction is related to the Acme application."
            ],
            "use_case": "Track job search memory and relationships",
        }
    ]


def test_synthesize_coerces_string_source_id_into_list():
    predictor = FakePredictor(
        SimpleNamespace(
            answer="I only have one grounded source for that.",
            cited_source_ids="node-9",
            abstain=True,
        )
    )
    synthesizer = DspyRagSynthesizer(predictor=predictor)

    result = synthesizer.synthesize(build_rag_context())

    assert result.cited_source_ids == ["node-9"]
    assert result.abstain is True
