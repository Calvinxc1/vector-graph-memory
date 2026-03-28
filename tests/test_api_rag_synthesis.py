"""Tests for the feature-flagged DSPy synthesis API path."""

import asyncio
from types import SimpleNamespace

from vgm.api import server
from vgm.rag import RagContext, RagSynthesisResult


class FakeAgent:
    """Minimal fallback agent stub."""

    def __init__(self, output="fallback answer"):
        self.output = output
        self.calls = []
        self.pending_proposals = {}

    def run(self, prompt, session_id=None):
        self.calls.append({"prompt": prompt, "session_id": session_id})
        return SimpleNamespace(output=self.output)


class FakeRagContextBuilder:
    """Return a fixed context and record calls."""

    def __init__(self, context: RagContext):
        self.context = context
        self.calls = []

    def build_from_messages(self, messages, session_id):
        self.calls.append({"messages": messages, "session_id": session_id})
        return self.context


class FakeRagSynthesizer:
    """Return a fixed synthesis result or raise a configured error."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def synthesize(self, context):
        self.calls.append(context)
        if self.error:
            raise self.error
        return self.result


def build_rag_context() -> RagContext:
    """Create a fixed context for API routing tests."""
    return RagContext(
        session_id="api-session",
        project_id="project-1",
        use_case_description="Track conversational memory",
        current_question="What do you remember about Acme?",
        retrieval_query="What do you remember about Acme?",
    )


def test_chat_completions_uses_dspy_synthesizer_when_enabled(monkeypatch):
    test_state = server.AppState()
    test_state.agent = FakeAgent(output="should not be used")
    test_state.rag_context_builder = FakeRagContextBuilder(build_rag_context())
    test_state.rag_context_enabled = True
    test_state.rag_synthesis_enabled = True
    test_state.rag_synthesizer = FakeRagSynthesizer(
        result=RagSynthesisResult(
            answer="This answer came from DSPy.",
            cited_source_ids=["node-1"],
            abstain=False,
            backend="dspy-baseline",
        )
    )
    monkeypatch.setattr(server, "state", test_state)

    request = server.ChatCompletionRequest(
        model="vector-graph-memory",
        messages=[server.ChatMessage(role="user", content="What do you remember about Acme?")],
        user="api-session",
    )

    response = asyncio.run(server.chat_completions(request))

    assert response.choices[0].message.content == "This answer came from DSPy."
    assert test_state.agent.calls == []
    assert len(test_state.rag_synthesizer.calls) == 1


def test_chat_completions_falls_back_to_memory_agent_on_synthesis_error(monkeypatch):
    test_state = server.AppState()
    test_state.agent = FakeAgent(output="Fallback from MemoryAgent.")
    test_state.rag_context_builder = FakeRagContextBuilder(build_rag_context())
    test_state.rag_context_enabled = True
    test_state.rag_synthesis_enabled = True
    test_state.rag_synthesizer = FakeRagSynthesizer(error=RuntimeError("DSPy failure"))
    monkeypatch.setattr(server, "state", test_state)

    request = server.ChatCompletionRequest(
        model="vector-graph-memory",
        messages=[server.ChatMessage(role="user", content="What do you remember about Acme?")],
        user="api-session",
    )

    response = asyncio.run(server.chat_completions(request))

    assert response.choices[0].message.content == "Fallback from MemoryAgent."
    assert len(test_state.agent.calls) == 1
    assert len(test_state.rag_synthesizer.calls) == 1
