"""Tests for the feature-flagged DSPy synthesis API path."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("dotenv")
from fastapi import HTTPException

from vgm.api import server
from vgm.rules import (
    LivePilotRulingInspection,
    PilotCaseMatch,
    PilotSeedInference,
    PilotSeedScore,
    RetrievedRulingEvidence,
    PrecedenceEntry,
    RuleCitation,
    RuleReference,
    RulesRulingResult,
)
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


class FakeCompileManager:
    """Track whether the API tries to queue background compilation."""

    def __init__(self, should_begin=True):
        self.should_begin = should_begin
        self.begin_calls = 0

    def begin_auto_compile(self):
        self.begin_calls += 1
        return self.should_begin


class FakeRulesRulingEngine:
    """Return a fixed pilot ruling and record the incoming request."""

    def __init__(self, result: RulesRulingResult):
        self.result = result
        self.calls = []
        self.inspect_calls = []

    def answer(self, request):
        self.calls.append(request)
        return self.result

    def inspect_request(self, request):
        self.inspect_calls.append(request)
        return build_live_ruling_inspection(request.question)


def build_rag_context() -> RagContext:
    """Create a fixed context for API routing tests."""
    return RagContext(
        session_id="api-session",
        project_id="project-1",
        use_case_description="Track conversational memory",
        current_question="What do you remember about Acme?",
        retrieval_query="What do you remember about Acme?",
    )


def build_rules_ruling_result() -> RulesRulingResult:
    """Create a fixed ruling result for OpenAI-model routing tests."""
    return RulesRulingResult(
        question="If another player already has an orbiter there, do I still get the cheaper landing cost?",
        question_id="seti-rules-002-opponent-orbiter-discount",
        seed_id="seti_landing_orbiter_seed_v1",
        subsystem="landing",
        ruling=(
            "Yes. The FAQ says the discount applies if any orbiter is already at the planet, "
            "including an opponent's orbiter."
        ),
        primary_rule=RuleReference(
            rule_node_id="rule-core-discount",
            title="Landing discount with existing orbiter",
            rule_kind="core_rule",
            normalized_statement="If an orbiter is already at the planet, landing there costs 1 less energy.",
        ),
        primary_citation=RuleCitation(
            source_node_id="src-core-discount",
            citation_label="Core Rulebook, Land on Planet or Moon",
            citation_short="Land on Planet or Moon",
            title="Core Rulebook",
            locator="p. 12",
            authority_scope="core",
            source_excerpt="If an orbiter is already at the planet, landing there costs 1 less energy.",
        ),
        modifying_citations=[
            RuleCitation(
                source_node_id="src-faq-discount",
                citation_label="Official FAQ Q6",
                citation_short="FAQ Q6",
                title="Official FAQ",
                locator="Q6",
                authority_scope="faq",
                source_excerpt="The discount applies regardless of which player owns the orbiter.",
            )
        ],
        precedence_order=[
            PrecedenceEntry(
                order=1,
                summary="Start with the core landing discount rule.",
                rule_node_id="rule-core-discount",
                source_node_id="src-core-discount",
                precedence_kind="primary",
            ),
            PrecedenceEntry(
                order=2,
                summary="Apply the FAQ clarification that ownership does not matter.",
                rule_node_id=None,
                source_node_id="src-faq-discount",
                precedence_kind="modifier",
            )
        ],
        abstain=False,
        backend="live-pilot",
    )


def build_live_ruling_inspection(question: str) -> LivePilotRulingInspection:
    """Create a fixed inspection trace for streamed thinking tests."""
    return LivePilotRulingInspection(
        question=question,
        normalized_question="if opponent orbiter discount applies",
        evidence=RetrievedRulingEvidence(
            question=question,
            seed_id="seti_landing_orbiter_seed_v1",
            subsystem="landing",
        ),
        seed_inference=PilotSeedInference(
            normalized_question="if opponent orbiter discount applies",
            selected_seed_id="seti_landing_orbiter_seed_v1",
            selected_score=0.98,
            candidates=[
                PilotSeedScore(seed_id="seti_landing_orbiter_seed_v1", score=0.98),
                PilotSeedScore(seed_id="seti_free_action_authority_seed_v1", score=0.11),
            ],
        ),
        selected_seed_id="seti_landing_orbiter_seed_v1",
        selected_case=PilotCaseMatch(
            question_id="seti-rules-002-opponent-orbiter-discount",
            seed_id="seti_landing_orbiter_seed_v1",
            question_score=0.94,
            evidence_score=3,
            total_score=3.94,
            matched_reference_ids=["src-core-discount", "src-faq-discount"],
        ),
        candidate_cases=[
            PilotCaseMatch(
                question_id="seti-rules-002-opponent-orbiter-discount",
                seed_id="seti_landing_orbiter_seed_v1",
                question_score=0.94,
                evidence_score=3,
                total_score=3.94,
                matched_reference_ids=["src-core-discount", "src-faq-discount"],
            )
        ],
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


def test_chat_completions_queues_background_compile_once(monkeypatch):
    test_state = server.AppState()
    test_state.agent = FakeAgent(output="should not be used")
    test_state.rag_context_builder = FakeRagContextBuilder(build_rag_context())
    test_state.rag_context_enabled = True
    test_state.rag_synthesis_enabled = True
    test_state.rag_synthesizer = FakeRagSynthesizer(
        result=RagSynthesisResult(
            answer="DSPy answer while compile runs in the background.",
            cited_source_ids=["node-1"],
            abstain=False,
            backend="dspy-baseline",
        )
    )
    test_state.rag_compile_manager = FakeCompileManager(should_begin=True)
    monkeypatch.setattr(server, "state", test_state)

    scheduled = {}

    def fake_create_task(coro):
        scheduled["started"] = True
        coro.close()
        return "task-sentinel"

    monkeypatch.setattr(server.asyncio, "create_task", fake_create_task)

    request = server.ChatCompletionRequest(
        model="vector-graph-memory",
        messages=[server.ChatMessage(role="user", content="What do you remember about Acme?")],
        user="api-session",
    )

    response = asyncio.run(server.chat_completions(request))

    assert response.choices[0].message.content == "DSPy answer while compile runs in the background."
    assert test_state.rag_compile_manager.begin_calls == 1
    assert scheduled["started"] is True
    assert test_state.rag_compile_task == "task-sentinel"


def test_chat_completions_routes_rules_model_to_ruling_engine(monkeypatch):
    test_state = server.AppState()
    test_state.rules_ruling_engine = FakeRulesRulingEngine(build_rules_ruling_result())
    monkeypatch.setattr(server, "state", test_state)

    request = server.ChatCompletionRequest(
        model=server.RULES_CHAT_MODEL_ID,
        messages=[
            server.ChatMessage(role="system", content="You answer rules questions."),
            server.ChatMessage(
                role="user",
                content="If another player already has an orbiter there, do I still get the cheaper landing cost?",
            ),
        ],
        user="rules-session",
    )

    response = asyncio.run(server.chat_completions(request))

    assert len(test_state.rules_ruling_engine.calls) == 1
    assert test_state.rules_ruling_engine.inspect_calls == []
    assert test_state.rules_ruling_engine.calls[0].question == request.messages[-1].content
    assert "Primary authority: Core Rulebook, Land on Planet or Moon (p. 12)" in response.choices[0].message.content
    assert "Modifiers:" in response.choices[0].message.content
    assert "Precedence:" in response.choices[0].message.content
    assert response.model == server.RULES_CHAT_MODEL_ID


def test_chat_completions_rejects_unknown_model(monkeypatch):
    test_state = server.AppState()
    test_state.agent = FakeAgent()
    monkeypatch.setattr(server, "state", test_state)

    request = server.ChatCompletionRequest(
        model="not-a-real-model",
        messages=[server.ChatMessage(role="user", content="Hello")],
        user="api-session",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.chat_completions(request))

    assert exc_info.value.status_code == 400
    assert server.MEMORY_CHAT_MODEL_ID in exc_info.value.detail
    assert server.RULES_CHAT_MODEL_ID in exc_info.value.detail


async def _collect_streaming_response_body(response) -> str:
    parts = []
    async for chunk in response.body_iterator:
        parts.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(parts)


def test_chat_completions_streams_rules_model_as_sse(monkeypatch):
    test_state = server.AppState()
    test_state.rules_ruling_engine = FakeRulesRulingEngine(build_rules_ruling_result())
    monkeypatch.setattr(server, "state", test_state)

    request = server.ChatCompletionRequest(
        model=server.RULES_CHAT_MODEL_ID,
        messages=[
            server.ChatMessage(
                role="user",
                content="If another player already has an orbiter there, do I still get the cheaper landing cost?",
            )
        ],
        stream=True,
        user="rules-stream-session",
    )

    response = asyncio.run(server.chat_completions(request))
    body = asyncio.run(_collect_streaming_response_body(response))

    assert response.media_type == "text/event-stream"
    assert len(test_state.rules_ruling_engine.calls) == 1
    assert len(test_state.rules_ruling_engine.inspect_calls) == 1
    assert '"object": "chat.completion.chunk"' in body
    assert '"role": "assistant"' in body
    assert "<think>" in body
    assert "Trace summary:" in body
    assert "Matched case: seti-rules-002-opponent-orbiter-discount" in body
    assert "Primary authority: Core Rulebook, Land on Planet or Moon (p. 12)" in body
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body
