"""Tests for the OpenAI-compatible rules-lawyer API surface."""

import asyncio

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
            issue_type="opponent_orbiter_discount",
            question_score=0.94,
            evidence_score=3,
            total_score=3.94,
            matched_reference_ids=["src-core-discount", "src-faq-discount"],
        ),
        candidate_cases=[
            PilotCaseMatch(
                question_id="seti-rules-002-opponent-orbiter-discount",
                seed_id="seti_landing_orbiter_seed_v1",
                issue_type="opponent_orbiter_discount",
                question_score=0.94,
                evidence_score=3,
                total_score=3.94,
                matched_reference_ids=["src-core-discount", "src-faq-discount"],
            )
        ],
    )


@pytest.fixture(autouse=True)
def isolate_request_trace_log(monkeypatch, tmp_path):
    monkeypatch.setenv("API_TRACE_LOG_PATH", str(tmp_path / "api_traces"))


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
    assert len(test_state.rules_ruling_engine.inspect_calls) == 1
    assert test_state.rules_ruling_engine.calls[0].question == request.messages[-1].content
    assert "Primary authority: Core Rulebook, Land on Planet or Moon (p. 12)" in response.choices[0].message.content
    assert "Modifiers:" in response.choices[0].message.content
    assert "Precedence:" in response.choices[0].message.content
    assert response.model == server.RULES_CHAT_MODEL_ID


def test_list_models_only_exposes_rules_model():
    response = asyncio.run(server.list_models())

    assert [model.id for model in response.data] == [server.RULES_CHAT_MODEL_ID]


def test_chat_completions_rejects_disabled_memory_model():
    request = server.ChatCompletionRequest(
        model=server.MEMORY_CHAT_MODEL_ID,
        messages=[server.ChatMessage(role="user", content="Hello")],
        user="api-session",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.chat_completions(request))

    assert exc_info.value.status_code == 400
    assert "temporarily disabled" in exc_info.value.detail
    assert server.RULES_CHAT_MODEL_ID in exc_info.value.detail


def test_chat_completions_rejects_unknown_model(monkeypatch):
    request = server.ChatCompletionRequest(
        model="not-a-real-model",
        messages=[server.ChatMessage(role="user", content="Hello")],
        user="api-session",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.chat_completions(request))

    assert exc_info.value.status_code == 400
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
    assert "Log file:" in body
    assert "Route: seti-rules-lawyer -> live pilot ruling" in body
    assert "Matched case: seti-rules-002-opponent-orbiter-discount" in body
    assert "Top candidates:" in body
    assert "Retrieval:" in body
    assert "Primary authority: Core Rulebook, Land on Planet or Moon (p. 12)" in body
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body


def test_streaming_response_moves_model_thinking_under_trace_summary():
    response = server._build_streaming_chat_response(
        model=server.RULES_CHAT_MODEL_ID,
        session_id="thinking-session",
        trace_summary="<think>\nTrace summary:\n- Route: test\n</think>",
        assistant_response="<think>Internal model chain.</think>\n\nVisible answer.",
    )

    body = asyncio.run(_collect_streaming_response_body(response))

    assert "Trace summary:" in body
    assert "Model thinking:" in body
    assert "Internal model chain." in body
    assert "Visible answer." in body
    assert body.index("Trace summary:") < body.index("Model thinking:")
    assert body.index("Model thinking:") < body.index("Visible answer.")
    assert "<think>Internal model chain.</think>" not in body
