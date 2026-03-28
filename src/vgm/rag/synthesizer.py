"""Baseline DSPy synthesis for answer-time RAG."""

from typing import Any, Optional

import dspy

from .models import RagContext, RagSynthesisResult


def normalize_dspy_model_name(model_name: str) -> str:
    """Translate provider:model strings into DSPy provider/model names."""
    if "/" in model_name:
        return model_name
    if ":" in model_name:
        provider, model = model_name.split(":", 1)
        return f"{provider}/{model}"
    return model_name


def build_dspy_lm(
    llm_model: str,
    *,
    model_name_override: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model_type: Optional[str] = None,
) -> Any:
    """Create a DSPy LM using the current repo model configuration."""
    model_name = model_name_override or normalize_dspy_model_name(llm_model)
    kwargs: dict[str, Any] = {}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if api_base is not None:
        kwargs["api_base"] = api_base
    if model_type is not None:
        kwargs["model_type"] = model_type
    return dspy.LM(model_name, **kwargs)


class BaselineRagAnswerSignature(dspy.Signature):
    """Generate a grounded answer from retrieved vector and graph context."""

    conversation_history: list[str] = dspy.InputField(
        desc="Recent conversation turns, oldest to newest."
    )
    question: str = dspy.InputField(desc="The current user question.")
    passages: list[str] = dspy.InputField(
        desc="Retrieved passages. Use them as the primary evidence."
    )
    graph_facts: list[str] = dspy.InputField(
        desc="Graph-derived facts. This list may be empty."
    )
    use_case: str = dspy.InputField(desc="The memory system's use case.")
    answer: str = dspy.OutputField(
        desc="A grounded answer that uses only the provided evidence."
    )
    cited_source_ids: list[str] = dspy.OutputField(
        desc="A list of source node IDs used in the answer."
    )
    abstain: bool = dspy.OutputField(
        desc="True if the evidence is insufficient for a grounded answer."
    )


class DspyRagSynthesizer:
    """Baseline DSPy synthesizer over a typed RAG context."""

    def __init__(
        self,
        predictor: Any,
        backend_name: str = "dspy-baseline",
    ):
        self.predictor = predictor
        self.backend_name = backend_name

    @classmethod
    def from_lm(cls, lm: Any) -> "DspyRagSynthesizer":
        """Build a synthesizer bound to a DSPy LM."""
        predictor = dspy.Predict(BaselineRagAnswerSignature)
        predictor.set_lm(lm)
        return cls(predictor=predictor)

    def synthesize(self, context: RagContext) -> RagSynthesisResult:
        """Generate a grounded answer from a structured RAG context."""
        prediction = self.predictor(
            conversation_history=self._format_conversation_history(context),
            question=context.current_question,
            passages=self._format_passages(context),
            graph_facts=self._format_graph_facts(context),
            use_case=context.use_case_description,
        )

        cited_source_ids = getattr(prediction, "cited_source_ids", [])
        if isinstance(cited_source_ids, str):
            cited_source_ids = [cited_source_ids]
        elif cited_source_ids is None:
            cited_source_ids = []

        return RagSynthesisResult(
            answer=str(getattr(prediction, "answer", "")).strip(),
            cited_source_ids=[str(source_id) for source_id in cited_source_ids],
            abstain=bool(getattr(prediction, "abstain", False)),
            backend=self.backend_name,
        )

    @staticmethod
    def _format_conversation_history(context: RagContext) -> list[str]:
        return [
            f"{turn.role}: {turn.content.strip()}"
            for turn in context.conversation_history
            if turn.content.strip()
        ]

    @staticmethod
    def _format_passages(context: RagContext) -> list[str]:
        return [
            (
                f"[source_id={passage.node_id}] "
                f"[node_type={passage.node_type}] "
                f"[similarity={passage.similarity_score:.3f}] "
                f"{passage.content}"
            )
            for passage in context.retrieved_passages
        ]

    @staticmethod
    def _format_graph_facts(context: RagContext) -> list[str]:
        return [
            (
                f"[source_id={fact.source_node_id}] "
                f"[target_id={fact.target_node_id}] "
                f"[relationship={fact.relationship_type}] "
                f"{fact.description}".strip()
            )
            for fact in context.graph_facts
        ]
