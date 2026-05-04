"""Runtime model-provider construction for chat and embedding models."""

from __future__ import annotations

import os
from typing import Literal

from pydantic_ai import EmbeddingModel
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


ModelProviderName = Literal["openai", "ollama"]

OPENAI_DEFAULT_CHAT_MODEL = "openai:gpt-4o-mini"
OPENAI_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
OLLAMA_DEFAULT_CHAT_MODEL = "llama3.1:8b"
OLLAMA_DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DUMMY_OPENAI_COMPATIBLE_API_KEY = "api-key-not-set"


def _normalized_provider(value: str | None, *, env_name: str) -> ModelProviderName:
    provider = (value or "openai").strip().lower()
    if provider not in {"openai", "ollama"}:
        raise ValueError(f"{env_name} must be one of: openai, ollama")
    return provider  # type: ignore[return-value]


def llm_provider_from_env() -> ModelProviderName:
    """Return the configured chat/runtime LLM provider."""

    return _normalized_provider(os.getenv("LLM_PROVIDER"), env_name="LLM_PROVIDER")


def embedding_provider_from_env() -> ModelProviderName:
    """Return the configured embedding provider.

    If EMBEDDING_PROVIDER is unset, embeddings follow LLM_PROVIDER. This makes
    full local-provider switches concise while preserving OpenAI as the default.
    """

    return _normalized_provider(
        os.getenv("EMBEDDING_PROVIDER") or os.getenv("LLM_PROVIDER"),
        env_name="EMBEDDING_PROVIDER",
    )


def _strip_provider_prefix(model_name: str, provider: ModelProviderName) -> str:
    for prefix in (f"{provider}:", f"{provider}/"):
        if model_name.startswith(prefix):
            return model_name[len(prefix) :]
    if provider == "ollama":
        for prefix in ("openai:", "openai/"):
            if model_name.startswith(prefix):
                return model_name[len(prefix) :]
    return model_name


def chat_model_name_from_env(model_name: str | None = None) -> str:
    """Return the configured chat model label, including provider prefix."""

    provider = llm_provider_from_env()
    if provider == "ollama":
        raw_model = (
            model_name
            or os.getenv("OLLAMA_CHAT_MODEL")
            or os.getenv("LLM_MODEL")
            or OLLAMA_DEFAULT_CHAT_MODEL
        )
        return f"ollama:{_strip_provider_prefix(raw_model, provider)}"
    return model_name or os.getenv("LLM_MODEL", OPENAI_DEFAULT_CHAT_MODEL)


def embedding_model_name_from_env(model_name: str | None = None) -> str:
    """Return the configured embedding model label, including provider prefix when useful."""

    provider = embedding_provider_from_env()
    if provider == "ollama":
        raw_model = (
            model_name
            or os.getenv("OLLAMA_EMBEDDING_MODEL")
            or os.getenv("EMBEDDING_MODEL")
            or OLLAMA_DEFAULT_EMBEDDING_MODEL
        )
        return f"ollama:{_strip_provider_prefix(raw_model, provider)}"
    return model_name or os.getenv("EMBEDDING_MODEL", OPENAI_DEFAULT_EMBEDDING_MODEL)


def _ollama_base_url(*, embedding: bool = False) -> str:
    value = (
        os.getenv("OLLAMA_EMBEDDING_BASE_URL")
        if embedding
        else os.getenv("OLLAMA_CHAT_BASE_URL")
    )
    value = value or os.getenv("OLLAMA_BASE_URL")
    if not value:
        env_hint = "OLLAMA_EMBEDDING_BASE_URL or OLLAMA_BASE_URL" if embedding else "OLLAMA_CHAT_BASE_URL or OLLAMA_BASE_URL"
        raise ValueError(f"{env_hint} must be set when using the ollama provider")
    return value


def _ollama_api_key() -> str | None:
    return os.getenv("OLLAMA_API_KEY") or os.getenv("LLM_API_KEY") or DUMMY_OPENAI_COMPATIBLE_API_KEY


def build_chat_model_from_env(model_name: str | None = None) -> str | Model:
    """Build a PydanticAI chat model from runtime provider configuration."""

    provider = llm_provider_from_env()
    resolved_name = chat_model_name_from_env(model_name)
    if provider == "ollama":
        model_id = _strip_provider_prefix(resolved_name, provider)
        return OpenAIChatModel(
            model_id,
            provider=OpenAIProvider(base_url=_ollama_base_url(), api_key=_ollama_api_key()),
        )
    return resolved_name


def build_embedding_model_from_env(model_name: str | None = None) -> EmbeddingModel:
    """Build a PydanticAI embedding model from runtime provider configuration."""

    provider = embedding_provider_from_env()
    resolved_name = embedding_model_name_from_env(model_name)
    if provider == "ollama":
        model_id = _strip_provider_prefix(resolved_name, provider)
        return OpenAIEmbeddingModel(
            model_id,
            provider=OpenAIProvider(
                base_url=_ollama_base_url(embedding=True),
                api_key=_ollama_api_key(),
            ),
        )

    openai_base_url = os.getenv("EMBEDDING_API_BASE") or os.getenv("OPENAI_BASE_URL")
    if openai_base_url:
        return OpenAIEmbeddingModel(
            _strip_provider_prefix(resolved_name, "openai"),
            provider=OpenAIProvider(
                base_url=openai_base_url,
                api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            ),
        )
    return OpenAIEmbeddingModel(_strip_provider_prefix(resolved_name, "openai"))


def dspy_model_name_for_provider(model_name: str) -> str:
    """Translate a repo model label into the DSPy/LiteLLM model label."""

    if model_name.startswith(("ollama:", "ollama/")):
        model_id = _strip_provider_prefix(model_name, "ollama")
        return f"openai/{model_id}"
    if "/" in model_name:
        return model_name
    if ":" in model_name:
        provider_name, model_id = model_name.split(":", 1)
        return f"{provider_name}/{model_id}"
    if llm_provider_from_env() == "ollama":
        return f"openai/{model_name}"
    return model_name


def dspy_api_base_from_env(api_base: str | None = None) -> str | None:
    """Return DSPy API base, inheriting external Ollama configuration."""

    if api_base is not None:
        return api_base
    if llm_provider_from_env() == "ollama":
        return os.getenv("DSPY_API_BASE") or _ollama_base_url()
    return os.getenv("DSPY_API_BASE")


def dspy_api_key_from_env(api_key: str | None = None) -> str | None:
    """Return DSPy API key, using a placeholder for unauthenticated Ollama."""

    if api_key is not None:
        return api_key
    if llm_provider_from_env() == "ollama":
        return os.getenv("DSPY_API_KEY") or _ollama_api_key()
    return os.getenv("DSPY_API_KEY")
