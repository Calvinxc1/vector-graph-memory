"""Tests for runtime model provider selection."""

from __future__ import annotations

import pytest
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.models.openai import OpenAIChatModel

from vgm.model_provider import (
    build_chat_model_from_env,
    build_embedding_model_from_env,
    chat_model_name_from_env,
    dspy_api_base_from_env,
    dspy_api_key_from_env,
    dspy_model_name_for_provider,
    embedding_model_name_from_env,
)


def test_openai_provider_preserves_existing_model_string(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o-mini")

    assert chat_model_name_from_env() == "openai:gpt-4o-mini"
    assert build_chat_model_from_env() == "openai:gpt-4o-mini"
    assert dspy_model_name_for_provider("openai:gpt-4o-mini") == "openai/gpt-4o-mini"


def test_ollama_provider_builds_openai_compatible_chat_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.example.test/v1")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen2.5:14b")

    model = build_chat_model_from_env()

    assert chat_model_name_from_env() == "ollama:qwen2.5:14b"
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "qwen2.5:14b"
    assert model._provider.base_url == "https://ollama.example.test/v1/"
    assert dspy_model_name_for_provider("ollama:qwen2.5:14b") == "openai/qwen2.5:14b"
    assert dspy_api_base_from_env() == "https://ollama.example.test/v1"
    assert dspy_api_key_from_env() == "api-key-not-set"


def test_ollama_embedding_provider_builds_openai_compatible_embedding_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.example.test/v1")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    model = build_embedding_model_from_env()

    assert embedding_model_name_from_env() == "ollama:nomic-embed-text"
    assert isinstance(model, OpenAIEmbeddingModel)
    assert model.model_name == "nomic-embed-text"
    assert model.base_url == "https://ollama.example.test/v1/"


def test_ollama_provider_requires_base_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_CHAT_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="OLLAMA_CHAT_BASE_URL or OLLAMA_BASE_URL"):
        build_chat_model_from_env()
