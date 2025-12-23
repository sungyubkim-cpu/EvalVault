"""Tests for OpenAI LLM adapter."""

import os
import pytest
from unittest.mock import MagicMock, patch

from evalvault.adapters.outbound.llm.openai_adapter import OpenAIAdapter
from evalvault.config.settings import Settings
from tests.unit.conftest import get_test_model


class TestOpenAIAdapter:
    """OpenAI adapter 테스트."""

    @pytest.fixture
    def model_name(self):
        """Get model name from environment."""
        return get_test_model()

    @pytest.fixture
    def settings(self, model_name):
        """Test settings fixture."""
        return Settings(
            openai_api_key="test-api-key",
            openai_model=model_name,
        )

    def test_get_model_name(self, settings, model_name):
        """get_model_name이 올바른 모델명을 반환하는지 테스트."""
        adapter = OpenAIAdapter(settings)
        assert adapter.get_model_name() == model_name

    def test_custom_base_url(self, model_name):
        """커스텀 base_url이 올바르게 설정되는지 테스트."""
        settings = Settings(
            openai_api_key="test-key",
            openai_base_url="https://custom-api.example.com/v1",
            openai_model=model_name,
        )
        adapter = OpenAIAdapter(settings)
        # Verify adapter was created successfully
        assert adapter.get_model_name() == model_name

    def test_as_ragas_llm_returns_langchain_instance(self, settings, model_name):
        """as_ragas_llm이 LangChain ChatOpenAI 인스턴스를 반환하는지 테스트."""
        adapter = OpenAIAdapter(settings)
        ragas_llm = adapter.as_ragas_llm()

        # Check that it returns a ChatOpenAI instance
        assert ragas_llm is not None
        assert hasattr(ragas_llm, "model_name")
        assert ragas_llm.model_name == model_name

    def test_as_ragas_llm_with_custom_base_url(self):
        """커스텀 base_url이 Ragas LLM에 전달되는지 테스트."""
        settings = Settings(
            openai_api_key="test-key",
            openai_base_url="https://custom-api.example.com/v1",
            openai_model="gpt-4o",
        )
        adapter = OpenAIAdapter(settings)
        ragas_llm = adapter.as_ragas_llm()

        assert ragas_llm.model_name == "gpt-4o"
        assert ragas_llm.openai_api_base == "https://custom-api.example.com/v1"
