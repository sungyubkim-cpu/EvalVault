"""Unit tests for OllamaAdapter.

모델명 참조:
  - 개발 LLM: gemma3:1b
  - 운영 LLM: gpt-oss-safeguard:20b
  - 개발 임베딩: qwen3-embedding:0.6b
  - 운영 임베딩: qwen3-embedding:8b
"""

import pytest
from unittest.mock import MagicMock, patch

from evalvault.config.settings import Settings
from evalvault.adapters.outbound.llm.ollama_adapter import OllamaAdapter
from evalvault.adapters.outbound.llm import get_llm_adapter, OllamaAdapter


class TestOllamaAdapter:
    """OllamaAdapter 단위 테스트."""

    @pytest.fixture
    def dev_settings(self) -> Settings:
        """개발 환경 설정 (gemma3:1b)."""
        settings = Settings()
        settings.llm_provider = "ollama"
        settings.ollama_model = "gemma3:1b"
        settings.ollama_embedding_model = "qwen3-embedding:0.6b"
        settings.ollama_base_url = "http://localhost:11434"
        settings.ollama_timeout = 120
        return settings

    @pytest.fixture
    def prod_settings(self) -> Settings:
        """운영 환경 설정 (gpt-oss-safeguard:20b)."""
        settings = Settings()
        settings.llm_provider = "ollama"
        settings.ollama_model = "gpt-oss-safeguard:20b"
        settings.ollama_embedding_model = "qwen3-embedding:8b"
        settings.ollama_base_url = "http://localhost:11434"
        settings.ollama_timeout = 180
        settings.ollama_think_level = "medium"
        return settings

    @patch("evalvault.adapters.outbound.llm.ollama_adapter.llm_factory")
    @patch("evalvault.adapters.outbound.llm.ollama_adapter.RagasOpenAIEmbeddings")
    def test_adapter_initialization_dev(
        self, mock_embeddings, mock_llm_factory, dev_settings
    ):
        """개발 환경 어댑터 초기화 테스트."""
        adapter = OllamaAdapter(dev_settings)

        assert adapter.get_model_name() == "ollama/gemma3:1b"
        assert adapter.get_embedding_model_name() == "qwen3-embedding:0.6b"
        assert adapter.get_base_url() == "http://localhost:11434"
        assert adapter.get_think_level() is None

    @patch("evalvault.adapters.outbound.llm.ollama_adapter.llm_factory")
    @patch("evalvault.adapters.outbound.llm.ollama_adapter.RagasOpenAIEmbeddings")
    def test_adapter_initialization_prod(
        self, mock_embeddings, mock_llm_factory, prod_settings
    ):
        """운영 환경 어댑터 초기화 테스트."""
        adapter = OllamaAdapter(prod_settings)

        assert adapter.get_model_name() == "ollama/gpt-oss-safeguard:20b"
        assert adapter.get_embedding_model_name() == "qwen3-embedding:8b"
        assert adapter.get_think_level() == "medium"

    @patch("evalvault.adapters.outbound.llm.ollama_adapter.llm_factory")
    @patch("evalvault.adapters.outbound.llm.ollama_adapter.RagasOpenAIEmbeddings")
    def test_as_ragas_llm(self, mock_embeddings, mock_llm_factory, dev_settings):
        """Ragas LLM 반환 테스트."""
        mock_ragas_llm = MagicMock()
        mock_llm_factory.return_value = mock_ragas_llm

        adapter = OllamaAdapter(dev_settings)
        result = adapter.as_ragas_llm()

        assert result == mock_ragas_llm
        mock_llm_factory.assert_called_once()

    @patch("evalvault.adapters.outbound.llm.ollama_adapter.llm_factory")
    @patch("evalvault.adapters.outbound.llm.ollama_adapter.RagasOpenAIEmbeddings")
    def test_as_ragas_embeddings(self, mock_embeddings, mock_llm_factory, dev_settings):
        """Ragas 임베딩 반환 테스트."""
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance

        adapter = OllamaAdapter(dev_settings)
        result = adapter.as_ragas_embeddings()

        assert result == mock_embedding_instance

    @patch("evalvault.adapters.outbound.llm.ollama_adapter.llm_factory")
    @patch("evalvault.adapters.outbound.llm.ollama_adapter.RagasOpenAIEmbeddings")
    def test_token_usage_tracking(self, mock_embeddings, mock_llm_factory, dev_settings):
        """토큰 사용량 추적 테스트."""
        adapter = OllamaAdapter(dev_settings)

        # 초기값 확인
        assert adapter.get_token_usage() == (0, 0, 0)

        # 리셋 테스트
        adapter.reset_token_usage()
        assert adapter.get_token_usage() == (0, 0, 0)

    @patch("evalvault.adapters.outbound.llm.ollama_adapter.llm_factory")
    @patch("evalvault.adapters.outbound.llm.ollama_adapter.RagasOpenAIEmbeddings")
    def test_get_and_reset_token_usage(
        self, mock_embeddings, mock_llm_factory, dev_settings
    ):
        """토큰 사용량 조회 및 리셋 테스트."""
        adapter = OllamaAdapter(dev_settings)

        # 수동으로 토큰 추가 (내부 상태 테스트)
        adapter._token_usage.add(100, 50, 150)

        result = adapter.get_and_reset_token_usage()
        assert result == (100, 50, 150)

        # 리셋 확인
        assert adapter.get_token_usage() == (0, 0, 0)


class TestGetLLMAdapter:
    """get_llm_adapter 팩토리 함수 테스트."""

    def test_openai_provider(self):
        """OpenAI 프로바이더 테스트."""
        settings = Settings()
        settings.llm_provider = "openai"
        settings.openai_api_key = "test-key"

        adapter = get_llm_adapter(settings)

        from evalvault.adapters.outbound.llm.openai_adapter import OpenAIAdapter

        assert isinstance(adapter, OpenAIAdapter)

    @patch("evalvault.adapters.outbound.llm.ollama_adapter.llm_factory")
    @patch("evalvault.adapters.outbound.llm.ollama_adapter.RagasOpenAIEmbeddings")
    def test_ollama_provider(self, mock_embeddings, mock_llm_factory):
        """Ollama 프로바이더 테스트."""
        settings = Settings()
        settings.llm_provider = "ollama"
        settings.ollama_model = "gemma3:1b"
        settings.ollama_embedding_model = "qwen3-embedding:0.6b"

        adapter = get_llm_adapter(settings)

        assert isinstance(adapter, OllamaAdapter)
        assert adapter.get_model_name() == "ollama/gemma3:1b"

    def test_unsupported_provider(self):
        """지원하지 않는 프로바이더 테스트."""
        settings = Settings()
        settings.llm_provider = "unsupported"

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm_adapter(settings)

    @patch("evalvault.adapters.outbound.llm.ollama_adapter.llm_factory")
    @patch("evalvault.adapters.outbound.llm.ollama_adapter.RagasOpenAIEmbeddings")
    def test_provider_case_insensitive(self, mock_embeddings, mock_llm_factory):
        """프로바이더 이름 대소문자 무시 테스트."""
        settings = Settings()
        settings.llm_provider = "OLLAMA"
        settings.ollama_model = "gemma3:1b"
        settings.ollama_embedding_model = "qwen3-embedding:0.6b"

        adapter = get_llm_adapter(settings)

        assert isinstance(adapter, OllamaAdapter)
