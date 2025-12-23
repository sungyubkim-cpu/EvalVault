"""OpenAI LLM adapter for Ragas evaluation."""

from langchain_openai import ChatOpenAI

from evalvault.config.settings import Settings
from evalvault.ports.outbound.llm_port import LLMPort


class OpenAIAdapter(LLMPort):
    """OpenAI LLM adapter using LangChain.

    This adapter wraps LangChain's ChatOpenAI to provide
    a consistent interface for Ragas metrics evaluation.
    """

    def __init__(self, settings: Settings):
        """Initialize OpenAI adapter.

        Args:
            settings: Application settings containing OpenAI configuration
        """
        self._settings = settings
        self._model_name = settings.openai_model

        # Build kwargs for ChatOpenAI
        kwargs = {
            "model": self._model_name,
            "temperature": 0.0,  # Deterministic for evaluation
        }

        # Add API key if provided
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key

        # Add custom base URL if provided
        if settings.openai_base_url:
            kwargs["openai_api_base"] = settings.openai_base_url

        self._llm = ChatOpenAI(**kwargs)

    def get_model_name(self) -> str:
        """Get the model name being used.

        Returns:
            Model identifier (e.g., 'gpt-4o-mini')
        """
        return self._model_name

    def as_ragas_llm(self) -> ChatOpenAI:
        """Return the LangChain ChatOpenAI instance for Ragas.

        Ragas metrics expect LangChain LLM instances.

        Returns:
            ChatOpenAI instance configured with settings
        """
        return self._llm
