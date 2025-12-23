"""LLM adapter port for Ragas evaluation."""

from abc import ABC, abstractmethod


class LLMPort(ABC):
    """LLM adapter interface for Ragas metrics evaluation.

    This port provides the necessary abstraction for LLM calls
    that will be used by Ragas metrics.
    """

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name being used.

        Returns:
            Model identifier (e.g., 'gpt-4o', 'gpt-4o-mini')
        """
        pass

    @abstractmethod
    def as_ragas_llm(self):
        """Return the LLM instance compatible with Ragas.

        Ragas expects langchain LLM instances. This method should
        return the appropriate LangChain LLM wrapper.

        Returns:
            LangChain-compatible LLM instance for Ragas
        """
        pass
