"""Dataset entities for RAG evaluation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestCase:
    """단일 평가 케이스 (Ragas SingleTurnSample과 매핑)."""

    id: str
    question: str  # user_input
    answer: str  # response
    contexts: list[str]  # retrieved_contexts
    ground_truth: str | None = None  # reference
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_ragas_dict(self) -> dict[str, Any]:
        """Ragas EvaluationDataset 형식으로 변환."""
        result = {
            "user_input": self.question,
            "response": self.answer,
            "retrieved_contexts": self.contexts,
        }
        if self.ground_truth:
            result["reference"] = self.ground_truth
        return result


@dataclass
class Dataset:
    """평가용 데이터셋."""

    name: str
    version: str
    test_cases: list[TestCase]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str | None = None  # CSV/Excel 원본 파일 경로

    def __len__(self) -> int:
        return len(self.test_cases)

    def __iter__(self):
        return iter(self.test_cases)

    def to_ragas_list(self) -> list[dict[str, Any]]:
        """Ragas EvaluationDataset.from_list() 형식으로 변환."""
        return [tc.to_ragas_dict() for tc in self.test_cases]
