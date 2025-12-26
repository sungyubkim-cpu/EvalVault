"""Dataset entities for RAG evaluation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestCase:
    """단일 평가 케이스 (Ragas SingleTurnSample과 매핑)."""

    __test__ = False

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

    __test__ = False

    name: str
    version: str
    test_cases: list[TestCase]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str | None = None  # CSV/Excel 원본 파일 경로
    thresholds: dict[str, float] = field(default_factory=dict)  # 메트릭별 임계값

    def get_threshold(self, metric_name: str, default: float = 0.7) -> float:
        """특정 메트릭의 임계값 반환.

        Args:
            metric_name: 메트릭 이름 (예: 'faithfulness')
            default: 임계값이 없을 때 기본값

        Returns:
            임계값 (0.0 ~ 1.0)
        """
        return self.thresholds.get(metric_name, default)

    def __len__(self) -> int:
        return len(self.test_cases)

    def __iter__(self):
        return iter(self.test_cases)

    def to_ragas_list(self) -> list[dict[str, Any]]:
        """Ragas EvaluationDataset.from_list() 형식으로 변환."""
        return [tc.to_ragas_dict() for tc in self.test_cases]
