"""평가 실행 인터페이스."""

from typing import Protocol

from evalvault.domain.entities import Dataset, EvaluationRun


class EvaluatorPort(Protocol):
    """평가 실행을 위한 포트 인터페이스.

    데이터셋과 메트릭을 사용하여 모델 평가를 실행합니다.
    """

    def evaluate(
        self,
        dataset: Dataset,
        metrics: list[str],
        model: str,
    ) -> EvaluationRun:
        """데이터셋에 대해 평가를 실행합니다.

        Args:
            dataset: 평가할 데이터셋
            metrics: 사용할 메트릭 목록 (예: ["faithfulness", "answer_relevancy"])
            model: 평가에 사용할 모델 이름

        Returns:
            EvaluationRun 객체 (평가 결과 포함)
        """
        ...
