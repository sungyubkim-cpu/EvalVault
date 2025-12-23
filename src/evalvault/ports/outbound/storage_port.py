"""결과 저장 인터페이스."""

from typing import Protocol

from evalvault.domain.entities import EvaluationRun


class StoragePort(Protocol):
    """평가 결과 저장을 위한 포트 인터페이스.

    SQLite, PostgreSQL 등 다양한 저장소에 평가 결과를 저장합니다.
    """

    def save_run(self, run: EvaluationRun) -> str:
        """평가 실행 결과를 저장합니다.

        Args:
            run: 저장할 평가 실행 결과

        Returns:
            저장된 run의 ID
        """
        ...

    def get_run(self, run_id: str) -> EvaluationRun:
        """저장된 평가 실행 결과를 조회합니다.

        Args:
            run_id: 조회할 run의 ID

        Returns:
            EvaluationRun 객체

        Raises:
            KeyError: run_id에 해당하는 결과가 없는 경우
        """
        ...

    def list_runs(
        self,
        limit: int = 100,
        dataset_name: str | None = None,
        model_name: str | None = None,
    ) -> list[EvaluationRun]:
        """저장된 평가 실행 결과 목록을 조회합니다.

        Args:
            limit: 최대 조회 개수
            dataset_name: 필터링할 데이터셋 이름 (선택)
            model_name: 필터링할 모델 이름 (선택)

        Returns:
            EvaluationRun 객체 리스트 (최신순)
        """
        ...
