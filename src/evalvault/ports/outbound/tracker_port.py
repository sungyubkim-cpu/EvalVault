"""Tracker port interface for logging evaluation traces."""

from typing import Any, Protocol

from evalvault.domain.entities import EvaluationRun


class TrackerPort(Protocol):
    """Port for tracking evaluation runs (e.g., Langfuse, MLflow).

    평가 실행을 Langfuse에 추적하고 메트릭, 아티팩트를 저장합니다.
    """

    def start_trace(self, name: str, metadata: dict[str, Any] | None = None) -> str:
        """새로운 trace를 시작합니다.

        Args:
            name: trace 이름
            metadata: trace에 첨부할 메타데이터 (선택)

        Returns:
            trace_id: trace 고유 식별자
        """
        ...

    def add_span(
        self,
        trace_id: str,
        name: str,
        input_data: Any | None = None,
        output_data: Any | None = None,
    ) -> None:
        """Add a span to an existing trace.

        Args:
            trace_id: ID of the trace to add the span to
            name: Name of the span
            input_data: Optional input data for the span
            output_data: Optional output data for the span
        """
        ...

    def log_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        """trace에 점수를 기록합니다.

        Args:
            trace_id: 점수를 기록할 trace ID
            name: 점수 이름 (예: 메트릭 이름)
            value: 점수 값 (일반적으로 0.0 ~ 1.0)
            comment: 점수에 대한 선택적 코멘트
        """
        ...

    def save_artifact(
        self,
        trace_id: str,
        name: str,
        data: Any,
        artifact_type: str = "json",
    ) -> None:
        """trace에 아티팩트를 저장합니다.

        Args:
            trace_id: 아티팩트를 저장할 trace ID
            name: 아티팩트 이름
            data: 아티팩트 데이터
            artifact_type: 아티팩트 타입 (json, text 등)
        """
        ...

    def end_trace(self, trace_id: str) -> None:
        """trace를 종료하고 대기 중인 데이터를 flush합니다.

        Args:
            trace_id: 종료할 trace ID
        """
        ...

    def log_evaluation_run(self, run: EvaluationRun) -> str:
        """Log a complete evaluation run as a trace.

        Args:
            run: EvaluationRun entity containing all evaluation results

        Returns:
            trace_id: ID of the created trace
        """
        ...
