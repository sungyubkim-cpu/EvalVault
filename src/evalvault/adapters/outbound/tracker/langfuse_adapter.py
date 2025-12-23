"""Langfuse tracker adapter implementation."""

from typing import Any

from langfuse import Langfuse

from evalvault.domain.entities import EvaluationRun
from evalvault.ports.outbound.tracker_port import TrackerPort


class LangfuseAdapter(TrackerPort):
    """Langfuse implementation of TrackerPort."""

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
    ):
        """
        Initialize Langfuse adapter.

        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL (default: https://cloud.langfuse.com)
        """
        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        self._traces: dict[str, Any] = {}

    def start_trace(self, name: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Start a new trace.

        Args:
            name: Name of the trace
            metadata: Optional metadata to attach to the trace

        Returns:
            trace_id: Unique identifier for the trace
        """
        trace = self._client.trace(
            name=name,
            metadata=metadata,
        )
        self._traces[trace.id] = trace
        return trace.id

    def add_span(
        self,
        trace_id: str,
        name: str,
        input_data: Any | None = None,
        output_data: Any | None = None,
    ) -> None:
        """
        Add a span to an existing trace.

        Args:
            trace_id: ID of the trace to add the span to
            name: Name of the span
            input_data: Optional input data for the span
            output_data: Optional output data for the span

        Raises:
            ValueError: If trace_id is not found
        """
        if trace_id not in self._traces:
            raise ValueError(f"Trace not found: {trace_id}")

        trace = self._traces[trace_id]
        trace.span(
            name=name,
            input=input_data,
            output=output_data,
        )

    def log_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        """
        Log a score to a trace.

        Args:
            trace_id: ID of the trace to log the score to
            name: Name of the score (e.g., metric name)
            value: Score value (typically 0.0 to 1.0)
            comment: Optional comment about the score

        Raises:
            ValueError: If trace_id is not found
        """
        if trace_id not in self._traces:
            raise ValueError(f"Trace not found: {trace_id}")

        trace = self._traces[trace_id]
        trace.score(
            name=name,
            value=value,
            comment=comment,
        )

    def save_artifact(
        self,
        trace_id: str,
        name: str,
        data: Any,
        artifact_type: str = "json",
    ) -> None:
        """
        Save an artifact to a trace.

        Langfuse doesn't have native artifact support, so we store it in metadata.

        Args:
            trace_id: ID of the trace to save the artifact to
            name: Name of the artifact
            data: Artifact data
            artifact_type: Type of artifact (json, text, etc.)

        Raises:
            ValueError: If trace_id is not found
        """
        if trace_id not in self._traces:
            raise ValueError(f"Trace not found: {trace_id}")

        trace = self._traces[trace_id]
        # Store artifact in metadata with special prefix
        artifact_metadata = {
            f"artifact_{name}": data,
            f"artifact_{name}_type": artifact_type,
        }
        trace.update(metadata=artifact_metadata)

    def end_trace(self, trace_id: str) -> None:
        """
        End a trace and flush any pending data.

        Args:
            trace_id: ID of the trace to end

        Raises:
            ValueError: If trace_id is not found
        """
        if trace_id not in self._traces:
            raise ValueError(f"Trace not found: {trace_id}")

        # Flush all pending data to Langfuse
        self._client.flush()

        # Remove trace from active traces
        del self._traces[trace_id]

    def log_evaluation_run(self, run: EvaluationRun) -> str:
        """
        Log a complete evaluation run as a trace.

        Args:
            run: EvaluationRun entity containing all evaluation results

        Returns:
            trace_id: ID of the created trace
        """
        # Calculate per-metric pass rates
        metric_pass_rates = {}
        for metric_name in run.metrics_evaluated:
            passed_count = sum(
                1 for r in run.results
                if r.get_metric(metric_name) and r.get_metric(metric_name).passed
            )
            metric_pass_rates[metric_name] = {
                "passed": passed_count,
                "total": len(run.results),
                "rate": passed_count / len(run.results) if run.results else 0.0,
            }

        # Create trace with run metadata
        metadata = {
            "run_id": run.run_id,
            "dataset_name": run.dataset_name,
            "dataset_version": run.dataset_version,
            "model_name": run.model_name,
            "started_at": run.started_at.isoformat(),
            "total_test_cases": run.total_test_cases,
            "passed_test_cases": run.passed_test_cases,
            "pass_rate": run.pass_rate,
            "total_tokens": run.total_tokens,
            "metrics_evaluated": run.metrics_evaluated,
            "thresholds": run.thresholds,
            "metric_pass_rates": metric_pass_rates,
        }

        if run.finished_at:
            metadata["finished_at"] = run.finished_at.isoformat()
            metadata["duration_seconds"] = run.duration_seconds

        if run.total_cost_usd:
            metadata["total_cost_usd"] = run.total_cost_usd

        trace_id = self.start_trace(
            name=f"evaluation-run-{run.run_id}",
            metadata=metadata,
        )

        # Log average scores and pass rates for each metric
        for metric_name in run.metrics_evaluated:
            avg_score = run.get_avg_score(metric_name)
            if avg_score is not None:
                # Calculate pass rate for this metric
                passed_count = sum(
                    1 for r in run.results
                    if r.get_metric(metric_name) and r.get_metric(metric_name).passed
                )
                metric_pass_rate = passed_count / len(run.results) if run.results else 0.0
                threshold = run.thresholds.get(metric_name, 0.7)

                self.log_score(
                    trace_id=trace_id,
                    name=f"avg_{metric_name}",
                    value=avg_score,
                    comment=f"Average {metric_name}: {avg_score:.2f} | Pass rate: {metric_pass_rate:.1%} ({passed_count}/{len(run.results)}) | Threshold: {threshold}",
                )

        # Log individual test case results as spans
        for result in run.results:
            # Build per-metric pass/fail status
            metric_results = {
                m.name: {
                    "score": m.score,
                    "threshold": m.threshold,
                    "passed": m.passed,
                }
                for m in result.metrics
            }

            span_metadata = {
                "test_case_id": result.test_case_id,
                "tokens_used": result.tokens_used,
                "latency_ms": result.latency_ms,
                "all_passed": result.all_passed,
                "metrics": metric_results,
            }
            if result.cost_usd:
                span_metadata["cost_usd"] = result.cost_usd

            # Add span for each test case
            self.add_span(
                trace_id=trace_id,
                name=f"test-case-{result.test_case_id}",
                input_data={"test_case_id": result.test_case_id},
                output_data=span_metadata,
            )

            # Log individual metric scores with pass/fail status
            for metric in result.metrics:
                status = "PASS" if metric.passed else "FAIL"
                self.log_score(
                    trace_id=trace_id,
                    name=f"{result.test_case_id}_{metric.name}",
                    value=metric.score,
                    comment=f"[{status}] {metric.name}: {metric.score:.2f} (threshold: {metric.threshold})",
                )

        # Flush the trace
        self._client.flush()

        return trace_id
