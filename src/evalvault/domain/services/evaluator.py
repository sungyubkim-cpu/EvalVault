"""Ragas evaluation service."""

from datetime import datetime

from ragas import SingleTurnSample
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from evalvault.domain.entities import (
    Dataset,
    EvaluationRun,
    MetricScore,
    TestCaseResult,
)
from evalvault.ports.outbound.llm_port import LLMPort


class RagasEvaluator:
    """Ragas 기반 RAG 평가 서비스.

    Ragas 메트릭을 사용하여 RAG 시스템의 품질을 평가합니다.
    """

    # Ragas 메트릭 매핑
    METRIC_MAP = {
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
    }

    async def evaluate(
        self,
        dataset: Dataset,
        metrics: list[str],
        llm: LLMPort,
        thresholds: dict[str, float] | None = None,
    ) -> EvaluationRun:
        """데이터셋을 Ragas로 평가.

        Args:
            dataset: 평가할 데이터셋
            metrics: 평가할 메트릭 리스트 (예: ['faithfulness', 'answer_relevancy'])
            llm: LLM 어댑터 (Ragas가 사용)
            thresholds: 메트릭별 임계값 (기본값: 0.7)

        Returns:
            평가 결과가 담긴 EvaluationRun
        """
        # Initialize evaluation run
        run = EvaluationRun(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            model_name=llm.get_model_name(),
            started_at=datetime.now(),
            metrics_evaluated=metrics,
            thresholds=thresholds or {},
        )

        # Handle empty dataset
        if len(dataset.test_cases) == 0:
            run.finished_at = datetime.now()
            return run

        # Set default thresholds
        if thresholds is None:
            thresholds = dict.fromkeys(metrics, 0.7)

        # Evaluate with Ragas
        scores_by_test_case = await self._evaluate_with_ragas(
            dataset=dataset, metrics=metrics, llm=llm
        )

        # Aggregate results
        for test_case in dataset.test_cases:
            tc_scores = scores_by_test_case.get(test_case.id, {})

            metric_scores = []
            for metric_name in metrics:
                score_value = tc_scores.get(metric_name, 0.0)
                threshold = thresholds.get(metric_name, 0.7)

                metric_scores.append(
                    MetricScore(
                        name=metric_name,
                        score=score_value,
                        threshold=threshold,
                    )
                )

            test_case_result = TestCaseResult(
                test_case_id=test_case.id,
                metrics=metric_scores,
            )
            run.results.append(test_case_result)

        # Finalize run
        run.finished_at = datetime.now()
        return run

    async def _evaluate_with_ragas(
        self, dataset: Dataset, metrics: list[str], llm: LLMPort
    ) -> dict[str, dict[str, float]]:
        """Ragas로 실제 평가 수행.

        Args:
            dataset: 평가할 데이터셋
            metrics: 평가할 메트릭 리스트
            llm: LLM 어댑터

        Returns:
            테스트 케이스 ID별 메트릭 점수
            예: {"tc-001": {"faithfulness": 0.9, "answer_relevancy": 0.85}}
        """
        # Convert dataset to Ragas format
        ragas_samples = []
        for test_case in dataset.test_cases:
            sample = SingleTurnSample(
                user_input=test_case.question,
                response=test_case.answer,
                retrieved_contexts=test_case.contexts,
                reference=test_case.ground_truth,
            )
            ragas_samples.append(sample)

        # Initialize Ragas metrics
        ragas_metrics = []
        for metric_name in metrics:
            metric_class = self.METRIC_MAP.get(metric_name)
            if metric_class:
                ragas_metrics.append(metric_class())

        # Get LangChain LLM for Ragas
        ragas_llm = llm.as_ragas_llm()

        # Evaluate using Ragas
        # Note: We need to pass the LLM to each metric
        results = {}
        for idx, sample in enumerate(ragas_samples):
            test_case_id = dataset.test_cases[idx].id
            results[test_case_id] = {}

            for metric in ragas_metrics:
                # Set LLM for the metric
                metric.llm = ragas_llm

                # Evaluate the sample
                score = await metric.single_turn_ascore(sample)
                results[test_case_id][metric.name] = score

        return results
