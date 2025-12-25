"""Tests for CLI interface."""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from evalvault.adapters.inbound.cli import app
from evalvault.domain.entities import (
    Dataset,
    TestCase,
    EvaluationRun,
    TestCaseResult,
    MetricScore,
)
from tests.unit.conftest import get_test_model


runner = CliRunner()


class TestCLIVersion:
    """CLI 버전 명령 테스트."""

    def test_version_command(self):
        """--version 플래그 테스트."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout


class TestCLIRun:
    """CLI run 명령 테스트."""

    @pytest.fixture
    def mock_dataset(self):
        """테스트용 데이터셋."""
        return Dataset(
            name="test-dataset",
            version="1.0.0",
            test_cases=[
                TestCase(
                    id="tc-001",
                    question="What is Python?",
                    answer="Python is a programming language.",
                    contexts=["Python is a high-level language."],
                    ground_truth="A programming language",
                ),
            ],
        )

    @pytest.fixture
    def mock_evaluation_run(self):
        """테스트용 평가 결과."""
        from datetime import datetime, timedelta

        start = datetime.now()
        end = start + timedelta(seconds=10)

        return EvaluationRun(
            dataset_name="test-dataset",
            dataset_version="1.0.0",
            model_name=get_test_model(),
            metrics_evaluated=["faithfulness"],
            started_at=start,
            finished_at=end,
            thresholds={"faithfulness": 0.7},
            results=[
                TestCaseResult(
                    test_case_id="tc-001",
                    metrics=[
                        MetricScore(name="faithfulness", score=0.85, threshold=0.7),
                    ],
                ),
            ],
        )

    def test_run_help(self):
        """run 명령 help 테스트."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "dataset" in result.stdout.lower()
        assert "metrics" in result.stdout.lower()

    def test_run_missing_dataset(self):
        """데이터셋 파일 누락 시 에러."""
        result = runner.invoke(app, ["run", "nonexistent.csv"])
        assert result.exit_code != 0

    @patch("evalvault.adapters.inbound.cli.get_loader")
    @patch("evalvault.adapters.inbound.cli.RagasEvaluator")
    @patch("evalvault.adapters.inbound.cli.get_llm_adapter")
    @patch("evalvault.adapters.inbound.cli.Settings")
    def test_run_with_valid_dataset(
        self,
        mock_settings_cls,
        mock_get_llm_adapter,
        mock_evaluator_cls,
        mock_get_loader,
        mock_dataset,
        mock_evaluation_run,
        tmp_path,
    ):
        """유효한 데이터셋으로 run 명령 테스트."""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "test-key"
        mock_settings.openai_model = get_test_model()
        mock_settings.llm_provider = "openai"
        mock_settings.evalvault_profile = None
        mock_settings_cls.return_value = mock_settings

        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_loader.return_value = mock_loader

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(return_value=mock_evaluation_run)
        mock_evaluator_cls.return_value = mock_evaluator

        mock_llm = MagicMock()
        mock_get_llm_adapter.return_value = mock_llm

        # Create test file
        test_file = tmp_path / "test.csv"
        test_file.write_text("id,question,answer,contexts\n")

        # Run command
        result = runner.invoke(
            app, ["run", str(test_file), "--metrics", "faithfulness"]
        )

        # Assert
        assert result.exit_code == 0
        assert "test-dataset" in result.stdout or "faithfulness" in result.stdout

    @patch("evalvault.adapters.inbound.cli.get_loader")
    @patch("evalvault.adapters.inbound.cli.RagasEvaluator")
    @patch("evalvault.adapters.inbound.cli.get_llm_adapter")
    @patch("evalvault.adapters.inbound.cli.Settings")
    def test_run_with_multiple_metrics(
        self,
        mock_settings_cls,
        mock_get_llm_adapter,
        mock_evaluator_cls,
        mock_get_loader,
        tmp_path,
    ):
        """여러 메트릭으로 run 명령 테스트."""
        from datetime import datetime, timedelta

        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "test-key"
        mock_settings.openai_model = get_test_model()
        mock_settings.llm_provider = "openai"
        mock_settings.evalvault_profile = None
        mock_settings_cls.return_value = mock_settings

        mock_dataset = Dataset(
            name="test",
            version="1.0.0",
            test_cases=[
                TestCase(
                    id="tc-001",
                    question="Q1",
                    answer="A1",
                    contexts=["C1"],
                ),
            ],
        )

        start = datetime.now()
        end = start + timedelta(seconds=5)
        mock_run = EvaluationRun(
            dataset_name="test",
            dataset_version="1.0.0",
            model_name=get_test_model(),
            metrics_evaluated=["faithfulness", "answer_relevancy"],
            started_at=start,
            finished_at=end,
            thresholds={"faithfulness": 0.7, "answer_relevancy": 0.7},
            results=[
                TestCaseResult(
                    test_case_id="tc-001",
                    metrics=[
                        MetricScore(name="faithfulness", score=0.9, threshold=0.7),
                        MetricScore(name="answer_relevancy", score=0.85, threshold=0.7),
                    ],
                ),
            ],
        )

        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_loader.return_value = mock_loader

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(return_value=mock_run)
        mock_evaluator_cls.return_value = mock_evaluator

        mock_llm = MagicMock()
        mock_get_llm_adapter.return_value = mock_llm

        # Create test file
        test_file = tmp_path / "test.csv"
        test_file.write_text("id,question,answer,contexts\n")

        # Run command with multiple metrics
        result = runner.invoke(
            app,
            ["run", str(test_file), "--metrics", "faithfulness,answer_relevancy"],
        )

        assert result.exit_code == 0


class TestCLIMetrics:
    """CLI metrics 명령 테스트."""

    def test_metrics_list(self):
        """metrics 명령으로 사용 가능한 메트릭 목록 출력."""
        result = runner.invoke(app, ["metrics"])
        assert result.exit_code == 0
        assert "faithfulness" in result.stdout.lower()
        assert "answer_relevancy" in result.stdout.lower()
        assert "context_precision" in result.stdout.lower()
        assert "context_recall" in result.stdout.lower()


class TestCLIConfig:
    """CLI config 명령 테스트."""

    @patch("evalvault.adapters.inbound.cli.Settings")
    def test_config_show(self, mock_settings_cls):
        """config 명령으로 현재 설정 출력."""
        test_model = get_test_model()
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "test-key"
        mock_settings.openai_model = test_model
        mock_settings.openai_embedding_model = "text-embedding-3-small"
        mock_settings.openai_base_url = None
        mock_settings.llm_provider = "openai"
        mock_settings.evalvault_profile = None  # No profile set
        mock_settings.langfuse_public_key = None
        mock_settings.langfuse_secret_key = None
        mock_settings.langfuse_host = "https://cloud.langfuse.com"
        mock_settings_cls.return_value = mock_settings

        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        # Check for configuration related text
        assert "Configuration" in result.stdout
