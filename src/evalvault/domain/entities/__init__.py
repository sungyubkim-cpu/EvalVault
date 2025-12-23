"""Domain entities."""

from evalvault.domain.entities.dataset import Dataset, TestCase
from evalvault.domain.entities.result import (
    EvaluationRun,
    MetricScore,
    MetricType,
    TestCaseResult,
)

__all__ = [
    "Dataset",
    "TestCase",
    "EvaluationRun",
    "MetricScore",
    "MetricType",
    "TestCaseResult",
]
