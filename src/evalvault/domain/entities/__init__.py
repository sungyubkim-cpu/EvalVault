"""Domain entities."""

from evalvault.domain.entities.dataset import Dataset, TestCase
from evalvault.domain.entities.experiment import Experiment, ExperimentGroup
from evalvault.domain.entities.kg import EntityModel, RelationModel
from evalvault.domain.entities.result import (
    EvaluationRun,
    MetricScore,
    MetricType,
    TestCaseResult,
)

__all__ = [
    "EntityModel",
    "RelationModel",
    "Dataset",
    "TestCase",
    "Experiment",
    "ExperimentGroup",
    "EvaluationRun",
    "MetricScore",
    "MetricType",
    "TestCaseResult",
]
