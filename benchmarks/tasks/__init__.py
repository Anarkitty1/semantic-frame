"""
Benchmark Tasks

Task-specific evaluation modules for different benchmark categories.
"""

from benchmarks.tasks.anomaly import AnomalyTask
from benchmarks.tasks.base import BaseTask
from benchmarks.tasks.comparative import ComparativeTask
from benchmarks.tasks.multi_step import MultiStepTask
from benchmarks.tasks.scaling import ScalingTask
from benchmarks.tasks.statistical import StatisticalTask
from benchmarks.tasks.trend import TrendTask

__all__ = [
    "BaseTask",
    "StatisticalTask",
    "TrendTask",
    "AnomalyTask",
    "ComparativeTask",
    "MultiStepTask",
    "ScalingTask",
]
