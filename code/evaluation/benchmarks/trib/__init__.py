"""
TritonBench Evaluation Module

Evaluates agent performance on the TritonBench benchmark for Triton GPU kernel generation.

Two evaluation channels:
- G channel (184 tasks): Real Triton kernels from GitHub repos
- T channel (166 tasks): PyTorch-to-Triton conversion tasks

References:
- Repository: https://github.com/thunlp/TritonBench
"""

from code.evaluation.benchmarks.trib.dataset import TritonBenchDataset
from code.evaluation.benchmarks.trib.evaluator import TritonBenchEvaluator
from code.evaluation.benchmarks.trib.metrics import TritonBenchMetrics

__all__ = [
    "TritonBenchDataset",
    "TritonBenchEvaluator",
    "TritonBenchMetrics",
]
