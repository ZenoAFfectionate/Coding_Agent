"""
SWE-bench (Software Engineering Benchmark) Evaluation Module

SWE-bench is a software engineering benchmark containing real GitHub issue fixing tasks.

Main features:
- Dataset loading (dev / test / train)
- Repository cloning and environment setup
- Patch generation and comparison
- Optional test execution verification

References:
- Paper: https://arxiv.org/abs/2310.06770
- Leaderboard: https://www.swebench.com
- Dataset: https://huggingface.co/datasets/princeton-nlp/SWE-bench
"""

from code.evaluation.benchmarks.swe.dataset import SWEDataset
from code.evaluation.benchmarks.swe.evaluator import SWEEvaluator
from code.evaluation.benchmarks.swe.metrics import SWEMetrics

__all__ = [
    "SWEDataset",
    "SWEEvaluator",
    "SWEMetrics",
]
