"""
Data Generation Evaluation Module

Module for evaluating data generation quality, including:
- LLM Judge: Uses an LLM as a judge to evaluate generation quality
- Win Rate: Computes win rates through pairwise comparisons
"""

from code.evaluation.benchmarks.data_generation.dataset import AIDataset
from code.evaluation.benchmarks.data_generation.llm_judge import LLMJudgeEvaluator
from code.evaluation.benchmarks.data_generation.win_rate import WinRateEvaluator

__all__ = [
    "AIDataset",
    "LLMJudgeEvaluator",
    "WinRateEvaluator",
]
