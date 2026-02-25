"""
BFCL (Berkeley Function Calling Leaderboard) Evaluation Module

Berkeley Function Calling Leaderboard is an authoritative benchmark for evaluating
the tool-calling capabilities of large language models.

Main features:
- Dataset loading and processing
- Tool-calling accuracy evaluation
- Multiple calling mode evaluation (simple, multiple, parallel, irrelevance detection)

References:
- Paper: https://arxiv.org/abs/2408.xxxxx
- Leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html
- Dataset: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
"""

from code.evaluation.benchmarks.bfcl.dataset import BFCLDataset
from code.evaluation.benchmarks.bfcl.evaluator import BFCLEvaluator
from code.evaluation.benchmarks.bfcl.metrics import BFCLMetrics
from code.evaluation.benchmarks.bfcl.bfcl_integration import BFCLIntegration

__all__ = [
    "BFCLDataset",
    "BFCLEvaluator",
    "BFCLMetrics",
    "BFCLIntegration",
]
