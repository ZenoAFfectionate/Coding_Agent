"""
SWE-bench (Software Engineering Benchmark) 评估模块

SWE-bench 是一个软件工程评估基准，包含真实的 GitHub Issue 修复任务。

主要功能:
- 数据集加载 (dev / test / train)
- 仓库克隆与环境准备
- 补丁生成与比较
- 可选的测试执行验证

参考:
- 论文: https://arxiv.org/abs/2310.06770
- 排行榜: https://www.swebench.com
- 数据集: https://huggingface.co/datasets/princeton-nlp/SWE-bench
"""

from code.evaluation.benchmarks.swe.dataset import SWEDataset
from code.evaluation.benchmarks.swe.evaluator import SWEEvaluator
from code.evaluation.benchmarks.swe.metrics import SWEMetrics

__all__ = [
    "SWEDataset",
    "SWEEvaluator",
    "SWEMetrics",
]
