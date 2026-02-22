"""
HelloAgents 智能体评估模块

本模块提供了完整的智能体评估框架,包括:
- BFCL (Berkeley Function Calling Leaderboard): 工具调用能力评估
- GAIA (General AI Assistants): 通用AI助手能力评估
- Data Generation: 数据生成质量评估（LLM Judge & Win Rate）

主要组件:
- benchmarks: 各种评估基准测试
  - bfcl: BFCL评估（包含专用metrics）
  - gaia: GAIA评估（包含专用metrics）
  - data_generation: 数据生成质量评估

使用示例:
    >>> from hello_agents.evaluation import BFCLDataset, BFCLEvaluator
    >>> from hello_agents import SimpleAgent
    >>>
    >>> agent = SimpleAgent(name="TestAgent")
    >>> dataset = BFCLDataset(category="simple_python")
    >>> evaluator = BFCLEvaluator(dataset=dataset)
    >>> results = evaluator.evaluate(agent, max_samples=5)
    >>> print(f"准确率: {results['overall_accuracy']:.2%}")
"""

# 导出benchmark评估器和数据集
from code.evaluation.benchmarks.bfcl.dataset import BFCLDataset
from code.evaluation.benchmarks.bfcl.evaluator import BFCLEvaluator
from code.evaluation.benchmarks.gaia.dataset import GAIADataset
from code.evaluation.benchmarks.gaia.evaluator import GAIAEvaluator
from code.evaluation.benchmarks.data_generation.dataset import AIDataset
from code.evaluation.benchmarks.data_generation.llm_judge import LLMJudgeEvaluator
from code.evaluation.benchmarks.data_generation.win_rate import WinRateEvaluator
from code.evaluation.benchmarks.swe.dataset import SWEDataset
from code.evaluation.benchmarks.swe.evaluator import SWEEvaluator

__version__ = "0.1.0"

__all__ = [
    # Benchmark数据集
    "BFCLDataset",
    "GAIADataset",
    "AIDataset",
    "SWEDataset",

    # Benchmark评估器
    "BFCLEvaluator",
    "GAIAEvaluator",
    "LLMJudgeEvaluator",
    "WinRateEvaluator",
    "SWEEvaluator",
]

