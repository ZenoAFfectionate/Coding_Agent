"""工具系统"""

from .base import Tool, ToolParameter
from .registry import ToolRegistry, global_registry

# 内置工具
from .builtin.search_tool import SearchTool
from .builtin.calculator import CalculatorTool
from .builtin.memory_tool import MemoryTool
from .builtin.rag_tool import RAGTool
from .builtin.note_tool import NoteTool
from .builtin.terminal_tool import TerminalTool
from .builtin.git_tool import GitTool
from .builtin.test_runner_tool import TestRunnerTool
from .builtin.file_tool import FileTool
from .builtin.code_execution_tool import CodeExecutionTool
from .builtin.code_search_tool import CodeSearchTool

# 协议工具
from .builtin.protocol_tools import MCPTool, A2ATool, ANPTool

# 评估工具（第12章）— optional, depends on hello_agents package
try:
    from .builtin.bfcl_evaluation_tool import BFCLEvaluationTool
except ImportError:
    BFCLEvaluationTool = None

try:
    from .builtin.gaia_evaluation_tool import GAIAEvaluationTool
except ImportError:
    GAIAEvaluationTool = None

try:
    from .builtin.llm_judge_tool import LLMJudgeTool
except ImportError:
    LLMJudgeTool = None

try:
    from .builtin.win_rate_tool import WinRateTool
except ImportError:
    WinRateTool = None

# RL训练工具（第11章）
from .builtin.rl_training_tool import RLTrainingTool

# 高级功能
from .chain import ToolChain, ToolChainManager, create_research_chain, create_simple_chain
from .async_executor import AsyncToolExecutor, run_parallel_tools, run_batch_tool, run_parallel_tools_sync, run_batch_tool_sync

__all__ = [
    # 基础工具系统
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "global_registry",

    # 内置工具
    "SearchTool",
    "CalculatorTool",
    "MemoryTool",
    "RAGTool",
    "NoteTool",
    "TerminalTool",
    "GitTool",
    "TestRunnerTool",
    "FileTool",
    "CodeExecutionTool",
    "CodeSearchTool",

    # 协议工具
    "MCPTool",
    "A2ATool",
    "ANPTool",

    # 评估工具
    "BFCLEvaluationTool",
    "GAIAEvaluationTool",
    "LLMJudgeTool",
    "WinRateTool",

    # RL训练工具
    "RLTrainingTool",

    # 工具链功能
    "ToolChain",
    "ToolChainManager",
    "create_research_chain",
    "create_simple_chain",

    # 异步执行功能
    "AsyncToolExecutor",
    "run_parallel_tools",
    "run_batch_tool",
    "run_parallel_tools_sync",
    "run_batch_tool_sync",
]
