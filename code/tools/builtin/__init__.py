"""内置工具模块

HelloAgents框架的内置工具集合，包括：
- SearchTool: 网页搜索工具
- CalculatorTool: 数学计算工具
- MemoryTool: 记忆工具
- RAGTool: 检索增强生成工具
- NoteTool: 结构化笔记工具（第9章）
- TerminalTool: 命令行工具（第9章）
- GitTool: Git操作工具（Coding Agent）
- TestRunnerTool: 测试执行工具（Coding Agent）
- FileTool: 文件读写编辑工具（Coding Agent）
- CodeExecutionTool: 沙箱代码执行工具（Coding Agent）
- CodeSearchTool: 代码搜索工具（Coding Agent）
- LinterTool: 代码检查与格式化工具（Coding Agent）
- ProfilerTool: 性能分析工具（Coding Agent）
- MCPTool: MCP 协议工具（第10章）
- A2ATool: A2A 协议工具（第10章）
- ANPTool: ANP 协议工具（第10章）
- BFCLEvaluationTool: BFCL评估工具（第12章）
- GAIAEvaluationTool: GAIA评估工具（第12章）
- LLMJudgeTool: LLM Judge评估工具（第12章）
- WinRateTool: Win Rate评估工具（第12章）
"""

from .search_tool import SearchTool
from .calculator import CalculatorTool
from .memory_tool import MemoryTool
from .rag_tool import RAGTool
from .note_tool import NoteTool
from .terminal_tool import TerminalTool
from .git_tool import GitTool
from .test_runner_tool import TestRunnerTool
from .file_tool import FileTool
from .code_execution_tool import CodeExecutionTool
from .code_search_tool import CodeSearchTool
from .linter_tool import LinterTool
from .profiler_tool import ProfilerTool
from .protocol_tools import MCPTool, A2ATool, ANPTool

# Evaluation tools depend on the external hello_agents package.
# Guard imports so core coding tools work without the full installation.
try:
    from .bfcl_evaluation_tool import BFCLEvaluationTool
except ImportError:
    BFCLEvaluationTool = None

try:
    from .gaia_evaluation_tool import GAIAEvaluationTool
except ImportError:
    GAIAEvaluationTool = None

try:
    from .llm_judge_tool import LLMJudgeTool
except ImportError:
    LLMJudgeTool = None

try:
    from .win_rate_tool import WinRateTool
except ImportError:
    WinRateTool = None

__all__ = [
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
    "LinterTool",
    "ProfilerTool",
    "MCPTool",
    "A2ATool",
    "ANPTool",
    "BFCLEvaluationTool",
    "GAIAEvaluationTool",
    "LLMJudgeTool",
    "WinRateTool",
]
