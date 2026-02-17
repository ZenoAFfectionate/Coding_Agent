import argparse
import atexit
import glob
import os
import select
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap: ensure the project root is on sys.path so ``code.*`` imports work
# even when running ``python run.py`` directly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from code.core.llm import HelloAgentsLLM
from code.agents.react_agent import ReActAgent
from code.tools.registry import ToolRegistry
from code.tools.builtin.file_tool import FileTool
from code.tools.builtin.code_execution_tool import CodeExecutionTool
from code.tools.builtin.code_search_tool import CodeSearchTool
from code.tools.builtin.test_runner_tool import TestRunnerTool
from code.tools.builtin.git_tool import GitTool
from code.tools.builtin.linter_tool import LinterTool
from code.tools.builtin.profiler_tool import ProfilerTool

PROMPTS_DIR = PROJECT_ROOT / "prompts"


class PromptManager:
    """Loads and manages task-specific prompts from the prompts/ directory.

    Prompt files are plain text with a .prompt extension. The system prompt
    is always loaded as the base. Task-specific prompts are prepended to the
    user message when a slash command is used, giving the LLM focused
    instructions for that task type.
    """

    # Maps slash command names to prompt file stems
    TASK_PROMPTS = {
        "review":   "code_review",
        "test":     "test_generation",
        "optimize": "optimization",
        "debug":    "debug",
    }

    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
        self._cache: dict[str, str] = {}

    def load(self, name: str) -> str:
        """Load a prompt file by stem name, with caching.

        Args:
            name: Prompt file stem (e.g. "system", "code_review").

        Returns:
            Prompt text, or empty string if not found.
        """
        if name in self._cache:
            return self._cache[name]

        path = self.prompts_dir / f"{name}.prompt"
        try:
            text = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            print(f"Warning: prompt file not found: {path}")
            text = ""

        self._cache[name] = text
        return text

    @property
    def system_prompt(self) -> str:
        """Load the base system prompt."""
        return self.load("system") or "You're an expert Python coding assistant."

    def build_task_message(self, command: str, user_args: str) -> str:
        """Build a user message with task-specific prompt prepended.

        Args:
            command: Slash command name (e.g. "review").
            user_args: The rest of the user's input after the command.

        Returns:
            Combined message with task prompt + user instruction.
        """
        prompt_name = self.TASK_PROMPTS.get(command)
        if not prompt_name:
            return user_args

        task_prompt = self.load(prompt_name)
        if not task_prompt:
            return user_args

        return (
            f"<task-instructions>\n{task_prompt}\n</task-instructions>\n\n"
            f"User request: {user_args}"
        )


def build_agent(
    workspace: str = ".",
    max_iterations: int = 15,
    temperature: float = 0.2,
) -> ReActAgent:
    """Create a fully-equipped Python Coding Agent.

    Args:
        workspace: Root directory the agent operates in.
        max_iterations: Max tool-calling rounds per query.
        temperature: LLM sampling temperature (low = deterministic).

    Returns:
        A ready-to-use ReActAgent.
    """
    workspace = str(Path(workspace).resolve())

    llm = HelloAgentsLLM(temperature=temperature)

    # --- Tools ---
    registry = ToolRegistry()
    registry.register_tool(FileTool(workspace=workspace))
    registry.register_tool(CodeExecutionTool(workspace=workspace, timeout=30))
    registry.register_tool(CodeSearchTool(workspace=workspace))
    registry.register_tool(TestRunnerTool(project_path=workspace, timeout=120))
    registry.register_tool(GitTool(repo_path=workspace))
    registry.register_tool(LinterTool(workspace=workspace, timeout=30))
    registry.register_tool(ProfilerTool(workspace=workspace, timeout=60))

    # --- System prompt ---
    prompt_manager = PromptManager(PROMPTS_DIR)
    system_prompt = prompt_manager.system_prompt

    # --- Agent ---
    agent = ReActAgent(
        name="CodingAgent",
        llm=llm,
        system_prompt=system_prompt,
        tool_registry=registry,
        max_steps=max_iterations,
    )

    return agent


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

HELP_TEXT = """\
Commands:
  /review <path>         Structured code review
  /test <path>           Generate tests for a module
  /optimize <path>       Profile and optimize code
  /debug <description>   Systematic debugging
  /help                  Show this help
  quit / exit / q        Exit the agent

Tip: Multi-line paste is auto-detected. You can also use:
  python run.py --task "$(cat problem.txt)"
"""


def _print_sandbox_code(workspace: str) -> None:
    """Print all .py files created in the sandbox after agent finishes."""
    py_files = sorted(glob.glob(os.path.join(workspace, "**", "*.py"), recursive=True))
    # Exclude test files and __pycache__
    py_files = [f for f in py_files if "__pycache__" not in f]

    if not py_files:
        return

    print("=" * 60)
    print("  Generated Code Files")
    print("=" * 60)
    for filepath in py_files:
        rel = os.path.relpath(filepath, workspace)
        try:
            content = open(filepath, encoding="utf-8").read()
        except Exception:
            continue
        print(f"\n--- {rel} ---")
        print(content)
    print("=" * 60)


def _read_user_input() -> str:
    """Read user input with multi-line paste detection.

    Uses select() to check if more lines are buffered on stdin
    (indicating a paste operation) and collects them all into a
    single input string.
    """
    first_line = input("You > ")
    lines = [first_line]

    # Collect additional buffered lines from a multi-line paste
    try:
        while select.select([sys.stdin], [], [], 0.05)[0]:
            line = sys.stdin.readline()
            if not line:  # EOF
                break
            lines.append(line.rstrip("\n"))
    except (ValueError, OSError):
        pass

    return "\n".join(lines).strip()


def repl(agent: ReActAgent, sandbox_dir: str = None) -> None:
    """Run an interactive read-eval-print loop with slash-command support."""
    prompt_manager = PromptManager(PROMPTS_DIR)

    print("=" * 60)
    print("  Python Coding Agent")
    print(f"  Model   : {agent.llm.model}")
    print(f"  Provider: {agent.llm.provider}")
    print(f"  Tools   : {', '.join(agent.tool_registry.list_tools())}")
    if sandbox_dir:
        print(f"  Sandbox : {sandbox_dir}")
        print(f"  Mode    : sandbox (auto-cleanup on exit)")
    else:
        print(f"  Workspace: {agent.tool_registry.get_tool('file').workspace}")
        print(f"  Mode    : direct (operating on real files)")
    print("=" * 60)
    print("Type your request, use /commands, or 'quit' to stop.")
    print("Type /help to see available commands.\n")

    while True:
        try:
            user_input = _read_user_input()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        # --- Slash commands ---
        if user_input.startswith("/"):
            parts = user_input[1:].split(None, 1)
            command = parts[0].lower() if parts else ""
            args = parts[1] if len(parts) > 1 else ""

            if command == "help":
                print(HELP_TEXT)
                continue

            if command in PromptManager.TASK_PROMPTS:
                if not args:
                    print(f"Usage: /{command} <path or description>")
                    continue
                message = prompt_manager.build_task_message(command, args)
                print(f"[{command}] Processing: {args}")
            else:
                print(f"Unknown command: /{command}. Type /help for options.")
                continue
        else:
            message = user_input

        # --- Run agent ---
        try:
            response = agent.run(message)
            print(f"\nAgent > {response}\n")
            if sandbox_dir:
                _print_sandbox_code(sandbox_dir)
        except Exception as e:
            print(f"\n[Error] {e}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Python Coding Agent — an AI-powered software engineering assistant."
    )
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default=None,
        help=(
            "Root directory the agent operates in. "
            "If not specified, a temporary sandbox is created and cleaned up on exit. "
            "Use this for code review / debugging on real projects."
        ),
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default=None,
        help="Single-shot mode: run one task and exit.",
    )
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=15,
        help="Max tool-calling iterations per query (default: 15).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature (default: 0.2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine workspace: sandbox (default) or user-specified directory
    sandbox_dir = None
    if args.workspace:
        # Direct mode: operate on user-specified directory
        workspace = args.workspace
    else:
        # Sandbox mode: create a temp directory, auto-cleanup on exit
        sandbox_dir = tempfile.mkdtemp(prefix="codingagent_sandbox_")
        workspace = sandbox_dir
        atexit.register(_cleanup_sandbox, sandbox_dir)
        print(f"[Sandbox] Created temporary workspace: {sandbox_dir}")

    agent = build_agent(
        workspace=workspace,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
    )

    if args.task:
        # Single-shot mode
        response = agent.run(args.task)
        print(response)
        if sandbox_dir:
            _print_sandbox_code(sandbox_dir)
    else:
        # Interactive REPL
        repl(agent, sandbox_dir=sandbox_dir)


def _cleanup_sandbox(sandbox_dir: str) -> None:
    """Remove the temporary sandbox directory."""
    try:
        shutil.rmtree(sandbox_dir)
        print(f"\n[Sandbox] Cleaned up temporary workspace: {sandbox_dir}")
    except Exception as e:
        print(f"\n[Sandbox] Warning: failed to clean up {sandbox_dir}: {e}")


if __name__ == "__main__":
    main()
