import argparse
import atexit
import json
import logging
import os
import re
import select
import shutil
import sys
import tempfile
import time
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from code.core.llm import HelloAgentsLLM
from code.core.config import Config
from code.core.agent import Agent
from code.agents.function_call_agent import FunctionCallAgent
from code.tools.registry import ToolRegistry
from code.tools.builtin.file_tool import FileTool
from code.tools.builtin.code_execution_tool import CodeExecutionTool
from code.tools.builtin.code_search_tool import CodeSearchTool
from code.tools.builtin.test_runner_tool import TestRunnerTool
from code.tools.builtin.git_tool import GitTool
from code.tools.builtin.linter_tool import LinterTool
from code.tools.builtin.profiler_tool import ProfilerTool
from code.tools.builtin.finish_tool import FinishTool

PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "xCode"
DEFAULT_SYSTEM_PROMPT = "You're an expert Python coding assistant."
DEFAULT_BATCH_TASK_PROMPT = (
    "{problem}\n\n"
    "Write a complete Python program that reads from stdin and writes to "
    "stdout. Save it as a single .py file using the file tool."
)

logger = logging.getLogger("coding_agent_fc")


# ======================================================================
#  Helpers
# ======================================================================

def setup_logging(log_dir: Path = None) -> Path:
    """Configure file-only logging under results/logs/."""
    log_dir = log_dir or (RESULTS_DIR / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "tool_agent.log"

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)
    return log_file


def _load_prompt(path: Path, fallback: str | None = None) -> str | None:
    """Load a prompt file, returning *fallback* if the file is missing or empty."""
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or fallback
    except FileNotFoundError:
        logger.warning("Prompt file not found: %s", path)
        return fallback


def _collect_py_files(directory: str) -> list[tuple[str, str]]:
    """Collect all .py files under *directory*, returning (rel_path, content) pairs."""
    root = Path(directory)
    results = []
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        try:
            results.append((str(p.relative_to(root)), p.read_text(encoding="utf-8").strip()))
        except Exception:
            continue
    return results


def _extract_solution(sandbox_dir: str, response: str) -> str:
    """Extract code: prefer sandbox .py files, fall back to response code blocks."""
    files = _collect_py_files(sandbox_dir)
    if files:
        if len(files) == 1:
            return files[0][1]
        return "\n\n".join(f"# --- {rel} ---\n{content}" for rel, content in files)

    for pattern in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

    return ""


# ======================================================================
#  Agent builder
# ======================================================================

def build_agent(
    workspace: str = ".",
    max_iterations: int = 32,
    temperature: float = 0.2,
    debug: bool = True,
    log_file: str = None,
    enable_reflection: bool = True,
    max_reflection_retries: int = 1,
    reflection_prompt: str | None = None,
    enable_planning: bool = False,
    system_prompt: str | None = None,
    exclude_tools: list[str] | None = None,
) -> FunctionCallAgent:
    """Create a fully-equipped FunctionCallAgent.

    Args:
        exclude_tools: Tool names to exclude (e.g. ["code_exec", "test_runner"]).
    """
    workspace = str(Path(workspace).resolve())
    config = Config(debug=debug, temperature=temperature)
    llm = HelloAgentsLLM(
        model=config.default_model,
        provider=config.default_provider,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    exclude = set(exclude_tools or [])

    all_tools = [
        FileTool(workspace=workspace),
        CodeExecutionTool(workspace=workspace, timeout=30),
        CodeSearchTool(workspace=workspace),
        TestRunnerTool(project_path=workspace, timeout=120),
        GitTool(repo_path=workspace),
        LinterTool(workspace=workspace, timeout=30),
        ProfilerTool(workspace=workspace, timeout=60),
        FinishTool(),
    ]

    registry = ToolRegistry()
    for tool in all_tools:
        if tool.name not in exclude:
            registry.register_tool(tool)

    agent = FunctionCallAgent(
        name="CodingAgent-FC",
        llm=llm,
        system_prompt=system_prompt or _load_prompt(PROMPTS_DIR / "system.prompt", DEFAULT_SYSTEM_PROMPT),
        tool_registry=registry,
        max_steps=max_iterations,
        config=config,
        enable_logging=bool(log_file),
        log_file=log_file,
        enable_reflection=enable_reflection,
        max_reflection_retries=max_reflection_retries,
        reflection_prompt=reflection_prompt,
        enable_planning=enable_planning,
    )

    # Silence AgentLogger's console handler in non-debug mode
    if agent.logger and not debug:
        agent.logger._logger.handlers = [
            h for h in agent.logger._logger.handlers
            if isinstance(h, logging.FileHandler)
        ]

    logger.info(
        "FC Agent built: model=%s provider=%s workspace=%s steps=%d reflection=%s",
        llm.model, llm.provider, workspace, max_iterations,
        f"on(retries={max_reflection_retries})" if enable_reflection else "off",
    )
    return agent


# ======================================================================
#  Interactive REPL
# ======================================================================

HELP_TEXT = """\
Commands:
  /help                  Show this help
  /save                  Save session history to file
  /history               Show conversation history summary
  /compact               Manually trigger context compaction
  quit / exit / q        Exit the agent

Just describe what you need in natural language.
"""


def _read_user_input() -> str:
    """Read user input with multi-line paste detection."""
    try:
        while select.select([sys.stdin], [], [], 0)[0]:
            if not sys.stdin.readline():
                break
    except (ValueError, OSError):
        pass

    lines = [input("You > ")]
    try:
        while select.select([sys.stdin], [], [], 0.05)[0]:
            line = sys.stdin.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    except (ValueError, OSError):
        pass
    return "\n".join(lines).strip()


def repl(agent: FunctionCallAgent, sandbox_dir: str = None, session_file: str = None) -> None:
    """Run an interactive read-eval-print loop."""
    # Restore previous session if available
    if session_file and os.path.exists(session_file):
        if agent.load_session(session_file):
            print(f"[Session] Restored from {session_file} ({len(agent._history)} messages)")

    # Banner
    reflect = (f"on (max_retries={agent.max_reflection_retries})"
               if agent.enable_reflection else "off")
    print("=" * 60)
    print("  Python Coding Agent (Function Calling)")
    print(f"  Model   : {agent.llm.model}")
    print(f"  Provider: {agent.llm.provider}")
    print(f"  Tools   : {', '.join(agent.list_tools())}")
    print(f"  Reflect : {reflect}")
    if sandbox_dir:
        print(f"  Sandbox : {sandbox_dir}")
    print("=" * 60)
    print("Type your request or 'quit' to stop. /help for commands.\n")

    def save_session():
        if session_file:
            agent.save_session(session_file)
            print(f"[Session] Saved to {session_file}")

    while True:
        try:
            user_input = _read_user_input()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            save_session()
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("quit", "exit", "q"):
            print("Goodbye.")
            save_session()
            break
        if cmd == "/help":
            print(HELP_TEXT)
            continue
        if cmd == "/save":
            save_session()
            continue
        if cmd == "/history":
            history = agent.get_history()
            if not history:
                print("[History] No conversation history yet.")
            else:
                print(f"[History] {len(history)} messages:")
                for i, msg in enumerate(history):
                    print(f"  [{i}] {msg.role}: {msg.content[:100].replace(chr(10), ' ')}...")
            continue
        if cmd == "/compact":
            print("[Compact] This command works automatically when context_max_tokens is set.")
            continue

        logger.info("User input (%d chars): %.200s", len(user_input), user_input)
        try:
            response = agent.run(user_input)
            print(f"\nAgent > {response}\n")
            logger.info("Agent response (%d chars): %.500s", len(response), response)

            # Save trajectory
            if agent.trajectory and agent.trajectory.steps:
                traj_dir = RESULTS_DIR / "trajectories"
                traj_dir.mkdir(parents=True, exist_ok=True)
                traj_path = traj_dir / "fc_agent_trajectory.json"
                agent.save_trajectory(str(traj_path), fmt="json")
                agent.print_trajectory()
                print(f"[Trajectory] Saved to {traj_path}")

            # Display generated code files
            if sandbox_dir:
                files = _collect_py_files(sandbox_dir)
                if files:
                    print("=" * 60)
                    print("  Generated Code Files")
                    print("=" * 60)
                    for rel, content in files:
                        print(f"\n--- {rel} ---")
                        print(content)
                    print("=" * 60)

        except Exception as e:
            print(f"\n[Error] {e}\n")
            logger.error("Agent error: %s", e, exc_info=True)


# ======================================================================
#  Batch mode
# ======================================================================

def run_batch(agent: FunctionCallAgent, sandbox_dir: str,
              input_path: Path, output_path: Path,
              start: int = 0, limit: int = None,
              total_samples: int = 500) -> None:
    """Run the agent on a batch of problems from a JSONL file."""
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    task_template = _load_prompt(PROMPTS_DIR / "batch_task.prompt", DEFAULT_BATCH_TASK_PROMPT)

    total = sum(1 for _ in open(input_path, encoding="utf-8"))
    effective_limit = limit or total_samples
    num_to_process = min(effective_limit, total - start)

    header = f"[Batch] Problems: {total}  |  Start: {start}  |  Target: {num_to_process}"
    print(f"{header}\n[Batch] Output: {output_path}\n")

    processed = failed = 0
    open_mode = "a" if start > 0 else "w"

    pbar = tqdm(total=num_to_process, desc="Processing",
                unit="problem",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, open_mode, encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            if idx < start:
                continue
            if processed >= num_to_process:
                break

            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            problem = record.get("problem", "")
            problem = re.split(r"\n###\s*Note\b", problem, maxsplit=1)[0].rstrip()
            input_output = record.get("input_output", "")
            task_id = record.get("id", f"task_{idx}")

            # Clear sandbox for each problem
            for entry in Path(sandbox_dir).iterdir():
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
            agent.clear_history()

            t0 = time.time()
            try:
                response = agent.run(task_template.format(problem=problem))
            except Exception as e:
                logger.error("Agent failed on %s: %s", task_id, e)
                response = ""
                failed += 1

            solution = _extract_solution(sandbox_dir, response)
            if not solution:
                logger.warning("No code extracted for %s", task_id)
                failed += 1

            fout.write(json.dumps({
                "problem": problem,
                "solution": solution,
                "input_output": input_output,
            }, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            elapsed = time.time() - t0
            pbar.set_postfix(
                id=task_id[:15],
                time=f"{elapsed:.1f}s",
                failed=failed
            )
            pbar.update(1)

    pbar.close()
    print(f"\n{'=' * 60}")
    print(f"[Batch] Done. Processed: {processed}, Failed: {failed}")
    print(f"[Batch] Results: {output_path}")
    print(f"{'=' * 60}")


# ======================================================================
#  Entry point
# ======================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Python Coding Agent (Function Calling) -- interactive REPL or batch inference."
    )

    p.add_argument("--workspace", "-w", default=None,
                   help="Agent workspace dir (default: temp sandbox).")
    p.add_argument("--max-iterations", "-n", type=int, default=32,
                   help="Max tool-calling iterations per query.")
    p.add_argument("--temperature", type=float, default=0.2,
                   help="LLM sampling temperature (default: 0.2).")
    p.add_argument("--debug", action="store_true",
                   help="Verbose console output.")
    p.add_argument("--no-reflection", action="store_true",
                   help="Disable reflection/self-verification.")
    p.add_argument("--max-reflection-retries", type=int, default=1,
                   help="Max reflection revision attempts.")
    p.add_argument("--plan", action="store_true",
                   help="Enable plan-then-execute mode.")
    p.add_argument("--restore", action="store_true",
                   help="Restore previous session history on startup.")
    p.add_argument("--session-file", default=None,
                   help="Path to session history file (default: auto-generated).")

    p.add_argument("--batch", "-b", action="store_true",
                   help="Batch mode: process JSONL problems.")
    p.add_argument("--input", "-i", default=str(DATA_DIR / "valid.jsonl"),
                   help="Batch input JSONL.")
    p.add_argument("--output", "-o", default=str(DATA_DIR / "result.jsonl"),
                   help="Batch output JSONL.")
    p.add_argument("--start", "-s", type=int, default=0,
                   help="Batch start index (0-based).")
    p.add_argument("--limit", "-l", type=int, default=None,
                   help="Batch max problems to process.")

    args = p.parse_args()

    log_file = setup_logging()
    logger.info("Session started (mode=%s)", "batch" if args.batch else "repl")

    # Workspace: use provided path or create a temp sandbox
    sandbox_dir = None
    if args.workspace:
        workspace = args.workspace
    else:
        sandbox_dir = tempfile.mkdtemp(prefix="codingagent_fc_sandbox_")
        workspace = sandbox_dir
        atexit.register(lambda: shutil.rmtree(sandbox_dir, ignore_errors=True))
        print(f"[Sandbox] Created: {sandbox_dir}")

    agent = build_agent(
        workspace=workspace,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        debug=args.debug,
        log_file=str(log_file),
        enable_reflection=not args.no_reflection,
        max_reflection_retries=args.max_reflection_retries,
        reflection_prompt=_load_prompt(PROMPTS_DIR / "reflection.prompt"),
        enable_planning=args.plan,
    )

    if args.batch:
        run_batch(agent, workspace, Path(args.input), Path(args.output),
                  args.start, args.limit)
    else:
        session_file = None
        if args.restore or args.session_file:
            session_file = args.session_file or Agent.get_default_session_path(agent.name)
        repl(agent, sandbox_dir=sandbox_dir, session_file=session_file)

    logger.info("Session ended")
