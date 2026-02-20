"""Inference entry point for the Python Coding Agent.

Supports two modes:
  1. Interactive REPL (default)      — python inference.py
  2. Batch (JSONL)                   — python inference.py --batch --input data.jsonl
"""

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
from glob import glob
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from code.core.llm import HelloAgentsLLM
from code.core.config import Config
from code.agents.react_agent import ReActAgent
from code.tools.registry import ToolRegistry
from code.tools.builtin.file_tool import FileTool
from code.tools.builtin.code_execution_tool import CodeExecutionTool
from code.tools.builtin.code_search_tool import CodeSearchTool
from code.tools.builtin.test_runner_tool import TestRunnerTool
from code.tools.builtin.git_tool import GitTool
from code.tools.builtin.linter_tool import LinterTool
from code.tools.builtin.profiler_tool import ProfilerTool


SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system.prompt"
BATCH_TASK_PROMPT_PATH = PROJECT_ROOT / "prompts" / "batch_task.prompt"
REFLECTION_PROMPT_PATH = PROJECT_ROOT / "prompts" / "reflection.prompt"
DEFAULT_SYSTEM_PROMPT = "You're an expert Python coding assistant."
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "xCode"

logger = logging.getLogger("coding_agent")


# ======================================================================
#  Helpers
# ======================================================================

def setup_logging(log_dir: Path = None) -> Path:
    """Configure file-only logging under results/logs/.

    We don't use ``code.utils.logging.setup_logger`` here because it always
    adds a console handler — we want stdout clean for the interactive REPL.
    """
    log_dir = log_dir or (RESULTS_DIR / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "inference.log"

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


def _load_system_prompt() -> str:
    """Load the system prompt from file, with a fallback default."""
    try:
        text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        return text or DEFAULT_SYSTEM_PROMPT
    except FileNotFoundError:
        logger.warning("System prompt not found: %s", SYSTEM_PROMPT_PATH)
        return DEFAULT_SYSTEM_PROMPT


def _load_batch_task_prompt() -> str:
    """Load the batch task prompt template from file.

    The template should contain a ``{problem}`` placeholder that will be
    substituted with the actual problem text at runtime.
    """
    fallback = (
        "{problem}\n\n"
        "Write a complete Python program that reads from stdin and writes to "
        "stdout. Save it as a single .py file using the file tool."
    )
    try:
        text = BATCH_TASK_PROMPT_PATH.read_text(encoding="utf-8").strip()
        if not text:
            return fallback
        return text
    except FileNotFoundError:
        logger.warning("Batch task prompt not found: %s", BATCH_TASK_PROMPT_PATH)
        return fallback


def _load_reflection_prompt() -> str | None:
    """Load the reflection prompt template from file.

    Returns ``None`` if the file is missing, letting the agent fall back
    to its built-in default.
    """
    try:
        text = REFLECTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
        return text or None
    except FileNotFoundError:
        logger.warning("Reflection prompt not found: %s", REFLECTION_PROMPT_PATH)
        return None


def _collect_py_files(directory: str) -> list[tuple[str, str]]:
    """Collect all .py files under *directory*, returning ``(rel_path, content)`` pairs.

    Shared by sandbox display (REPL) and code extraction (batch).
    """
    results = []
    for fp in sorted(glob(os.path.join(directory, "**", "*.py"), recursive=True)):
        if "__pycache__" in fp:
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                results.append((os.path.relpath(fp, directory), f.read().strip()))
        except Exception:
            continue
    return results


def _save_trajectory(agent: ReActAgent) -> None:
    """Persist the agent's trajectory to ``results/trajectories/`` and print to console."""
    if not agent.trajectory or not agent.trajectory.steps:
        return
    traj_dir = RESULTS_DIR / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    path = traj_dir / "single_agent_trajectory.json"
    agent.save_trajectory(str(path), fmt="json")
    agent.print_trajectory()
    print(f"[Trajectory] Saved to {path}")
    logger.info("Trajectory saved: %s", path)



def build_agent(
    workspace: str = ".",
    max_iterations: int = 32,
    temperature: float = 0.2,
    debug: bool = True,
    log_file: str = None,
    enable_reflection: bool = True,
    max_reflection_retries: int = 1,
    reflection_prompt: str | None = None,
) -> ReActAgent:
    """Create a fully-equipped single-agent CodingAgent."""
    workspace = str(Path(workspace).resolve())
    config = Config(debug=debug, temperature=temperature)
    llm = HelloAgentsLLM(
        model=config.default_model,
        provider=config.default_provider,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    registry = ToolRegistry()
    for tool in [
        FileTool(workspace=workspace),
        CodeExecutionTool(workspace=workspace, timeout=30),
        CodeSearchTool(workspace=workspace),
        TestRunnerTool(project_path=workspace, timeout=120),
        GitTool(repo_path=workspace),
        LinterTool(workspace=workspace, timeout=30),
        ProfilerTool(workspace=workspace, timeout=60),
    ]:
        registry.register_tool(tool)

    agent = ReActAgent(
        name="CodingAgent",
        llm=llm,
        system_prompt=_load_system_prompt(),
        tool_registry=registry,
        max_steps=max_iterations,
        config=config,
        enable_logging=bool(log_file),
        log_file=log_file,
        enable_reflection=enable_reflection,
        max_reflection_retries=max_reflection_retries,
        reflection_prompt=reflection_prompt,
    )

    # Silence AgentLogger's console handler in non-debug mode
    if agent.logger and not debug:
        agent.logger._logger.handlers = [
            h for h in agent.logger._logger.handlers
            if isinstance(h, logging.FileHandler)
        ]

    logger.info(
        "Agent built: model=%s provider=%s workspace=%s steps=%d reflection=%s",
        llm.model, llm.provider, workspace, max_iterations,
        f"on(retries={max_reflection_retries})" if enable_reflection else "off",
    )
    return agent



HELP_TEXT = """\
Commands:
  /help                  Show this help
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


def _print_sandbox_code(workspace: str) -> None:
    """Display all .py files generated in the sandbox."""
    files = _collect_py_files(workspace)
    if not files:
        return
    print("=" * 60)
    print("  Generated Code Files")
    print("=" * 60)
    for rel, content in files:
        print(f"\n--- {rel} ---")
        print(content)
        logger.info("Generated file: %s (%d chars)", rel, len(content))
    print("=" * 60)


def repl(agent: ReActAgent, sandbox_dir: str = None) -> None:
    """Run an interactive read-eval-print loop."""
    print("=" * 60)
    print("  Python Coding Agent")
    print(f"  Model   : {agent.llm.model}")
    print(f"  Provider: {agent.llm.provider}")
    print(f"  Tools   : {', '.join(agent.tool_registry.list_tools())}")
    reflect = (f"on (max_retries={agent.max_reflection_retries})"
               if agent.enable_reflection else "off")
    print(f"  Reflect : {reflect}")
    if sandbox_dir:
        print(f"  Sandbox : {sandbox_dir}")
    else:
        print(f"  Workspace: {agent.tool_registry.get_tool('file').workspace}")
    print("=" * 60)
    print("Type your request or 'quit' to stop. /help for commands.\n")

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
        if user_input.strip().lower() == "/help":
            print(HELP_TEXT)
            continue

        logger.info("User input (%d chars): %.200s", len(user_input), user_input)
        try:
            response = agent.run(user_input)
            print(f"\nAgent > {response}\n")
            logger.info("Agent response (%d chars): %.500s", len(response), response)
            _save_trajectory(agent)
            if sandbox_dir:
                _print_sandbox_code(sandbox_dir)
        except Exception as e:
            print(f"\n[Error] {e}\n")
            logger.error("Agent error: %s", e, exc_info=True)



def _extract_solution(sandbox_dir: str, response: str) -> str:
    """Extract code: prefer sandbox .py files, fall back to response code blocks."""
    # 1. Collect from sandbox
    files = _collect_py_files(sandbox_dir)
    if files:
        if len(files) == 1:
            return files[0][1]
        return "\n\n".join(f"# --- {rel} ---\n{content}" for rel, content in files)

    # 2. Fall back to fenced code blocks in the response text
    for pattern in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

    return ""


def run_batch(agent: ReActAgent, sandbox_dir: str,
              input_path: Path, output_path: Path,
              start: int = 0, limit: int = None) -> None:
    """Run the agent on a batch of problems from a JSONL file."""
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    task_template = _load_batch_task_prompt()

    total = sum(1 for _ in open(input_path, encoding="utf-8"))
    header = f"[Batch] Problems: {total}  |  Start: {start}"
    if limit:
        header += f"  |  Limit: {limit}"
    print(f"{header}\n[Batch] Output: {output_path}\n")

    processed = failed = 0
    open_mode = "a" if start > 0 else "w"

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, open_mode, encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            if idx < start:
                continue
            if limit and processed >= limit:
                break

            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            problem = record.get("problem", "")
            # Strip "### Note" sections — they are supplementary explanations
            # that add tokens without aiding the solution.
            problem = re.split(r"\n###\s*Note\b", problem, maxsplit=1)[0].rstrip()
            input_output = record.get("input_output", "")
            task_id = record.get("id", f"task_{idx}")

            print(f"{'=' * 60}\n[{idx}/{total}] {task_id}\n{'=' * 60}")

            for entry in os.listdir(sandbox_dir):
                ep = os.path.join(sandbox_dir, entry)
                shutil.rmtree(ep) if os.path.isdir(ep) else os.remove(ep)
            agent.clear_history()

            task_msg = task_template.format(problem=problem)

            t0 = time.time()
            try:
                response = agent.run(task_msg)
            except Exception as e:
                print(f"[Error] Agent failed on {task_id}: {e}")
                response = ""
                failed += 1

            solution = _extract_solution(sandbox_dir, response)
            if not solution:
                print(f"[Warning] No code extracted for {task_id}")
                failed += 1

            fout.write(json.dumps({
                "problem": problem,
                "solution": solution,
                "input_output": input_output,
            }, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            print(f"[Done] {task_id} — {time.time() - t0:.1f}s — "
                  f"{len(solution)} chars\n")

    print(f"{'=' * 60}")
    print(f"[Batch] Done. Processed: {processed}, Failed: {failed}")
    print(f"[Batch] Results: {output_path}")
    print(f"{'=' * 60}")



if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Python Coding Agent — interactive REPL or batch inference."
    )

    p.add_argument("--workspace", "-w", default=None,
                   help="Agent workspace dir (default: temp sandbox).")
    p.add_argument("--max-iterations", "-n", type=int, default=16,
                   help="Max tool-calling iterations per query.")
    p.add_argument("--temperature", type=float, default=0.2,
                   help="LLM sampling temperature (default: 0.2).")
    p.add_argument("--debug", action="store_true",
                   help="Verbose console output.")
    p.add_argument("--no-reflection", action="store_true",
                   help="Disable reflection/self-verification.")
    p.add_argument("--max-reflection-retries", type=int, default=1,
                   help="Max reflection revision attempts.")

    # Batch
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
    logger.info("Session started (mode=%s)",
                "batch" if args.batch else "repl")

    # Workspace
    sandbox_dir = None
    if args.workspace:
        workspace = args.workspace
    else:
        sandbox_dir = tempfile.mkdtemp(prefix="codingagent_sandbox_")
        workspace = sandbox_dir
        atexit.register(lambda: shutil.rmtree(sandbox_dir, ignore_errors=True))
        print(f"[Sandbox] Created: {sandbox_dir}")
        logger.info("Sandbox created: %s", sandbox_dir)

    # Build & dispatch
    agent = build_agent(
        workspace=workspace,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        debug=args.debug,
        log_file=str(log_file),
        enable_reflection=not args.no_reflection,
        max_reflection_retries=args.max_reflection_retries,
        reflection_prompt=_load_reflection_prompt(),
    )

    if args.batch:
        run_batch(agent, workspace, Path(args.input), Path(args.output),
                  args.start, args.limit)
    else:
        repl(agent, sandbox_dir=sandbox_dir)

    logger.info("Session ended")
