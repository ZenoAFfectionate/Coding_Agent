"""Execution modes — batch runner and interactive REPL."""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import select
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .worker_factory import WORKER_SPECS

if TYPE_CHECKING:
    from code.agents.orchestrator_agent import OrchestratorAgent

logger = logging.getLogger("multi_agent")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _print_sandbox_code(workspace: str) -> None:
    """Print all .py files created in the sandbox after agent finishes."""
    py_files = sorted(glob.glob(os.path.join(workspace, "**", "*.py"), recursive=True))
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


def _cleanup_sandbox(sandbox_dir: str) -> None:
    """Remove the temporary sandbox directory."""
    try:
        shutil.rmtree(sandbox_dir)
        print(f"\n[Sandbox] Cleaned up temporary workspace: {sandbox_dir}")
    except Exception as e:
        print(f"\n[Sandbox] Warning: failed to clean up {sandbox_dir}: {e}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

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


def run_batch(
    orchestrator: "OrchestratorAgent",
    sandbox_dir: str,
    input_path: Path,
    output_path: Path,
    start: int = 0,
    limit: int = None,
    total_samples: int = 500,
) -> None:
    """Run the multi-agent orchestrator on a batch of problems from a JSONL file."""
    from tqdm import tqdm
    from . import load_swarm_prompt, DEFAULT_BATCH_TASK_PROMPT

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    task_template = load_swarm_prompt("batch_task") or DEFAULT_BATCH_TASK_PROMPT

    total = sum(1 for _ in open(input_path, encoding="utf-8"))
    effective_limit = limit or total_samples
    num_to_process = min(effective_limit, total - start)

    header = f"[Batch] Problems: {total}  |  Start: {start}  |  Target: {num_to_process}"
    print(f"{header}\n[Batch] Output: {output_path}\n")
    logger.info(header)

    processed = failed = 0
    open_mode = "a" if start > 0 else "w"

    pbar = tqdm(
        total=num_to_process, desc="Processing",
        unit="problem",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, open_mode, encoding="utf-8") as fout,
    ):
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
            orchestrator._history.clear()

            logger.info("=" * 80)
            logger.info("[Batch] Processing %s (idx=%d)", task_id, idx)

            t0 = time.time()
            try:
                response = orchestrator.run(task_template.format(problem=problem))
            except Exception as e:
                logger.error("[Batch] Agent failed on %s: %s", task_id, e, exc_info=True)
                response = ""
                failed += 1

            solution = _extract_solution(sandbox_dir, response)
            if not solution:
                logger.warning("[Batch] No code extracted for %s", task_id)
                failed += 1

            fout.write(json.dumps({
                "problem": problem,
                "solution": solution,
                "input_output": input_output,
            }, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            elapsed = time.time() - t0
            logger.info("[Batch] %s done in %.1fs, solution_len=%d", task_id, elapsed, len(solution))
            pbar.set_postfix(id=task_id[:15], time=f"{elapsed:.1f}s", failed=failed)
            pbar.update(1)

    pbar.close()
    print(f"\n{'=' * 60}")
    print(f"[Batch] Done. Processed: {processed}, Failed: {failed}")
    print(f"[Batch] Results: {output_path}")
    print(f"{'=' * 60}")
    logger.info("[Batch] Done. Processed: %d, Failed: %d", processed, failed)


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

HELP_TEXT = """\
Commands:
  /help                  Show this help
  /workers               List available worker types
  quit / exit / q        Exit

This is a multi-agent system. The orchestrator automatically decomposes
your request and dispatches it to specialized workers (review, test,
optimize, debug).

Just describe what you need in natural language.

Tip: Multi-line paste is auto-detected. You can also use:
  python run_multi.py --task "$(cat problem.txt)"
"""


def _read_user_input() -> str:
    """Read user input with multi-line paste detection."""
    first_line = input("You > ")
    lines = [first_line]

    try:
        while select.select([sys.stdin], [], [], 0.05)[0]:
            line = sys.stdin.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    except (ValueError, OSError):
        pass

    return "\n".join(lines).strip()


def repl(orchestrator: "OrchestratorAgent", sandbox_dir: str = None) -> None:
    """Run an interactive read-eval-print loop."""

    print("=" * 60)
    print("  Multi-Agent Coding System")
    print(f"  Model   : {orchestrator.llm.model}")
    print(f"  Provider: {orchestrator.llm.provider}")
    print(f"  Workers : {', '.join(WORKER_SPECS.keys())}")
    if sandbox_dir:
        print(f"  Sandbox : {sandbox_dir}")
        print(f"  Mode    : sandbox (auto-cleanup on exit)")
    else:
        print(f"  Workspace: {orchestrator.workspace}")
        print(f"  Mode    : direct (operating on real files)")
    print("=" * 60)
    print("Type your request or 'quit' to stop.")
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

        if user_input.strip().lower() == "/help":
            print(HELP_TEXT)
            continue

        if user_input.strip().lower() == "/workers":
            print("\nAvailable workers:")
            for wtype, spec in WORKER_SPECS.items():
                print(f"  {wtype:10s} — {spec['description']}")
            print()
            continue

        try:
            response = orchestrator.run(user_input)
            print(f"\nAgent > {response}\n")
            if sandbox_dir:
                _print_sandbox_code(sandbox_dir)
        except Exception as e:
            print(f"\n[Error] {e}\n")
