"""Batch runner: feeds each problem from valid.jsonl through the CodingAgent
and saves results (problem, solution, input_output) to result.jsonl."""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from run import build_agent

DATA_DIR = PROJECT_ROOT / "data" / "xCode"
INPUT_FILE = DATA_DIR / "valid.jsonl"
OUTPUT_FILE = DATA_DIR / "result.jsonl"


def extract_code_from_sandbox(sandbox_dir: str) -> str:
    """Collect all .py files written by the agent in the sandbox.

    If there is exactly one file, return its content.
    If there are multiple, concatenate them (separated by a comment header).
    """
    py_files = sorted(
        glob.glob(os.path.join(sandbox_dir, "**", "*.py"), recursive=True)
    )
    py_files = [f for f in py_files if "__pycache__" not in f]

    if not py_files:
        return ""

    if len(py_files) == 1:
        return open(py_files[0], encoding="utf-8").read().strip()

    parts = []
    for fp in py_files:
        rel = os.path.relpath(fp, sandbox_dir)
        content = open(fp, encoding="utf-8").read().strip()
        parts.append(f"# --- {rel} ---\n{content}")
    return "\n\n".join(parts)


def extract_code_from_response(response: str) -> str:
    """Fallback: extract python code block from the agent's text response."""
    matches = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    matches = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    return ""


def clean_sandbox(sandbox_dir: str) -> None:
    """Remove all files inside the sandbox without deleting the dir itself."""
    for entry in os.listdir(sandbox_dir):
        path = os.path.join(sandbox_dir, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def build_task_message(problem: str) -> str:
    """Build the user message for a coding problem.

    The system prompt (set in run.py's build_agent) already establishes the
    agent's role and available tools — so the user message only needs the
    problem itself and the output format requirement.
    """
    return (
        f"{problem}\n\n"
        "Write a complete Python program that reads from stdin and writes to "
        "stdout. Save it as a single .py file using the file tool."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run CodingAgent on xCode problems."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(INPUT_FILE),
        help=f"Input JSONL file (default: {INPUT_FILE})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(OUTPUT_FILE),
        help=f"Output JSONL file (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=15,
        help="Max tool-calling iterations per problem (default: 15).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature (default: 0.2).",
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        help="Start index (0-based) — skip problems before this index.",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Max number of problems to process (default: all).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    # Create a persistent sandbox directory for the agent
    sandbox_dir = tempfile.mkdtemp(prefix="codingagent_batch_")
    print(f"[Batch] Sandbox: {sandbox_dir}")

    # Build agent once, reuse for all problems
    agent = build_agent(
        workspace=sandbox_dir,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        debug=False,
    )

    # Count total lines for progress reporting
    total_lines = sum(1 for _ in open(input_path, encoding="utf-8"))
    print(f"[Batch] Total problems in file: {total_lines}")
    print(f"[Batch] Starting from index: {args.start}")
    if args.limit:
        print(f"[Batch] Processing at most: {args.limit} problems")
    print(f"[Batch] Output: {output_path}")
    print()

    # If resuming (start > 0), keep existing output; otherwise start fresh
    open_mode = "a" if args.start > 0 else "w"

    processed = 0
    failed = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, open_mode, encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            if idx < args.start:
                continue

            if args.limit and processed >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            problem = record.get("problem", "")
            input_output = record.get("input_output", "")
            task_id = record.get("id", f"task_{idx}")

            print(f"{'='*60}")
            print(f"[{idx}/{total_lines}] {task_id}")
            print(f"{'='*60}")

            # Clean sandbox from previous run
            clean_sandbox(sandbox_dir)
            # Reset agent conversation history to prevent cross-problem pollution
            agent.clear_history()

            # Build prompt and run agent
            task_message = build_task_message(problem)

            start_time = time.time()
            try:
                response = agent.run(task_message)
            except Exception as e:
                print(f"[Error] Agent failed on {task_id}: {e}")
                response = ""
                failed += 1

            elapsed = time.time() - start_time

            # Extract solution: prefer sandbox files, fall back to response text
            solution = extract_code_from_sandbox(sandbox_dir)
            if not solution:
                solution = extract_code_from_response(response)

            if not solution:
                print(f"[Warning] No code extracted for {task_id}")
                failed += 1

            # Write result
            result = {
                "problem": problem,
                "solution": solution,
                "input_output": input_output,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            print(f"[Done] {task_id} — {elapsed:.1f}s — "
                  f"solution: {len(solution)} chars")
            print()

    # Cleanup
    shutil.rmtree(sandbox_dir, ignore_errors=True)

    print(f"{'='*60}")
    print(f"[Batch] Finished. Processed: {processed}, Failed: {failed}")
    print(f"[Batch] Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
