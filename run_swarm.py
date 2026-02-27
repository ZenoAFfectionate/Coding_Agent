"""Multi-Agent Coding System — CLI entry point.

Thin wrapper that wires together the swarm package modules and
provides the argparse CLI for batch, single-task, and REPL modes.
"""

import argparse
import atexit
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from code.core.config import Config
from code.agents.orchestrator_agent import OrchestratorAgent
from code.swarm import (
    setup_logging, logger, DATA_DIR,
    run_batch, repl, _cleanup_sandbox, _print_sandbox_code,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Agent Coding System — orchestrator + specialized workers."
    )
    parser.add_argument(
        "--workspace", "-w", type=str, default=None,
        help="Root directory the agents operate in. "
             "If not specified, a temporary sandbox is created and cleaned up on exit.",
    )
    parser.add_argument(
        "--task", "-t", type=str, default=None,
        help="Single-shot mode: run one task and exit.",
    )
    parser.add_argument(
        "--max-worker-steps", "-n", type=int, default=10,
        help="Max tool-calling iterations per worker agent (default: 10).",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=16,
        help="Max orchestrator reasoning rounds (default: 16).",
    )
    parser.add_argument(
        "--max-result-chars", type=int, default=4000,
        help="Max chars per worker result before truncation (default: 4000).",
    )
    parser.add_argument(
        "--context-max-tokens", type=int, default=0,
        help="Context budget in tokens. 0 = unlimited (default: 0).",
    )
    parser.add_argument("--no-fc", action="store_true",
                        help="Disable function calling; use text-based JSON parsing instead.")
    parser.add_argument("--no-summarize", action="store_true",
                        help="Disable LLM-based result summarization (use truncation instead).")
    parser.add_argument("--no-reflect", action="store_true",
                        help="Disable reflection / self-verification step on final answer.")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="LLM sampling temperature (default: 0.2).")
    parser.add_argument("--no-debug", action="store_true",
                        help="Suppress step-by-step debug output.")

    # Batch mode arguments
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Batch mode: process problems from a JSONL file.")
    parser.add_argument("--input", "-i", default=str(DATA_DIR / "valid.jsonl"),
                        help="Batch input JSONL file (default: data/xCode/valid.jsonl).")
    parser.add_argument("--output", "-o", default=str(DATA_DIR / "result_multi.jsonl"),
                        help="Batch output JSONL file.")
    parser.add_argument("--start", "-s", type=int, default=0,
                        help="Batch start index (0-based).")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Batch max problems to process.")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for log files (default: results/logs/).")

    args = parser.parse_args()

    # Setup logging
    log_dir = Path(args.log_dir) if args.log_dir else None
    log_file = setup_logging(log_dir)
    logger.info("Session started (mode=%s)", "batch" if args.batch else ("task" if args.task else "repl"))

    # Determine workspace
    sandbox_dir = None
    if args.workspace:
        workspace = args.workspace
    else:
        sandbox_dir = tempfile.mkdtemp(prefix="codingagent_multi_sandbox_")
        workspace = sandbox_dir
        atexit.register(_cleanup_sandbox, sandbox_dir)
        print(f"[Sandbox] Created temporary workspace: {sandbox_dir}")

    config = Config(debug=not args.no_debug, temperature=args.temperature)

    orchestrator = OrchestratorAgent(
        workspace=workspace,
        config=config,
        max_worker_steps=args.max_worker_steps,
        max_orchestrator_rounds=args.max_rounds,
        max_result_chars=args.max_result_chars,
        context_max_tokens=args.context_max_tokens,
        enable_summarization=not args.no_summarize,
        enable_reflection=not args.no_reflect,
        debug=not args.no_debug,
    )
    if args.no_fc:
        orchestrator._use_function_calling = False

    print(f"[Log] Detailed interaction log: {log_file}")

    if args.batch:
        run_batch(
            orchestrator, workspace,
            Path(args.input), Path(args.output),
            args.start, args.limit,
        )
    elif args.task:
        response = orchestrator.run(args.task)
        print(response)
        if sandbox_dir:
            _print_sandbox_code(sandbox_dir)
    else:
        repl(orchestrator, sandbox_dir=sandbox_dir)

    logger.info("Session ended")
