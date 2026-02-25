import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def _get_agent_factory(agent_type: str):
    """Import and return the build_agent function for the chosen agent type."""
    if agent_type == "funca":
        from funca_agent import build_agent
    else:
        from react_agent import build_agent
    return build_agent


def run_bfcl(args):
    """Run BFCL (Berkeley Function Calling Leaderboard) evaluation.

    Uses direct LLM invocation (single-shot) instead of the full ReAct agent
    loop, which is faster and avoids prompt conflicts for function-calling tasks.
    """
    from code.evaluation.benchmarks.bfcl.dataset import BFCLDataset
    from code.evaluation.benchmarks.bfcl.evaluator import BFCLEvaluator
    from code.core.llm import HelloAgentsLLM

    llm = HelloAgentsLLM(temperature=args.temperature)

    dataset = BFCLDataset(
        bfcl_data_dir=args.data_dir or "data/BFCL",
        category=args.category,
    )
    evaluator = BFCLEvaluator(
        dataset=dataset,
        evaluation_mode=args.eval_mode,
        llm=llm,
    )

    results = evaluator.evaluate(max_samples=args.max_samples)

    # Save results
    output_path = Path(args.output or f"results/bfcl_{args.category or 'all'}_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Export to BFCL official format
    if args.export:
        export_path = Path(f"results/bfcl_{args.category or 'all'}_bfcl_format.jsonl")
        evaluator.export_to_bfcl_format(results, export_path)


def run_gaia(args):
    """Run GAIA (General AI Assistants) evaluation.

    Uses direct LLM invocation by default, which supports file attachments
    (text, PDF, Excel, images).  Falls back to a ReAct agent when --agent
    mode is requested.
    """
    from code.evaluation.benchmarks.gaia.dataset import GAIADataset
    from code.evaluation.benchmarks.gaia.evaluator import GAIAEvaluator
    from code.core.llm import HelloAgentsLLM

    llm = HelloAgentsLLM(temperature=args.temperature)

    dataset = GAIADataset(
        level=args.level,
        local_data_dir=args.data_dir or "data/gaia",
    )
    evaluator = GAIAEvaluator(
        dataset=dataset,
        llm=llm,
        strict_mode=not args.lenient,
    )

    results = evaluator.evaluate(max_samples=args.max_samples)

    # Save results
    level_tag = f"level{args.level}" if args.level else "all"
    output_path = Path(args.output or f"results/gaia_{level_tag}_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Export to GAIA official format
    if args.export:
        export_path = Path(f"results/gaia_{level_tag}_gaia_format.jsonl")
        evaluator.export_to_gaia_format(results, export_path)


def run_swev(args):
    """Run SWE-bench Verified evaluation."""
    from code.evaluation.benchmarks.swe.dataset import SWEDataset
    from code.evaluation.benchmarks.swe.evaluator import SWEEvaluator

    build_agent = _get_agent_factory(args.agent_type)

    split = args.split or "test"
    dataset = SWEDataset(
        split=split,
        data_dir=args.data_dir or "data/SWEV",
        repo_filter=args.repo_filter,
    )
    evaluator = SWEEvaluator(
        dataset=dataset,
        timeout_per_instance=args.timeout_per_instance,
        run_tests=args.run_tests,
    )

    results = evaluator.evaluate(
        agent_factory=build_agent,
        max_samples=args.max_samples,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
    )

    # Save results
    repo_tag = args.repo_filter.replace("/", "_") if args.repo_filter else "all"
    output_path = Path(
        args.output or f"results/swev_{split}_{repo_tag}_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    if args.export:
        export_path = Path(f"results/swev_{split}_{repo_tag}_swe_format.jsonl")
        evaluator.export_to_swe_format(results, export_path)


def run_trib(args):
    """Run TritonBench evaluation (Triton GPU kernel generation)."""
    from code.evaluation.benchmarks.trib.dataset import TritonBenchDataset
    from code.evaluation.benchmarks.trib.evaluator import TritonBenchEvaluator

    build_agent = _get_agent_factory(args.agent_type)

    channel = args.channel or "G"
    dataset = TritonBenchDataset(
        channel=channel,
        data_dir=args.data_dir or "data/TRIB",
        instruction_mode=args.instruction_mode or "simple",
        difficulty=args.difficulty,
    )
    evaluator = TritonBenchEvaluator(
        dataset=dataset,
        timeout_per_instance=args.timeout_per_instance,
    )

    results = evaluator.evaluate(
        agent_factory=build_agent,
        max_samples=args.max_samples,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
    )

    # Save results
    mode_tag = args.instruction_mode or "simple"
    diff_tag = f"d{args.difficulty}" if args.difficulty else "all"
    output_path = Path(
        args.output or f"results/trib_{channel}_{mode_tag}_{diff_tag}_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


def run_data_gen(args):
    """Run Data Generation quality evaluation (LLM Judge / Win Rate)."""
    from code.evaluation.benchmarks.data_generation.dataset import AIDataset
    from code.evaluation.benchmarks.data_generation.llm_judge import LLMJudgeEvaluator
    from code.core.llm import HelloAgentsLLM

    if args.data_path:
        dataset = AIDataset(dataset_type="generated", data_path=args.data_path)
    elif args.year:
        dataset = AIDataset(dataset_type="real", year=args.year)
    else:
        print("Error: --data-path or --year is required for data_gen benchmark")
        sys.exit(1)

    problems = dataset.load()
    if not problems:
        print("No problems loaded. Exiting.")
        sys.exit(1)

    # Limit samples
    if args.max_samples:
        problems = problems[:args.max_samples]

    judge_llm = HelloAgentsLLM(model=args.judge_model) if args.judge_model else None
    evaluator = LLMJudgeEvaluator(llm=judge_llm)

    print(f"\nEvaluating {len(problems)} problems with LLM Judge...")
    results = []
    for i, problem in enumerate(problems):
        if i % 5 == 0:
            print(f"  Progress: {i+1}/{len(problems)}")
        result = evaluator.evaluate_single(problem)
        results.append(result)

    # Save results
    output_path = Path(args.output or "results/data_gen_judge_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run agent evaluation benchmarks (BFCL, GAIA, SWE-bench Verified, Data Generation, TritonBench)"
    )
    parser.add_argument(
        "--benchmark", "-b",
        choices=["bfcl", "gaia", "swev", "data_gen", "trib"],
        required=True,
        help="Benchmark to run",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int, default=None,
        help="Max samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Output file path for results JSON",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Also export results in official benchmark format",
    )

    # Agent configuration
    parser.add_argument("--workspace", "-w", type=str, default=".")
    parser.add_argument("--max-iterations", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument(
        "--agent-type",
        type=str,
        default="react",
        choices=["react", "funca"],
        help="Agent type: react (ReActAgent) or funca (FunctionCallAgent)",
    )

    # BFCL-specific
    parser.add_argument(
        "--category", "-c",
        type=str, default=None,
        help="[BFCL] Test category (e.g., simple_python, multiple, parallel)",
    )
    parser.add_argument(
        "--eval-mode",
        type=str, default="ast",
        choices=["ast", "execution"],
        help="[BFCL] Evaluation mode (default: ast)",
    )
    parser.add_argument(
        "--data-dir",
        type=str, default=None,
        help="[BFCL/GAIA/SWEV/TRIB] Data directory path",
    )

    # GAIA-specific
    parser.add_argument(
        "--level", "-l",
        type=int, default=None,
        choices=[1, 2, 3],
        help="[GAIA] Difficulty level (1-3, default: all)",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="[GAIA] Use lenient matching instead of strict",
    )

    # Data generation-specific
    parser.add_argument(
        "--data-path",
        type=str, default=None,
        help="[data_gen] Path to generated problems JSON file",
    )
    parser.add_argument(
        "--year",
        type=int, default=None,
        help="[data_gen] AIME year for real problems (e.g., 2025)",
    )
    parser.add_argument(
        "--judge-model",
        type=str, default=None,
        help="[data_gen] Model name for LLM Judge (default: from .env)",
    )

    # SWEV-specific
    parser.add_argument(
        "--split",
        type=str, default=None,
        choices=["dev", "test", "train"],
        help="[SWEV] Dataset split (default: test)",
    )
    parser.add_argument(
        "--repo-filter",
        type=str, default=None,
        help="[SWEV] Only evaluate instances from this repo (e.g., django/django)",
    )
    parser.add_argument(
        "--timeout-per-instance",
        type=int, default=600,
        help="[SWEV] Timeout in seconds per instance (default: 600)",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="[SWEV] Run FAIL_TO_PASS tests for verification (requires repo deps)",
    )

    # TritonBench-specific
    parser.add_argument(
        "--channel",
        type=str, default=None,
        choices=["G", "T"],
        help="[TRIB] Evaluation channel: G (GitHub kernels) or T (PyTorch-to-Triton) (default: G)",
    )
    parser.add_argument(
        "--instruction-mode",
        type=str, default=None,
        choices=["simple", "complex"],
        help="[TRIB] Instruction mode for G channel (default: simple)",
    )
    parser.add_argument(
        "--difficulty",
        type=int, default=None,
        choices=[1, 2, 3, 4, 5],
        help="[TRIB] Filter by difficulty level 1-5 (default: all)",
    )

    args = parser.parse_args()

    dispatch = {
        "bfcl": run_bfcl,
        "gaia": run_gaia,
        "swev": run_swev,
        "data_gen": run_data_gen,
        "trib": run_trib,
    }
    dispatch[args.benchmark](args)


if __name__ == "__main__":
    main()
