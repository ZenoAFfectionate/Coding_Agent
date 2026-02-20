import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def run_bfcl(args):
    """Run BFCL (Berkeley Function Calling Leaderboard) evaluation."""
    from code.evaluation.benchmarks.bfcl.dataset import BFCLDataset
    from code.evaluation.benchmarks.bfcl.evaluator import BFCLEvaluator
    from inference import build_agent

    agent = build_agent(
        workspace=args.workspace,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
    )

    dataset = BFCLDataset(
        bfcl_data_dir=args.data_dir or "data/BFCL",
        category=args.category,
    )
    evaluator = BFCLEvaluator(
        dataset=dataset,
        evaluation_mode=args.eval_mode,
    )

    results = evaluator.evaluate(agent, max_samples=args.max_samples)

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
    """Run GAIA (General AI Assistants) evaluation."""
    from code.evaluation.benchmarks.gaia.dataset import GAIADataset
    from code.evaluation.benchmarks.gaia.evaluator import GAIAEvaluator
    from inference import build_agent

    agent = build_agent(
        workspace=args.workspace,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
    )

    dataset = GAIADataset(
        level=args.level,
        local_data_dir=args.data_dir or "data/gaia",
    )
    evaluator = GAIAEvaluator(
        dataset=dataset,
        strict_mode=not args.lenient,
    )

    results = evaluator.evaluate(agent, max_samples=args.max_samples)

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
        description="Run agent evaluation benchmarks (BFCL, GAIA, Data Generation)"
    )
    parser.add_argument(
        "--benchmark", "-b",
        choices=["bfcl", "gaia", "data_gen"],
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
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.2)

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
        help="[BFCL/GAIA] Data directory path",
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

    args = parser.parse_args()

    dispatch = {
        "bfcl": run_bfcl,
        "gaia": run_gaia,
        "data_gen": run_data_gen,
    }
    dispatch[args.benchmark](args)


if __name__ == "__main__":
    main()
