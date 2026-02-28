"""
SWE-bench Evaluator Module

Evaluates agent performance on the SWE-bench benchmark.

Each instance represents a real GitHub issue. The evaluator will:
1. Clone the corresponding repository and checkout base_commit
2. Build the prompt (issue + hints)
3. Run the agent
4. Collect the agent's patch via git diff
5. Compare the predicted patch against the gold patch (optionally: run tests)
"""

from typing import Dict, Any, List, Optional, Union, Callable
import sys
import time
import json
import logging
import subprocess
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

from code.evaluation.benchmarks.swe.dataset import SWEDataset
from code.evaluation.benchmarks.swe.metrics import SWEMetrics

logger = logging.getLogger(__name__)

# Evaluation prompts directory (benchmark-specific prompts)
EVAL_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _load_prompt(path: Path, fallback: str | None = None) -> str | None:
    """Load a prompt file, returning *fallback* if the file is missing or empty."""
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or fallback
    except FileNotFoundError:
        logger.warning("Prompt file not found: %s", path)
        return fallback


class SWEEvaluator:
    """SWE-bench evaluator

    Evaluates an agent's ability to fix real GitHub issues.

    Attributes:
        dataset: SWE-bench dataset
        metrics: Metrics calculator
        workspace_base: Parent directory for temporary workspaces
        timeout_per_instance: Timeout in seconds per instance
        run_tests: Whether to run FAIL_TO_PASS tests
    """

    def __init__(
        self,
        dataset: Optional[SWEDataset] = None,
        workspace_base: Optional[str] = None,
        timeout_per_instance: int = 600,
        run_tests: bool = False,
    ):
        """Initialize the SWE-bench evaluator.

        Args:
            dataset: SWE-bench dataset; if None, auto-creates with dev split
            workspace_base: Base directory for cloned repos; defaults to system temp dir
            timeout_per_instance: Timeout in seconds per instance
            run_tests: Whether to execute FAIL_TO_PASS test verification
        """
        self.dataset = dataset if dataset is not None else SWEDataset()
        self.metrics = SWEMetrics()
        self.workspace_base = workspace_base
        self.timeout_per_instance = timeout_per_instance
        self.run_tests = run_tests

        # Load benchmark-specific prompts
        self.system_prompt = _load_prompt(EVAL_PROMPTS_DIR / "swev_system.prompt")
        self.task_template = _load_prompt(EVAL_PROMPTS_DIR / "swev_task.prompt")

    def evaluate(
        self,
        agent_factory: Callable[..., Any],
        max_samples: Optional[int] = None,
        **agent_kwargs,
    ) -> Dict[str, Any]:
        """Evaluate the agent.

        Since each SWE-bench instance requires a different workspace (a cloned repo),
        an agent_factory (e.g. build_agent) is passed in, and the evaluator creates
        a new agent for each instance.

        Args:
            agent_factory: Factory function that accepts a workspace keyword argument
                           and returns an agent
            max_samples: Maximum number of samples to evaluate; None means all
            **agent_kwargs: Additional keyword arguments passed to agent_factory

        Returns:
            Evaluation results dictionary
        """
        print(f"\n[SWE-bench] Starting evaluation...")

        # Load dataset
        dataset = self.dataset.load()
        if not dataset:
            print("   [Warning] Dataset is empty, skipping evaluation")
            return self._create_empty_results()

        # Limit sample count
        if max_samples:
            dataset = dataset[:max_samples]

        print(f"   Samples   : {len(dataset)}")
        print(f"   Run tests : {'yes' if self.run_tests else 'no'}")

        results: List[Dict[str, Any]] = []
        for i, sample in enumerate(dataset):
            instance_id = sample.get("instance_id", "")
            print(
                f"   Progress: {i + 1}/{len(dataset)} - {instance_id}"
            )

            try:
                sample_result = self.evaluate_sample(
                    agent_factory, sample, **agent_kwargs
                )
                results.append(sample_result)
            except Exception as e:
                print(f"   [Warning] Instance {sample.get('instance_id')} evaluation failed: {e}")
                results.append(
                    {
                        "instance_id": sample.get("instance_id", ""),
                        "repo": sample.get("repo", ""),
                        "exact_match": False,
                        "tests_passed": False,
                        "patch_metrics": {},
                        "predicted_patch": "",
                        "error": str(e),
                        "score": 0.0,
                        "model": "unknown",
                        "steps_used": 0,
                        "max_steps": None,
                        "finish_reason": "error",
                        "agent_answer": "",
                        "gold_patch": sample.get("patch", ""),
                        "trajectory_summary": [],
                        "tool_calls_summary": {},
                    }
                )

            # Print per-instance result line
            last = results[-1]
            status = "PASS" if last.get("score", 0) > 0 else "FAIL"
            reason = last.get("finish_reason", "?")
            steps = last.get("steps_used", "?")
            t = last.get("execution_time", 0)
            overlap = last.get("patch_metrics", {}).get("line_overlap", 0)
            print(f"     -> {status} | {reason} | {steps} steps | {t:.1f}s | overlap={overlap:.0%}")
            print()  # blank line between instances

        # Compute aggregate metrics
        overall_metrics = self.metrics.compute_metrics(results)

        # Determine model name from first result
        model = next(
            (r.get("model") for r in results if r.get("model") and r["model"] != "unknown"),
            "unknown",
        )

        final_results = {
            "benchmark": "SWE-bench",
            "total_samples": len(results),
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "max_steps": results[0].get("max_steps") if results else None,
            "resolved_rate": overall_metrics["resolved_rate"],
            "exact_match_rate": overall_metrics["exact_match_rate"],
            "average_line_overlap": overall_metrics["average_line_overlap"],
            "average_execution_time": overall_metrics["average_execution_time"],
            "average_steps_used": overall_metrics["average_steps_used"],
            "finish_reason_counts": overall_metrics["finish_reason_counts"],
            "files_matched_count": overall_metrics["files_matched_count"],
            "repo_metrics": overall_metrics["repo_metrics"],
            "detailed_results": results,
        }

        print(f"[Done] SWE-bench evaluation complete")
        print(f"   Resolved rate       : {overall_metrics['resolved_rate']:.2%}")
        print(f"   Exact match rate    : {overall_metrics['exact_match_rate']:.2%}")
        print(f"   Avg line overlap    : {overall_metrics['average_line_overlap']:.2%}")
        print(f"   Avg steps used      : {overall_metrics['average_steps_used']:.1f} / {final_results.get('max_steps', '?')}")
        print(f"   Finish reasons      : {overall_metrics['finish_reason_counts']}")

        return final_results

    def evaluate_sample(
        self,
        agent_factory: Callable[..., Any],
        sample: Dict[str, Any],
        **agent_kwargs,
    ) -> Dict[str, Any]:
        """Evaluate a single instance.

        Args:
            agent_factory: Agent factory function
            sample: SWE-bench instance
            **agent_kwargs: Additional keyword arguments passed to agent_factory

        Returns:
            Evaluation result for this instance
        """
        instance_id = sample.get("instance_id", "")
        workspace = None

        try:
            # 1. Clone repo and checkout base_commit
            workspace = self._setup_repo(sample)

            # 2. Create agent (workspace points to the cloned repo)
            factory_kwargs = dict(agent_kwargs)
            if self.system_prompt:
                factory_kwargs["system_prompt"] = self.system_prompt
            # Exclude execution tools for SWE-bench: repo dependencies are not
            # installed, so both code_exec and test_runner will always raise
            # ImportErrors (e.g. "No module named 'erfa'") and waste steps.
            # The evaluator handles test verification separately via _run_tests().
            for tool in ("code_exec", "test_runner"):
                factory_kwargs.setdefault("exclude_tools", []).append(tool)
            # Use lenient text-only policy for SWE-bench: the model is free to
            # return text-only responses at any point (states A/B pass through).
            # Only nudge when confirmed edits exist but the model returned an
            # empty response (state C), because that likely means the model
            # finished thinking without calling `finish` explicitly.
            factory_kwargs.setdefault("text_only_policy", "lenient")
            agent = agent_factory(workspace=str(workspace), **factory_kwargs)

            # 3. Build prompt
            prompt = self._build_prompt(sample)

            # 4. Run agent
            start_time = time.time()
            agent_answer = agent.run(prompt)
            execution_time = time.time() - start_time

            # 4b. Extract trajectory data before cleanup
            model = getattr(getattr(agent, "llm", None), "model", "unknown")
            max_steps = getattr(agent, "max_steps", None)
            trajectory = getattr(agent, "trajectory", None)

            finish_reason = "unknown"
            steps_used = 0
            traj_summary: List[Dict[str, Any]] = []
            tool_counts: Dict[str, int] = {}

            if trajectory and trajectory.steps:
                stats = trajectory.get_stats()
                steps_used = stats.get("total_steps", len(trajectory.steps))

                # Derive finish_reason
                last_step = trajectory.steps[-1]
                if last_step.step_type == "final_answer":
                    finish_actions = [
                        s for s in trajectory.steps
                        if s.step_type == "action"
                        and s.metadata.get("tool") == "finish"
                    ]
                    finish_reason = "finish_tool" if finish_actions else "text_only"
                elif last_step.step_type == "error" and "Max steps" in (last_step.content or ""):
                    finish_reason = "max_steps_reached"
                elif steps_used >= (max_steps or float("inf")):
                    finish_reason = "max_steps_reached"
                else:
                    finish_reason = "text_only"

                # Build trajectory_summary and tool_calls_summary
                for step in trajectory.steps:
                    entry: Dict[str, Any] = {
                        "step": step.step_number,
                        "type": step.step_type,
                    }
                    if step.duration_ms is not None:
                        entry["duration_ms"] = step.duration_ms
                    tool = step.metadata.get("tool") if step.metadata else None
                    if tool:
                        entry["tool"] = tool
                        tool_counts[tool] = tool_counts.get(tool, 0) + 1
                    traj_summary.append(entry)

            # 5. Collect the agent's patch
            predicted_patch = self._collect_patch(workspace)

            # 6. Compute patch metrics
            gold_patch = sample.get("patch", "")
            patch_metrics = self.metrics.calculate_patch_metrics(
                predicted_patch, gold_patch
            )

            # 7. Optional: run tests
            tests_passed = False
            test_output = ""
            if self.run_tests and sample.get("FAIL_TO_PASS"):
                tests_passed, test_output = self._run_tests(
                    workspace, sample["FAIL_TO_PASS"]
                )

            score = 1.0 if patch_metrics["exact_match"] or tests_passed else 0.0

            return {
                "instance_id": instance_id,
                "repo": sample.get("repo", ""),
                "exact_match": patch_metrics["exact_match"],
                "tests_passed": tests_passed,
                "patch_metrics": patch_metrics,
                "predicted_patch": predicted_patch,
                "score": score,
                "execution_time": execution_time,
                "test_output": test_output,
                "model": model,
                "steps_used": steps_used,
                "max_steps": max_steps,
                "finish_reason": finish_reason,
                "agent_answer": agent_answer,
                "gold_patch": gold_patch,
                "trajectory_summary": traj_summary,
                "tool_calls_summary": tool_counts,
            }

        except Exception as e:
            return {
                "instance_id": instance_id,
                "repo": sample.get("repo", ""),
                "exact_match": False,
                "tests_passed": False,
                "patch_metrics": {},
                "predicted_patch": "",
                "score": 0.0,
                "error": str(e),
                "model": "unknown",
                "steps_used": 0,
                "max_steps": None,
                "finish_reason": "error",
                "agent_answer": "",
                "gold_patch": sample.get("patch", ""),
                "trajectory_summary": [],
                "tool_calls_summary": {},
            }
        finally:
            if workspace:
                self._cleanup_workspace(workspace)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_repo(self, sample: Dict[str, Any]) -> Path:
        """Clone the repository and checkout base_commit.

        Args:
            sample: SWE-bench instance

        Returns:
            Path to the cloned repository
        """
        repo = sample["repo"]
        base_commit = sample["base_commit"]

        workspace = Path(
            tempfile.mkdtemp(
                prefix=f"swe_{sample.get('instance_id', 'unknown')}_",
                dir=self.workspace_base,
            )
        )

        repo_url = f"git@github.com:{repo}.git"

        # Clone (no checkout) + checkout target commit
        subprocess.run(
            ["git", "clone", "--no-checkout", repo_url, str(workspace)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        subprocess.run(
            ["git", "checkout", base_commit],
            check=True,
            capture_output=True,
            cwd=str(workspace),
            timeout=60,
        )

        # Install repo dependencies only when tests will be executed;
        # otherwise skip to save time (up to 300s per instance).
        if self.run_tests:
            for install_cmd in [
                # 1. Try installing just the test extras (fastest if it works)
                [sys.executable, "-m", "pip", "install", "-e", ".[test]",
                 "--quiet", "--no-build-isolation"],
                # 2. Fallback: install the repo itself (handles C extensions etc.)
                [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
            ]:
                try:
                    result = subprocess.run(
                        install_cmd,
                        cwd=str(workspace),
                        capture_output=True,
                        timeout=300,
                    )
                    if result.returncode == 0:
                        break
                except Exception:
                    continue

        return workspace

    def _collect_patch(self, workspace: Path) -> str:
        """Collect all changes made by the agent in the workspace.

        Args:
            workspace: Repository working directory

        Returns:
            Unified diff string
        """
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(workspace),
            timeout=30,
        )
        return result.stdout

    def _build_prompt(self, sample: Dict[str, Any]) -> str:
        """Build the prompt to send to the agent.

        Uses the swev_task.prompt template if available, otherwise falls back
        to a hardcoded default.

        Args:
            sample: SWE-bench instance

        Returns:
            Prompt string
        """
        problem = sample.get("problem_statement", "")
        hints = sample.get("hints_text", "")
        repo = sample.get("repo", "")
        instance_id = sample.get("instance_id", "")

        if self.task_template:
            hints_section = f"\n## Hints\n\n{hints}\n" if hints else ""
            return self.task_template.format(
                repo=repo,
                instance_id=instance_id,
                problem_statement=problem,
                hints_section=hints_section,
            )

        # Fallback: hardcoded prompt (kept for backwards compatibility)
        prompt = (
            f"You are working on the repository: {repo}\n"
            f"Instance ID: {instance_id}\n\n"
            f"## GitHub Issue\n\n{problem}\n"
        )

        if hints:
            prompt += f"\n## Hints\n\n{hints}\n"

        prompt += (
            "\n## Instructions\n\n"
            "Please investigate this issue in the codebase and produce a fix. "
            "Explore the relevant source files, understand the root cause, and "
            "make the necessary code changes to resolve the issue. "
            "Do NOT run tests or create new test files — only modify source code."
        )

        return prompt

    def _run_tests(
        self, workspace: Path, fail_to_pass: List[str]
    ) -> tuple:
        """Run the FAIL_TO_PASS test list.

        Args:
            workspace: Repository working directory
            fail_to_pass: List of tests that should change from FAIL to PASS

        Returns:
            (all_passed, test_output)
        """
        if not fail_to_pass:
            return False, ""

        # Ensure common test dependencies are available
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "hypothesis", "--quiet"],
                capture_output=True, timeout=60,
            )
        except Exception:
            pass

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest"] + fail_to_pass + ["-x", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=str(workspace),
                timeout=self.timeout_per_instance,
            )
            all_passed = result.returncode == 0
            return all_passed, result.stdout + result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return False, str(e)

    def _cleanup_workspace(self, workspace: Path) -> None:
        """Clean up the temporary workspace directory."""
        try:
            shutil.rmtree(str(workspace), ignore_errors=True)
        except Exception:
            pass

    def _create_empty_results(self) -> Dict[str, Any]:
        """Create empty evaluation results."""
        return {
            "benchmark": "SWE-bench",
            "total_samples": 0,
            "resolved_rate": 0.0,
            "exact_match_rate": 0.0,
            "average_line_overlap": 0.0,
            "average_execution_time": 0.0,
            "repo_metrics": {},
            "detailed_results": [],
        }

    def export_to_swe_format(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> None:
        """Export results in SWE-bench official submission format.

        JSONL format with each line containing instance_id and model_patch.

        Args:
            results: Results dictionary returned by evaluate()
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        detailed = results.get("detailed_results", [])

        with open(output_path, "w", encoding="utf-8") as f:
            for r in detailed:
                entry = {
                    "instance_id": r.get("instance_id", ""),
                    "model_patch": r.get("predicted_patch", ""),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[Done] SWE-bench format results exported")
        print(f"   Output file : {output_path}")
        print(f"   Samples     : {len(detailed)}")
