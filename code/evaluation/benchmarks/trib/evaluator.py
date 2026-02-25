"""
TritonBench Evaluator Module

Evaluates agent performance on the TritonBench benchmark.

Each instance requires the agent to write a Triton GPU kernel. The evaluator:
1. Creates a temporary workspace for the agent
2. Builds a prompt from the task instruction
3. Runs the agent (which writes code using file/code_exec tools)
4. Extracts the generated code from workspace files or agent response
5. Checks call accuracy (code + test runs without error)
6. Checks execution accuracy (stdout matches reference)
"""

from typing import Dict, Any, List, Optional, Callable
import sys
import time
import re
import logging
import subprocess
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

from code.evaluation.benchmarks.trib.dataset import TritonBenchDataset, SEPARATOR
from code.evaluation.benchmarks.trib.metrics import TritonBenchMetrics

logger = logging.getLogger(__name__)

# Evaluation prompts directory
EVAL_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _load_prompt(path: Path, fallback: Optional[str] = None) -> Optional[str]:
    """Load a prompt file, returning fallback if missing or empty."""
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or fallback
    except FileNotFoundError:
        logger.warning("Prompt file not found: %s", path)
        return fallback


class TritonBenchEvaluator:
    """TritonBench evaluator

    Evaluates an agent's ability to write Triton GPU kernels.

    Attributes:
        dataset: TritonBench dataset
        metrics: Metrics calculator
        timeout_per_instance: Timeout in seconds for subprocess execution
    """

    def __init__(
        self,
        dataset: Optional[TritonBenchDataset] = None,
        timeout_per_instance: int = 120,
    ):
        """Initialize the TritonBench evaluator.

        Args:
            dataset: TritonBench dataset; if None, auto-creates with G channel
            timeout_per_instance: Timeout in seconds for running generated code
        """
        self.dataset = dataset if dataset is not None else TritonBenchDataset()
        self.metrics = TritonBenchMetrics()
        self.timeout_per_instance = timeout_per_instance

        # Load prompts
        self.system_prompt = _load_prompt(EVAL_PROMPTS_DIR / "trib_system.prompt")
        self.task_template_g = _load_prompt(EVAL_PROMPTS_DIR / "trib_task_G.prompt")
        self.task_template_t = _load_prompt(EVAL_PROMPTS_DIR / "trib_task_T.prompt")

    def evaluate(
        self,
        agent_factory: Callable[..., Any],
        max_samples: Optional[int] = None,
        **agent_kwargs,
    ) -> Dict[str, Any]:
        """Evaluate the agent on TritonBench tasks.

        Args:
            agent_factory: Factory function that creates an agent (accepts workspace kwarg)
            max_samples: Maximum number of samples to evaluate; None means all
            **agent_kwargs: Additional keyword arguments for agent_factory

        Returns:
            Evaluation results dictionary
        """
        channel = self.dataset.channel
        print(f"\n[TritonBench] Starting evaluation (channel={channel})...")

        # Load dataset
        dataset = self.dataset.load()
        if not dataset:
            print("   [Warning] Dataset is empty, skipping evaluation")
            return self._create_empty_results()

        # Limit sample count
        if max_samples:
            dataset = dataset[:max_samples]

        print(f"   Samples: {len(dataset)}")

        results: List[Dict[str, Any]] = []
        for i, sample in enumerate(dataset):
            task_id = sample.get("task_id", "")
            print(f"   Progress: {i + 1}/{len(dataset)} - {task_id}")

            try:
                sample_result = self.evaluate_sample(
                    agent_factory, sample, **agent_kwargs
                )
                results.append(sample_result)
            except Exception as e:
                print(f"   [Warning] Task {task_id} evaluation failed: {e}")
                results.append({
                    "task_id": task_id,
                    "channel": sample.get("channel", channel),
                    "difficulty": sample.get("difficulty"),
                    "call_pass": False,
                    "exec_pass": False,
                    "score": 0.0,
                    "generated_code": "",
                    "error_output": str(e),
                    "execution_time": 0.0,
                    "model": "unknown",
                    "steps_used": 0,
                    "max_steps": None,
                    "finish_reason": "error",
                    "trajectory_summary": [],
                    "tool_calls_summary": {},
                })

            # Print per-instance result
            last = results[-1]
            call_s = "CALL_PASS" if last.get("call_pass") else "CALL_FAIL"
            exec_s = "EXEC_PASS" if last.get("exec_pass") else "EXEC_FAIL"
            reason = last.get("finish_reason", "?")
            steps = last.get("steps_used", "?")
            t = last.get("execution_time", 0)
            print(f"     -> {call_s} | {exec_s} | {reason} | {steps} steps | {t:.1f}s")

        # Compute aggregate metrics
        overall_metrics = self.metrics.compute_metrics(results)

        # Determine model name
        model = next(
            (r.get("model") for r in results if r.get("model") and r["model"] != "unknown"),
            "unknown",
        )

        final_results = {
            "benchmark": "TritonBench",
            "channel": channel,
            "total_samples": len(results),
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "max_steps": results[0].get("max_steps") if results else None,
            "call_accuracy": overall_metrics["call_accuracy"],
            "execution_accuracy": overall_metrics["execution_accuracy"],
            "difficulty_breakdown": overall_metrics["difficulty_breakdown"],
            "average_execution_time": overall_metrics["average_execution_time"],
            "average_steps_used": overall_metrics["average_steps_used"],
            "finish_reason_counts": overall_metrics["finish_reason_counts"],
            "detailed_results": results,
        }

        print(f"[Done] TritonBench evaluation complete")
        print(f"   Call accuracy      : {overall_metrics['call_accuracy']:.2%}")
        print(f"   Execution accuracy : {overall_metrics['execution_accuracy']:.2%}")
        print(f"   Avg steps used     : {overall_metrics['average_steps_used']:.1f}")
        print(f"   Finish reasons     : {overall_metrics['finish_reason_counts']}")

        return final_results

    def evaluate_sample(
        self,
        agent_factory: Callable[..., Any],
        sample: Dict[str, Any],
        **agent_kwargs,
    ) -> Dict[str, Any]:
        """Evaluate a single TritonBench task.

        Args:
            agent_factory: Agent factory function
            sample: TritonBench task instance
            **agent_kwargs: Additional keyword arguments for agent_factory

        Returns:
            Evaluation result dict
        """
        task_id = sample.get("task_id", "")
        channel = sample.get("channel", "G")
        workspace = None

        try:
            # 1. Create temporary workspace
            workspace = Path(tempfile.mkdtemp(prefix=f"trib_{task_id}_"))

            # 2. Create agent with workspace
            factory_kwargs = dict(agent_kwargs)
            if self.system_prompt:
                factory_kwargs["system_prompt"] = self.system_prompt
            factory_kwargs["enable_reflection"] = False
            agent = agent_factory(workspace=str(workspace), **factory_kwargs)

            # 3. Build prompt
            prompt = self._build_prompt(sample)

            # 4. Run agent
            start_time = time.time()
            agent_answer = agent.run(prompt)
            execution_time = time.time() - start_time

            # 4b. Extract trajectory data
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

            # 5. Extract generated code
            generated_code = self._extract_code(workspace, agent_answer)

            # 6. Evaluate: call accuracy + execution accuracy
            test_code = sample.get("test_code", "")
            call_pass, call_error = self._check_call_accuracy(
                generated_code, test_code
            )

            exec_pass = False
            if call_pass:
                ref_file = self._get_reference_file_path(sample)
                exec_pass = self._check_execution_accuracy(
                    generated_code, test_code, ref_file
                )

            # Compute score
            if exec_pass:
                score = 1.0
            elif call_pass:
                score = 0.5
            else:
                score = 0.0

            return {
                "task_id": task_id,
                "channel": channel,
                "difficulty": sample.get("difficulty"),
                "call_pass": call_pass,
                "exec_pass": exec_pass,
                "score": score,
                "generated_code": generated_code,
                "error_output": call_error,
                "execution_time": execution_time,
                "model": model,
                "steps_used": steps_used,
                "max_steps": max_steps,
                "finish_reason": finish_reason,
                "trajectory_summary": traj_summary,
                "tool_calls_summary": tool_counts,
            }

        except Exception as e:
            return {
                "task_id": task_id,
                "channel": channel,
                "difficulty": sample.get("difficulty"),
                "call_pass": False,
                "exec_pass": False,
                "score": 0.0,
                "generated_code": "",
                "error_output": str(e),
                "execution_time": 0.0,
                "model": "unknown",
                "steps_used": 0,
                "max_steps": None,
                "finish_reason": "error",
                "trajectory_summary": [],
                "tool_calls_summary": {},
            }
        finally:
            if workspace:
                self._cleanup_workspace(workspace)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, sample: Dict[str, Any]) -> str:
        """Build the task prompt for the agent.

        Uses channel-specific templates if available.
        """
        channel = sample.get("channel", "G")

        if channel == "T" and self.task_template_t:
            return self.task_template_t.format(
                instruction=sample.get("instruction", ""),
                func_inputs=sample.get("func_inputs", ""),
                description=sample.get("description", ""),
                math=sample.get("math", ""),
                torch_code=sample.get("torch_code", ""),
                example=sample.get("example", ""),
            )

        if channel == "G" and self.task_template_g:
            return self.task_template_g.format(
                instruction=sample.get("instruction", ""),
            )

        # Fallback: inline prompt
        instruction = sample.get("instruction", "")
        return (
            f"## Task: Write a Triton GPU Kernel\n\n"
            f"{instruction}\n\n"
            f"## Requirements\n"
            f"- Write complete, runnable Triton code including all imports "
            f"(torch, triton, triton.language as tl).\n"
            f"- Include both the @triton.jit kernel function(s) AND the Python "
            f"wrapper function(s) that launch them.\n"
            f"- Save the code as `solution.py` using the file tool.\n"
            f"- Do NOT include any test code or main blocks.\n"
        )

    def _extract_code(self, workspace: Path, agent_answer: str) -> str:
        """Extract generated code from workspace files or agent response.

        Priority:
        1. Look for .py files in workspace (prefer solution.py)
        2. Fall back to code blocks in agent response text
        """
        # 1. Check for solution.py first
        solution_file = workspace / "solution.py"
        if solution_file.exists():
            code = solution_file.read_text(encoding="utf-8")
            return self._strip_test_code(code)

        # 2. Check for any .py files in workspace
        py_files = sorted(workspace.glob("*.py"))
        if py_files:
            # Use the largest .py file (most likely to be the solution)
            best = max(py_files, key=lambda f: f.stat().st_size)
            code = best.read_text(encoding="utf-8")
            return self._strip_test_code(code)

        # 3. Fall back to extracting code blocks from agent response
        if agent_answer:
            code_blocks = re.findall(
                r"```(?:python)?\s*\n(.*?)```",
                agent_answer,
                re.DOTALL,
            )
            if code_blocks:
                # Use the longest code block
                code = max(code_blocks, key=len)
                return self._strip_test_code(code)

        return ""

    def _strip_test_code(self, code: str) -> str:
        """Remove test code below the separator if the agent accidentally included it."""
        if SEPARATOR in code:
            return code.split(SEPARATOR)[0].strip()
        return code.strip()

    def _check_call_accuracy(
        self, generated_code: str, test_code: str
    ) -> tuple:
        """Check call accuracy: generated code + test code runs without error.

        Args:
            generated_code: The agent's generated Triton code
            test_code: Test code extracted from the reference file

        Returns:
            (call_pass, error_output) tuple
        """
        if not generated_code or not test_code:
            return False, "Empty generated code or test code"

        # Combine generated code with test code
        combined = generated_code + "\n" + SEPARATOR + "\n" + test_code

        # Write to temp file and run
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(combined)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_per_instance,
            )
            call_pass = result.returncode == 0
            error_output = result.stderr if not call_pass else ""
            return call_pass, error_output
        except subprocess.TimeoutExpired:
            return False, "Timeout expired"
        except Exception as e:
            return False, str(e)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _check_execution_accuracy(
        self,
        generated_code: str,
        test_code: str,
        reference_file: Optional[Path],
    ) -> bool:
        """Check execution accuracy: stdout of generated code matches reference.

        Only called when call_pass is True.

        Args:
            generated_code: The agent's generated Triton code
            test_code: Test code from reference file
            reference_file: Path to the full reference .py file

        Returns:
            True if stdout matches
        """
        if reference_file is None or not reference_file.exists():
            return False

        # Run reference file
        try:
            ref_result = subprocess.run(
                [sys.executable, str(reference_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout_per_instance,
            )
            if ref_result.returncode != 0:
                return False
            ref_stdout = ref_result.stdout
        except (subprocess.TimeoutExpired, Exception):
            return False

        # Run generated code + test code
        combined = generated_code + "\n" + SEPARATOR + "\n" + test_code
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(combined)
            temp_path = f.name

        try:
            gen_result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_per_instance,
            )
            gen_stdout = gen_result.stdout
            return gen_stdout == ref_stdout
        except (subprocess.TimeoutExpired, Exception):
            return False
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _get_reference_file_path(self, sample: Dict[str, Any]) -> Optional[Path]:
        """Get the full path to the reference .py file for a sample."""
        channel = sample.get("channel", "G")
        ref_filename = sample.get("reference_file", "")
        if not ref_filename:
            return None

        if channel == "G":
            ref_dir = self.dataset.data_dir / "TritonBench_G_v1"
        else:
            ref_dir = self.dataset.data_dir / "TritonBench_T_v1"

        ref_path = ref_dir / ref_filename
        return ref_path if ref_path.exists() else None

    def _cleanup_workspace(self, workspace: Path) -> None:
        """Clean up the temporary workspace directory."""
        try:
            shutil.rmtree(str(workspace), ignore_errors=True)
        except Exception:
            pass

    def _create_empty_results(self) -> Dict[str, Any]:
        """Create empty evaluation results."""
        return {
            "benchmark": "TritonBench",
            "channel": self.dataset.channel,
            "total_samples": 0,
            "call_accuracy": 0.0,
            "execution_accuracy": 0.0,
            "difficulty_breakdown": {},
            "average_execution_time": 0.0,
            "detailed_results": [],
        }
