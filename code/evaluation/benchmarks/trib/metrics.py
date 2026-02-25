"""
TritonBench Evaluation Metrics Module

Computes TritonBench evaluation metrics: call accuracy, execution accuracy,
and per-difficulty breakdowns.
"""

from typing import Dict, Any, List


class TritonBenchMetrics:
    """TritonBench evaluation metrics calculator

    Metrics:
    - Call accuracy: generated code + test code runs without error (returncode == 0)
    - Execution accuracy: generated code produces identical stdout as the reference
    - Per-difficulty breakdown of both metrics
    """

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate metrics from per-sample results.

        Args:
            results: List of per-sample evaluation result dicts

        Returns:
            Aggregate metrics dictionary
        """
        if not results:
            return self._empty_metrics()

        n = len(results)
        call_pass_count = sum(1 for r in results if r.get("call_pass", False))
        exec_pass_count = sum(1 for r in results if r.get("exec_pass", False))

        # Per-difficulty breakdown
        difficulty_groups: Dict[Any, List[Dict[str, Any]]] = {}
        for r in results:
            d = r.get("difficulty")
            difficulty_groups.setdefault(d, []).append(r)

        difficulty_breakdown = {}
        for d, group in sorted(difficulty_groups.items(), key=lambda x: (x[0] is None, x[0])):
            total = len(group)
            d_call = sum(1 for r in group if r.get("call_pass", False))
            d_exec = sum(1 for r in group if r.get("exec_pass", False))
            difficulty_breakdown[d] = {
                "total": total,
                "call_acc": d_call / total if total else 0.0,
                "exec_acc": d_exec / total if total else 0.0,
            }

        # Execution time statistics
        exec_times = [
            r.get("execution_time", 0.0) for r in results if "execution_time" in r
        ]
        avg_execution_time = sum(exec_times) / len(exec_times) if exec_times else 0.0

        # Finish reason counts
        finish_reason_counts: Dict[str, int] = {}
        for r in results:
            reason = r.get("finish_reason", "unknown")
            finish_reason_counts[reason] = finish_reason_counts.get(reason, 0) + 1

        # Average steps used
        steps_list = [
            r.get("steps_used", 0) for r in results if "steps_used" in r
        ]
        average_steps_used = sum(steps_list) / len(steps_list) if steps_list else 0.0

        return {
            "total_samples": n,
            "call_accuracy": call_pass_count / n,
            "execution_accuracy": exec_pass_count / n,
            "difficulty_breakdown": difficulty_breakdown,
            "average_execution_time": avg_execution_time,
            "finish_reason_counts": finish_reason_counts,
            "average_steps_used": average_steps_used,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics."""
        return {
            "total_samples": 0,
            "call_accuracy": 0.0,
            "execution_accuracy": 0.0,
            "difficulty_breakdown": {},
            "average_execution_time": 0.0,
            "finish_reason_counts": {},
            "average_steps_used": 0.0,
        }
