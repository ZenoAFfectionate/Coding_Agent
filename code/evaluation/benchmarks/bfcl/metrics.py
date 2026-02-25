"""
BFCL Metrics Module

Calculates BFCL-related evaluation metrics.
"""

from typing import Dict, Any, List, Optional
import json
import ast
import numpy as np


class BFCLMetrics:
    """BFCL Evaluation Metrics Calculator

    Calculates tool-calling related evaluation metrics:
    - Accuracy: Proportion of completely correct results
    - AST Match: Abstract Syntax Tree match score
    - Parameter Accuracy: Proportion of correct parameters
    - F1 Score: Harmonic mean of precision and recall
    - Execution Success Rate: Success rate of executable function calls
    """

    @staticmethod
    def calculate_accuracy(predictions: List[Any], references: List[Any]) -> float:
        """Calculate accuracy.

        Args:
            predictions: List of predictions.
            references: List of reference answers.

        Returns:
            Accuracy (0-1).
        """
        if not predictions or not references:
            return 0.0

        min_len = min(len(predictions), len(references))
        correct = sum(1 for p, r in zip(predictions[:min_len], references[:min_len]) if p == r)
        return correct / min_len

    @staticmethod
    def calculate_ast_match(predicted: str, expected: str) -> float:
        """Calculate AST match score.

        Args:
            predicted: Predicted function call.
            expected: Expected function call.

        Returns:
            Match score (0-1).
        """
        try:
            # Try to parse as AST
            pred_ast = ast.parse(predicted, mode='eval')
            exp_ast = ast.parse(expected, mode='eval')

            # Compare AST structure
            pred_dump = ast.dump(pred_ast)
            exp_dump = ast.dump(exp_ast)

            if pred_dump == exp_dump:
                return 1.0

            # Calculate structural similarity
            similarity = BFCLMetrics._calculate_string_similarity(pred_dump, exp_dump)
            return similarity

        except SyntaxError:
            # If parsing fails, use string similarity
            return BFCLMetrics._calculate_string_similarity(predicted, expected)

    @staticmethod
    def _calculate_string_similarity(s1: str, s2: str) -> float:
        """Calculate string similarity (simplified Levenshtein distance)."""
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Use set intersection to calculate similarity
        set1 = set(s1.split())
        set2 = set(s2.split())

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def calculate_parameter_accuracy(
        predicted_params: Dict[str, Any],
        expected_params: Dict[str, Any]
    ) -> float:
        """Calculate parameter accuracy.

        Args:
            predicted_params: Predicted parameters.
            expected_params: Expected parameters.

        Returns:
            Parameter accuracy (0-1).
        """
        if not expected_params:
            return 1.0 if not predicted_params else 0.0

        if not predicted_params:
            return 0.0

        correct = 0
        for key, expected_value in expected_params.items():
            if key in predicted_params:
                predicted_value = predicted_params[key]
                if BFCLMetrics._values_match(predicted_value, expected_value):
                    correct += 1

        return correct / len(expected_params)

    @staticmethod
    def _values_match(v1: Any, v2: Any) -> bool:
        """Compare whether two values match."""
        # Handle numeric types
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            return abs(v1 - v2) < 1e-6

        # Handle string types
        if isinstance(v1, str) and isinstance(v2, str):
            return v1.strip().lower() == v2.strip().lower()

        # Handle list types
        if isinstance(v1, list) and isinstance(v2, list):
            if len(v1) != len(v2):
                return False
            return all(BFCLMetrics._values_match(a, b) for a, b in zip(v1, v2))

        # Handle dict types
        if isinstance(v1, dict) and isinstance(v2, dict):
            if set(v1.keys()) != set(v2.keys()):
                return False
            return all(BFCLMetrics._values_match(v1[k], v2[k]) for k in v1.keys())

        # Default to equality comparison
        return v1 == v2

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comprehensive metrics.

        Args:
            results: List of evaluation results.

        Returns:
            Metrics dictionary containing various evaluation metrics.
        """
        if not results:
            return self._empty_metrics()

        total = len(results)

        # Basic metrics
        success_count = sum(1 for r in results if r.get("success", False))
        accuracy = success_count / total

        # Score statistics
        scores = [r.get("score", 0.0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Execution time statistics
        execution_times = [r.get("execution_time", 0.0) for r in results if "execution_time" in r]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0

        # Per-category statistics
        category_metrics = self._compute_category_metrics(results)

        # Function call statistics
        function_call_stats = self._compute_function_call_stats(results)

        return {
            "total_samples": total,
            "success_count": success_count,
            "accuracy": accuracy,
            "average_score": avg_score,
            "average_execution_time": avg_execution_time,
            "category_metrics": category_metrics,
            "function_call_stats": function_call_stats,
            "score_distribution": self._compute_score_distribution(scores)
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics."""
        return {
            "total_samples": 0,
            "success_count": 0,
            "accuracy": 0.0,
            "average_score": 0.0,
            "average_execution_time": 0.0,
            "category_metrics": {},
            "function_call_stats": {},
            "score_distribution": {}
        }

    def _compute_category_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Compute per-category metrics."""
        categories = {}

        for result in results:
            category = result.get("category", "unknown")
            if category not in categories:
                categories[category] = {
                    "total": 0,
                    "success": 0,
                    "scores": []
                }

            categories[category]["total"] += 1
            if result.get("success", False):
                categories[category]["success"] += 1
            categories[category]["scores"].append(result.get("score", 0.0))

        # Calculate statistics for each category
        category_metrics = {}
        for category, stats in categories.items():
            accuracy = stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
            avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0

            category_metrics[category] = {
                "total": stats["total"],
                "success": stats["success"],
                "accuracy": accuracy,
                "average_score": avg_score
            }

        return category_metrics

    def _compute_function_call_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute function call statistics."""
        total_calls = 0
        successful_calls = 0
        function_names = set()

        for result in results:
            predicted = result.get("predicted", [])
            if isinstance(predicted, list):
                total_calls += len(predicted)
                for call in predicted:
                    if isinstance(call, dict) and "name" in call:
                        function_names.add(call["name"])
                        if result.get("success", False):
                            successful_calls += 1

        return {
            "total_function_calls": total_calls,
            "successful_calls": successful_calls,
            "unique_functions": len(function_names),
            "function_names": sorted(list(function_names)),
            "avg_calls_per_sample": total_calls / len(results) if results else 0.0
        }

    def _compute_score_distribution(self, scores: List[float]) -> Dict[str, Any]:
        """Compute score distribution."""
        if not scores:
            return {}

        return {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
            "median": sorted(scores)[len(scores) // 2],
            "std": np.std(scores) if len(scores) > 1 else 0.0,
            "quartiles": {
                "q1": sorted(scores)[len(scores) // 4],
                "q2": sorted(scores)[len(scores) // 2],
                "q3": sorted(scores)[3 * len(scores) // 4]
            }
        }

    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """Calculate F1 score.

        Args:
            precision: Precision.
            recall: Recall.

        Returns:
            F1 score.
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def calculate_precision_recall(
        predicted: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> tuple[float, float]:
        """Calculate precision and recall.

        Args:
            predicted: List of predicted function calls.
            expected: List of expected function calls.

        Returns:
            (precision, recall) tuple.
        """
        if not expected:
            return 1.0 if not predicted else 0.0, 1.0

        if not predicted:
            return 0.0, 0.0

        # Simplified version: based on function name matching
        pred_names = set(call.get("name", "") for call in predicted if isinstance(call, dict))
        exp_names = set(call.get("name", "") for call in expected if isinstance(call, dict))

        true_positives = len(pred_names & exp_names)

        precision = true_positives / len(pred_names) if pred_names else 0.0
        recall = true_positives / len(exp_names) if exp_names else 0.0

        return precision, recall
