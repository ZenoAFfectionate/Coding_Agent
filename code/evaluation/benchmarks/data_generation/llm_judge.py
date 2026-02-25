"""
LLM Judge Evaluator

Uses an LLM as a judge to evaluate data generation quality.
"""

import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
from code.core.llm import HelloAgentsLLM


class LLMJudgeEvaluator:
    """LLM Judge evaluator"""

    # Evaluation dimensions
    EVALUATION_DIMENSIONS = [
        "correctness",      # Mathematical correctness
        "clarity",          # Clarity of expression
        "difficulty_match", # Difficulty alignment
        "completeness"      # Completeness of solution
    ]

    def __init__(
        self,
        llm: Optional[HelloAgentsLLM] = None,
        judge_model: str = "gpt-4o"
    ):
        """
        Initialize the LLM Judge evaluator.

        Args:
            llm: LLM instance; if None, a new instance will be created
            judge_model: Name of the judge model
        """
        self.llm = llm or HelloAgentsLLM(model=judge_model)
        self.judge_model = judge_model

    def evaluate_single(
        self,
        problem: Dict[str, Any],
        reference: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single problem.

        Args:
            problem: The problem to evaluate
            reference: Reference problem (optional, for comparison)

        Returns:
            Evaluation result containing dimension scores and total score
        """
        start_time = time.time()

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(problem, reference)

        # Call LLM for evaluation
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages)

        # Parse evaluation result
        scores = self._parse_evaluation_response(response)

        # Compute total score
        total_score = sum(scores.values()) / len(scores)

        execution_time = time.time() - start_time

        return {
            "problem_id": problem.get("problem_id", "unknown"),
            "scores": scores,
            "total_score": total_score,
            "evaluation_text": response,
            "execution_time": execution_time
        }

    def evaluate_batch(
        self,
        problems: List[Dict[str, Any]],
        references: Optional[List[Dict[str, Any]]] = None,
        total_samples: int = 500
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of problems.

        Args:
            problems: List of problems to evaluate
            references: List of reference problems (optional)
            total_samples: Total number of samples for progress bar display (default: 500)

        Returns:
            Aggregated evaluation results
        """
        num_problems = len(problems)
        print(f"\n[LLM Judge] Starting evaluation")
        print(f"   Judge model : {self.judge_model}")
        print(f"   Num samples : {num_problems}")
        print(f"   Dimensions  : {', '.join(self.EVALUATION_DIMENSIONS)}")

        results = []
        pbar = tqdm(total=min(num_problems, total_samples),
                     desc="Evaluating", unit="sample",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        for idx, problem in enumerate(problems):
            if idx >= total_samples:
                break

            reference = references[idx] if references and idx < len(references) else None
            result = self.evaluate_single(problem, reference)
            results.append(result)

            pbar.set_postfix(
                score=f"{result['total_score']:.2f}",
                id=problem.get('problem_id', 'unknown')[:15]
            )
            pbar.update(1)

        pbar.close()

        # Compute statistics
        metrics = self._compute_metrics(results)

        return {
            "results": results,
            "metrics": metrics,
            "evaluation_date": datetime.now().isoformat(),
            "judge_model": self.judge_model,
            "num_problems": len(results)
        }

    def _build_evaluation_prompt(
        self,
        problem: Dict[str, Any],
        reference: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the evaluation prompt."""
        prompt = f"""You are a professional mathematics problem evaluation expert. Please evaluate the quality of the following AIME-style math problem.

[Problem to Evaluate]
Problem: {problem.get('problem', '')}
Answer: {problem.get('answer', '')}
Solution: {problem.get('solution', '')}
"""

        if reference:
            prompt += f"""
[Reference Problem (AIME Real)]
Problem: {reference.get('problem', '')}
Answer: {reference.get('answer', '')}
Solution: {reference.get('solution', '')}
"""

        prompt += """
Please evaluate the problem quality on the following four dimensions (1-5 points each):

1. **Correctness**: Is the mathematical logic correct? Is the answer accurate?
2. **Clarity**: Is the problem statement clear? Is the solution easy to understand?
3. **Difficulty Match**: Does the difficulty match AIME standards (6-9/15)?
4. **Completeness**: Are the solution steps complete? Does it include necessary reasoning?

Please output the scores in the following JSON format:
```json
{
    "correctness": 5,
    "clarity": 4,
    "difficulty_match": 4,
    "completeness": 5,
    "comments": "Detailed evaluation..."
}
```
"""
        return prompt

    def _parse_evaluation_response(self, response: str) -> Dict[str, float]:
        """Parse the LLM evaluation response."""
        try:
            # Extract JSON portion
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            # Parse JSON
            data = json.loads(json_str)

            # Extract scores
            scores = {}
            for dim in self.EVALUATION_DIMENSIONS:
                scores[dim] = float(data.get(dim, 3.0))  # Default to 3.0

            return scores

        except Exception as e:
            print(f"[Warning] Failed to parse evaluation response: {e}")
            # Return default scores
            return {dim: 3.0 for dim in self.EVALUATION_DIMENSIONS}

    def _compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        if not results:
            return {}

        # Compute average score per dimension
        dimension_scores = {dim: [] for dim in self.EVALUATION_DIMENSIONS}
        total_scores = []

        for result in results:
            total_scores.append(result["total_score"])
            for dim in self.EVALUATION_DIMENSIONS:
                dimension_scores[dim].append(result["scores"][dim])

        metrics = {
            "average_total_score": sum(total_scores) / len(total_scores),
            "dimension_averages": {
                dim: sum(scores) / len(scores)
                for dim, scores in dimension_scores.items()
            },
            "pass_rate": sum(1 for s in total_scores if s >= 3.5) / len(total_scores),
            "excellent_rate": sum(1 for s in total_scores if s >= 4.5) / len(total_scores)
        }

        return metrics

    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """Export evaluation results."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[Done] Evaluation results saved to: {output_path}")
