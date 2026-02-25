"""
Win Rate Evaluator

Computes win rates through pairwise comparisons.
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from code.core.llm import HelloAgentsLLM


class WinRateEvaluator:
    """Win Rate evaluator"""

    def __init__(
        self,
        llm: Optional[HelloAgentsLLM] = None,
        judge_model: str = "gpt-4o"
    ):
        """
        Initialize the Win Rate evaluator.

        Args:
            llm: LLM instance; if None, a new instance will be created
            judge_model: Name of the judge model
        """
        self.llm = llm or HelloAgentsLLM(model=judge_model)
        self.judge_model = judge_model

    def compare_pair(
        self,
        problem_a: Dict[str, Any],
        problem_b: Dict[str, Any],
        label_a: str = "A",
        label_b: str = "B"
    ) -> Dict[str, Any]:
        """
        Compare two problems to determine which is better.

        Args:
            problem_a: Problem A
            problem_b: Problem B
            label_a: Label for problem A
            label_b: Label for problem B

        Returns:
            Comparison result containing the winner and reasoning
        """
        start_time = time.time()

        # Build comparison prompt
        prompt = self._build_comparison_prompt(problem_a, problem_b, label_a, label_b)

        # Call LLM for comparison
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages)

        # Parse comparison result
        winner, reason = self._parse_comparison_response(response, label_a, label_b)

        execution_time = time.time() - start_time

        return {
            "problem_a_id": problem_a.get("problem_id", "unknown"),
            "problem_b_id": problem_b.get("problem_id", "unknown"),
            "winner": winner,
            "reason": reason,
            "comparison_text": response,
            "execution_time": execution_time
        }

    def evaluate_win_rate(
        self,
        generated_problems: List[Dict[str, Any]],
        reference_problems: List[Dict[str, Any]],
        num_comparisons: Optional[int] = None,
        total_samples: int = 500
    ) -> Dict[str, Any]:
        """
        Evaluate the win rate of generated data against reference data.

        Args:
            generated_problems: List of generated problems
            reference_problems: List of reference problems (e.g. AIME real problems)
            num_comparisons: Number of comparisons; if None, compares all possible pairs
            total_samples: Total number of samples for progress bar display (default: 500)

        Returns:
            Win rate evaluation results
        """
        print(f"\n[Win Rate] Starting evaluation")
        print(f"   Judge model     : {self.judge_model}")
        print(f"   Generated data  : {len(generated_problems)} problems")
        print(f"   Reference data  : {len(reference_problems)} problems")

        # Determine number of comparisons
        if num_comparisons is None:
            num_comparisons = min(len(generated_problems), len(reference_problems))

        # Limit comparisons to available generated problems and total_samples
        num_comparisons = min(num_comparisons, len(generated_problems), total_samples)

        print(f"   Num comparisons : {num_comparisons}")

        # Randomly sample generated problem indices
        import random
        gen_indices = random.sample(range(len(generated_problems)), num_comparisons)

        print(f"   Sampling method : Random")

        # Perform pairwise comparisons
        comparisons = []
        wins = 0
        losses = 0
        ties = 0

        pbar = tqdm(total=num_comparisons, desc="Comparing",
                     unit="pair",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        for i, gen_idx in enumerate(gen_indices):
            gen_problem = generated_problems[gen_idx]
            # Randomly select a reference problem
            ref_idx = random.randint(0, len(reference_problems) - 1)
            ref_problem = reference_problems[ref_idx]

            # Randomize problem order to avoid position bias
            if random.random() < 0.5:
                # Generated first
                result = self.compare_pair(
                    gen_problem,
                    ref_problem,
                    label_a="Problem A",
                    label_b="Problem B"
                )
                # Record actual order
                result["actual_order"] = {"A": "Generated", "B": "Reference"}

                # Convert winner
                if result["winner"] == "Problem A":
                    actual_winner = "Generated"
                elif result["winner"] == "Problem B":
                    actual_winner = "Reference"
                else:
                    actual_winner = "Tie"
            else:
                # Reference first
                result = self.compare_pair(
                    ref_problem,
                    gen_problem,
                    label_a="Problem A",
                    label_b="Problem B"
                )
                # Record actual order
                result["actual_order"] = {"A": "Reference", "B": "Generated"}

                # Convert winner
                if result["winner"] == "Problem A":
                    actual_winner = "Reference"
                elif result["winner"] == "Problem B":
                    actual_winner = "Generated"
                else:
                    actual_winner = "Tie"

            result["actual_winner"] = actual_winner
            comparisons.append(result)

            # Track wins/losses
            if actual_winner == "Generated":
                wins += 1
            elif actual_winner == "Reference":
                losses += 1
            else:
                ties += 1

            pbar.set_postfix(
                W=wins, L=losses, T=ties,
                wr=f"{wins/(i+1):.0%}"
            )
            pbar.update(1)

        pbar.close()

        # Compute win rate
        win_rate = wins / num_comparisons if num_comparisons > 0 else 0
        loss_rate = losses / num_comparisons if num_comparisons > 0 else 0
        tie_rate = ties / num_comparisons if num_comparisons > 0 else 0

        metrics = {
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "tie_rate": tie_rate,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "total_comparisons": num_comparisons
        }

        print(f"\n[Results] Win Rate evaluation:")
        print(f"   Win rate  : {win_rate:.2%}")
        print(f"   Loss rate : {loss_rate:.2%}")
        print(f"   Tie rate  : {tie_rate:.2%}")

        return {
            "comparisons": comparisons,
            "metrics": metrics,
            "evaluation_date": datetime.now().isoformat(),
            "judge_model": self.judge_model
        }

    def _build_comparison_prompt(
        self,
        problem_a: Dict[str, Any],
        problem_b: Dict[str, Any],
        label_a: str,
        label_b: str
    ) -> str:
        """Build the comparison prompt."""
        # Check if solution field exists
        has_solution_a = bool(problem_a.get('solution', '').strip())
        has_solution_b = bool(problem_b.get('solution', '').strip())

        # Build problem display
        problem_a_text = f"""**{label_a}**
Problem: {problem_a.get('problem', '')}
Answer: {problem_a.get('answer', '')}"""

        if has_solution_a:
            problem_a_text += f"\nSolution: {problem_a.get('solution', '')}"

        problem_b_text = f"""**{label_b}**
Problem: {problem_b.get('problem', '')}
Answer: {problem_b.get('answer', '')}"""

        if has_solution_b:
            problem_b_text += f"\nSolution: {problem_b.get('solution', '')}"

        # Adjust evaluation criteria based on whether solutions are provided
        if has_solution_a and has_solution_b:
            criteria = """**Evaluation Criteria:**
Please evaluate comprehensively from the following dimensions:
1. **Mathematical Correctness**: Are the problem, solution, and answer mathematically correct?
2. **Clarity**: Is the problem statement clear and unambiguous?
3. **Difficulty Appropriateness**: Does the difficulty match AIME standards (challenging but solvable)?
4. **Solution Completeness**: Is the solution complete with clear reasoning steps?"""
        else:
            criteria = """**Evaluation Criteria:**
Please evaluate comprehensively from the following dimensions:
1. **Mathematical Correctness**: Are the problem and answer mathematically correct and reasonable?
2. **Clarity**: Is the problem statement clear and unambiguous?
3. **Difficulty Appropriateness**: Does the difficulty match AIME standards (challenging but solvable)?
4. **Problem Quality**: Is the problem well-designed with appropriate complexity?

Note: Some problems may not have solutions provided. Focus on the problem statement and answer quality."""

        prompt = f"""You are a professional mathematics problem evaluator. Please compare the following two AIME-style math problems and determine which one has higher quality.

{problem_a_text}

{problem_b_text}

{criteria}

**Important Guidelines:**
- Be objective and fair in your evaluation
- Consider all dimensions equally
- If both problems are of similar quality, choose "Tie"
- Do not favor one problem just because it appears first or second
- If one problem has a solution and the other doesn't, focus on the problem statement and answer quality

Please output your judgment in the following JSON format:
```json
{{
    "winner": "{label_a}",  // or "{label_b}" or "Tie"
    "reason": "Detailed explanation of why you chose this answer, covering the evaluation dimensions..."
}}
```
"""
        return prompt

    def _parse_comparison_response(
        self,
        response: str,
        label_a: str,
        label_b: str
    ) -> Tuple[str, str]:
        """Parse the comparison response."""
        try:
            # Extract JSON portion
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            # Fix LaTeX escape issues
            import re
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Fix LaTeX escapes: convert \frac to \\frac etc.
                fixed_json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', json_str)
                data = json.loads(fixed_json_str)

            winner = data.get("winner", "Tie")
            reason = data.get("reason", "No reason provided")

            # Validate winner
            if winner not in [label_a, label_b, "Tie"]:
                winner = "Tie"

            return winner, reason

        except Exception as e:
            print(f"[Warning] Failed to parse comparison response: {e}")
            return "Tie", "Failed to parse response"

    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """Export evaluation results."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[Done] Win rate results saved to: {output_path}")
